from typing import Collection, Callable
from typing import NamedTuple

import numba
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange, tnrange

from .rnd import rand
from .strategy import BaseStrategy
from .util import distribute_equal_weight


class SourceParams(NamedTuple):
    max_weight: np.ndarray
    min_weight: np.ndarray
    initial_weight: np.ndarray
    initial_visits: np.ndarray
    initial_conversions: np.ndarray
    true_conversion: Callable[[], np.ndarray]


def generate_top_k(n, k, cr_sampler: Callable[[], np.ndarray]):
    return SourceParams(
        max_weight=np.full(n, 1 / k),
        min_weight=np.zeros(n),
        initial_weight=np.full(n, 1 / n),
        initial_visits=np.zeros(n, dtype=np.int32),
        initial_conversions=np.zeros(n, dtype=np.int32),
        true_conversion=cr_sampler
    )


def proportional_weights(conv_rates, cr_sampler: Callable[[], np.ndarray], cr_expectation, mean_max_weight=0.08):
    n = len(conv_rates)
    max_weight = 1 / conv_rates * cr_expectation * mean_max_weight
    initial_weight = distribute_equal_weight(max_weight)

    return SourceParams(
        max_weight=max_weight,
        min_weight=np.zeros(n),
        initial_weight=initial_weight,
        initial_visits=np.zeros(n, dtype=np.int32),
        initial_conversions=np.zeros(n, dtype=np.int32),
        true_conversion=cr_sampler
    )


class SimulationResult(NamedTuple):
    total_efficiency: np.ndarray
    last_efficiency: np.ndarray
    total_gain: np.ndarray
    last_gain: np.ndarray
    ideal_conversion_rate: np.ndarray
    base_conversion_rate: np.ndarray
    day_conversion: np.ndarray
    day_weights: np.ndarray


class Simulator:
    def __init__(self, n_sources, visits_per_day, n_days, sp_generator: Callable[[int], SourceParams]):
        self.N = n_sources
        self.T = visits_per_day
        self.max_days = n_days
        self.sp_generator = sp_generator

    def run_simulation(self, n_rounds, strategy_generator: Callable[[], BaseStrategy],
                       n_jobs=None, day_weights=1, progress=None) -> SimulationResult:
        result = SimulationResult(
            total_efficiency=np.zeros(n_rounds),
            last_efficiency=np.zeros(n_rounds),
            total_gain=np.zeros(n_rounds),
            last_gain=np.zeros(n_rounds),
            ideal_conversion_rate=np.zeros((n_rounds, self.max_days)),
            base_conversion_rate=np.zeros((n_rounds, self.max_days)),
            day_conversion=np.zeros((n_rounds, self.max_days)),
            day_weights=np.zeros((day_weights, self.N, self.max_days)) if day_weights else None
        )
        if progress == 'console':
            rounds = trange(n_rounds, leave=False)
        elif progress == 'notebook':
            rounds = tnrange(n_rounds, leave=False)
        else:
            rounds = range(n_rounds)
        results: Collection[RoundResult] = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(simulation_round)
                                                                                              (self.N, self.max_days,
                                                                                               self.T,
                                                                                               strategy_generator(),
                                                                                               self.sp_generator(
                                                                                                   self.N),
                                                                                               day_weights > 0) for _ in
                                                                                              rounds)
        for i, r in enumerate(results):
            # Сколько конверсий на одну единицу трафика за все дни
            sum_ideal, sum_real, sum_base = r.ideal_cr.sum(), r.real_cr.sum(), r.base_cr.sum()
            result.last_efficiency[i] = (1 - (r.ideal_cr[-1] - r.last_cr) / r.ideal_cr[-1]) * 100
            # Какой процент от макс. возможного кол-ва конверсий получился
            result.total_efficiency[i] = (1 - (sum_ideal - sum_real) / sum_ideal) * 100
            result.total_gain[i] = ((sum_real - sum_base) / (sum_ideal - sum_base)) * 100
            result.last_gain[i] = ((r.last_cr - r.base_cr[-1]) / (r.ideal_cr[-1] - r.base_cr[-1])) * 100
            result.ideal_conversion_rate[i] = r.ideal_cr
            result.base_conversion_rate[i] = r.base_cr
            result.day_conversion[i] = r.real_cr
            if day_weights > i:
                result.day_weights[i, :, :] = r.day_weights
        return result


class RoundResult(NamedTuple):
    # Максимально возможная латентная конверсия по дням (по весам лучших источников каждый день)
    ideal_cr: np.ndarray
    # Средняя латентная конверсия по дням по всем источникам
    base_cr: np.ndarray
    # Средняя латентная конверсия по весам последнего дня
    last_cr: float
    # Реально наблюдаемая конверсия по дням
    real_cr: np.ndarray
    day_weights: np.ndarray


@numba.njit(fastmath=True)
def calc_ideal_cr(latent_cr, max_weights):
    """
    :param latent_cr: [n_days, n_sources]
    :param max_weights: [n_sources]
    :return: [n_days]
    """
    n_days = latent_cr.shape[0]
    n_sources = latent_cr.shape[1]
    best_weights = np.zeros_like(latent_cr)
    for d in numba.prange(n_days):
        sum_best_weight = 0
        best_idx = np.argsort(-latent_cr[d])
        for s in range(n_sources):
            idx = best_idx[s]
            w = max_weights[idx]
            if (sum_best_weight + w) > 1:
                w = 1 - sum_best_weight
            sum_best_weight += w
            best_weights[d, idx] = w
            if np.abs(sum_best_weight - 1) < 1e-5:
                break
    ideal_cr = (latent_cr * best_weights).sum(axis=1)
    return ideal_cr


def simulation_round(n, max_days, visits_per_day, strategy, sp: SourceParams, store_day_weights: bool = False):
    from .strategy import StepContext
    weights = sp.initial_weight.copy()
    initial_weights = weights.copy()
    observed_cr = np.zeros(max_days)
    day_weights = np.zeros((n, max_days), dtype=np.float32)
    latent_cr = np.empty((max_days, n))
    context = StepContext(sp.initial_visits, sp.initial_conversions, sp.max_weight)
    for day in range(max_days):
        visits = rand.poisson(weights * visits_per_day)
        conv_rates = sp.true_conversion()
        latent_cr[day] = conv_rates
        # noinspection PyTypeChecker
        conversions = rand.binomial(visits, conv_rates)
        # If poisson and binomial samplers works correctly, then no real need to calculate CR from samples.
        # We can calculate it from latent values to reduce variance.
        # observed_cr[day] = conversions.sum() / visits.sum()
        observed_cr[day] = (weights * conv_rates).sum()
        context.step(visits, conversions)
        step_results = strategy.step(context)
        if step_results is not None:
            assert np.allclose(step_results.sum(), 1)
            weights = step_results
        if store_day_weights:
            day_weights[:, day] = weights
    base_cr = (latent_cr * initial_weights[None, :]).sum(axis=1)
    result = RoundResult(
        ideal_cr=calc_ideal_cr(latent_cr, sp.max_weight),
        base_cr=base_cr,
        last_cr=(weights * latent_cr[-1]).sum(),
        real_cr=observed_cr,
        day_weights=day_weights if store_day_weights else None
    )
    return result
