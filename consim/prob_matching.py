import numba
import numpy as np

from .rnd import rand
from .strategy import BaseStrategy, StepContext
from .util import best_weights


@numba.njit(fastmath=True, nogil=False)
def multipass_select(x, limit, budget, bad_source_mask, power=1.0):
    n_samples = x.shape[0]
    n_sources = x.shape[1]
    counts = np.zeros(n_sources, dtype=np.int64)
    good_sources = (~bad_source_mask).nonzero()[0]
    for i in range(n_samples):
        mx = 0.0
        max_idx = 0
        for s in good_sources:
            if x[i, s] > mx:
                mx = x[i, s]
                max_idx = s
        counts[max_idx] += 1
    weights = np.power(counts.astype(np.float64) / counts.sum(), power)
    weights = weights / weights.sum() * budget
    mask = weights > limit
    if np.any(mask):
        good_mask = ~mask
        remaining_budget = budget - limit[mask].sum()
        new_weights = multipass_select(x, limit, remaining_budget, mask | bad_source_mask, power)
        weights[mask] = limit[mask]
        weights[good_mask] = new_weights[good_mask]
    return weights


def max_select(x, limit):
    n_samples = x.shape[0]
    n_sources = x.shape[1]
    selects = np.argmax(x, axis=1)
    counts = np.bincount(selects, minlength=n_sources)
    weights = best_weights(limit, counts)
    return weights


class ProbMatchingStrategy(BaseStrategy):
    def __init__(self, a_prior, b_prior, sample_size=2048, sampler='cupy', power=1, decay=1):
        self.a_prior = a_prior
        self.b_prior = b_prior
        self.sample_size = sample_size
        self.sampler = sampler
        self.power = power
        self.decay = decay
        self.prev_negatives = 0
        self.prev_positives = 0

    def prepare_ab(self, ctx: StepContext):
        if self.decay == 1 or self.decay is None:
            total_positives = ctx.total_conversions
            total_negatives = ctx.total_visits - total_positives
        else:
            positives = ctx.last_conversions
            negatives = ctx.last_visits - positives
            total_negatives = self.prev_negatives * self.decay + negatives
            total_positives = self.prev_positives * self.decay + positives
            self.prev_positives = total_positives
            self.prev_negatives = total_negatives
        a = total_positives + self.a_prior
        b = total_negatives + self.b_prior - self.a_prior
        return a, b

    def step(self, ctx: StepContext):
        a, b = self.prepare_ab(ctx)
        n = len(a)
        sample_size = (self.sample_size, n)
        if self.sampler == 'cupy':
            import cupy as cp
            cp.random.seed()
            x = cp.random.beta(a, b, size=sample_size, dtype=cp.float32).get()
        elif self.sampler == 'mkl':
            x = rand.beta(a, b, sample_size)
        elif self.sampler == 'numpy':
            x = np.random.beta(a, b, sample_size)
        else:
            raise RuntimeError('Unsupported sampler ' + self.sampler)
        if self.power == np.inf:
            weights = max_select(x, ctx.max_weight)
        else:
            weights = multipass_select(x, ctx.max_weight, 1.0, np.zeros(n, dtype=np.bool_), self.power)
        return weights
