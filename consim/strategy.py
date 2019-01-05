from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

from .util import best_weights, distribute_equal_weight


class StepContext:
    def __init__(self, initial_visits, initial_conversions, max_weight):
        self.total_visits = initial_visits.copy()
        self.total_conversions = initial_conversions.copy()
        self.last_visits = None
        self.last_conversions = None
        self.time = 0
        self.max_weight = max_weight
        self.N = len(max_weight)
        self.equal_weights = distribute_equal_weight(max_weight)

    def step(self, visits, conversions):
        self.last_visits = visits
        self.last_conversions = conversions
        self.time += 1
        self.total_visits += visits
        self.total_conversions += conversions

    @property
    def total_cr(self):
        return self.total_conversions / np.clip(self.total_visits, 1, None)

    def select_best(self, scores=None):
        if scores is None:
            scores = self.total_cr
        return best_weights(self.max_weight, scores)


class BaseStrategy(ABC):
    @abstractmethod
    def step(self, ctx: StepContext):
        pass


class NaiveStrategy(BaseStrategy):
    def __init__(self, v_threshold):
        self.v_threshold = v_threshold
        self.final_weights = None

    def step(self, ctx: StepContext):
        if self.final_weights is None:
            if ctx.total_visits.mean() > self.v_threshold:
                self.final_weights = ctx.select_best()
            else:
                return ctx.equal_weights
        return self.final_weights


class HalvingStrategy(BaseStrategy):
    def __init__(self, v_threshold):
        self.next_threshold = v_threshold
        self.current_weights = None
        self.finished = False

    def step(self, ctx: StepContext):
        if self.current_weights is None:
            self.current_weights = ctx.equal_weights
        mask = self.current_weights > 0
        if not self.finished and np.mean(ctx.total_visits[mask]) > self.next_threshold:
            self.next_threshold = self.next_threshold * 2
            n_retain = max(1, mask.sum() // 2)
            best_idx = np.argsort(-ctx.total_cr)[:n_retain]
            if ctx.max_weight[best_idx].sum() <= 1:
                self.current_weights = ctx.select_best()
                self.finished = True
            else:
                best_weights = distribute_equal_weight(ctx.max_weight[best_idx])
                new_weights = np.zeros_like(ctx.max_weight)
                new_weights[best_idx] = best_weights
                self.current_weights = new_weights
        return self.current_weights


class EpsilonGreedyTopKStrategy(BaseStrategy):
    def __init__(self, epsilon, decay_power=1):
        self.epsilon = epsilon
        self.decay_power = decay_power

    def step(self, ctx: StepContext):
        if np.array_equiv(ctx.max_weight, ctx.max_weight[0]):
            k = int(np.round(1 / ctx.max_weight[0]))
        else:
            raise RuntimeError("All max weights should have same value")
        epsilon = min(1, self.epsilon / np.power((ctx.time + 1), self.decay_power))
        leader_weight = (1 - epsilon) / k
        non_leader_weight = epsilon / (ctx.N - k)
        if leader_weight > non_leader_weight:
            weights = np.full(ctx.N, non_leader_weight)
            # Indexes of retained sources, get last k
            idx = np.argpartition(ctx.total_cr, ctx.N - k)[-k:]
            weights[idx] = leader_weight
        else:
            weights = ctx.equal_weights
        return weights


class EpsilonGreedyStrategy(BaseStrategy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def step(self, ctx: StepContext):
        epsilon = min(1, self.epsilon / (ctx.time + 1))
        used_budget = 0
        remaining_budget = 1 - epsilon
        weights = np.empty_like(ctx.max_weight)
        idx = np.argsort(-ctx.total_cr)
        for i in range(len(idx)):
            ew = distribute_equal_weight(ctx.max_weight[idx[i:]], 1 - used_budget)
            if ew[0] > remaining_budget:
                weights[idx[i:]] = ew
                break
            else:
                mw = ctx.max_weight[idx[i]]
                to_distribute = min(remaining_budget, mw)
                weights[idx[i]] = to_distribute
                remaining_budget -= to_distribute
                used_budget += to_distribute
        return weights


class SoftmaxStrategy(BaseStrategy):
    def __init__(self, init_temp=10):
        self.init_temp = init_temp

    def step(self, ctx: StepContext):
        # temp = self.init_temp/self.time * np.log(self.time + 1)
        temp = self.init_temp / (ctx.time + 1) / 90

        def calc_w(r, budget, max_weight):
            logw = r - logsumexp(r)
            w = np.exp(logw) * budget
            overflow = w > max_weight
            if np.any(overflow):
                good = ~overflow
                budget_remain = budget - max_weight[overflow].sum()
                w[good] = calc_w(r[good], budget_remain, max_weight[good])
                w[overflow] = max_weight[overflow]
            return w

        rates = ctx.total_cr / temp
        result = calc_w(rates, 1, ctx.max_weight)
        return result


class CBRacingStrategy(BaseStrategy):
    def __init__(self, a, b, quantile):
        self.quantile = quantile
        self.a_prior = a
        self.b_prior = b

    def step(self, ctx: StepContext):
        failures = ctx.total_visits - ctx.total_conversions
        successes = ctx.total_conversions
        quantiles = np.array([self.quantile, 1 - self.quantile])[:, None]

        lower, upper = beta.ppf(quantiles,
                                (self.a_prior + successes)[None, :], (self.b_prior + failures)[None, :])
        # lower = beta.ppf(self.quantile, self.a_prior + successes, self.b_prior + failures)
        # upper = beta.ppf(1 - self.quantile, self.a_prior + successes, self.b_prior + failures)
        n = len(lower)
        # Если у источника верхняя граница credible interval меньше, чем нижняя граница любого другого, исключаем его
        gap_matrix = lower[:, None] - upper[None, :]
        bad_mask = np.any(gap_matrix > 0, axis=0)
        # Не исключаем самые переспективные источники, в противном случае
        # можем не распределить весь бюджет
        top_items = best_weights(ctx.max_weight, upper) > 0
        bad_mask = bad_mask & ~top_items

        weights = np.zeros(n)
        good_mask = ~bad_mask
        weights[good_mask] = distribute_equal_weight(ctx.max_weight[good_mask])
        return weights
