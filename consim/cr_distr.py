from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from .rnd import rand


class ConversionDistribution(ABC):
    def ppf(self, quantiles):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, sample_size):
        raise NotImplementedError()


class LogNormalConversion(ConversionDistribution):
    def __init__(self, median, sigma, cutoff=.5):
        self.sigma = sigma
        self.median = median
        mu = np.log(median)
        self.rv = norm(loc=mu, scale=sigma)
        # Analytical mean of lognormal distribution
        self.mean = self.median_to_mean(median, sigma)
        self.cutoff = cutoff

    def sample(self, sample_size):
        return np.clip(np.exp(self.rv.rvs(sample_size)), None, self.cutoff)

    def ppf(self, quantiles):
        return np.exp(self.rv.ppf(quantiles))

    @staticmethod
    def mean_to_median(mean, sigma):
        # mean = np.exp(np.log(median) + sigma**2 / 2)
        # log_mean = log_median + s
        # log_median = log_mean - s
        return np.exp(np.log(mean) - sigma ** 2 / 2)

    @staticmethod
    def median_to_mean(median, sigma):
        return np.exp(np.log(median) + sigma ** 2 / 2)

    @classmethod
    def from_mean(cls, mean, sigma, cutoff=0.5):
        median = cls.mean_to_median(mean, sigma)
        return cls(median, sigma, cutoff)


class ConstantConversion:
    def __init__(self, initial_state):
        self.current_state = initial_state

    def __call__(self):
        return self.current_state


class RandomDrift:
    def __init__(self, initial_state, eta=0.01, sigma=0.05, clip_max=.15):
        self.eta = eta
        self.sigma = sigma
        self.original_values = np.log(initial_state)
        self.current_state = self.original_values
        self.clip_max = clip_max

    def __call__(self):
        drift = rand.normal((self.original_values - self.current_state) * self.eta, self.sigma)
        self.current_state = np.clip(self.current_state + drift, None, self.clip_max)
        return np.exp(self.current_state)


class DecayingConversion:
    def __init__(self, initial_state, n_days, lamb=1):
        self.current_state = initial_state
        self.lamb = lamb
        self.n_days = n_days
        decays = rand.exponential(self.lamb, size=len(initial_state)) + 1
        self.decay_k = np.power(1 / decays, 1 / self.n_days)
        self.decay_k += 0

    def __call__(self):
        self.current_state = self.current_state * self.decay_k
        return self.current_state
