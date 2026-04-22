"""PL1 — Thompson Sampling (Bayesian bandit agent)."""
from __future__ import annotations

import numpy as np

from AR1.core.base import BanditAgent


class ThompsonSampling(BanditAgent):
    def __init__(self, k: int = 10):
        super().__init__(k)

    def reset(self) -> None:
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0
        self.mu = np.zeros(self.k)      # Estimated mean
        self.prec = np.ones(self.k)     # Precision (1/variance)

    def select_action(self) -> int:
        samples = np.random.normal(self.mu, 1.0 / np.sqrt(self.prec))
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        # Bayesian update (normal conjugate)
        old_prec = self.prec[action]
        self.prec[action] += 1.0
        self.mu[action] = (old_prec * self.mu[action] + reward) / self.prec[action]
