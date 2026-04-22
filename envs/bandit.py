"""
PL1 — K-Armed Bandit Environment
Supports stationary and non-stationary (random walk) variants.
"""
from __future__ import annotations

import numpy as np


class KArmedBandit:
    """Multi-armed bandit testbed (Sutton & Barto, Chapter 2)."""

    def __init__(self, k: int = 10, stationary: bool = True, walk_std: float = 0.01):
        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std
        self.reset()

    def reset(self) -> None:
        self.q_true = np.random.randn(self.k)          # true action values
        self.optimal_action = np.argmax(self.q_true)

    def step(self, action: int) -> float:
        reward = np.random.randn() + self.q_true[action]

        # Non-stationary random walk
        if not self.stationary:
            self.q_true += np.random.normal(0, self.walk_std, self.k)
            self.optimal_action = np.argmax(self.q_true)

        return reward
