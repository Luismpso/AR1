"""PL1 — ε-Greedy Agent."""
from __future__ import annotations

import numpy as np

from AR1.core.base import BanditAgent


class EpsilonGreedy(BanditAgent):
    def __init__(self, k: int = 10, epsilon: float = 0.1, alpha: float | None = None, optimistic: float = 0.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic = optimistic
        super().__init__(k)

    def reset(self) -> None:
        self.Q = np.full(self.k, self.optimistic, dtype=float)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        max_q = np.max(self.Q)
        return np.random.choice(np.where(self.Q == max_q)[0])

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.N[action] += 1
        if self.alpha is None:
            # Sample average
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            # Constant step-size (for non-stationary problems)
            self.Q[action] += self.alpha * (reward - self.Q[action])
