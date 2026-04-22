"""PL1 — UCB (Upper Confidence Bound) Agent."""
from __future__ import annotations

import numpy as np

from AR1.core.base import BanditAgent


class UCB(BanditAgent):
    def __init__(self, k: int = 10, c: float = 2.0):
        self.c = c
        super().__init__(k)

    def reset(self) -> None:
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        # If any action has not been tried yet, select it
        for a in range(self.k):
            if self.N[a] == 0:
                return a

        # UCB: Q(a) + c * sqrt(ln(t) / N(a))
        uncertainty = self.c * np.sqrt(np.log(self.t) / self.N)
        ucb_values = self.Q + uncertainty

        max_val = np.max(ucb_values)
        return np.random.choice(np.where(ucb_values == max_val)[0])

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
