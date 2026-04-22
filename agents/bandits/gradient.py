"""PL1 — Gradient Bandit Agent (Softmax preferences with SGD)."""
from __future__ import annotations

import numpy as np

from AR1.core.base import BanditAgent


class GradientBandit(BanditAgent):
    def __init__(self, k: int = 10, alpha: float = 0.1, baseline: bool = True):
        self.alpha = alpha
        self.baseline = baseline
        super().__init__(k)

    def reset(self) -> None:
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0
        self.H = np.zeros(self.k)       # Numerical preferences
        self.avg_reward = 0.0

    def _policy(self) -> np.ndarray:
        """Numerically stable softmax."""
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self) -> int:
        probs = self._policy()
        return np.random.choice(self.k, p=probs)

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        probs = self._policy()

        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline_val = self.avg_reward
        else:
            baseline_val = 0

        # Stochastic Gradient Ascent
        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline_val) * (1 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline_val) * probs[a]
