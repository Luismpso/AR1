"""PL1 — Exp3 (Exponential-weight algorithm for adversarial bandits)."""
from __future__ import annotations

import numpy as np

from AR1.core.base import BanditAgent


class Exp3(BanditAgent):
    def __init__(self, k: int = 10, gamma: float = 0.1):
        self.gamma = gamma
        super().__init__(k)

    def reset(self) -> None:
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.t = 0
        self.weights = np.ones(self.k)

    def _get_probs(self) -> np.ndarray:
        total_weight = np.sum(self.weights)
        return (1 - self.gamma) * (self.weights / total_weight) + (self.gamma / self.k)

    def select_action(self) -> int:
        probs = self._get_probs()
        return np.random.choice(self.k, p=probs)

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        probs = self._get_probs()

        # Sigmoid to map Gaussian reward to (0, 1)
        scaled_reward = 1.0 / (1.0 + np.exp(-reward))

        # Importance sampling
        estimated_reward = scaled_reward / probs[action]

        # Multiplicative update
        self.weights[action] *= np.exp(self.gamma * estimated_reward / self.k)

        # Overflow protection
        if np.any(self.weights > 1e100):
            self.weights /= np.max(self.weights)
