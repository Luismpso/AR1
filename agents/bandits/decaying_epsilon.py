"""PL1 — Decaying ε-Greedy (exponential decay from full exploration to near-greedy)."""
from __future__ import annotations

from AR1.agents.bandits.epsilon_greedy import EpsilonGreedy


class DecayingEpsilonGreedy(EpsilonGreedy):
    def __init__(
        self,
        k: int = 10,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995,
    ):
        super().__init__(k, epsilon=initial_epsilon)
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_action(self) -> int:
        action = super().select_action()

        # Decaimento exponencial do epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate
        return action
