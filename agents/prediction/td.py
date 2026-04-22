"""PL4 — TD(0) Prediction (Temporal-Difference with 1-step bootstrap)."""
from __future__ import annotations

from collections import defaultdict

from AR1.core.base import Episode, PredictionAgent
from AR1.envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update V using TD(0): V(S) += α [R + γ V(S') − V(S)]."""

        for transition in episode.transitions:
            state = transition.state
            reward = transition.reward
            next_state = transition.next_state

            # Bootstrap value (0 if terminal)
            v_next = 0.0 if transition.done else self.V[next_state]

            # TD target and update
            td_target = reward + self.gamma * v_next
            td_error = td_target - self.V[state]
            self.V[state] += self.alpha * td_error

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
