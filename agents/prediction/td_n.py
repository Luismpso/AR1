"""PL4 — n-step TD Prediction (generalization of TD(0) with n-step returns)."""
from __future__ import annotations

from collections import defaultdict

from AR1.core.base import Episode, PredictionAgent
from AR1.envs.blackjack import BlackjackAction, BlackjackState


class TDnPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, n: int, alpha: float = 0.05, gamma: float = 1.0):
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update V using n-step TD returns."""
        T = len(episode.transitions)

        for tau in range(T):
            state_tau = episode.transitions[tau].state

            # 1. Sum discounted rewards up to step tau + n
            G = 0.0
            max_step = min(tau + self.n, T)

            for i in range(tau, max_step):
                G += (self.gamma ** (i - tau)) * episode.transitions[i].reward

            # 2. Bootstrap if the window does not reach the end of the episode
            last_transition_idx = max_step - 1
            if tau + self.n <= T and not episode.transitions[last_transition_idx].done:
                next_state = episode.transitions[last_transition_idx].next_state
                G += (self.gamma ** self.n) * self.V[next_state]

            # 3. Update
            td_error = G - self.V[state_tau]
            self.V[state_tau] += self.alpha * td_error

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
