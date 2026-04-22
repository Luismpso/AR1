"""PL4 — First-Visit Monte Carlo Prediction."""
from __future__ import annotations

from collections import defaultdict

from AR1.core.base import Episode, PredictionAgent
from AR1.envs.blackjack import BlackjackAction, BlackjackState


class FirstVisitMonteCarloPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def reset(self) -> None:
        self.V = defaultdict(float)
        self.N = defaultdict(int)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update V using first-visit Monte Carlo prediction."""

        # 1. Identify FIRST visit to each state
        first_visits = {}
        for t, transition in enumerate(episode.transitions):
            if transition.state not in first_visits:
                first_visits[transition.state] = t

        G = 0.0

        # 2. Walk the episode backwards
        for t in range(len(episode.transitions) - 1, -1, -1):
            transition = episode.transitions[t]
            G = transition.reward + G

            # 3. If this is the first visit to this state, update V(S)
            if first_visits[transition.state] == t:
                state = transition.state
                self.N[state] += 1
                # Incremental mean update
                self.V[state] += (G - self.V[state]) / self.N[state]

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
