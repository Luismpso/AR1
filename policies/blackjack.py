"""PL4 — Threshold Policy for Blackjack."""
from __future__ import annotations

from AR1.core.base import Policy
from AR1.envs.blackjack import BlackjackAction, BlackjackState


class ThresholdPolicy(Policy[BlackjackState, BlackjackAction]):
    """Hit below threshold, stick otherwise."""

    def __init__(self, threshold: int = 20):
        self.threshold = threshold

    def select_action(self, state: BlackjackState) -> BlackjackAction:
        player_sum, _, _ = state
        return "hit" if player_sum < self.threshold else "stick"
