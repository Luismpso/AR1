"""
PL2/PL3 — Gridworld Environments
Includes deterministic, trap, and stochastic variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

ACTIONS = ["U", "D", "L", "R"]

ACTION_TO_DELTA = {
    "U": (-1, 0),
    "D": ( 1, 0),
    "L": ( 0,-1),
    "R": ( 0, 1),
}


@dataclass(frozen=True)
class Gridworld:
    """Deterministic 4×4 gridworld (Sutton & Barto, Example 4.1)."""
    n_rows: int = 4
    n_cols: int = 4
    terminal_states: Tuple[Tuple[int, int], ...] = ((0, 0), (3, 3))
    step_reward: float = -1.0

    def states(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        return state in self.terminal_states

    def step(self, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float, bool]:
        if self.is_terminal(state):
            return state, 0.0, True

        # Get row and column delta based on the action
        delta_r, delta_c = ACTION_TO_DELTA[action]
        new_r = state[0] + delta_r
        new_c = state[1] + delta_c

        # Check if the new position is outside the grid bounds
        if new_r < 0 or new_r >= self.n_rows or new_c < 0 or new_c >= self.n_cols:
            next_state = state            # hit the wall -> stay in place
        else:
            next_state = (new_r, new_c)

        reward = self.step_reward
        done_flag = self.is_terminal(next_state)
        return next_state, reward, done_flag


class GridworldTrap(Gridworld):
    """Gridworld variant where stepping into cell (1,2) gives reward -10 (Exercise B)."""

    def step(self, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float, bool]:
        if self.is_terminal(state):
            return state, 0.0, True

        delta_r, delta_c = ACTION_TO_DELTA[action]
        new_r = state[0] + delta_r
        new_c = state[1] + delta_c

        if new_r < 0 or new_r >= self.n_rows or new_c < 0 or new_c >= self.n_cols:
            next_state = state
        else:
            next_state = (new_r, new_c)

        # Trap reward
        if next_state == (1, 2):
            reward = -10.0
        else:
            reward = self.step_reward

        done_flag = self.is_terminal(next_state)
        return next_state, reward, done_flag


# ============================================================
# Stochastic transitions (Exercise C)
# ============================================================

def get_stochastic_transitions(
    env: Gridworld, state: Tuple[int, int], action: str
) -> List[Tuple[float, Tuple[int, int], float]]:
    """Return list of (probability, next_state, reward) for stochastic dynamics.

    With prob 0.8 the intended action happens, with 0.1 each the agent slips
    to a perpendicular direction.
    """
    if env.is_terminal(state):
        return [(1.0, state, 0.0)]

    slips = {
        "U": ["L", "R"],
        "D": ["R", "L"],
        "L": ["D", "U"],
        "R": ["U", "D"],
    }

    transitions = []

    # Intended movement (probability 0.8)
    ns, r, _ = env.step(state, action)
    transitions.append((0.8, ns, r))

    # Slips (probability 0.1 each)
    for slip_action in slips[action]:
        ns_slip, r_slip, _ = env.step(state, slip_action)
        transitions.append((0.1, ns_slip, r_slip))

    return transitions
