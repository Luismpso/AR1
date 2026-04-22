"""
PL2/PL3 — Gridworld MDP Utilities
Helper functions for value arrays and policies.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from AR1.envs.gridworld import Gridworld, ACTIONS


def zeros_V(env: Gridworld) -> np.ndarray:
    """Initialize state-value array to zeros."""
    return np.zeros((env.n_rows, env.n_cols), dtype=float)


def zeros_Q(env: Gridworld) -> np.ndarray:
    """Initialize action-value array to zeros: Q[r, c, a_index]."""
    return np.zeros((env.n_rows, env.n_cols, len(ACTIONS)), dtype=float)


def all_states(env: Gridworld) -> List[Tuple[int, int]]:
    return env.states()


def uniform_random_policy(env: Gridworld) -> Dict[Tuple[int, int], Dict[str, float]]:
    """π(a|s) = 1/4 for all non-terminal states."""
    policy = {}
    for s in env.states():
        if env.is_terminal(s):
            policy[s] = {a: 0.0 for a in ACTIONS}
        else:
            policy[s] = {a: 1.0 / len(ACTIONS) for a in ACTIONS}
    return policy
