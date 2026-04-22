"""
PL3.2 — Jack's Car Rental MDP (Sutton & Barto, Example 4.2)
Stochastic continuing MDP with Poisson-distributed requests and returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class CarRentalParams:
    max_cars_1: int = 20
    max_cars_2: int = 20
    max_moveable: int = 5
    revenue_per_rental: float = 10.0
    cost_per_moved: float = 2.0
    # lambdas: (req1, req2, ret1, ret2)
    lambdas: Tuple[float, float, float, float] = (3.0, 4.0, 3.0, 2.0)
    # Truncation caps for Poisson r.v.s (last bucket is a tail bucket)
    max_requests_1: int = 8
    max_requests_2: int = 10
    max_returns_1: int = 8
    max_returns_2: int = 8


# ============================================================
# Poisson helper
# ============================================================

def poisson_pmf_truncated(lam: float, max_k: int) -> np.ndarray:
    """Poisson(lam) pmf over k=0..max_k; last bucket includes tail mass P(K>=max_k)."""
    probs = np.zeros(max_k + 1, dtype=float)
    p0 = np.exp(-lam)
    probs[0] = p0
    for k in range(1, max_k):
        probs[k] = probs[k - 1] * lam / k
    probs[max_k] = max(0.0, 1.0 - probs[:max_k].sum())
    probs /= probs.sum()
    return probs


# ============================================================
# MDP
# ============================================================

class CarRentalMDP:
    def __init__(self, params: CarRentalParams):
        self.params = params

        # Precompute truncated distributions (with tail bucket)
        self.req1 = poisson_pmf_truncated(params.lambdas[0], params.max_requests_1)
        self.req2 = poisson_pmf_truncated(params.lambdas[1], params.max_requests_2)
        self.ret1 = poisson_pmf_truncated(params.lambdas[2], params.max_returns_1)
        self.ret2 = poisson_pmf_truncated(params.lambdas[3], params.max_returns_2)

        # Cache: (loc_id, cars_available_after_move) -> (p_next_cars, expected_rentals)
        self._loc_cache: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

    def states(self) -> List[Tuple[int, int]]:
        return [
            (i, j)
            for i in range(self.params.max_cars_1 + 1)
            for j in range(self.params.max_cars_2 + 1)
        ]

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        return False  # Continuing task

    def possible_actions(self, s: Tuple[int, int]) -> List[int]:
        """Actions bounded by max_moveable AND available cars / capacity."""
        n1, n2 = s
        a_min = -min(self.params.max_moveable, n2, self.params.max_cars_1 - n1)
        a_max = min(self.params.max_moveable, n1, self.params.max_cars_2 - n2)
        return list(range(a_min, a_max + 1))

    def _loc_outcomes(self, loc_id: int, cars_after_move: int) -> Tuple[np.ndarray, float]:
        """For one location: (distribution over next cars, expected rentals)."""
        key = (loc_id, cars_after_move)
        if key in self._loc_cache:
            return self._loc_cache[key]

        if loc_id == 1:
            req, ret, cap = self.req1, self.ret1, self.params.max_cars_1
        else:
            req, ret, cap = self.req2, self.ret2, self.params.max_cars_2

        p_next = np.zeros(cap + 1, dtype=float)
        exp_rented = 0.0

        for k_req, p_req in enumerate(req):
            rented = min(cars_after_move, k_req)
            exp_rented += p_req * rented
            cars_left = cars_after_move - rented

            for k_ret, p_ret in enumerate(ret):
                next_cars = min(cap, cars_left + k_ret)
                p_next[next_cars] += p_req * p_ret

        p_next /= p_next.sum()
        self._loc_cache[key] = (p_next, exp_rented)
        return p_next, exp_rented

    def after_move(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        n1, n2 = s
        return (n1 - a, n2 + a)

    def expected_transition(
        self, s: Tuple[int, int], a: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (p_next_1, p_next_2, expected_revenue)."""
        if a not in self.possible_actions(s):
            raise ValueError("Illegal action")

        n1m, n2m = self.after_move(s, a)
        p_next_1, e_rent1 = self._loc_outcomes(1, n1m)
        p_next_2, e_rent2 = self._loc_outcomes(2, n2m)

        exp_revenue = (e_rent1 + e_rent2) * self.params.revenue_per_rental
        return p_next_1, p_next_2, exp_revenue
