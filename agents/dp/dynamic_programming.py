"""
PL2/PL3 — Dynamic Programming Algorithms
Policy Evaluation, Value Iteration, Policy Iteration for Gridworld and Car Rental.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from AR1.envs.gridworld import Gridworld, ACTIONS, get_stochastic_transitions
from AR1.mdps.gridworld_mdp import zeros_V, zeros_Q


# ============================================================
# Bellman backups (Gridworld)
# ============================================================

def bellman_expectation_update(
    env: Gridworld,
    V: np.ndarray,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    state: Tuple[int, int],
    gamma: float,
) -> float:
    """Return the updated V(s) using the Bellman expectation backup."""
    if env.is_terminal(state):
        return 0.0

    v_new = 0.0
    for a in ACTIONS:
        prob = policy[state][a]
        if prob > 0:
            ns, r, done = env.step(state, a)
            v_new += prob * (r + gamma * V[ns[0], ns[1]])
    return v_new


def bellman_optimality_update(
    env: Gridworld,
    V: np.ndarray,
    s: Tuple[int, int],
    gamma: float,
) -> float:
    """Return the updated V(s) using the Bellman optimality backup (max over actions)."""
    if env.is_terminal(s):
        return 0.0
    best = -np.inf
    for a in ACTIONS:
        ns, r, done = env.step(s, a)
        best = max(best, r + gamma * V[ns[0], ns[1]])
    return best


# ============================================================
# Policy Evaluation (PL2)
# ============================================================

def policy_evaluation(
    env: Gridworld,
    policy: Dict[Tuple[int, int], Dict[str, float]],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Iterative policy evaluation → V^π."""
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for s in env.states():
            if env.is_terminal(s):
                continue
            v_new = bellman_expectation_update(env, V_old, policy, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new

        if delta < theta:
            return V, it + 1

    return V, max_iters


# ============================================================
# Value Iteration (PL2)
# ============================================================

def value_iteration(
    env: Gridworld,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Value iteration → V*."""
    V = zeros_V(env)
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in env.states():
            v_new = bellman_optimality_update(env, V_old, s, gamma)
            delta = max(delta, abs(v_new - V[s[0], s[1]]))
            V[s[0], s[1]] = v_new
        if delta < theta:
            return V, it + 1
    return V, max_iters


# ============================================================
# Greedy policy extraction (PL2)
# ============================================================

def greedy_policy_from_V(
    env: Gridworld,
    V: np.ndarray,
    gamma: float,
) -> Dict[Tuple[int, int], str]:
    """Extract greedy deterministic policy: π(s) = argmax_a [r + γ V(s')]."""
    pi_greedy = {}
    for s in env.states():
        if env.is_terminal(s):
            pi_greedy[s] = "·"
            continue

        best_q = float("-inf")
        best_a = None
        for a in ACTIONS:
            ns, r, done = env.step(s, a)
            q = r + gamma * V[ns[0], ns[1]]
            if q > best_q:
                best_q = q
                best_a = a
        pi_greedy[s] = best_a
    return pi_greedy


# ============================================================
# Q^π evaluation (PL2)
# ============================================================

def policy_evaluation_Q(
    env: Gridworld,
    pi: Dict[Tuple[int, int], Dict[str, float]],
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Iterative evaluation of Q^π(s, a)."""
    Q = zeros_Q(env)

    for it in range(max_iters):
        delta = 0.0
        Q_old = Q.copy()

        for (r, c) in env.states():
            s = (r, c)
            if env.is_terminal(s):
                Q[r, c, :] = 0.0
                continue

            for ai, a in enumerate(ACTIONS):
                ns, reward, done = env.step(s, a)
                nr, nc = ns

                exp_next = 0.0
                for aj, a2 in enumerate(ACTIONS):
                    exp_next += pi[ns][a2] * Q_old[nr, nc, aj]

                q_new = reward + gamma * exp_next
                delta = max(delta, abs(q_new - Q[r, c, ai]))
                Q[r, c, ai] = q_new

        if delta < theta:
            return Q, it + 1

    return Q, max_iters


# ============================================================
# Policy Improvement (PL3.1)
# ============================================================

def greedy_action_from_V(
    env: Gridworld,
    V: np.ndarray,
    s: Tuple[int, int],
    gamma: float,
) -> str:
    """Return argmax_a [r + γ V(s')]."""
    best_a = None
    best_q = -np.inf
    for a in ACTIONS:
        ns, r, done = env.step(s, a)
        q = r + gamma * V[ns[0], ns[1]]
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def policy_improvement(
    env: Gridworld,
    V: np.ndarray,
    old_policy_actions: Optional[Dict[Tuple[int, int], str]] = None,
    gamma: float = 0.9,
) -> Tuple[Dict[Tuple[int, int], str], bool]:
    """Greedify policy w.r.t. V. Returns (new_policy_actions, stable)."""
    new_policy_actions: Dict[Tuple[int, int], str] = {}
    stable = True

    for s in env.states():
        if env.is_terminal(s):
            new_policy_actions[s] = "·"
            continue

        best_a = greedy_action_from_V(env, V, s, gamma)
        if old_policy_actions and old_policy_actions[s] != best_a:
            stable = False
        new_policy_actions[s] = best_a

    return new_policy_actions, stable


# ============================================================
# Policy Iteration (PL3.1)
# ============================================================

def policy_iteration(
    env: Gridworld,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_outer: int = 100,
):
    """Full policy iteration: Eval → Improve → repeat until stable.

    Returns (V*, π*_actions, history).
    """
    from AR1.mdps.gridworld_mdp import uniform_random_policy

    pi_stochastic = uniform_random_policy(env)
    pi_actions = {s: ("·" if env.is_terminal(s) else None) for s in env.states()}
    history = []

    for outer in range(max_outer):
        # 1) Evaluate current policy
        V, iters = policy_evaluation(env, pi_stochastic, gamma=gamma, theta=theta)

        # 2) Improve
        new_actions, stable = policy_improvement(env, V, old_policy_actions=pi_actions, gamma=gamma)
        history.append((outer, iters, V.copy(), new_actions.copy()))

        pi_actions = new_actions

        # Convert to stochastic representation
        pi_stochastic = {}
        for s in env.states():
            if env.is_terminal(s):
                pi_stochastic[s] = {a: 0.0 for a in ACTIONS}
            else:
                chosen = pi_actions[s]
                pi_stochastic[s] = {a: (1.0 if a == chosen else 0.0) for a in ACTIONS}

        if stable:
            return V, pi_actions, history

    return V, pi_actions, history


# ============================================================
# Stochastic Value Iteration (PL2, Exercise C)
# ============================================================

def stochastic_value_iteration(
    env: Gridworld,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    """Value iteration for stochastic transitions (slip model)."""
    V = zeros_V(env)
    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in env.states():
            if env.is_terminal(s):
                continue

            best = float("-inf")
            for a in ACTIONS:
                expected_value = 0.0
                for prob, ns, r in get_stochastic_transitions(env, s, a):
                    expected_value += prob * (r + gamma * V_old[ns[0], ns[1]])
                best = max(best, expected_value)

            delta = max(delta, abs(best - V[s[0], s[1]]))
            V[s[0], s[1]] = best

        if delta < theta:
            return V, it + 1
    return V, max_iters


# ============================================================
# Car Rental DP (PL3.2)
# ============================================================

def q_from_v_car(mdp, V: np.ndarray, s: Tuple[int, int], a: int, gamma: float) -> float:
    """Compute q(s,a) = E[reward + γ V(s')] for the Car Rental MDP."""
    p_next_1, p_next_2, exp_revenue = mdp.expected_transition(s, a)

    exp_next = 0.0
    for n1p, p1v in enumerate(p_next_1):
        if p1v == 0:
            continue
        for n2p, p2v in enumerate(p_next_2):
            if p2v == 0:
                continue
            exp_next += p1v * p2v * V[n1p, n2p]

    moving_cost = mdp.params.cost_per_moved * abs(a)
    reward = exp_revenue - moving_cost
    return reward + gamma * exp_next


def car_rental_policy_evaluation(
    mdp, policy: Dict[Tuple[int, int], int], gamma: float,
    theta: float = 1e-6, max_iters: int = 10_000,
) -> Tuple[np.ndarray, int]:
    V = np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=float)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for s in mdp.states():
            a = policy[s]
            v_new = q_from_v_car(mdp, V_old, s, a, gamma)
            V[s[0], s[1]] = v_new
            delta = max(delta, abs(v_new - V_old[s[0], s[1]]))

        if delta < theta:
            return V, it + 1
    return V, max_iters


def car_rental_policy_improvement(
    mdp, V: np.ndarray,
    old_policy: Optional[Dict[Tuple[int, int], int]],
    gamma: float,
) -> Tuple[Dict[Tuple[int, int], int], bool]:
    new_policy: Dict[Tuple[int, int], int] = {}
    stable = True

    for s in mdp.states():
        best_a = None
        best_q = -np.inf
        for a in mdp.possible_actions(s):
            q = q_from_v_car(mdp, V, s, a, gamma)
            if q > best_q:
                best_q = q
                best_a = a
        new_policy[s] = best_a
        if old_policy is not None and old_policy.get(s) != best_a:
            stable = False
        if old_policy is None:
            stable = False

    return new_policy, stable


def car_rental_policy_iteration(
    mdp, gamma: float = 0.9, theta: float = 1e-6, max_outer: int = 50,
):
    policy = {s: 0 for s in mdp.states()}
    history = []

    for outer in range(max_outer):
        V, eval_iters = car_rental_policy_evaluation(mdp, policy, gamma, theta)
        new_policy, stable = car_rental_policy_improvement(mdp, V, policy, gamma)
        history.append((outer, eval_iters, V.copy(), new_policy.copy()))
        policy = new_policy
        if stable:
            break

    return V, policy, history


def car_rental_value_iteration(
    mdp, gamma: float = 0.9, theta: float = 1e-6, max_iters: int = 10_000,
):
    V = np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=float)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for s in mdp.states():
            best = -float("inf")
            for a in mdp.possible_actions(s):
                best = max(best, q_from_v_car(mdp, V_old, s, a, gamma))
            delta = max(delta, abs(best - V_old[s[0], s[1]]))
            V[s[0], s[1]] = best
        if delta < theta:
            break

    pi = {}
    for s in mdp.states():
        best_a, best_q = None, -np.inf
        for a in mdp.possible_actions(s):
            q = q_from_v_car(mdp, V, s, a, gamma)
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a

    return V, pi, it + 1
