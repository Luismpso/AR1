"""PL8 — REINFORCE (Monte Carlo Policy Gradient) for Tic-Tac-Toe.

Two variants:
  1. Vanilla REINFORCE — uses raw returns G_t
  2. REINFORCE with baseline (Sutton & Barto, Sec. 13.4) — subtracts a
     learned state-value V(s) from the returns, reducing variance without
     introducing bias:
         delta_t = G_t - V(s_t)                  (advantage estimate)
         theta  += alpha * gamma^t * delta_t * grad log pi(a_t|s_t)  (policy)
         w      += alpha_w * delta_t * phi(s_t)                      (value)
"""
from __future__ import annotations

import numpy as np

from AR1.features.tictactoe import STATE_FEATURE_DIM

N_ACTIONS: int = 9


class ReinforceAgent:
    """REINFORCE (Monte Carlo policy gradient) for TicTacToe.

    Policy: pi(a|s) = softmax over available actions of h(s,a) = theta[a] . phi(s)
    Baseline (optional): V(s) = w . phi(s)  — linear state-value function

    When use_baseline=True, the policy gradient uses the advantage
    delta_t = G_t - V(s_t) instead of the raw return G_t.  This reduces
    variance significantly while keeping the gradient estimate unbiased
    (Sutton & Barto, Sec. 13.4).
    """

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        n_features: int = STATE_FEATURE_DIM,
        alpha: float = 0.01,
        gamma: float = 1.0,
        entropy_beta: float = 0.0,
        use_baseline: bool = False,
        alpha_w: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.use_baseline = use_baseline
        self.alpha_w = alpha_w if alpha_w is not None else alpha * 5.0
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Initialise / reinitialise weights to zero."""
        self.theta = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        if self.use_baseline:
            self.w = np.zeros(self.n_features, dtype=np.float64)
        self._episode: list[tuple[np.ndarray, int, list[int], float]] = []

    def _probs(self, phi: np.ndarray, available: list[int]) -> np.ndarray:
        """Softmax probabilities over available actions only."""
        logits = self.theta[available] @ phi
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def _value(self, phi: np.ndarray) -> float:
        """Linear state-value estimate V(s) = w . phi(s)."""
        return float(self.w @ phi)

    # ── Action selection ───────────────────────────────────────────────────

    def select_action(self, phi: np.ndarray, available: list[int]) -> int:
        """Sample an action proportionally to the current stochastic policy."""
        probs = self._probs(phi, available)
        idx = self._rng.choice(len(available), p=probs)
        return available[int(idx)]

    def greedy_action(self, phi: np.ndarray, available: list[int]) -> int:
        """Return the most probable available action (for evaluation / play)."""
        probs = self._probs(phi, available)
        return available[int(np.argmax(probs))]

    # ── Learning ───────────────────────────────────────────────────────────

    def store_step(self, phi, action, available, reward) -> None:
        """Append one environment step to the episode buffer."""
        self._episode.append((phi, action, available, reward))

    def update_episode(
        self,
        trajectory: list[tuple[np.ndarray, int, list[int], float]] | None = None,
    ) -> float:
        """Compute Monte Carlo returns and apply REINFORCE update.

        When use_baseline=True, subtracts V(s_t) from G_t (advantage)
        and updates the value weights w alongside the policy weights theta.
        """
        episode = trajectory if trajectory is not None else self._episode
        if not episode:
            return 0.0

        T = len(episode)

        # Backward pass — discounted returns
        returns = np.empty(T)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = episode[t][3] + self.gamma * G
            returns[t] = G

        total_loss = 0.0
        for t, (phi, action, available, _) in enumerate(episode):
            probs = self._probs(phi, available)
            action_idx = available.index(action)

            # Advantage: delta = G_t - V(s_t) if baseline, else just G_t
            if self.use_baseline:
                v = self._value(phi)
                delta = returns[t] - v
                # Update value weights: w += alpha_w * delta * phi
                self.w += self.alpha_w * delta * phi
            else:
                delta = returns[t]

            total_loss -= delta * np.log(probs[action_idx] + 1e-8)

            # Policy gradient: score function * advantage
            scale = self.alpha * (self.gamma ** t) * delta
            for i, a in enumerate(available):
                if a == action:
                    self.theta[a] += scale * phi * (1.0 - probs[i])
                else:
                    self.theta[a] -= scale * phi * probs[i]

            # Optional entropy regularisation
            if self.entropy_beta > 0.0:
                H = -float(np.sum(probs * np.log(probs + 1e-8)))
                for i, a in enumerate(available):
                    log_p = np.log(probs[i] + 1e-8)
                    self.theta[a] += self.alpha * self.entropy_beta * (-probs[i] * (log_p + H)) * phi

        if trajectory is None:
            self._episode.clear()
        return total_loss / T
