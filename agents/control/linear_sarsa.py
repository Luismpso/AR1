from __future__ import annotations

import random
from typing import Callable

import numpy as np

from AR1.agents.control.base import ActionT, ControlAgent, StateT
from AR1.core.base import Transition


class LinearSarsaControl(ControlAgent[StateT, ActionT]):
    """Semi-gradient SARSA with linear function approximation (NumPy).

    Model:      q_hat(s, a) = w · phi(s, a)
    TD error:   delta = r + gamma * q_hat(s', a') - q_hat(s, a)
    Update:     w += alpha * delta * phi(s, a)

    phi(s, a) uses an action-specific block encoding so that each action
    gets its own independent weight slice (the only non-zero block is the
    one for the current action).
    """

    def __init__(
        self,
        actions: tuple[ActionT, ...],
        phi: Callable[[StateT, ActionT], np.ndarray],
        n_features: int,
        alpha: float = 0.01,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.phi = phi
        self.n_features = n_features
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.w = np.zeros(self.n_features)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._td_errors: list[float] = []

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            q_values = [self.action_value_of(state, a) for a in self.actions]
            best_value = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == best_value]
            action = self.rng.choice(best_actions) if best_actions else self.rng.choice(self.actions)
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        # 1. Compute feature vector (phi)
        phi = self.phi(transition.state, transition.action)

        # 2. Compute bootstrap value
        bootstrap = 0.0
        if not transition.done and transition.next_state is not None:
            next_action = self._selected_actions[transition.next_state]
            bootstrap = self.action_value_of(transition.next_state, next_action)

        # 3. Compute TD error (delta)
        delta = transition.reward + self.gamma * bootstrap - float(self.w @ phi)

        # 4. Weight update (semi-gradient)
        self.w += self.alpha * delta * phi

        # 5. Store absolute error
        self._td_errors.append(abs(delta))

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.w @ self.phi(state, action))

    def greedy_action(self, state: StateT) -> ActionT:
        q_values = [self.action_value_of(state, a) for a in self.actions]
        best_value = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == best_value]
        return self.rng.choice(best_actions) if best_actions else self.rng.choice(self.actions)

    def flush_td_errors(self) -> list[float]:
        """Return and clear the accumulated per-step TD errors since last flush."""
        errors = list(self._td_errors)
        self._td_errors.clear()
        return errors
