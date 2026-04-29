"""PL5/PL6 — Tabular Q-Learning control (off-policy TD).

Q-Learning differs from SARSA in the bootstrap target:
    - SARSA uses the action actually selected by the (epsilon-greedy)
      behaviour policy at the next state — on-policy.
    - Q-Learning uses ``max_a' Q(s', a')`` regardless of the action that
      will be played next — off-policy. It learns the value of the
      greedy/optimal policy directly while still exploring.

Q-Learning therefore tends to converge to the optimal policy faster in
problems where exploration would otherwise drag the on-policy estimate
down (e.g. cliff walking, or tic-tac-toe vs a random opponent).
"""
from __future__ import annotations

import random
from collections import defaultdict

from AR1.agents.control.base import ActionT, ControlAgent, StateT
from AR1.core.base import Transition


class QLearningControl(ControlAgent[StateT, ActionT]):
    """Tabular off-policy Q-Learning (Watkins, 1989).

    Update rule::

        Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    The behaviour policy is epsilon-greedy w.r.t. the current Q-table,
    matching the convention used by :class:`SarsaControl` so the two
    agents are drop-in interchangeable in training loops.
    """

    def __init__(
        self,
        actions: tuple[ActionT, ...],
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)

    def select_action(self, state: StateT) -> ActionT:
        """Epsilon-greedy action selection (behaviour policy)."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        max_q = max(self.Q[(state, a)] for a in self.actions)
        best_actions = [a for a in self.actions if self.Q[(state, a)] == max_q]
        return self.rng.choice(best_actions)

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Off-policy update: bootstrap from ``max_a' Q(s', a')``."""
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        done = transition.done

        # 1. Terminal transitions have no future return to bootstrap from.
        if done:
            q_next = 0.0
        else:
            # 2. Off-policy: use the *greedy* value at the next state,
            #    independently of which action will actually be taken.
            q_next = max(self.Q[(next_state, a)] for a in self.actions)

        # 3. TD target and incremental update.
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
