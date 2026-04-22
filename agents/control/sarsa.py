from __future__ import annotations

import random
from collections import defaultdict

from AR1.agents.control.base import ActionT, ControlAgent, StateT
from AR1.core.base import Transition


class SarsaControl(ControlAgent[StateT, ActionT]):
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
        self._selected_actions: dict[StateT, ActionT] = {}

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the SARSA bootstrap."""
        # 1. Explore: choose a random action with probability epsilon
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        # 2. Exploit (Greedy): choose the action with highest Q-value
        else:
            max_q = max(self.Q[(state, a)] for a in self.actions)
            # Good practice: find all actions with max_q to break ties fairly
            best_actions = [a for a in self.actions if self.Q[(state, a)] == max_q]
            action = self.rng.choice(best_actions)
            
        # 3. Cache selected action (used for SARSA bootstrapping)
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Apply the SARSA update using the cached next action for the next state."""
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        done = transition.done
        
        # 1. Use a bootstrap value of 0.0 on terminal transitions.
        if done:
            q_next = 0.0
        else:
            # 2. Read the cached next action
            next_action = self._selected_actions[next_state]
            q_next = self.Q[(next_state, next_action)]
            
        # 3. Compute the SARSA target
        sarsa_target = reward + self.gamma * q_next
        
        # 4. Apply the incremental update
        td_error = sarsa_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
