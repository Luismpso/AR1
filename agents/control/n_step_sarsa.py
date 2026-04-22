from __future__ import annotations

import random
from collections import defaultdict

from AR1.agents.control.base import ActionT, ControlAgent, StateT
from AR1.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._pending_transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the n-step bootstrap."""
        # 1. With probability epsilon, choose a random action (exploration)
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        # 2. Otherwise, choose the greedy action (exploitation)
        else:
            max_q = max(self.Q[(state, a)] for a in self.actions)
            best_actions = [a for a in self.actions if self.Q[(state, a)] == max_q]
            action = self.rng.choice(best_actions)
            
        # 3. Cache the selected action
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Buffer the next transition and update the oldest state-action when possible."""
        # 1. Add transition to the buffer
        self._pending_transitions.append(transition)
        
        # 2. If episode ended, flush buffer (update all pending states)
        if transition.done:
            while len(self._pending_transitions) > 0:
                self._update_oldest_transition()
                
        # 3. If not done but buffer reached n_steps, update the oldest
        elif len(self._pending_transitions) >= self.n_steps:
            self._update_oldest_transition()

    def _update_oldest_transition(self) -> None:
        """Compute the n-step Sarsa target for the oldest transition in the buffer."""
        # The oldest transition is always the first in the list (index 0)
        oldest_transition = self._pending_transitions[0]
        state = oldest_transition.state
        action = oldest_transition.action
        
        # 1 & 2. Sum discounted rewards within the current window
        G = 0.0
        for i, t in enumerate(self._pending_transitions):
            G += (self.gamma ** i) * t.reward
            
        # 3. Bootstrap only if the window has exactly n_steps size
        # and the last transition in the window is NOT terminal
        last_transition = self._pending_transitions[-1]
        if len(self._pending_transitions) == self.n_steps and not last_transition.done:
            next_state = last_transition.next_state
            next_action = self._selected_actions[next_state]
            G += (self.gamma ** self.n_steps) * self.Q[(next_state, next_action)]
            
        # 4. Apply incremental update to the oldest state-action pair
        td_error = G - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error
        
        # Finally, remove the oldest transition so the window can slide
        self._pending_transitions.pop(0)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))