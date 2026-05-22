"""PL6/PL9 (extra) — Deep Q-Network for the Windy Gridworld.

A small replay-buffer + target-network DQN trained on a 2-dim normalised
(row, col) input.  The point is to compare against the tile-coded
``LinearSarsaControl`` agent on the same environment: same task, same
episodes budget, different function-approximation strategy.

Architecture
------------
    input  (2,)         → normalised (row/(H-1), col/(W-1))
    hidden (2 × 64)     → ReLU + ReLU
    output (n_actions)  → linear (one Q-value per action)

Training loop
-------------
    1.  Choose action with epsilon-greedy on Q(s)         (online net)
    2.  Step the environment, store (s, a, r, s', done)   (replay buffer)
    3.  Sample a mini-batch and compute the DQN target:
            y = r + (1 - done) * gamma * max_a' Q_target(s', a')
        Loss = MSE(Q_online(s, a), y).backward(), optimiser.step()
    4.  Every ``target_sync_every`` updates, copy weights
        Q_online → Q_target.

This is intentionally minimal; the only "modern" ingredient besides the
target network is the replay buffer (Mnih et al., 2015).
"""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from AR1.agents.control.base import ActionT, ControlAgent
from AR1.core.base import Transition
from AR1.envs.windy_gridworld import (
    ACTIONS,
    WindyGridworldAction,
    WindyGridworldEnv,
    WindyGridworldState,
)

torch.set_num_threads(1)


class QNetwork(nn.Module):
    """Small MLP: 2 → 64 → 64 → n_actions."""

    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNControl(ControlAgent[WindyGridworldState, WindyGridworldAction]):
    """DQN for the Windy Gridworld (deterministic 7×10 grid).

    Uses ``state_normaliser`` to map a (row, col) tuple to a 2-dim float
    tensor in [0, 1]^2.  The action space is the standard 4 directions.
    """

    def __init__(
        self,
        actions: tuple[WindyGridworldAction, ...] = ACTIONS,
        env: WindyGridworldEnv | None = None,
        alpha: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        gamma: float = 1.0,
        buffer_size: int = 10_000,
        batch_size: int = 32,
        warmup_steps: int = 200,
        target_sync_every: int = 100,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_sync_every = target_sync_every
        self.hidden = hidden
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # Grid dimensions (used for normalisation).  Defaults match the
        # standard Windy Gridworld 7×10 grid.
        if env is not None:
            self._rows = env.rows
            self._cols = env.cols
        else:
            self._rows = 7
            self._cols = 10

        super().__init__(gamma=gamma)

    # ── ControlAgent API ────────────────────────────────────────────────

    def reset(self) -> None:
        self.online_net = QNetwork(2, len(self.actions), self.hidden)
        self.target_net = QNetwork(2, len(self.actions), self.hidden)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.alpha)
        self.buffer: Deque[Tuple] = deque(maxlen=self.buffer_size)
        self._step = 0
        self._updates = 0
        self._losses: list[float] = []
        # Backwards-compat for code that introspects Q-tables; not used by DQN.
        self.Q: dict = {}

    def _state_tensor(self, state: WindyGridworldState) -> torch.Tensor:
        row, col = state
        return torch.tensor(
            [row / max(self._rows - 1, 1), col / max(self._cols - 1, 1)],
            dtype=torch.float32,
        )

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: WindyGridworldState) -> WindyGridworldAction:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        with torch.no_grad():
            q_values = self.online_net(self._state_tensor(state).unsqueeze(0)).squeeze(0)
        best = int(torch.argmax(q_values).item())
        return self.actions[best]

    def greedy_action(self, state: WindyGridworldState) -> WindyGridworldAction:
        with torch.no_grad():
            q_values = self.online_net(self._state_tensor(state).unsqueeze(0)).squeeze(0)
        return self.actions[int(torch.argmax(q_values).item())]

    def action_value_of(
        self, state: WindyGridworldState, action: WindyGridworldAction
    ) -> float:
        with torch.no_grad():
            q_values = self.online_net(self._state_tensor(state).unsqueeze(0)).squeeze(0)
        return float(q_values[self.actions.index(action)].item())

    def update_transition(
        self, transition: Transition[WindyGridworldState, WindyGridworldAction]
    ) -> None:
        self._step += 1
        action_idx = self.actions.index(transition.action)
        next_state = transition.next_state if transition.next_state is not None else transition.state
        self.buffer.append((
            transition.state,
            action_idx,
            float(transition.reward),
            next_state,
            bool(transition.done),
        ))

        if len(self.buffer) < max(self.warmup_steps, self.batch_size):
            return

        batch = self.rng.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_batch  = torch.stack([self._state_tensor(s) for s in states])
        ns_batch = torch.stack([self._state_tensor(s) for s in next_states])
        a_batch  = torch.tensor(actions, dtype=torch.long)
        r_batch  = torch.tensor(rewards, dtype=torch.float32)
        d_batch  = torch.tensor(dones, dtype=torch.float32)

        # Q(s, a) — gather along action dim
        q_sa = self.online_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

        # target: r + (1-done) * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next_max = self.target_net(ns_batch).max(dim=1).values
            target = r_batch + (1.0 - d_batch) * self.gamma * q_next_max

        loss = F.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping helps with stability on small batches.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._updates += 1
        self._losses.append(float(loss.item()))
        if self._updates % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def end_episode(self) -> None:
        """No-op — DQN updates per step."""
        return

    def flush_losses(self) -> list[float]:
        out = list(self._losses)
        self._losses.clear()
        return out

    # Same contract as LinearSarsaControl.flush_td_errors so the agent can be
    # plugged into experiments.fa_training.train_fa_agent without changes.
    def flush_td_errors(self) -> list[float]:
        return self.flush_losses()

    # Same contract as LinearSarsaControl.flush_td_errors so the agent can be
    # plugged into experiments.fa_training.train_fa_agent without changes.
    def flush_td_errors(self) -> list[float]:
        return self.flush_losses()
