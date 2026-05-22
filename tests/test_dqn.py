"""Smoke tests for the DQN agent (PL6 extra).

Skipped automatically when PyTorch is not installed — DQN is an optional
extension that requires torch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from AR1.agents.control.dqn import DQNControl, QNetwork
from AR1.core.base import Transition
from AR1.envs.windy_gridworld import ACTIONS, WindyGridworldEnv


def _make_agent(seed: int = 0) -> DQNControl:
    return DQNControl(
        actions=ACTIONS,
        env=WindyGridworldEnv(),
        alpha=1e-3,
        batch_size=4,
        buffer_size=200,
        warmup_steps=8,
        target_sync_every=10,
        hidden=16,
        epsilon_start=0.5,
        epsilon_end=0.05,
        epsilon_decay_steps=200,
        seed=seed,
    )


class TestQNetwork:
    def test_forward_shape(self):
        net = QNetwork(n_inputs=2, n_actions=4, hidden=16)
        x = torch.zeros(3, 2)
        out = net(x)
        assert out.shape == (3, 4)

    def test_two_hidden_layers(self):
        net = QNetwork(n_inputs=2, n_actions=4, hidden=64)
        # fc1, fc2, fc3 → two hidden layers
        weights = [p for n, p in net.named_parameters() if "weight" in n]
        assert len(weights) == 3
        # Hidden layer widths must be 64
        assert weights[0].shape == (64, 2)
        assert weights[1].shape == (64, 64)
        assert weights[2].shape == (4, 64)


class TestDQNAgent:
    def test_select_action_returns_valid_action(self):
        agent = _make_agent()
        agent.reset()
        a = agent.select_action((3, 0))
        assert a in ACTIONS

    def test_buffer_grows_on_update(self):
        agent = _make_agent()
        agent.reset()
        for i in range(10):
            agent.update_transition(Transition(
                state=(0, 0), action="right", reward=-1.0,
                next_state=(0, 1), done=False,
            ))
        assert len(agent.buffer) == 10

    def test_training_runs_without_error(self):
        env = WindyGridworldEnv()
        agent = _make_agent()
        agent.reset()
        # Tiny rollout — just confirm gradient steps succeed
        for ep in range(3):
            state = env.reset()
            done, steps = False, 0
            while not done and steps < 30:
                action = agent.select_action(state)
                ns, r, done = env.step(action)
                agent.update_transition(Transition(
                    state=state, action=action, reward=r,
                    next_state=None if done else ns, done=done,
                ))
                state = ns
                steps += 1
            agent.end_episode()
        # At least one update must have happened
        assert agent._updates > 0

    def test_target_network_starts_in_sync(self):
        agent = _make_agent()
        agent.reset()
        for op, tp in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(op.data, tp.data)
        # target net is frozen
        for p in agent.target_net.parameters():
            assert not p.requires_grad

    def test_epsilon_decay(self):
        agent = _make_agent()
        agent.reset()
        assert agent.epsilon == pytest.approx(0.5, abs=1e-6)
        agent._step = 200
        assert agent.epsilon == pytest.approx(0.05, abs=1e-6)
        agent._step = 1_000_000  # clamps to 1.0
        assert agent.epsilon == pytest.approx(0.05, abs=1e-6)

    def test_action_value_of_returns_float(self):
        agent = _make_agent()
        agent.reset()
        v = agent.action_value_of((2, 2), "up")
        assert isinstance(v, float)

    def test_flush_td_errors_alias(self):
        agent = _make_agent()
        agent.reset()
        agent._losses.append(0.123)
        out = agent.flush_td_errors()
        assert out == [0.123]
        # flushing again returns empty
        assert agent.flush_td_errors() == []
