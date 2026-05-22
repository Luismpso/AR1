"""Smoke tests for the AlphaZero-style agent (PL9 extra).

Skipped automatically when PyTorch is not installed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from AR1.agents.planning.alphazero import (
    AlphaZeroMCTS,
    AlphaZeroPolicy,
    PolicyValueNet,
    _apply,
    _available,
    _is_terminal,
    train_alphazero,
)


class TestPolicyValueNet:
    def test_forward_shapes(self):
        net = PolicyValueNet()
        x = torch.zeros(5, 27)
        logits, v = net(x)
        assert logits.shape == (5, 9)
        assert v.shape == (5,)
        # Value head is tanh-bounded
        assert torch.all(v >= -1.0) and torch.all(v <= 1.0)

    def test_two_hidden_layers_and_two_heads(self):
        net = PolicyValueNet()
        weights = {n: p for n, p in net.named_parameters() if "weight" in n}
        # Trunk: two Linear layers (27→64, 64→64)
        assert "trunk.0.weight" in weights and weights["trunk.0.weight"].shape == (64, 27)
        assert "trunk.2.weight" in weights and weights["trunk.2.weight"].shape == (64, 64)
        # Heads
        assert "policy_head.weight" in weights and weights["policy_head.weight"].shape == (9, 64)
        assert "value_head.weight"  in weights and weights["value_head.weight"].shape  == (1, 64)


class TestAlphaZeroMCTS:
    def test_evaluate_returns_legal_distribution(self):
        net = PolicyValueNet()
        mcts = AlphaZeroMCTS(net=net, n_simulations=4, dirichlet_alpha=None)
        board = (1, 0, 0, 0, -1, 0, 0, 0, 0)
        probs, v = mcts._evaluate(board, player=1)
        assert probs.shape == (9,)
        # Illegal cells must be 0
        assert probs[0] == 0.0 and probs[4] == 0.0
        # Legal cells sum to 1
        assert abs(probs.sum() - 1.0) < 1e-5
        assert -1.0 <= v <= 1.0

    def test_visit_distribution_sums_to_one(self):
        net = PolicyValueNet()
        mcts = AlphaZeroMCTS(net=net, n_simulations=8, dirichlet_alpha=None)
        pi = mcts.visit_distribution((0,) * 9, player=1)
        assert abs(pi.sum() - 1.0) < 1e-5

    def test_select_action_returns_legal(self):
        net = PolicyValueNet()
        mcts = AlphaZeroMCTS(net=net, n_simulations=8, dirichlet_alpha=None,
                             rng=np.random.default_rng(0))
        board = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        action, pi = mcts.select_action(board, player=-1, temperature=0.0)
        assert action in _available(board)
        # Argmax with temperature=0 should be a single deterministic action
        assert pi.shape == (9,)


class TestSelfPlayAndTraining:
    def test_one_iteration_runs_and_loss_finite(self):
        net = PolicyValueNet()
        # Tiny budget — just confirm we get out without errors and produce metrics
        history = train_alphazero(
            net=net,
            n_iterations=1,
            games_per_iter=2,
            n_simulations=4,
            epochs_per_iter=1,
            batch_size=8,
            verbose=False,
            seed=0,
        )
        assert len(history["policy_loss"]) == 1
        assert len(history["value_loss"]) == 1
        for k in ("policy_loss", "value_loss", "total_loss"):
            assert np.isfinite(history[k][0])

    def test_training_changes_weights(self):
        net = PolicyValueNet()
        before = net.policy_head.weight.detach().clone()
        train_alphazero(
            net=net, n_iterations=1, games_per_iter=2,
            n_simulations=4, epochs_per_iter=1, batch_size=8,
            verbose=False, seed=0,
        )
        after = net.policy_head.weight.detach().clone()
        assert not torch.equal(before, after)


class TestAlphaZeroPolicy:
    def test_callable_returns_legal_action(self):
        net = PolicyValueNet()
        policy = AlphaZeroPolicy(net=net, n_simulations=4)

        class _E:
            current_player = 1

        action = policy(_E, (0,) * 9)
        assert 0 <= action < 9
