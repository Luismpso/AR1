"""Unit tests for the environments (PL1–PL9)."""
from __future__ import annotations

import numpy as np
import pytest

from AR1.envs.bandit import KArmedBandit
from AR1.envs.blackjack import BlackjackEnv, is_bust, sum_hand, usable_ace
from AR1.envs.gridworld import Gridworld, GridworldTrap, get_stochastic_transitions
from AR1.envs.tictactoe import TicTacToeEnv, _winner
from AR1.envs.windy_gridworld import ACTIONS as WINDY_ACTIONS
from AR1.envs.windy_gridworld import WindyGridworldEnv


class TestKArmedBandit:
    def test_default_shape(self):
        np.random.seed(0)
        b = KArmedBandit(k=10)
        assert b.q_true.shape == (10,)
        assert 0 <= int(b.optimal_action) < 10

    def test_reward_is_noisy_estimate(self):
        np.random.seed(0)
        b = KArmedBandit(k=5)
        rewards = [b.step(2) for _ in range(1000)]
        assert abs(np.mean(rewards) - b.q_true[2]) < 0.1

    def test_non_stationary_drift(self):
        np.random.seed(0)
        b = KArmedBandit(k=10, stationary=False, walk_std=0.1)
        q0 = b.q_true.copy()
        for _ in range(200):
            b.step(0)
        assert not np.allclose(b.q_true, q0)


class TestGridworld:
    def test_terminal_corners(self):
        g = Gridworld()
        assert g.is_terminal((0, 0))
        assert g.is_terminal((3, 3))
        assert not g.is_terminal((1, 1))

    def test_step_reward_minus_one(self):
        g = Gridworld()
        _, r, _ = g.step((1, 1), "U")
        assert r == -1.0

    def test_walls_keep_state(self):
        g = Gridworld()
        ns, _, _ = g.step((0, 1), "U")
        assert ns == (0, 1)
        ns, _, _ = g.step((3, 1), "D")
        assert ns == (3, 1)

    def test_terminal_self_loop(self):
        g = Gridworld()
        ns, r, done = g.step((0, 0), "R")
        assert ns == (0, 0)
        assert r == 0.0
        assert done

    def test_trap_reward(self):
        g = GridworldTrap()
        _, r, _ = g.step((1, 1), "R")
        assert r == -10.0

    def test_stochastic_transitions_sum_to_one(self):
        g = Gridworld()
        for state in [(1, 1), (2, 3), (0, 1)]:
            for a in ("U", "D", "L", "R"):
                transitions = get_stochastic_transitions(g, state, a)
                total = sum(p for p, _, _ in transitions)
                assert abs(total - 1.0) < 1e-9


class TestBlackjack:
    def test_sum_with_usable_ace(self):
        assert sum_hand([1, 6]) == 17
        assert usable_ace([1, 6])

    def test_no_usable_ace_when_bust(self):
        assert sum_hand([1, 10, 5]) == 16
        assert not usable_ace([1, 10, 5])

    def test_bust(self):
        assert is_bust([10, 10, 5])
        assert not is_bust([10, 10])

    def test_initial_state_player_sum_ge_12(self):
        env = BlackjackEnv(seed=0)
        for _ in range(50):
            state = env.reset()
            assert state[0] >= 12

    def test_hit_can_bust(self):
        env = BlackjackEnv(seed=123)
        bust_seen = False
        for _ in range(200):
            env.reset()
            done = False
            while not done:
                state, reward, done = env.step("hit")
                if done and reward == -1.0:
                    bust_seen = True
                    break
            if bust_seen:
                break
        assert bust_seen

    def test_stick_reaches_terminal(self):
        env = BlackjackEnv(seed=0)
        env.reset()
        _, _, done = env.step("stick")
        assert done


class TestWindyGridworld:
    def test_default_shape(self):
        env = WindyGridworldEnv()
        assert env.rows == 7 and env.cols == 10
        assert env.start == (3, 0)
        assert env.goal == (3, 7)

    def test_wind_pushes_upward(self):
        env = WindyGridworldEnv()
        # wind at col 5 = 1 -> from (3, 5) action "right": row=3-1=2, col=6
        ns, r, done = env.step_from_state((3, 5), "right")
        assert ns == (2, 6)
        assert r == -1.0
        assert not done

    def test_wind_strength_two_column(self):
        env = WindyGridworldEnv()
        # wind at col 6 = 2 -> from (5, 6) action "right": row=5-2=3, col=7 (goal)
        ns, _, done = env.step_from_state((5, 6), "right")
        assert ns == (3, 7)
        assert done

    def test_bounds_clamp(self):
        env = WindyGridworldEnv()
        ns, _, _ = env.step_from_state((0, 0), "up")
        assert ns == (0, 0)
        ns, _, _ = env.step_from_state((6, 9), "right")
        assert ns[1] == 9

    def test_actions_set(self):
        env = WindyGridworldEnv()
        actions = env.available_actions((0, 0))
        assert set(actions) == set(WINDY_ACTIONS)


class TestTicTacToe:
    def test_initial_state_empty_x_first(self):
        env = TicTacToeEnv()
        s = env.reset()
        assert s == (0,) * 9
        assert env.current_player == 1

    def test_available_actions_is_empty_indices(self):
        env = TicTacToeEnv()
        env.reset()
        env.step(4)
        env.step(0)
        actions = env.available_actions(env.board)
        assert 4 not in actions and 0 not in actions
        assert len(actions) == 7

    def test_winner_detection(self):
        assert _winner((1, 1, 1, 0, 0, 0, 0, 0, 0)) == 1
        assert _winner((-1, -1, -1, 0, 0, 0, 0, 0, 0)) == -1
        assert _winner((1, 0, 0, 1, 0, 0, 1, 0, 0)) == 1
        assert _winner((1, 0, 0, 0, 1, 0, 0, 0, 1)) == 1
        assert _winner((1, -1, 1, -1, 1, -1, -1, 1, -1)) == 0

    def test_reward_on_win(self):
        env = TicTacToeEnv()
        env.reset()
        env.step(0); env.step(3); env.step(1); env.step(4)
        _, reward, done = env.step(2)
        assert reward == 1.0
        assert done

    def test_illegal_move_raises(self):
        env = TicTacToeEnv()
        env.reset()
        env.step(4)
        with pytest.raises(ValueError):
            env.step(4)

    def test_alternation(self):
        env = TicTacToeEnv()
        env.reset()
        assert env.current_player == 1
        env.step(0)
        assert env.current_player == -1
        env.step(4)
        assert env.current_player == 1
