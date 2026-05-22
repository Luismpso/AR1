"""Unit tests for the learning agents (PL1-PL9)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from AR1.agents.bandits import (
    DecayingEpsilonGreedy,
    EpsilonGreedy,
    Exp3,
    GradientBandit,
    ThompsonSampling,
    UCB,
)
from AR1.agents.control.linear_sarsa import LinearSarsaControl
from AR1.agents.control.monte_carlo import MonteCarloControl
from AR1.agents.control.n_step_sarsa import NStepSarsaControl
from AR1.agents.control.q_learning import QLearningControl
from AR1.agents.control.reinforce import ReinforceAgent
from AR1.agents.control.sarsa import SarsaControl
from AR1.agents.dp.dynamic_programming import (
    greedy_policy_from_V,
    policy_evaluation,
    policy_iteration,
    value_iteration,
)
from AR1.agents.planning.mcts import MCTSAgent, _apply, _available, _is_terminal
from AR1.agents.prediction.monte_carlo import FirstVisitMonteCarloPrediction
from AR1.agents.prediction.td import TD0Prediction
from AR1.agents.prediction.td_n import TDnPrediction
from AR1.core.base import Transition
from AR1.envs.bandit import KArmedBandit
from AR1.envs.blackjack import BlackjackEnv
from AR1.envs.gridworld import Gridworld
from AR1.envs.windy_gridworld import ACTIONS as WINDY_ACTIONS
from AR1.envs.windy_gridworld import WindyGridworldEnv
from AR1.experiments.training import generate_episode
from AR1.features.tictactoe import STATE_FEATURE_DIM, encode_state
from AR1.features.windy_gridworld import (
    STATE_ACTION_FEATURE_DIM,
    TILE_STATE_DIM,
    state_action_features,
)
from AR1.mdps.gridworld_mdp import uniform_random_policy
from AR1.policies.blackjack import ThresholdPolicy


# Bandits (PL1)

@pytest.fixture
def bandit_seed():
    np.random.seed(0)


class TestBanditAgents:
    @pytest.mark.parametrize("AgentCls,kwargs", [
        (EpsilonGreedy,         {"k": 5, "epsilon": 0.1}),
        (UCB,                   {"k": 5, "c": 1.0}),
        (GradientBandit,        {"k": 5, "alpha": 0.1}),
        (ThompsonSampling,      {"k": 5}),
        (Exp3,                  {"k": 5}),
        (DecayingEpsilonGreedy, {"k": 5, "initial_epsilon": 1.0, "decay_rate": 0.99}),
    ])
    def test_select_action_in_range(self, bandit_seed, AgentCls, kwargs):
        agent = AgentCls(**kwargs)
        agent.reset()
        for _ in range(10):
            a = agent.select_action()
            assert 0 <= int(a) < 5

    def test_eps_greedy_converges_to_best_arm(self, bandit_seed):
        bandit = KArmedBandit(k=3)
        bandit.q_true = np.array([0.0, 0.0, 5.0])
        bandit.optimal_action = 2
        agent = EpsilonGreedy(k=3, epsilon=0.01)
        agent.reset()
        for _ in range(500):
            a = agent.select_action()
            r = bandit.step(a)
            agent.update(a, r)
        assert int(np.argmax(agent.Q)) == 2


# Dynamic Programming (PL2 / PL3)

class TestDP:
    def test_policy_evaluation_matches_textbook(self):
        env = Gridworld()
        pi = uniform_random_policy(env)
        V, _ = policy_evaluation(env, pi, gamma=1.0, theta=1e-6)
        assert math.isclose(V[0, 0], 0.0, abs_tol=1e-6)
        assert math.isclose(V[3, 3], 0.0, abs_tol=1e-6)
        assert -20.5 < V[2, 2] < -17.0

    def test_value_iteration_is_optimal(self):
        env = Gridworld()
        V, _ = value_iteration(env, gamma=1.0, theta=1e-9)
        pi = greedy_policy_from_V(env, V, gamma=1.0)
        for start in env.states():
            if env.is_terminal(start):
                continue
            s = start
            done = False
            for _ in range(10):
                a = pi[s]
                s, _, done = env.step(s, a)
                if done:
                    break
            assert done, f"Optimal policy failed to terminate from {start}"

    def test_policy_iteration_returns_optimal_value(self):
        env = Gridworld()
        V_pi, pi, history = policy_iteration(env, gamma=1.0, theta=1e-9)
        V_vi, _ = value_iteration(env, gamma=1.0, theta=1e-9)
        np.testing.assert_allclose(V_pi, V_vi, atol=1e-3)
        assert len(history) >= 1


# Prediction (PL4)

class TestPrediction:
    def test_mc_value_high_for_strong_hand(self):
        env = BlackjackEnv(seed=0)
        agent = FirstVisitMonteCarloPrediction()
        agent.reset()
        for _ in range(2000):
            ep = generate_episode(env, ThresholdPolicy(threshold=20))
            agent.update_episode(ep)
        v_20 = np.mean([agent.value_of((20, d, False)) for d in range(1, 11)])
        assert v_20 > 0.3, f"V(20,*) should be clearly winning, got {v_20}"

    def test_td0_converges_close_to_mc(self):
        env = BlackjackEnv(seed=1)
        policy = ThresholdPolicy(threshold=20)
        mc = FirstVisitMonteCarloPrediction()
        td = TD0Prediction(alpha=0.05)
        mc.reset(); td.reset()
        for _ in range(2000):
            ep = generate_episode(env, policy)
            mc.update_episode(ep)
            td.update_episode(ep)
        diffs = [abs(mc.value_of((20, d, False)) - td.value_of((20, d, False)))
                 for d in range(1, 11)]
        assert np.mean(diffs) < 0.5

    def test_tdn_with_n1_matches_td0(self):
        env = BlackjackEnv(seed=2)
        policy = ThresholdPolicy(threshold=20)
        td0 = TD0Prediction(alpha=0.05)
        tdn = TDnPrediction(n=1, alpha=0.05)
        td0.reset(); tdn.reset()
        for _ in range(200):
            ep = generate_episode(env, policy)
            td0.update_episode(ep)
            tdn.update_episode(ep)
        for state in td0.V:
            assert math.isclose(td0.V[state], tdn.V[state], abs_tol=1e-9)


# Tabular Control (PL5)

def _train_windy_control(agent, episodes=100, max_steps=200):
    env = WindyGridworldEnv()
    for _ in range(episodes):
        state = env.reset()
        action = agent.select_action(state)
        steps = 0
        done = False
        while not done and steps < max_steps:
            ns, r, done = env.step(action)
            if not done:
                next_action = agent.select_action(ns)
            else:
                next_action = None
            agent.update_transition(Transition(
                state=state, action=action, reward=r,
                next_state=None if done else ns, done=done,
            ))
            state = ns
            if next_action is not None:
                action = next_action
            steps += 1
        if hasattr(agent, "end_episode"):
            agent.end_episode()


class TestTabularControl:
    @pytest.mark.parametrize("AgentCls,kwargs", [
        (SarsaControl,      {}),
        (QLearningControl,  {}),
        (NStepSarsaControl, {"n_steps": 4}),
    ])
    def test_control_agents_train_without_error(self, AgentCls, kwargs):
        agent = AgentCls(actions=WINDY_ACTIONS, seed=0, **kwargs)
        agent.reset()
        _train_windy_control(agent, episodes=20)
        assert len(agent.Q) > 0

    def test_q_learning_learns_negative_values(self):
        agent = QLearningControl(actions=WINDY_ACTIONS, alpha=0.5, epsilon=0.1, seed=0)
        agent.reset()
        _train_windy_control(agent, episodes=80, max_steps=200)
        sample_q = [v for v in agent.Q.values() if v != 0.0]
        assert len(sample_q) > 0
        assert np.mean(sample_q) < 0.0

    def test_mc_control_train(self):
        agent = MonteCarloControl(actions=WINDY_ACTIONS, epsilon=0.2, seed=0)
        agent.reset()
        _train_windy_control(agent, episodes=20)
        assert len(agent.Q) > 0


# Function Approximation (PL6)

class TestFunctionApprox:
    def test_state_action_features_shape(self):
        env = WindyGridworldEnv()
        phi = state_action_features((3, 4), "up", env)
        assert phi.shape == (STATE_ACTION_FEATURE_DIM,)
        slice_size = TILE_STATE_DIM
        idx_up = WINDY_ACTIONS.index("up")
        non_zero = np.where(phi != 0)[0]
        assert np.all((non_zero >= idx_up * slice_size) & (non_zero < (idx_up + 1) * slice_size))

    def test_linear_sarsa_trains(self):
        env = WindyGridworldEnv()

        def phi(state, action):
            return state_action_features(state, action, env)

        agent = LinearSarsaControl(
            actions=WINDY_ACTIONS, phi=phi,
            n_features=STATE_ACTION_FEATURE_DIM,
            alpha=0.05, epsilon=0.1, seed=0,
        )
        agent.reset()
        _train_windy_control(agent, episodes=30)
        assert np.linalg.norm(agent.w) > 0.0


# TicTacToe Features (PL7)

class TestTicTacToeFeatures:
    def test_encode_state_shape_and_dtype(self):
        phi = encode_state((0,) * 9, current_player=1)
        assert phi.shape == (STATE_FEATURE_DIM,)
        assert phi.dtype == np.float32

    def test_encode_state_one_hot_perspective(self):
        board = (1, 0, 0, 0, -1, 0, 0, 0, 0)
        phi_x = encode_state(board, current_player=1)
        phi_o = encode_state(board, current_player=-1)
        assert phi_x[0 * 3 + 0] == 1.0
        assert phi_x[4 * 3 + 1] == 1.0
        assert phi_o[0 * 3 + 1] == 1.0
        assert phi_o[4 * 3 + 0] == 1.0


# Policy Gradient (PL8)

class TestReinforce:
    def test_probs_sum_to_one(self):
        agent = ReinforceAgent(seed=0)
        agent.reset()
        phi = encode_state((0,) * 9, current_player=1)
        probs = agent._probs(phi, list(range(9)))
        assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-6)
        assert (probs >= 0).all()

    def test_select_action_in_available_set(self):
        agent = ReinforceAgent(seed=0)
        agent.reset()
        phi = encode_state((0,) * 9, current_player=1)
        avail = [0, 4, 8]
        for _ in range(20):
            a = agent.select_action(phi, avail)
            assert a in avail

    def test_update_episode_changes_weights(self):
        agent = ReinforceAgent(seed=0)
        agent.reset()
        phi = encode_state((0,) * 9, current_player=1)
        episode = [(phi, 0, [0, 1, 2], 1.0)]
        before = agent.theta.copy()
        agent.update_episode(episode)
        assert not np.array_equal(before, agent.theta)


# Planning / MCTS (PL9)

class TestMCTS:
    def test_mcts_blocks_immediate_loss(self):
        # X X .   X is about to win at cell 2; O must block.
        # . . .
        # . . .
        board = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        agent = MCTSAgent(n_simulations=500)
        action = agent.select_action(board, player=-1)
        assert action == 2, f"MCTS should block at cell 2, picked {action}"

    def test_mcts_takes_immediate_win(self):
        # X X .
        # O O .  -> X to move, must complete top row
        # . . .
        board = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        agent = MCTSAgent(n_simulations=500)
        action = agent.select_action(board, player=1)
        assert action == 2

    def test_helpers_pure(self):
        s = (0,) * 9
        s2 = _apply(s, 4, 1)
        assert s == (0,) * 9
        assert s2[4] == 1
        assert 4 not in _available(s2)
        assert not _is_terminal(s2)
