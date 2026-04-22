"""PL8 — REINFORCE self-play training and evaluation for Tic-Tac-Toe."""
from __future__ import annotations

import random

import numpy as np

from AR1.agents.control.reinforce import ReinforceAgent
from AR1.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState, _winner
from AR1.features.tictactoe import encode_state
from AR1.policies.tictactoe import Policy, random_action


# ── Policy wrapper ────────────────────────────────────────────────────────────

def make_reinforce_policy(agent: ReinforceAgent, greedy: bool = True) -> Policy:
    """Wrap a trained ReinforceAgent as a Policy callable for play_game."""
    def policy(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
        phi = encode_state(state, env.current_player)
        available = env.available_actions(state)
        if greedy:
            return agent.greedy_action(phi, available)
        return agent.select_action(phi, available)

    return policy


# ── Self-play episode ─────────────────────────────────────────────────────────

def run_reinforce_episode(env: TicTacToeEnv, agent: ReinforceAgent) -> tuple[float, int]:
    """One self-play game: both sides controlled by agent, updated at the end.

    Trajectories are collected separately for X and O.  Terminal rewards:
        - Winner's last step:  +1  (returned by the environment)
        - Loser's last step:   -1  (injected here -- the env emits 0)
        - Draw / intermediate:  0

    Both trajectories are used to update the same agent.theta sequentially
    (X first, then O).

    Returns:
        (mean_loss, winner)   where winner is 1, -1, or 0.
    """
    state = env.reset()

    traj_x: list[tuple[np.ndarray, int, list[int], float]] = []
    traj_o: list[tuple[np.ndarray, int, list[int], float]] = []

    while not env.is_terminal(state):
        player = env.current_player
        phi = encode_state(state, player)
        available = env.available_actions(state)
        action = agent.select_action(phi, available)
        next_state, reward, done = env.step(action)

        step = (phi, action, available, reward)
        if player == 1:
            traj_x.append(step)
        else:
            traj_o.append(step)

        # Inject r = -1 into the loser's last step
        if done and reward == 1.0:
            # Current player won -> the OTHER player lost
            if player == 1:
                # X won -> O lost -> inject -1 into O's last step
                if traj_o:
                    last = traj_o[-1]
                    traj_o[-1] = (last[0], last[1], last[2], -1.0)
            else:
                # O won -> X lost -> inject -1 into X's last step
                if traj_x:
                    last = traj_x[-1]
                    traj_x[-1] = (last[0], last[1], last[2], -1.0)

        state = next_state

    loss_x = agent.update_episode(traj_x)
    loss_o = agent.update_episode(traj_o)
    return (loss_x + loss_o) / 2.0, _winner(state)


# ── vs-random episode ─────────────────────────────────────────────────────────

def run_vs_random_episode(env: TicTacToeEnv, agent: ReinforceAgent) -> tuple[float, int]:
    """One episode where the agent plays one side and a random policy plays the other.

    The side the agent plays is chosen at random each call.
    Only the agent's trajectory is collected and used for the update.

    Returns:
        (loss, winner)
    """
    state = env.reset()
    agent_player = random.choice([1, -1])
    traj: list[tuple[np.ndarray, int, list[int], float]] = []

    while not env.is_terminal(state):
        player = env.current_player
        available = env.available_actions(state)

        if player == agent_player:
            phi = encode_state(state, player)
            action = agent.select_action(phi, available)
            next_state, reward, done = env.step(action)
            traj.append((phi, action, available, reward))
        else:
            action = random_action(env, state)
            next_state, reward, done = env.step(action)
            if done and reward == 1.0:
                # Random player won -> inject -1 into agent's last step
                if traj:
                    last = traj[-1]
                    traj[-1] = (last[0], last[1], last[2], -1.0)

        state = next_state

    return agent.update_episode(traj), _winner(state)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_vs_random(
    env: TicTacToeEnv,
    agent: ReinforceAgent,
    n_games: int = 200,
    as_player: int = 1,
) -> tuple[float, float, float]:
    """Evaluate the greedy policy against a random opponent.

    Returns:
        (win_rate, draw_rate, loss_rate)
    """
    agent_policy = make_reinforce_policy(agent, greedy=True)
    wins, draws, losses = 0, 0, 0

    for _ in range(n_games):
        if as_player == 1:
            result = _play_silent(env, agent_policy, random_action)
        else:
            result = _play_silent(env, random_action, agent_policy)

        if result == as_player:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

    return wins / n_games, draws / n_games, losses / n_games


def _play_silent(
    env: TicTacToeEnv,
    policy_x: Policy,
    policy_o: Policy,
) -> int:
    """Play one silent game (no rendering). Returns 1/-1/0."""
    state = env.reset()
    while not env.is_terminal(state):
        policy = policy_x if env.current_player == 1 else policy_o
        action = policy(env, state)
        state, _, _ = env.step(action)
    return _winner(state)


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    agent: ReinforceAgent,
    num_episodes: int = 50_000,
    eval_every: int = 2_000,
    eval_episodes: int = 500,
    random_opp_fraction: float = 0.3,
    seed: int | None = None,
) -> dict:
    """Train ReinforceAgent via self-play and periodically evaluate vs random.

    Returns:
        dict with keys: losses, self_play_x_wins, eval_checkpoints,
        win_rates_as_x, win_rates_as_o, draw_rates_as_x, draw_rates_as_o
    """
    env = TicTacToeEnv()

    losses: list[float] = []
    self_play_x_wins: list[int] = []
    eval_checkpoints: list[int] = []
    win_rates_as_x: list[float] = []
    win_rates_as_o: list[float] = []
    draw_rates_as_x: list[float] = []
    draw_rates_as_o: list[float] = []

    rng = np.random.default_rng(seed)

    for ep in range(1, num_episodes + 1):
        if rng.random() < random_opp_fraction:
            loss, _ = run_vs_random_episode(env, agent)
        else:
            loss, winner = run_reinforce_episode(env, agent)
            self_play_x_wins.append(winner)
        losses.append(loss)

        if ep % eval_every == 0:
            wr_x, dr_x, _ = evaluate_vs_random(env, agent, eval_episodes, as_player=1)
            wr_o, dr_o, _ = evaluate_vs_random(env, agent, eval_episodes, as_player=-1)
            eval_checkpoints.append(ep)
            win_rates_as_x.append(wr_x)
            win_rates_as_o.append(wr_o)
            draw_rates_as_x.append(dr_x)
            draw_rates_as_o.append(dr_o)

    return {
        "losses": losses,
        "self_play_x_wins": self_play_x_wins,
        "eval_checkpoints": eval_checkpoints,
        "win_rates_as_x": win_rates_as_x,
        "win_rates_as_o": win_rates_as_o,
        "draw_rates_as_x": draw_rates_as_x,
        "draw_rates_as_o": draw_rates_as_o,
    }
