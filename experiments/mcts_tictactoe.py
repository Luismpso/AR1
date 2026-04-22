"""PL9 — MCTS on Tic-Tac-Toe: policy wrapper + head-to-head evaluation.

Evaluation helpers:
    - evaluate_vs_random        — MCTS vs uniform-random opponent.
    - evaluate_mcts_vs_reinforce — MCTS vs a trained REINFORCE policy.
    - evaluate_mcts_vs_mcts     — one MCTS vs another MCTS (e.g. 100 sims vs 1000 sims).

Every evaluation swaps nothing internally: ``as_player`` (or the role flag)
controls which side the MCTS-of-interest plays.  This makes it easy to report
win/draw/loss rates both as X and as O.
"""
from __future__ import annotations

from AR1.agents.planning.mcts import MCTSAgent
from AR1.envs.tictactoe import (
    TicTacToeAction,
    TicTacToeEnv,
    TicTacToeState,
    _winner,
)
from AR1.policies.tictactoe import Policy, random_action


# ── Policy wrapper ───────────────────────────────────────────────────────────

def make_mcts_policy(agent: MCTSAgent) -> Policy:
    """Wrap an ``MCTSAgent`` as a ``Policy`` callable compatible with ``play_game``.

    A fresh MCTS search is run from scratch on every move call; no tree is
    reused between moves.

    Args:
        agent: configured MCTSAgent (n_simulations, c already set).

    Returns:
        A ``Policy`` callable: ``(env, state) → action``.
    """

    def policy(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
        return agent.select_action(state, env.current_player)

    return policy


# ── MCTS vs Random ───────────────────────────────────────────────────────────

def evaluate_vs_random(
    env: TicTacToeEnv,
    agent: MCTSAgent,
    n_games: int = 200,
    as_player: int = 1,
) -> tuple[float, float, float]:
    """Evaluate MCTS win/draw/loss rates against a uniform-random opponent.

    Args:
        env:       TicTacToe environment instance.
        agent:     MCTSAgent to evaluate.
        n_games:   number of evaluation games.
        as_player: +1 → MCTS plays X (first mover); -1 → MCTS plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` — three fractions summing to 1.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == as_player:
                action = agent.select_action(state, env.current_player)  # MCTS move
            else:
                action = random_action(env, state)                       # random opponent
            state, _, done = env.step(action)

        winner = _winner(state)
        if winner == as_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins / n_games, draws / n_games, losses / n_games


# ── MCTS vs REINFORCE ────────────────────────────────────────────────────────

def evaluate_mcts_vs_reinforce(
    env: TicTacToeEnv,
    mcts_agent: MCTSAgent,
    reinforce_policy: Policy,
    n_games: int = 200,
    mcts_as_player: int = 1,
) -> tuple[float, float, float]:
    """Pit MCTS against a trained REINFORCE policy.

    Args:
        env:              TicTacToe environment instance.
        mcts_agent:       MCTSAgent (planning, no learning).
        reinforce_policy: trained REINFORCE policy from ``make_reinforce_policy``.
        n_games:          number of games to play.
        mcts_as_player:   +1 → MCTS plays X; -1 → MCTS plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` for MCTS.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == mcts_as_player:
                action = mcts_agent.select_action(state, env.current_player)
            else:
                action = reinforce_policy(env, state)
            state, _, done = env.step(action)

        winner = _winner(state)
        if winner == mcts_as_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins / n_games, draws / n_games, losses / n_games


# ── MCTS vs MCTS ─────────────────────────────────────────────────────────────

def evaluate_mcts_vs_mcts(
    env: TicTacToeEnv,
    agent_a: MCTSAgent,
    agent_b: MCTSAgent,
    n_games: int = 100,
    a_as_player: int = 1,
) -> tuple[float, float, float]:
    """Pit one MCTS agent against another MCTS agent.

    Useful to show that increasing ``n_simulations`` produces stronger play
    (e.g. MCTS-1000 should dominate MCTS-50) and to observe how often two
    strong planners agree and draw on a solved game like Tic-Tac-Toe.

    Args:
        env:          TicTacToe environment instance.
        agent_a:      MCTSAgent whose win rate is being reported.
        agent_b:      opposing MCTSAgent.
        n_games:      number of games to play.
        a_as_player:  +1 → ``agent_a`` plays X; -1 → ``agent_a`` plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` for ``agent_a``.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == a_as_player:
                action = agent_a.select_action(state, env.current_player)
            else:
                action = agent_b.select_action(state, env.current_player)
            state, _, done = env.step(action)

        winner = _winner(state)
        if winner == a_as_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins / n_games, draws / n_games, losses / n_games
