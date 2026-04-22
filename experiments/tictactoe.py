"""PL7 — Tic-Tac-Toe game runner (evaluate two policies against each other)."""
from __future__ import annotations

from typing import Callable

from AR1.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState, _winner

# Policy type: a callable that takes (env, state) and returns an action.
Policy = Callable[[TicTacToeEnv, TicTacToeState], TicTacToeAction]


def play_game(
    env: TicTacToeEnv, # the environment instance to use for the game
    policy_x: Policy, # policy for player X (the first player, represented by +1) , if you want to play as X, just pass a policy that selects your moves
    policy_o: Policy, # policy for player O (the second player, represented by -1), if you want to play as O, just pass a policy that selects your moves
    render: bool = True, # whether to print the board after each move
) -> int: 
    """Play one full game between two policies, optionally rendering each step.

    Args:
        env: the TicTacToeEnv instance.
        policy_x: callable (env, state) -> action for player X (+1).
        policy_o: callable (env, state) -> action for player O (-1).
        render: if True, print the board after every move.

    Returns:
        1 if X wins, -1 if O wins, 0 for a draw.
    """
    state = env.reset()
    if render:
        print("Initial board:")
        env.render(state)
        print()

    while not env.is_terminal(state):
        player_label = "X" if env.current_player == 1 else "O"

        # Select the correct policy for the current player and get an action
        policy = policy_x if env.current_player == 1 else policy_o
        action = policy(env, state)

        state, reward, done = env.step(action)

        if render:
            print(f"Player {player_label} plays cell {action}:")
            env.render(state)
            print()

    result = _winner(state)
    if render:
        if result == 1:
            print("X wins!")
        elif result == -1:
            print("O wins!")
        else:
            print("Draw!")
    return result

def play_game_vs_human(
    env: TicTacToeEnv,
    agent_policy: Policy,
    human_starts: bool = True,
    render: bool = True,
) -> int:
    """Play one full game against a trained agent, interacting via console."""
    state = env.reset()
    if render:
        print("Initial board:")
        env.render(state)
        print()

    human_player = 1 if human_starts else -1

    while not env.is_terminal(state):
        player_label = "X" if env.current_player == 1 else "O"
        is_human_turn = (env.current_player == human_player)

        if is_human_turn:
            valid_move = False
            while not valid_move:
                try:
                    # Input minimalista para não sujar muito o terminal
                    action_str = input(f"Enter move for Player {player_label} (0-8): ")
                    action = int(action_str)
                    
                    if 0 <= action <= 8 and state[action] == 0:
                        valid_move = True
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input.")
        else:
            action = agent_policy(env, state)

        state, reward, done = env.step(action)

        # Output rigorosamente igual ao do professor
        if render:
            print(f"Player {player_label} plays cell {action}:")
            env.render(state)
            print()

    result = _winner(state)
    
    # Output final rigorosamente igual ao do professor
    if render:
        if result == 1:
            print("X wins!")
        elif result == -1:
            print("O wins!")
        else:
            print("Draw!")
            
    return result