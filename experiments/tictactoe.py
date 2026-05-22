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


# ── Interactive widget-based human play (Jupyter only) ───────────────────────

def play_game_vs_human_widget(
    env,
    agent_policy,
    human_starts: bool = True,
    agent_label: str = "Agent",
):
    """Interactive Tic-Tac-Toe inside a Jupyter notebook (ipywidgets only).

    A 3x3 grid of buttons IS the board: each button shows the cell number
    while empty, an X / O once played.  Click a cell to play; the agent
    responds immediately and the board is redrawn.  No matplotlib backend,
    no extra dependencies beyond the ipywidgets that ship with Jupyter.

    Args:
        env:           a :class:`TicTacToeEnv` instance.
        agent_policy:  callable ``(env, state) -> action`` for the AI side.
        human_starts:  ``True`` if the human plays X (first); ``False`` for O.
        agent_label:   label used in status messages (e.g. ``"MCTS"``).
    """
    import ipywidgets as widgets
    from IPython.display import display

    human_player = 1 if human_starts else -1
    env.reset()

    # ── State holders ─────────────────────────────────────────────────
    finished = {"flag": False}
    status_label = widgets.HTML()
    cell_buttons = [
        widgets.Button(layout=widgets.Layout(width="70px", height="70px"))
        for _ in range(9)
    ]
    grid = widgets.GridBox(
        cell_buttons,
        layout=widgets.Layout(
            grid_template_columns="repeat(3, 70px)",
            grid_gap="6px",
        ),
    )
    new_game_btn = widgets.Button(description="Novo jogo", button_style="info")

    # ── Render helpers ────────────────────────────────────────────────
    def _render():
        for idx, b in enumerate(cell_buttons):
            v = env.board[idx]
            if v == 1:
                b.description = "X"
                b.button_style = "primary"            # blue
                b.disabled = True
            elif v == -1:
                b.description = "O"
                b.button_style = "danger"             # red
                b.disabled = True
            else:
                b.description = str(idx)
                b.button_style = ""
                b.disabled = finished["flag"]

    def _announce_outcome():
        w = _winner(env.board)
        if w == human_player:
            status_label.value = "<b style='color:#2e7d32; font-size:18px'>Ganhaste!</b>"
        elif w == -human_player:
            status_label.value = (
                f"<b style='color:#c62828; font-size:18px'>{agent_label} venceu.</b>"
            )
        else:
            status_label.value = "<b style='color:#ef6c00; font-size:18px'>Empate.</b>"

    def _check_end_after_move() -> bool:
        if env.is_terminal(env.board):
            finished["flag"] = True
            _announce_outcome()
            _render()
            return True
        return False

    def _agent_turn():
        if finished["flag"] or env.is_terminal(env.board):
            return
        status_label.value = f"<i>{agent_label} a pensar...</i>"
        action = agent_policy(env, env.board)
        env.step(action)
        _render()
        if _check_end_after_move():
            return
        status_label.value = "<b>A tua vez.</b>"

    def _on_click(b):
        if finished["flag"]:
            return
        try:
            action = int(b.description)
        except (TypeError, ValueError):
            return
        if env.board[action] != 0:
            return
        env.step(action)
        _render()
        if _check_end_after_move():
            return
        _agent_turn()

    def _on_new_game(_):
        env.reset()
        finished["flag"] = False
        status_label.value = (
            "<b>A tua vez.</b>" if human_starts
            else f"<i>{agent_label} a comecar...</i>"
        )
        _render()
        if not human_starts:
            _agent_turn()

    for b in cell_buttons:
        b.on_click(_on_click)
    new_game_btn.on_click(_on_new_game)

    # ── Initial render ────────────────────────────────────────────────
    status_label.value = (
        "<b>A tua vez.</b>" if human_starts
        else f"<i>{agent_label} a comecar...</i>"
    )
    _render()
    if not human_starts:
        _agent_turn()

    display(widgets.VBox([status_label, grid, new_game_btn]))
