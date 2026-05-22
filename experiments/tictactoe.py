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
    env: TicTacToeEnv,
    agent_policy: Policy,
    human_starts: bool = True,
    agent_label: str = "Agent",
    figsize: tuple[float, float] = (4.5, 4.5),
):
    """Interactive Tic-Tac-Toe: click a cell to play, the agent replies.

    Designed for Jupyter notebooks.  Renders the 3x3 board with matplotlib
    and exposes one ipywidgets Button per empty cell so the human plays by
    clicking the cell where they want to place their mark.  After every
    human move the agent ``agent_policy(env, state)`` is invoked, and the
    board is redrawn.

    Args:
        env:           a :class:`TicTacToeEnv` instance.
        agent_policy:  callable ``(env, state) -> action`` for the AI side.
        human_starts:  ``True`` if the human plays X (first); ``False`` for O.
        agent_label:   label used in the status messages (e.g. ``"MCTS"``).
        figsize:       matplotlib figure size for the board.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    human_player = 1 if human_starts else -1

    state = env.reset()

    # ── matplotlib board ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False

    def _draw():
        ax.clear()
        ax.set_xlim(-0.05, 3.05); ax.set_ylim(-0.05, 3.05)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        # Grid
        for i in (1, 2):
            ax.plot([0, 3], [i, i], color="#444", lw=2)
            ax.plot([i, i], [0, 3], color="#444", lw=2)
        # Outer border
        for s in ax.spines.values():
            s.set_color("#444"); s.set_linewidth(2)
        # Pieces
        for idx, v in enumerate(env.board):
            r, c = divmod(idx, 3)
            cx, cy = c + 0.5, 2.5 - r       # rows numbered top to bottom
            if v == 1:
                ax.plot([cx - 0.3, cx + 0.3], [cy - 0.3, cy + 0.3], color="#1e88e5", lw=4)
                ax.plot([cx - 0.3, cx + 0.3], [cy + 0.3, cy - 0.3], color="#1e88e5", lw=4)
            elif v == -1:
                circle = plt.Circle((cx, cy), 0.32, fill=False, color="#e53935", lw=4)
                ax.add_patch(circle)
            else:
                ax.text(cx, cy, str(idx), ha="center", va="center",
                        color="#aaa", fontsize=22, fontweight="bold")
        fig.canvas.draw_idle()

    out = widgets.Output()
    status_label = widgets.HTML(value="<b>Carrega numa célula para jogar.</b>")

    # ── click buttons (one per cell) ───────────────────────────────────────
    cell_buttons = [
        widgets.Button(description=str(i), layout=widgets.Layout(width="60px", height="60px"))
        for i in range(9)
    ]
    grid = widgets.GridBox(
        cell_buttons,
        layout=widgets.Layout(grid_template_columns="repeat(3, 60px)", grid_gap="4px"),
    )

    new_game_btn = widgets.Button(description="Novo jogo", button_style="info")

    finished = {"flag": False}

    def _sync_buttons():
        legal = set(env.available_actions(env.board)) if not finished["flag"] else set()
        for idx, b in enumerate(cell_buttons):
            b.disabled = (idx not in legal)
            # cosmetic label: piece if filled, else cell number
            v = env.board[idx]
            b.description = "X" if v == 1 else ("O" if v == -1 else str(idx))

    def _announce_outcome():
        w = _winner(env.board)
        if w == human_player:
            status_label.value = "<b style='color:#2e7d32'>Ganhaste!</b>"
        elif w == -human_player:
            status_label.value = f"<b style='color:#c62828'>{agent_label} venceu.</b>"
        else:
            status_label.value = "<b style='color:#ef6c00'>Empate.</b>"

    def _agent_turn():
        if env.is_terminal(env.board):
            finished["flag"] = True
            _announce_outcome(); _sync_buttons(); _draw()
            return
        status_label.value = f"<i>{agent_label} a pensar…</i>"
        with out:
            action = agent_policy(env, env.board)
        env.step(action)
        _draw()
        if env.is_terminal(env.board):
            finished["flag"] = True
            _announce_outcome(); _sync_buttons()
            return
        status_label.value = "<b>A tua vez.</b>"
        _sync_buttons()

    def _on_click(b):
        if finished["flag"]:
            return
        action = int(b.description) if b.description.isdigit() else None
        if action is None or env.board[action] != 0:
            return
        env.step(action)
        _draw()
        if env.is_terminal(env.board):
            finished["flag"] = True
            _announce_outcome(); _sync_buttons()
            return
        _sync_buttons()
        _agent_turn()

    def _on_new_game(b):
        env.reset()
        finished["flag"] = False
        status_label.value = (
            "<b>A tua vez.</b>" if human_starts else f"<i>{agent_label} a começar…</i>"
        )
        _draw(); _sync_buttons()
        if not human_starts:
            _agent_turn()

    for b in cell_buttons:
        b.on_click(_on_click)
    new_game_btn.on_click(_on_new_game)

    # ── initial render ────────────────────────────────────────────────────
    _draw(); _sync_buttons()
    if not human_starts:
        # If the agent goes first, fire its move immediately.
        # (status_label.value will be updated inside _agent_turn)
        _agent_turn()

    display(widgets.VBox([
        widgets.HBox([fig.canvas, widgets.VBox([status_label, grid, new_game_btn])]),
        out,
    ]))
