from __future__ import annotations
from random import random

from AR1.core.base import Environment

# ── Type aliases ────────────────────────────────────────────────────────────
# The board is a 9-tuple of ints (one per cell, row-major):
#   0 = empty, 1 = player X, -1 = player O
# Actions are integers 0-8 identifying the cell to mark.
TicTacToeState  = tuple[int, ...]   # length-9
TicTacToeAction = int               # 0 … 8

# Indices of every winning line (rows, columns, diagonals)
_WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def _winner(board: TicTacToeState) -> int:
    """Return 1 if X wins, -1 if O wins, 0 otherwise."""
    for i, j, k in _WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class TicTacToeEnv(Environment[TicTacToeState, TicTacToeAction]):
    """Two-player Tic-Tac-Toe environment.

    Conventions:
    - Player X always goes first (represented as +1 in the board).
    - Player O is represented as -1.
    - `current_player` alternates between 1 (X) and -1 (O) each step.
    - The state is a length-9 tuple representing all 9 cells row-major:
        indices  0 1 2
                 3 4 5
                 6 7 8
    - `step()` applies the current player's move, then switches turns.
    - Episode ends when a player wins or the board is full (draw).
    - Rewards from the perspective of the player who just moved:
        +1  for winning
        -1  for losing (opponent wins — not possible in one step, included for completeness)
         0  otherwise (ongoing or draw)

    For self-play, call `reset()` at the start of each game and alternate
    calling `step()` for player X and player O.
    """

    def __init__(self) -> None:
        self.board: TicTacToeState = (0,) * 9
        self.current_player: int = 1  # X starts

    def reset(self) -> TicTacToeState:
        # 1. Initialize board with nine zeros
        self.board = (0,) * 9
        # 2. Player X (1) goes first
        self.current_player = 1
        # 3. Return the initial state
        return self.board

    def available_actions(self, state: TicTacToeState) -> list[TicTacToeAction]:
        # Return indices of all empty cells (value == 0)
        return [i for i, val in enumerate(state) if val == 0]

    def is_terminal(self, state: TicTacToeState) -> bool:
        # 1. Check if someone won
        if _winner(state) != 0:
            return True
        # 2. Check for draw (no empty cells left)
        return 0 not in state

    def step(self, action: TicTacToeAction) -> tuple[TicTacToeState, float, bool]:
        # 1. Validate that the move is legal
        if self.board[action] != 0:
            raise ValueError(f"Illegal move: cell {action} is not empty.")

        # 2. Build new board (tuples are immutable, so convert to list and back)
        new_board_list = list(self.board)
        new_board_list[action] = self.current_player
        new_board = tuple(new_board_list)

        # 3. Check for a winner
        winner = _winner(new_board)
        
        # 4. Check if the game is over (winner or no empty cells)
        done = (winner != 0) or (0 not in new_board)

        # 5. Reward (+1 if current player won, 0 otherwise)
        reward = 1.0 if winner == self.current_player else 0.0

        # 6. Switch player (+1 becomes -1 and vice versa)
        self.current_player *= -1
        
        # 7. Update internal board state
        self.board = new_board

        # 8. Return (new_board, reward, done)
        return new_board, reward, done

    def render(self, state: TicTacToeState | None = None) -> None:
        # 1. Default to self.board if no state given
        if state is None:
            state = self.board
            
        # 2. Symbol mapping
        chars = {1: 'X', -1: 'O', 0: '.'}
        
        # 3. Print the 3x3 board
        for row in range(3):
            row_str = " ".join(chars[state[row * 3 + col]] for col in range(3))
            print(row_str)
        print()  # Blank line for readability