"""PL7 — Tic-Tac-Toe feature encoding (perspective-relative one-hot)."""
from __future__ import annotations

import random

import numpy as np

from AR1.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState

# Each of the 9 cells is encoded as a 3-dim one-hot vector (from the current
# player's perspective): [my piece, opponent's piece, empty].
# Total feature dimension: 9 × 3 = 27.
STATE_FEATURE_DIM: int = 27


def encode_state(board: TicTacToeState, current_player: int) -> np.ndarray:
    """Encode a board as a 27-dim one-hot vector from `current_player`'s perspective.

    For each cell the encoding is:
        [1, 0, 0]  if the cell contains current_player's mark
        [0, 1, 0]  if the cell contains the opponent's mark
        [0, 0, 1]  if the cell is empty

    Args:
        board: length-9 tuple of ints (0 = empty, 1 = X, -1 = O).
        current_player: +1 or -1, identifies whose turn it is.

    Returns:
        np.ndarray of shape (27,), dtype float32.
    """
    # 1. Create a zero array of shape (27,)
    features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)

    # 2. Loop over each cell
    for i, cell in enumerate(board):
        if cell == current_player:
            # Current player's piece
            features[i * 3 + 0] = 1.0
        elif cell == -current_player:
            # Opponent's piece
            features[i * 3 + 1] = 1.0
        else:
            # Empty cell
            features[i * 3 + 2] = 1.0

    # 4. Return the feature vector
    return features


def random_action(env: TicTacToeEnv, state: TicTacToeState) -> TicTacToeAction:
    """Choose a uniformly random legal action. Used as a baseline opponent."""
    return random.choice(env.available_actions(state))
