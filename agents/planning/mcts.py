"""PL9 — Monte Carlo Tree Search (MCTS) for Tic-Tac-Toe.

Model-based planning agent: no weights are learned.  On every move a fresh
search tree is built from scratch using the environment itself as a perfect
model of the dynamics.

Each simulation (tree iteration) has four phases:

    1. Selection   — descend with UCB1 until reaching a node that is not
                     fully expanded or is terminal.
    2. Expansion   — add one new child for an untried action.
    3. Simulation  — roll out a random game from the new node to the end.
    4. Backup      — propagate the outcome up the tree, alternating sign
                     at every level (parent is opponent).

Action selection: *most-visited child* of the root (robust best — less
sensitive to outlier rollouts than argmax-Q).
"""
from __future__ import annotations

import math
import random

from AR1.envs.tictactoe import TicTacToeAction, TicTacToeState, _winner


# ── Pure-function helpers (no env mutation) ──────────────────────────────────

def _apply(state: TicTacToeState, action: int, player: int) -> TicTacToeState:
    """Return a new board state after ``player`` marks ``action``."""
    board = list(state)
    board[action] = player
    return tuple(board)


def _available(state: TicTacToeState) -> list[int]:
    """Indices of empty cells in ``state``."""
    return [i for i, c in enumerate(state) if c == 0]


def _is_terminal(state: TicTacToeState) -> bool:
    """True if the game has ended (win or draw)."""
    return _winner(state) != 0 or 0 not in state


# ── MCTS Node ────────────────────────────────────────────────────────────────

class MCTSNode:
    """One node in the MCTS search tree.

    Each node represents a board *state* and the player whose turn it is
    to move from that state (``player``).

    ``value_sum`` accumulates simulation outcomes from **this node's player's
    perspective**: +1 win, -1 loss, 0 draw.

    UCB1 score (evaluated by the *parent*) negates ``value_sum`` because
    the parent is the opponent — what is good for the child is bad for the parent.
    """

    __slots__ = (
        "state",
        "player",
        "parent",
        "action",
        "children",
        "visit_count",
        "value_sum",
        "untried_actions",
    )

    def __init__(
        self,
        state: TicTacToeState,
        player: int,
        parent: MCTSNode | None = None,
        action: int | None = None,
    ) -> None:
        self.state = state
        self.player = player                 # player to move at this node (+1=X, -1=O)
        self.parent = parent                 # None for root
        self.action = action                 # action taken from parent to reach this node
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0          # cumulative outcome from self.player's perspective
        self.untried_actions: list[int] = _available(state)

    @property
    def is_terminal(self) -> bool:
        return _is_terminal(self.state)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb(self, c: float) -> float:
        """UCB1 score for this node, evaluated from the **parent's** perspective.

        The parent's player is the opponent of ``self.player``, so a high
        ``value_sum`` here (good for self) is bad for the parent → negate Q.

            UCB = −Q(s,a) + c · √( ln N_parent / N_child )
        """
        if self.visit_count == 0:
            return float("inf")  # always expand unvisited nodes first
        q = -self.value_sum / self.visit_count        # negate: opponent's gain is our loss
        u = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return q + u

    def best_child(self, c: float) -> MCTSNode:
        """Child with the highest UCB1 score."""
        return max(self.children.values(), key=lambda n: n.ucb(c))

    def expand(self) -> MCTSNode:
        """Add one unexplored child and return it."""
        action = self.untried_actions.pop()                 # pick any untried action
        next_state = _apply(self.state, action, self.player)
        child = MCTSNode(next_state, -self.player, parent=self, action=action)
        self.children[action] = child
        return child

    def backpropagate(self, value: float) -> None:
        """Propagate simulation result up the tree.

        ``value`` is from **this node's player's** perspective.
        Each level up switches player, so we negate the value at every step.
        """
        # TODO (1/3) — SOLVED
        # 1. Increment this node's visit count.
        self.visit_count += 1
        # 2. Accumulate the outcome from self.player's perspective.
        self.value_sum += value
        # 3. Propagate to the parent with the sign flipped (parent is opponent).
        if self.parent is not None:
            self.parent.backpropagate(-value)


# ── MCTS Agent ───────────────────────────────────────────────────────────────

class MCTSAgent:
    """Monte Carlo Tree Search agent for TicTacToe.

    Four phases per simulation:

    1. **Selection**   — traverse the tree with UCB1 until reaching a node
                        that is not fully expanded or is terminal.
    2. **Expansion**   — add one new child for an untried action.
    3. **Simulation**  — roll out a random game from the new node to the end.
    4. **Backup**      — propagate the outcome up to the root, alternating sign.

    No weights are ever learned; the environment is used as a perfect model.
    Action selection: **most-visited child** (robust best).

    Args:
        n_simulations: number of MCTS simulations (tree iterations) per move.
        c:             exploration constant in UCB1 (√2 is the theoretical value).
    """

    def __init__(self, n_simulations: int = 500, c: float = math.sqrt(2)) -> None:
        self.n_simulations = n_simulations
        self.c = c

    def reset(self) -> None:
        """Stateless between games — nothing to reset."""
        pass

    def select_action(self, state: TicTacToeState, player: int) -> TicTacToeAction:
        """Run MCTS and return the best action for ``player`` from ``state``."""
        action, _ = self.search(state, player)
        return action

    def search(
        self, state: TicTacToeState, player: int
    ) -> tuple[TicTacToeAction, MCTSNode]:
        """Run MCTS and return both the best action and the root node.

        The root node gives access to the full search tree for inspection
        (visit counts, value estimates, UCB scores).
        """
        root = MCTSNode(state, player)

        for _ in range(self.n_simulations):
            # 1 & 2. Selection + Expansion — descend and add one new node
            node = self._select(root)
            # 3. Simulation — random playout from the new node
            value = self._rollout(node.state, node.player)
            # 4. Backup — update visit counts and value sums
            node.backpropagate(value)

        # Robust best: most-visited child
        best = max(root.children.values(), key=lambda n: n.visit_count)
        return best.action, root

    # ── Private helpers ──────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend with UCB1 until we find a node we can expand (or a terminal)."""
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node.expand()         # expand and return the new child
            node = node.best_child(self.c)   # fully expanded: follow best UCB child
        return node                          # terminal: backprop from here directly

    def _rollout(self, state: TicTacToeState, player: int) -> float:
        """Random playout from ``state``; returns outcome from ``player``'s perspective.

        Uses pure-function helpers so the environment object is never mutated.
        """
        # TODO (2/3) — SOLVED: random default rollout policy.
        current = player
        while not _is_terminal(state):
            action = random.choice(_available(state))   # uniform random legal action
            state = _apply(state, action, current)      # apply it (pure function)
            current = -current                          # switch player

        # TODO (3/3) — SOLVED: return outcome from the *original* player's perspective.
        w = _winner(state)
        if w == player:
            return 1.0         # original player won
        if w == 0:
            return 0.0         # draw
        return -1.0            # opponent won
