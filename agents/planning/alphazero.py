"""PL9 (extra) — AlphaZero-style agent for Tic-Tac-Toe.

A miniature reproduction of the algorithm from Silver et al. (2017, 2018):

    1.  Policy + value network: a single MLP with two output heads.
            input  : encode_state(board, current_player)      shape (27,)
            hidden : (27 -> 64 -> 64) with ReLU
            policy : 64 -> 9 logits (one per cell)
            value  : 64 -> 1 with tanh                      in [-1, +1]

    2.  PUCT MCTS — exactly like the vanilla MCTS in agents/planning/mcts.py
        except:
          - on a leaf (non-terminal) we ask the network for (pi, v) and
            expand ALL legal children with prior probabilities P from pi,
            then back-propagate v (no random rollout).
          - selection uses the PUCT formula
                a* = argmax_a  Q(s, a) + c_puct * P(a|s) * sqrt(sum_b N_b) / (1 + N(s, a))

    3.  Self-play training loop — at every move:
          - run K simulations
          - sample the move from pi_t = N(s, .) ** (1/temperature), normalised
          - record (state, pi_t, current_player)
        At the end of the game we set z in {-1, 0, +1} (winner's reward) and
        train on  L = MSE(v_pred, z_t) + CE(pi_pred, pi_t) + l2 reg.

The goal is to *demonstrate the algorithm*, not to push the state of the
art on a 3x3 board — Tic-Tac-Toe is solved, so the metric to watch is
whether self-play learns to beat both Random and the trained REINFORCE
agent within a reasonable training budget.
"""
from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from AR1.envs.tictactoe import TicTacToeAction, TicTacToeState, _winner
from AR1.features.tictactoe import STATE_FEATURE_DIM, encode_state

torch.set_num_threads(1)


# ── Pure-function helpers (no env mutation) ───────────────────────────────

def _apply(state: TicTacToeState, action: int, player: int) -> TicTacToeState:
    board = list(state)
    board[action] = player
    return tuple(board)


def _available(state: TicTacToeState) -> list[int]:
    return [i for i, c in enumerate(state) if c == 0]


def _is_terminal(state: TicTacToeState) -> bool:
    return _winner(state) != 0 or 0 not in state


# ── Network: 27 -> 64 -> 64 -> (policy 9, value 1) ────────────────────────

class PolicyValueNet(nn.Module):
    """Small two-headed MLP.

    Trunk: encode_state(27) -> 64 -> 64 (ReLU, ReLU).
    Heads:  policy  : Linear(64, 9)         (logits — softmax applied externally)
            value   : Linear(64, 1) -> tanh in [-1, 1]
    """

    def __init__(self, n_inputs: int = STATE_FEATURE_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 9)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)


# ── PUCT MCTS Node ────────────────────────────────────────────────────────

class _AZNode:
    """One node of the PUCT search tree."""

    __slots__ = (
        "state", "player", "parent", "action_from_parent",
        "prior", "children",
        "N", "W", "Q",
    )

    def __init__(
        self,
        state: TicTacToeState,
        player: int,
        prior: float = 0.0,
        parent: "_AZNode | None" = None,
        action_from_parent: int | None = None,
    ) -> None:
        self.state = state
        self.player = player                 # player to move at this node
        self.prior = prior
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[int, _AZNode] = {}
        # Statistics for *this* node from the perspective of self.player
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        return self.Q


def _puct_score(parent_N: int, child: _AZNode, c_puct: float) -> float:
    """Q(parent, a) + c_puct * P(a) * sqrt(sum_b N_b) / (1 + N(a))."""
    # Q viewed from the parent: a high value for the child (good for child.player)
    # is bad for the parent (parent.player = -child.player), so we negate.
    u = c_puct * child.prior * math.sqrt(parent_N) / (1.0 + child.N)
    return -child.Q + u


# ── AlphaZero MCTS Agent ──────────────────────────────────────────────────

class AlphaZeroMCTS:
    """PUCT MCTS that uses a PolicyValueNet as the leaf evaluator.

    Stateless between games — the search tree is rebuilt at every move.

    Args:
        net               : a trained or partially trained PolicyValueNet.
        n_simulations     : number of MCTS simulations per move.
        c_puct            : PUCT exploration constant (typical: 1.0 - 2.5).
        dirichlet_alpha   : optional Dirichlet noise added to the root priors
                            during self-play (set to None to disable).
        dirichlet_epsilon : mixing weight of Dirichlet noise.
        temperature       : >0  ⇒ sample from N**(1/T); 0 ⇒ argmax N (eval).
        rng               : numpy random generator for Dirichlet noise.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        n_simulations: int = 64,
        c_puct: float = 1.4,
        dirichlet_alpha: float | None = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()

    # ── Network call helpers ──────────────────────────────────────────────

    def _evaluate(self, state: TicTacToeState, player: int) -> tuple[np.ndarray, float]:
        """Run the net on a single (state, player) pair; return (priors over 9, value)."""
        phi = encode_state(state, player)
        x = torch.from_numpy(phi).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(x)
        logits = logits.squeeze(0).numpy()
        # Mask illegal actions, renormalise
        legal = _available(state)
        mask = np.full(9, -np.inf, dtype=np.float32)
        for a in legal:
            mask[a] = logits[a]
        mask -= mask.max()
        probs = np.exp(mask)
        probs /= probs.sum()
        return probs, float(v.item())

    # ── Tree operations ───────────────────────────────────────────────────

    def _expand(self, node: _AZNode, priors: np.ndarray) -> None:
        for action in _available(node.state):
            next_state = _apply(node.state, action, node.player)
            child = _AZNode(
                state=next_state,
                player=-node.player,
                prior=float(priors[action]),
                parent=node,
                action_from_parent=action,
            )
            node.children[action] = child

    def _select(self, node: _AZNode) -> _AZNode:
        while node.expanded and not _is_terminal(node.state):
            node = max(
                node.children.values(),
                key=lambda c: _puct_score(node.N, c, self.c_puct),
            )
        return node

    def _backpropagate(self, node: _AZNode, value: float) -> None:
        """Propagate value (from `node`'s player's perspective) up the tree."""
        current = node
        v = value
        while current is not None:
            current.N += 1
            current.W += v
            current.Q = current.W / current.N
            v = -v   # parent is the opponent
            current = current.parent

    # ── Public API ────────────────────────────────────────────────────────

    def visit_distribution(
        self, state: TicTacToeState, player: int, add_dirichlet: bool = False
    ) -> np.ndarray:
        """Run MCTS from (state, player) and return N(s, a)/sum_b N_b over 9 actions."""
        root = _AZNode(state=state, player=player)
        priors, v = self._evaluate(state, player)

        # Optional Dirichlet noise at the root for self-play exploration
        if add_dirichlet and self.dirichlet_alpha is not None:
            legal = _available(state)
            if legal:
                noise = self.rng.dirichlet([self.dirichlet_alpha] * len(legal))
                for i, a in enumerate(legal):
                    priors[a] = (1 - self.dirichlet_epsilon) * priors[a] \
                                + self.dirichlet_epsilon * noise[i]

        self._expand(root, priors)
        self._backpropagate(root, v)

        for _ in range(self.n_simulations):
            leaf = self._select(root)
            if _is_terminal(leaf.state):
                w = _winner(leaf.state)
                if w == 0:
                    value = 0.0
                else:
                    # Reward from the perspective of `leaf.player` (who is to move
                    # but cannot, because the game is over).  If they ARE the winner
                    # the value is +1 (impossible here — the previous mover won),
                    # so leaf.player loses ⇒ -1 when w == -leaf.player.
                    value = 1.0 if w == leaf.player else -1.0
            else:
                p_leaf, v_leaf = self._evaluate(leaf.state, leaf.player)
                self._expand(leaf, p_leaf)
                value = v_leaf
            self._backpropagate(leaf, value)

        counts = np.zeros(9, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N
        if counts.sum() == 0:
            # No legal actions (shouldn't happen for non-terminal states)
            return counts
        return counts / counts.sum()

    def select_action(
        self,
        state: TicTacToeState,
        player: int,
        add_dirichlet: bool = False,
        temperature: float | None = None,
    ) -> tuple[int, np.ndarray]:
        """Choose an action via the (temperature-tempered) visit-count distribution.

        Returns (action, pi_target) where pi_target is the normalised visit
        distribution used as the policy training target (no temperature applied,
        always raw N/sum N — temperature only affects the sampling).
        """
        counts = self.visit_distribution(state, player, add_dirichlet=add_dirichlet)
        pi_target = counts.copy()              # used by the trainer
        T = self.temperature if temperature is None else temperature

        if T <= 1e-6:
            # Deterministic: most-visited action
            action = int(np.argmax(counts))
            return action, pi_target

        # Temperature-tempered sampling
        scaled = np.power(counts, 1.0 / T)
        s = scaled.sum()
        if s == 0:
            legal = _available(state)
            action = int(self.rng.choice(legal))
            return action, pi_target
        scaled /= s
        action = int(self.rng.choice(9, p=scaled))
        return action, pi_target


# ── Self-play and Training ────────────────────────────────────────────────

def _play_self_play_game(
    mcts: AlphaZeroMCTS, exploration_moves: int = 4
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """One self-play game; returns list of (phi, pi_target, player_at_state).

    The value target z is filled in after the game ends.
    """
    state: TicTacToeState = (0,) * 9
    player = 1
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    move_idx = 0
    while not _is_terminal(state):
        T = mcts.temperature if move_idx < exploration_moves else 0.0
        action, pi = mcts.select_action(
            state, player, add_dirichlet=True, temperature=T,
        )
        history.append((encode_state(state, player), pi, player))
        state = _apply(state, action, player)
        player = -player
        move_idx += 1

    # Outcome from the perspective of player +1 (X)
    w = _winner(state)              # +1, -1, or 0
    return [
        (phi, pi, p, float(w) * p)   # value target z = w·p (so winner sees +1, loser -1, draw 0)
        for (phi, pi, p) in history
    ]


def train_alphazero(
    net: PolicyValueNet,
    n_iterations: int = 5,
    games_per_iter: int = 30,
    n_simulations: int = 32,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    exploration_moves: int = 4,
    batch_size: int = 64,
    epochs_per_iter: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    buffer_size: int = 5_000,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Full AlphaZero training loop.

    Returns a dict with per-iteration metrics (policy loss, value loss, total loss).
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    optimiser = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    replay: list[tuple[np.ndarray, np.ndarray, float]] = []

    history = {"policy_loss": [], "value_loss": [], "total_loss": []}

    for it in range(1, n_iterations + 1):
        # ── 1. Self-play ────────────────────────────────────────────────
        mcts = AlphaZeroMCTS(
            net=net, n_simulations=n_simulations, c_puct=c_puct,
            temperature=temperature, rng=rng,
        )
        n_examples_before = len(replay)
        for _ in range(games_per_iter):
            samples = _play_self_play_game(mcts, exploration_moves=exploration_moves)
            for phi, pi, _player, z in samples:
                replay.append((phi.astype(np.float32), pi.astype(np.float32), float(z)))
        if len(replay) > buffer_size:
            replay = replay[-buffer_size:]
        n_new = len(replay) - max(n_examples_before, 0)
        if verbose:
            print(f"[iter {it}] self-play: +{n_new} samples (buffer={len(replay)})")

        # ── 2. Train ────────────────────────────────────────────────────
        net.train()
        running = {"p": 0.0, "v": 0.0, "t": 0.0, "n": 0}
        for _ in range(epochs_per_iter):
            indices = rng.permutation(len(replay))
            for start in range(0, len(replay), batch_size):
                batch_idx = indices[start:start + batch_size]
                phis  = torch.from_numpy(np.stack([replay[i][0] for i in batch_idx]))
                pis   = torch.from_numpy(np.stack([replay[i][1] for i in batch_idx]))
                zs    = torch.tensor([replay[i][2] for i in batch_idx], dtype=torch.float32)

                logits, v_pred = net(phis)
                # Policy loss = - sum_a pi_target · log_softmax(logits)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(pis * log_probs).sum(dim=1).mean()
                value_loss  = F.mse_loss(v_pred, zs)
                loss = policy_loss + value_loss

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                optimiser.step()

                running["p"] += float(policy_loss.item()) * len(batch_idx)
                running["v"] += float(value_loss.item())  * len(batch_idx)
                running["t"] += float(loss.item())        * len(batch_idx)
                running["n"] += len(batch_idx)
        if running["n"] > 0:
            history["policy_loss"].append(running["p"] / running["n"])
            history["value_loss"].append( running["v"] / running["n"])
            history["total_loss"].append( running["t"] / running["n"])
            if verbose:
                print(
                    f"[iter {it}] train: policy={history['policy_loss'][-1]:.4f}  "
                    f"value={history['value_loss'][-1]:.4f}  "
                    f"total={history['total_loss'][-1]:.4f}"
                )

    return history


# ── Greedy policy wrapper (for evaluation against other agents) ───────────

class AlphaZeroPolicy:
    """Convenience wrapper that exposes a fixed-policy API for evaluation.

    Calling ``policy(env, state)`` returns the most-visited action under the
    trained network and a configurable number of MCTS simulations.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        n_simulations: int = 100,
        c_puct: float = 1.4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._mcts = AlphaZeroMCTS(
            net=net, n_simulations=n_simulations, c_puct=c_puct,
            dirichlet_alpha=None, temperature=0.0, rng=rng,
        )

    def __call__(self, env, state: TicTacToeState) -> int:
        action, _ = self._mcts.select_action(
            state, env.current_player, add_dirichlet=False, temperature=0.0,
        )
        return action
