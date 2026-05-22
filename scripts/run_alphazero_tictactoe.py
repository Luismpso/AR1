"""PL9 (extra) — Train an AlphaZero-style agent on Tic-Tac-Toe and evaluate it.

Pipeline:
    1. Train a small (27→64→64→{9, 1}) policy+value network by self-play
       with PUCT MCTS as the search policy.
    2. Evaluate the trained agent against three opponents:
            (a) uniform-random
            (b) a freshly trained REINFORCE-with-baseline agent
            (c) the vanilla MCTS agent (rollout-based, no network)
       Each evaluation reports win/draw/loss as X and as O.
    3. Save loss curves and bar charts of the evaluation in
       outputs/alphazero_tictactoe/.

This script requires PyTorch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AlphaZero-style on Tic-Tac-Toe.")
    p.add_argument("--iterations",       type=int,   default=4)
    p.add_argument("--games-per-iter",   type=int,   default=20)
    p.add_argument("--sims",             type=int,   default=32)
    p.add_argument("--epochs-per-iter",  type=int,   default=4)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--c-puct",           type=float, default=1.4)
    p.add_argument("--temperature",      type=float, default=1.0)
    p.add_argument("--exploration-moves",type=int,   default=4)
    p.add_argument("--buffer-size",      type=int,   default=5_000)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--eval-games",       type=int,   default=50)
    p.add_argument("--eval-sims",        type=int,   default=64)
    p.add_argument(
        "--skip-reinforce", action="store_true",
        help="Skip training a fresh REINFORCE agent for evaluation.",
    )
    p.add_argument("--reinforce-episodes", type=int, default=1000)
    p.add_argument("--mcts-sims",        type=int,   default=200,
                   help="n_simulations for the vanilla MCTS opponent.")
    p.add_argument("--output-dir",       type=str,
                   default="outputs/alphazero_tictactoe")
    p.add_argument("--no-show",          action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import torch  # noqa: F401
        from AR1.agents.planning.alphazero import (
            AlphaZeroPolicy, PolicyValueNet, train_alphazero,
        )
    except ImportError as exc:
        print("Cannot import AlphaZero — PyTorch is required.")
        print(f"  {exc}")
        return

    from AR1.envs.tictactoe import TicTacToeEnv
    from AR1.agents.planning.mcts import MCTSAgent
    from AR1.experiments.mcts_tictactoe import (
        evaluate_vs_random,
        evaluate_mcts_vs_reinforce,
        evaluate_mcts_vs_mcts,
        make_mcts_policy,
    )

    # ── 1. Train AlphaZero ────────────────────────────────────────────
    print(f"Training AlphaZero — {args.iterations} iterations × "
          f"{args.games_per_iter} self-play games × {args.sims} sims/move")
    net = PolicyValueNet(hidden=64)
    history = train_alphazero(
        net=net,
        n_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        n_simulations=args.sims,
        c_puct=args.c_puct,
        temperature=args.temperature,
        exploration_moves=args.exploration_moves,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Loss curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["policy_loss"], label="policy loss (CE)",   marker="o")
    ax.plot(history["value_loss"],  label="value loss (MSE)",   marker="s")
    ax.plot(history["total_loss"],  label="total loss",         marker="^", alpha=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("AlphaZero — training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "training_loss.png", dpi=150, bbox_inches="tight")

    # ── 3. Build evaluation wrapper ──────────────────────────────────
    az_policy = AlphaZeroPolicy(
        net=net, n_simulations=args.eval_sims, c_puct=args.c_puct,
        rng=np.random.default_rng(args.seed + 1),
    )

    # Reuse the MCTS evaluation helpers — they take any `policy(env, state)` callable.
    # The trick: we wrap az_policy as an MCTS-like agent by passing it where the helpers
    # expect a "policy" or "agent" — both APIs are compatible.

    # ── 4. AlphaZero vs Random ───────────────────────────────────────
    print("\nEvaluating AlphaZero vs Random...")
    az_results: dict[str, tuple[float, float, float]] = {}
    env = TicTacToeEnv()

    # We need a generic helper because evaluate_vs_random takes MCTSAgent (calls select_action).
    # az_policy is callable as (env, state) → int; we adapt it here.
    class _PolicyAdapter:
        def __init__(self, p): self.p = p
        def select_action(self, state, current_player):
            class _E:
                current_player = None
            _E.current_player = current_player
            return self.p(_E, state)

    adapter = _PolicyAdapter(az_policy)

    for role, role_name in [(1, "X"), (-1, "O")]:
        wr, dr, lr = evaluate_vs_random(env, adapter, n_games=args.eval_games, as_player=role)
        print(f"  as {role_name}: win={wr:.1%}  draw={dr:.1%}  loss={lr:.1%}")
        az_results[f"random_{role_name}"] = (wr, dr, lr)

    # ── 5. AlphaZero vs vanilla MCTS ─────────────────────────────────
    print(f"\nEvaluating AlphaZero vs MCTS({args.mcts_sims}-sims)...")
    mcts = MCTSAgent(n_simulations=args.mcts_sims)
    for role, role_name in [(1, "X"), (-1, "O")]:
        wr, dr, lr = evaluate_mcts_vs_mcts(env, adapter, mcts, n_games=args.eval_games, a_as_player=role)
        print(f"  AlphaZero as {role_name} vs MCTS: win={wr:.1%}  draw={dr:.1%}  loss={lr:.1%}")
        az_results[f"mcts_{role_name}"] = (wr, dr, lr)

    # ── 6. AlphaZero vs REINFORCE ────────────────────────────────────
    if not args.skip_reinforce:
        print(f"\nTraining REINFORCE baseline ({args.reinforce_episodes} episodes)...")
        from AR1.agents.control.reinforce import ReinforceAgent
        from AR1.experiments.reinforce_tictactoe import train as train_reinforce

        reinforce_agent = ReinforceAgent(alpha=0.01, use_baseline=True, seed=args.seed + 2)
        train_reinforce(
            agent=reinforce_agent, num_episodes=args.reinforce_episodes, eval_every=10**9,

        )

        # Wrap REINFORCE as a policy callable
        from AR1.features.tictactoe import encode_state

        def reinforce_policy(env_, state_):
            phi = encode_state(state_, env_.current_player)
            avail = env_.available_actions(state_)
            return reinforce_agent.greedy_action(phi, avail)

        for role, role_name in [(1, "X"), (-1, "O")]:
            wr, dr, lr = evaluate_mcts_vs_reinforce(
                env, adapter, reinforce_policy,
                n_games=args.eval_games, mcts_as_player=role,
            )
            print(f"  AlphaZero as {role_name} vs REINFORCE: win={wr:.1%}  draw={dr:.1%}  loss={lr:.1%}")
            az_results[f"reinforce_{role_name}"] = (wr, dr, lr)

    # ── 7. Plot evaluation summary ────────────────────────────────────
    labels = list(az_results.keys())
    wins   = [az_results[l][0] for l in labels]
    draws  = [az_results[l][1] for l in labels]
    losses = [az_results[l][2] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, wins,                       label="win",  color="#4caf50")
    ax.bar(x, draws,  bottom=wins,        label="draw", color="#ffb300")
    ax.bar(x, losses, bottom=np.array(wins) + np.array(draws), label="loss", color="#e53935")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of games")
    ax.set_title("AlphaZero evaluation — win/draw/loss vs each opponent")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved plots to {output_dir}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
