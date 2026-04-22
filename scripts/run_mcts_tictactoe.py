"""PL9 — Monte Carlo Tree Search on Tic-Tac-Toe.

Model-based planning agent: no weights are learned. A fresh search tree is
built from scratch on every move using the environment itself as a perfect
model of the dynamics.

Three experiments:
    1. Sweep of ``n_simulations`` — MCTS vs random for a range of sim budgets,
       reporting win rate as X and as O.
    2. MCTS vs REINFORCE — reuses the vanilla REINFORCE agent from PL8.
    3. MCTS vs MCTS — one MCTS plays against another MCTS with a different
       ``n_simulations`` budget (e.g. 1000 vs 50) to show that more simulations
       produces strictly stronger play; two equally-strong MCTS draw most
       games, which matches the theoretical optimum for Tic-Tac-Toe.

Plots produced:
    - mcts_vs_random_sweep.png      — win rate as X and as O vs number of sims.
    - mcts_vs_reinforce.png         — win/draw/loss bars, MCTS as X and as O.
    - mcts_vs_mcts.png              — win/draw/loss bars for different sim pairings.

Usage:
    python -m AR1.scripts.run_mcts_tictactoe --no-show
    python -m AR1.scripts.run_mcts_tictactoe --no-reinforce --no-show
    python -m AR1.scripts.run_mcts_tictactoe --play
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="MCTS on Tic-Tac-Toe.")
    # MCTS vs Random sweep
    p.add_argument("--sim-counts", type=int, nargs="+",
                   default=[10, 50, 100, 200, 500, 1000],
                   help="Sim budgets to evaluate against Random.")
    p.add_argument("--eval-games", type=int, default=200,
                   help="Games per evaluation (each role).")
    # MCTS vs REINFORCE
    p.add_argument("--no-reinforce", action="store_true",
                   help="Skip the MCTS-vs-REINFORCE experiment (no training).")
    p.add_argument("--reinforce-episodes", type=int, default=30_000,
                   help="REINFORCE training episodes (used only if --no-reinforce not set).")
    p.add_argument("--mcts-sims-vs-reinforce", type=int, default=1_000,
                   help="n_simulations for the MCTS that faces REINFORCE.")
    # MCTS vs MCTS
    p.add_argument("--mcts-pairings", type=int, nargs="+",
                   default=[1000, 50, 1000, 200, 500, 500],
                   help="Flat list of (strong, weak) pairs. Default: 1000 vs 50, 1000 vs 200, 500 vs 500.")
    p.add_argument("--mcts-vs-mcts-games", type=int, default=50,
                   help="Games per (strong, weak) pairing and per role.")
    # Misc
    p.add_argument("--c", type=float, default=math.sqrt(2),
                   help="UCB1 exploration constant.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--play", action="store_true",
                   help="Play interactively against the MCTS agent.")
    p.add_argument("--play-sims", type=int, default=1_000,
                   help="n_simulations for the --play agent.")
    p.add_argument("--output-dir", type=str, default="outputs/mcts_tictactoe")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_show and not args.play:
        import matplotlib
        matplotlib.use("Agg")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    from AR1.envs.tictactoe import TicTacToeEnv
    from AR1.agents.planning.mcts import MCTSAgent
    from AR1.experiments.mcts_tictactoe import (
        make_mcts_policy,
        evaluate_vs_random,
        evaluate_mcts_vs_reinforce,
        evaluate_mcts_vs_mcts,
    )

    env = TicTacToeEnv()

    # ── Interactive mode: skip everything else ──────────────────────────────
    if args.play:
        agent = MCTSAgent(n_simulations=args.play_sims, c=args.c)
        _play_interactive(env, agent, args.play_sims)
        return

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # ── 1. MCTS vs Random — sweep of n_simulations ──────────────────────────
    print("=" * 60)
    print(" 1. MCTS vs Random — sweep of n_simulations")
    print("=" * 60)

    win_x, win_o, draw_x, draw_o, times = [], [], [], [], []
    for n in args.sim_counts:
        a = MCTSAgent(n_simulations=n, c=args.c)
        t0 = time.time()
        wx, dx, _ = evaluate_vs_random(env, a, n_games=args.eval_games, as_player=1)
        wo, do, _ = evaluate_vs_random(env, a, n_games=args.eval_games, as_player=-1)
        elapsed = time.time() - t0
        win_x.append(wx); win_o.append(wo)
        draw_x.append(dx); draw_o.append(do)
        times.append(elapsed)
        print(f"  n_sims={n:<5d}  win(X)={wx:.1%}  win(O)={wo:.1%}  "
              f"draw(X)={dx:.1%}  draw(O)={do:.1%}  ({elapsed:.1f}s)")

    _plot_sweep(args.sim_counts, win_x, win_o, draw_x, draw_o,
                output_dir / "mcts_vs_random_sweep.png")
    print(f"  ✔ saved {output_dir / 'mcts_vs_random_sweep.png'}")

    # ── 2. MCTS vs REINFORCE ────────────────────────────────────────────────
    reinforce_result = None
    if not args.no_reinforce:
        print()
        print("=" * 60)
        print(f" 2. MCTS({args.mcts_sims_vs_reinforce} sims) vs trained REINFORCE")
        print("=" * 60)

        from AR1.agents.control.reinforce import ReinforceAgent
        from AR1.experiments.reinforce_tictactoe import train, make_reinforce_policy

        print(f"  Training REINFORCE for {args.reinforce_episodes:,} episodes...")
        reinforce_agent = ReinforceAgent(
            alpha=0.02, gamma=1.0, entropy_beta=0.01, seed=args.seed,
        )
        train(reinforce_agent,
              num_episodes=args.reinforce_episodes,
              eval_every=max(1, args.reinforce_episodes),   # skip eval during training
              eval_episodes=1,
              random_opp_fraction=0.5,
              seed=args.seed)
        reinforce_policy = make_reinforce_policy(reinforce_agent, greedy=True)

        mcts_strong = MCTSAgent(n_simulations=args.mcts_sims_vs_reinforce, c=args.c)

        w_x, d_x, l_x = evaluate_mcts_vs_reinforce(
            env, mcts_strong, reinforce_policy,
            n_games=args.eval_games, mcts_as_player=1,
        )
        w_o, d_o, l_o = evaluate_mcts_vs_reinforce(
            env, mcts_strong, reinforce_policy,
            n_games=args.eval_games, mcts_as_player=-1,
        )
        print(f"  MCTS as X vs REINFORCE as O: win={w_x:.1%}  draw={d_x:.1%}  loss={l_x:.1%}")
        print(f"  MCTS as O vs REINFORCE as X: win={w_o:.1%}  draw={d_o:.1%}  loss={l_o:.1%}")

        reinforce_result = {
            "mcts_x": (w_x, d_x, l_x),
            "mcts_o": (w_o, d_o, l_o),
        }
        _plot_mcts_vs_reinforce(
            reinforce_result, args.mcts_sims_vs_reinforce,
            output_dir / "mcts_vs_reinforce.png",
        )
        print(f"  ✔ saved {output_dir / 'mcts_vs_reinforce.png'}")

    # ── 3. MCTS vs MCTS ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" 3. MCTS vs MCTS — effect of simulation budget")
    print("=" * 60)

    pairings = _parse_pairings(args.mcts_pairings)
    mcts_vs_mcts_rows: list[dict] = []
    for strong_n, weak_n in pairings:
        strong = MCTSAgent(n_simulations=strong_n, c=args.c)
        weak = MCTSAgent(n_simulations=weak_n, c=args.c)

        wx, dx, lx = evaluate_mcts_vs_mcts(
            env, strong, weak, n_games=args.mcts_vs_mcts_games, a_as_player=1,
        )
        wo, do, lo = evaluate_mcts_vs_mcts(
            env, strong, weak, n_games=args.mcts_vs_mcts_games, a_as_player=-1,
        )
        print(f"  MCTS({strong_n}) vs MCTS({weak_n})")
        print(f"     strong as X: win={wx:.1%}  draw={dx:.1%}  loss={lx:.1%}")
        print(f"     strong as O: win={wo:.1%}  draw={do:.1%}  loss={lo:.1%}")
        mcts_vs_mcts_rows.append({
            "label": f"{strong_n} vs {weak_n}",
            "x": (wx, dx, lx),
            "o": (wo, do, lo),
        })

    _plot_mcts_vs_mcts(mcts_vs_mcts_rows, output_dir / "mcts_vs_mcts.png")
    print(f"  ✔ saved {output_dir / 'mcts_vs_mcts.png'}")

    print()
    print(f"All plots saved to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


# ── Plot helpers ────────────────────────────────────────────────────────────

def _plot_sweep(sim_counts, win_x, win_o, draw_x, draw_o, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(sim_counts, win_x, "o-", color="#1f77b4", linewidth=2,
            label="Win as X")
    ax.plot(sim_counts, win_o, "s-", color="#ff7f0e", linewidth=2,
            label="Win as O")
    ax.plot(sim_counts, draw_x, "o--", color="#1f77b4", linewidth=1.2,
            alpha=0.5, label="Draw as X")
    ax.plot(sim_counts, draw_o, "s--", color="#ff7f0e", linewidth=1.2,
            alpha=0.5, label="Draw as O")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Rate vs random opponent")
    ax.set_title("MCTS vs Random — effect of simulation budget")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_mcts_vs_reinforce(result, n_sims, out_path):
    import matplotlib.pyplot as plt

    labels = ["MCTS as X", "MCTS as O"]
    wins = [result["mcts_x"][0], result["mcts_o"][0]]
    draws = [result["mcts_x"][1], result["mcts_o"][1]]
    losses = [result["mcts_x"][2], result["mcts_o"][2]]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.bar(x - w, wins, w, label="Win", color="#2ca02c")
    ax.bar(x, draws, w, label="Draw", color="#ff7f0e")
    ax.bar(x + w, losses, w, label="Loss", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"MCTS ({n_sims} sims) vs trained REINFORCE")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_mcts_vs_mcts(rows, out_path):
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True,
                             sharey=True)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        labels = ["strong as X", "strong as O"]
        wins = [row["x"][0], row["o"][0]]
        draws = [row["x"][1], row["o"][1]]
        losses = [row["x"][2], row["o"][2]]

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, wins, w, label="Win", color="#2ca02c")
        ax.bar(x, draws, w, label="Draw", color="#ff7f0e")
        ax.bar(x + w, losses, w, label="Loss", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"MCTS {row['label']}")
        ax.grid(alpha=0.25, axis="y")

    axes[0].set_ylabel("Rate (strong player's perspective)")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("MCTS vs MCTS — effect of simulation budget")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _parse_pairings(flat: list[int]) -> list[tuple[int, int]]:
    """Take a flat list [s1, w1, s2, w2, ...] and return [(s1, w1), (s2, w2), ...]."""
    if len(flat) % 2 != 0:
        raise ValueError("--mcts-pairings must have an even number of entries.")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


# ── Interactive play ─────────────────────────────────────────────────────────

def _play_interactive(env, agent, n_sims: int) -> None:
    print("\n" + "=" * 50)
    print(f"  You (O) vs MCTS with {n_sims} sims (X)")
    print("  Board cells: 0|1|2  3|4|5  6|7|8")
    print("=" * 50)

    while True:
        state = env.reset()
        done = False
        env.render(state)

        while not done:
            if env.current_player == 1:
                print(f"\nMCTS is thinking ({n_sims} sims)...")
                t0 = time.time()
                action = agent.select_action(state, env.current_player)
                print(f"MCTS plays cell {action}  ({time.time() - t0:.1f}s)")
                state, reward, done = env.step(action)
                env.render(state)
                if done:
                    print(">>> MCTS wins!" if reward > 0 else ">>> Draw!")
            else:
                available = env.available_actions(state)
                while True:
                    try:
                        cell = int(input(f"\nYour move (O) — available {available}: "))
                        if cell in available:
                            break
                        print(f"  Cell {cell} not available.")
                    except (ValueError, EOFError):
                        print("  Enter a number 0-8.")
                state, reward, done = env.step(cell)
                print(f"\nYou (O) play cell {cell}:")
                env.render(state)
                if done:
                    print(">>> You win!" if reward > 0 else ">>> Draw!")

        try:
            again = input("\nPlay again? (y/n): ").strip().lower()
            if again != "y":
                break
        except EOFError:
            break


if __name__ == "__main__":
    main()
