"""PL8 — REINFORCE on Tic-Tac-Toe: vanilla vs baseline comparison.

Trains two REINFORCE agents via self-play:
  1. Vanilla REINFORCE — raw returns G_t
  2. REINFORCE with baseline — advantage delta_t = G_t - V(s_t)
     using a learned linear value function (Sutton & Barto, Sec. 13.4)

The baseline reduces variance without introducing bias, which typically
leads to faster and more stable convergence.

Plots produced:
  - Win rate comparison (vanilla vs baseline)
  - Loss curve comparison
  - Final evaluation bar chart

Usage:
    python -m AR1.scripts.run_reinforce_tictactoe --no-show
    python -m AR1.scripts.run_reinforce_tictactoe --play
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE on Tic-Tac-Toe.")
    p.add_argument("--episodes", type=int, default=50_000)
    p.add_argument("--alpha", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--entropy-beta", type=float, default=0.01)
    p.add_argument("--random-frac", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--play", action="store_true", help="Play interactively against the best agent.")
    p.add_argument("--output-dir", type=str, default="outputs/reinforce_tictactoe")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_show and not args.play:
        import matplotlib
        matplotlib.use("Agg")

    from AR1.agents.control.reinforce import ReinforceAgent
    from AR1.envs.tictactoe import TicTacToeEnv
    from AR1.experiments.reinforce_tictactoe import train, evaluate_vs_random

    common = dict(
        alpha=args.alpha, gamma=args.gamma,
        entropy_beta=args.entropy_beta, seed=args.seed,
    )

    # Train both variants
    agents = {
        "Vanilla REINFORCE": ReinforceAgent(**common, use_baseline=False),
        "REINFORCE + baseline": ReinforceAgent(**common, use_baseline=True),
    }
    all_results = {}

    for name, agent in agents.items():
        print(f"Training {name} for {args.episodes:,} episodes...")
        results = train(
            agent, num_episodes=args.episodes,
            eval_every=args.eval_every, eval_episodes=args.eval_games,
            random_opp_fraction=args.random_frac, seed=args.seed,
        )
        all_results[name] = results

        env = TicTacToeEnv()
        wr_x, dr_x, lr_x = evaluate_vs_random(env, agent, 2000, as_player=1)
        wr_o, dr_o, lr_o = evaluate_vs_random(env, agent, 2000, as_player=-1)
        results["final_x"] = (wr_x, dr_x, lr_x)
        results["final_o"] = (wr_o, dr_o, lr_o)
        print(f"  As X: {wr_x:.1%} win, {dr_x:.1%} draw, {lr_x:.1%} loss")
        print(f"  As O: {wr_o:.1%} win, {dr_o:.1%} draw, {lr_o:.1%} loss")

    if args.play:
        best_name = max(all_results, key=lambda n: all_results[n]["final_x"][0])
        print(f"\nPlaying against {best_name}...")
        _play_interactive(agents[best_name], best_name)
        return

    # Plots
    import matplotlib.pyplot as plt

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Vanilla REINFORCE": "#1f77b4", "REINFORCE + baseline": "#ff7f0e"}

    # 1. Win rate comparison (as X)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for name, res in all_results.items():
        chk = res["eval_checkpoints"]
        ax.plot(chk, res["win_rates_as_x"], "o-", label=f"{name} (as X)",
                linewidth=2, markersize=3, color=colors[name])
        ax.plot(chk, res["win_rates_as_o"], "s--", label=f"{name} (as O)",
                linewidth=1.5, markersize=3, color=colors[name], alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate vs random")
    ax.set_title("REINFORCE: vanilla vs baseline — win rate over training")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "win_rate_comparison.png", dpi=150, bbox_inches="tight")

    # 2. Loss curve comparison
    fig2, ax2 = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    window = min(500, args.episodes // 10)
    for name, res in all_results.items():
        losses = np.array(res["losses"])
        rolling = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax2.plot(np.arange(window - 1, len(losses)), rolling,
                 linewidth=1.5, label=name, color=colors[name])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(f"Loss (rolling {window})")
    ax2.set_title("REINFORCE: policy gradient loss comparison")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)
    fig2.savefig(output_dir / "loss_comparison.png", dpi=150, bbox_inches="tight")

    # 3. Final evaluation bar chart
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, role, key in zip(axes3, ["As X", "As O"], ["final_x", "final_o"]):
        names = list(all_results.keys())
        wins = [all_results[n][key][0] for n in names]
        draws = [all_results[n][key][1] for n in names]
        losses_pct = [all_results[n][key][2] for n in names]
        x = np.arange(len(names))
        w = 0.22
        ax.bar(x - w, wins, w, label="Win", color="#2ca02c")
        ax.bar(x, draws, w, label="Draw", color="#ff7f0e")
        ax.bar(x + w, losses_pct, w, label="Loss", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("REINFORCE", "REINF.") for n in names], fontsize=9)
        ax.set_ylabel("Rate")
        ax.set_title(f"Greedy eval {role} (2000 games)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig3.savefig(output_dir / "eval_comparison.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


def _play_interactive(agent, agent_name):
    from AR1.envs.tictactoe import TicTacToeEnv
    from AR1.features.tictactoe import encode_state

    env = TicTacToeEnv()
    print("\n" + "=" * 40)
    print(f"  You (O) vs {agent_name} (X)")
    print("  Board:  0|1|2  3|4|5  6|7|8")
    print("=" * 40)

    while True:
        state = env.reset()
        done = False
        env.render(state)

        while not done:
            if env.current_player == 1:
                phi = encode_state(state, 1)
                available = env.available_actions(state)
                action = agent.greedy_action(phi, available)
                state, reward, done = env.step(action)
                print(f"\n{agent_name} (X) plays cell {action}:")
                env.render(state)
                if done:
                    print(">>> X wins!" if reward > 0 else ">>> Draw!")
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
