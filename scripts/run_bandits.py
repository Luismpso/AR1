"""
PL1 — Run Multi-Armed Bandit experiments.

Plots produced:
  - Parameter study: Epsilon-Greedy (epsilon = 0, 0.01, 0.1)
  - Battle of the Bandits: all six algorithms compared

Usage:
    python -m AR1.scripts.run_bandits
    python -m AR1.scripts.run_bandits --runs 500 --no-show
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit experiments.")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per run.")
    parser.add_argument("--runs", type=int, default=2000, help="Number of independent runs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bandits",
        help="Directory inside AR1 where plots are saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def plot_comparison(agents_dict, env, steps, runs, title, output_path, plt):
    from AR1.experiments.bandits import run_bandit_experiment

    print(f"Running: {title}  ({runs} runs x {steps} steps)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for name, agent in agents_dict.items():
        rewards, optimal = run_bandit_experiment(agent, env, steps, runs)
        ax1.plot(rewards, label=name, linewidth=1.8)
        ax2.plot(optimal * 100, label=name, linewidth=1.8)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Average Reward")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal Action")
    ax2.set_title("% Optimal Action")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(0, 100)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    return fig


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from AR1.envs.bandit import KArmedBandit
    from AR1.agents.bandits import (
        EpsilonGreedy, UCB, GradientBandit, ThompsonSampling, Exp3, DecayingEpsilonGreedy,
    )

    env = KArmedBandit()
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parameter study: Epsilon-Greedy
    eps_agents = {
        "epsilon=0 (Greedy)": EpsilonGreedy(epsilon=0),
        "epsilon=0.01": EpsilonGreedy(epsilon=0.01),
        "epsilon=0.1": EpsilonGreedy(epsilon=0.1),
    }
    fig1 = plot_comparison(
        eps_agents, env, args.steps, args.runs,
        "Parameter Study: Epsilon-Greedy",
        output_dir / "epsilon_greedy_study.png", plt,
    )

    # 2. Battle of the Bandits
    all_agents = {
        "Epsilon-Greedy (0.1)": EpsilonGreedy(epsilon=0.1),
        "UCB (c=2)": UCB(c=2),
        "Thompson Sampling": ThompsonSampling(),
        "Decaying Epsilon": DecayingEpsilonGreedy(initial_epsilon=1.0, decay_rate=0.99),
        "Gradient Bandit (0.1)": GradientBandit(alpha=0.1),
        "Exp3 (gamma=0.1)": Exp3(gamma=0.1),
    }
    fig2 = plot_comparison(
        all_agents, env, args.steps, args.runs,
        "Battle of the Bandits: All Algorithms",
        output_dir / "all_algorithms.png", plt,
    )

    print(f"All plots saved to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
