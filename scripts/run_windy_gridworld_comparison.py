"""PL5 — Windy Gridworld: compare MC Control, SARSA, and n-step SARSA.

Plots produced:
  - Episode lengths comparison
  - Episode rewards comparison
  - Overlaid greedy paths

Usage:
    python -m AR1.scripts.run_windy_gridworld_comparison
    python -m AR1.scripts.run_windy_gridworld_comparison --no-show
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
    parser = argparse.ArgumentParser(description="Windy Gridworld algorithm comparison.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/windy_gridworld_comparison",
        help="Directory inside AR1 where plots are saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from AR1.agents.control import MonteCarloControl, SarsaControl, NStepSarsaControl
    from AR1.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from AR1.experiments.control import greedy_path, greedy_policy_from_agent, train_control_agent

    env = WindyGridworldEnv()

    # 1. Initialize the three agents
    agent_mc = MonteCarloControl(actions=ACTIONS, epsilon=0.1, gamma=1.0, seed=args.seed)
    agent_sarsa = SarsaControl(actions=ACTIONS, alpha=0.5, epsilon=0.1, gamma=1.0, seed=args.seed)
    agent_n_step = NStepSarsaControl(actions=ACTIONS, n_steps=4, alpha=0.5, epsilon=0.1, gamma=1.0, seed=args.seed)

    # 2. Train all agents
    print("Training Monte Carlo Control...")
    len_mc, rew_mc = train_control_agent(env, agent_mc, args.episodes, max_steps=args.max_steps)

    print("Training SARSA...")
    len_sarsa, rew_sarsa = train_control_agent(env, agent_sarsa, args.episodes, max_steps=args.max_steps)

    print("Training 4-step SARSA...")
    len_n_step, rew_n_step = train_control_agent(env, agent_n_step, args.episodes, max_steps=args.max_steps)

    # 3. Extract greedy paths
    pol_mc = greedy_policy_from_agent(env, agent_mc)
    path_mc = greedy_path(env, pol_mc)
    pol_sarsa = greedy_policy_from_agent(env, agent_sarsa)
    path_sarsa = greedy_path(env, pol_sarsa)
    pol_nstep = greedy_policy_from_agent(env, agent_n_step)
    path_nstep = greedy_path(env, pol_nstep)

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # PLOT 1: Episode Lengths
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(len_mc, label="MC Control", alpha=0.8)
    ax.plot(len_sarsa, label="SARSA", alpha=0.8)
    ax.plot(len_n_step, label="4-step SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Comparison: Episode Length Over Training")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(output_dir / "combined_lengths.png", dpi=150, bbox_inches="tight")

    # PLOT 2: Episode Rewards
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(rew_mc, label="MC Control", alpha=0.8)
    ax.plot(rew_sarsa, label="SARSA", alpha=0.8)
    ax.plot(rew_n_step, label="4-step SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Comparison: Episode Reward Over Training")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(output_dir / "combined_rewards.png", dpi=150, bbox_inches="tight")

    # PLOT 3: Overlaid greedy paths
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    def get_coords(path):
        return [s[1] for s in path], [s[0] for s in path]

    xs_mc, ys_mc = get_coords(path_mc)
    xs_sarsa, ys_sarsa = get_coords(path_sarsa)
    xs_nstep, ys_nstep = get_coords(path_nstep)

    ax.plot(xs_mc, ys_mc, marker="o", label="MC Control", alpha=0.7, linewidth=2, markersize=6)
    ax.plot(xs_sarsa, ys_sarsa, marker="s", label="SARSA", alpha=0.7, linewidth=2, markersize=6)
    ax.plot(xs_nstep, ys_nstep, marker="^", label="4-step SARSA", alpha=0.7, linewidth=2, markersize=8)

    # Mark start and goal
    start_state = path_mc[0]
    goal_state = path_mc[-1]
    ax.scatter([start_state[1]], [start_state[0]], color="green", s=200, marker="*", zorder=5, label="Start")
    ax.scatter([goal_state[1]], [goal_state[0]], color="gold", s=200, marker="*", zorder=5, label="Goal")

    # In classic gridworlds the Y axis (rows) grows top-down
    ax.invert_yaxis()
    ax.set_title("Comparison: Greedy Paths Found")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(output_dir / "combined_paths.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved comparison plots to {output_dir}")
    print("Final greedy path lengths:")
    print(f"  MC Control:   {len(path_mc) - 1}")
    print(f"  SARSA:        {len(path_sarsa) - 1}")
    print(f"  4-step SARSA: {len(path_nstep) - 1}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
