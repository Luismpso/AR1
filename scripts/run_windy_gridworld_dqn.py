"""Run DQN on the Windy Gridworld and compare against Linear SARSA.

Goal: show that a small MLP (2 hidden layers x 64 units) can learn the same
task that Linear SARSA solves with tile coding, on the *same* episodes
budget.  The two agents are trained side-by-side and the script produces
side-by-side learning curves + greedy policies.

Plots produced (saved to outputs/windy_gridworld_dqn/):
    - lengths_comparison.png   - episode length per training episode
    - loss_curve.png           - DQN MSE loss per episode (mean)
    - td_errors_comparison.png - DQN loss vs Linear SARSA |TD error|
    - value_heatmaps.png       - V(s) heatmaps for both agents side by side
    - policies_comparison.png  - greedy policy grids for both agents
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
    p = argparse.ArgumentParser(description="DQN vs Linear SARSA on Windy Gridworld.")
    p.add_argument("--episodes", type=int, default=300, help="Episodes per agent.")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--dqn-lr", type=float, default=5e-4, help="Adam learning rate for DQN.")
    p.add_argument("--dqn-batch", type=int, default=32, help="DQN minibatch size.")
    p.add_argument("--dqn-buffer", type=int, default=10_000, help="Replay buffer size.")
    p.add_argument("--dqn-warmup", type=int, default=200, help="Steps before first DQN update.")
    p.add_argument("--dqn-sync", type=int, default=100, help="Target network sync period (updates).")
    p.add_argument("--dqn-hidden", type=int, default=64, help="Hidden width of the DQN MLP.")
    p.add_argument("--lin-alpha", type=float, default=0.5, help="Linear SARSA step-size.")
    p.add_argument("--lin-epsilon", type=float, default=0.1, help="Linear SARSA epsilon.")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/windy_gridworld_dqn",
        help="Output directory inside AR1/",
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    from AR1.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from AR1.experiments.control import greedy_path, greedy_policy_from_agent
    from AR1.experiments.fa_training import train_fa_agent
    from AR1.plots.windy_gridworld import plot_policy

    # Heavy imports gated so the missing-torch case prints a clean error.
    try:
        from AR1.agents.control.dqn import DQNControl
    except ImportError as exc:
        print("Cannot import DQN — PyTorch is required for this script.")
        print(f"  {exc}")
        return

    from AR1.agents.control.linear_sarsa import LinearSarsaControl
    from AR1.features.windy_gridworld import (
        STATE_ACTION_FEATURE_DIM,
        state_action_features,
    )

    env_dqn = WindyGridworldEnv()
    env_lin = WindyGridworldEnv()

    # ── DQN ────────────────────────────────────────────────────────────
    dqn = DQNControl(
        actions=ACTIONS,
        env=env_dqn,
        alpha=args.dqn_lr,
        batch_size=args.dqn_batch,
        buffer_size=args.dqn_buffer,
        warmup_steps=args.dqn_warmup,
        target_sync_every=args.dqn_sync,
        hidden=args.dqn_hidden,
        gamma=args.gamma,
        seed=args.seed,
    )

    # ── Linear SARSA (baseline) ───────────────────────────────────────
    _phi_cache: dict = {}

    def phi(s, a):
        key = (s, a)
        if key not in _phi_cache:
            _phi_cache[key] = state_action_features(s, a, env_lin)
        return _phi_cache[key]

    lin = LinearSarsaControl(
        actions=ACTIONS, phi=phi, n_features=STATE_ACTION_FEATURE_DIM,
        alpha=args.lin_alpha, epsilon=args.lin_epsilon, gamma=args.gamma,
        seed=args.seed,
    )

    print(f"Training DQN for {args.episodes} episodes...")
    dqn_lengths, _, dqn_errors = train_fa_agent(env_dqn, dqn, args.episodes, max_steps=args.max_steps)
    print(f"Training Linear SARSA for {args.episodes} episodes...")
    lin_lengths, _, lin_errors = train_fa_agent(env_lin, lin, args.episodes, max_steps=args.max_steps)

    dqn_policy = greedy_policy_from_agent(env_dqn, dqn)
    lin_policy = greedy_policy_from_agent(env_lin, lin)
    dqn_path = greedy_path(env_dqn, dqn_policy)
    lin_path = greedy_path(env_lin, lin_policy)

    print(f"Final greedy path length — DQN: {len(dqn_path)-1}, Linear SARSA: {len(lin_path)-1}")

    # ── Plots ─────────────────────────────────────────────────────────
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dqn_lengths, label="DQN (MLP 64×64)", alpha=0.8)
    ax.plot(lin_lengths, label="Linear SARSA (tile coding)", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps to goal")
    ax.set_title(f"Windy Gridworld — episode length over {args.episodes} episodes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "lengths_comparison.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dqn_errors, label="DQN loss (MSE)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean loss")
    ax.set_yscale("log")
    ax.set_title("DQN — MSE loss per episode")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    axes[0].plot(dqn_errors, color="C0")
    axes[0].set_title("DQN loss (MSE) per episode")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Mean loss")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3, which="both")
    axes[1].plot(lin_errors, color="C1")
    axes[1].set_title("Linear SARSA |TD error| per episode")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Mean |TD error|")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Learning signal — DQN vs Linear SARSA")
    fig.tight_layout()
    fig.savefig(output_dir / "td_errors_comparison.png", dpi=150, bbox_inches="tight")

    # Value heatmaps
    def v_dqn(s):
        return max(dqn.action_value_of(s, a) for a in ACTIONS)

    def v_lin(s):
        return max(lin.action_value_of(s, a) for a in ACTIONS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, vfn) in zip(axes, [("DQN", v_dqn), ("Linear SARSA", v_lin)]):
        V = np.zeros((env_dqn.rows, env_dqn.cols))
        for r in range(env_dqn.rows):
            for c in range(env_dqn.cols):
                V[r, c] = vfn((r, c))
        im = ax.imshow(V, cmap="viridis")
        ax.set_title(f"{name} — V(s) = max_a Q(s, a)")
        ax.set_xlabel("col"); ax.set_ylabel("row")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "value_heatmaps.png", dpi=150, bbox_inches="tight")

    fig_p_dqn, _ = plot_policy(env_dqn, dqn_policy, path=dqn_path, title="DQN greedy policy")
    fig_p_dqn.savefig(output_dir / "policy_dqn.png", dpi=150, bbox_inches="tight")
    fig_p_lin, _ = plot_policy(env_lin, lin_policy, path=lin_path, title="Linear SARSA greedy policy")
    fig_p_lin.savefig(output_dir / "policy_linear_sarsa.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
