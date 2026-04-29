"""PL4 — Blackjack prediction: First-Visit MC vs TD(0) vs TD(n).

Trains three prediction agents on the same fixed threshold policy and
plots the resulting value functions side-by-side, plus an RMSE bar
chart against a long-run MC estimate used as reference (V*).

The optional ``--n-step`` argument controls the n-step TD agent
(defaults to 4). Set it to 0 to skip TD(n) and reproduce the original
PL4 deliverable (MC vs TD(0) only).
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
    parser = argparse.ArgumentParser(description="Run Blackjack model-free prediction experiments.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size for TD(0) and TD(n).")
    parser.add_argument("--n-step", type=int, default=4,
                        help="n for the TD(n) agent. Use 0 to disable TD(n) and only compare MC vs TD(0).")
    parser.add_argument("--reference-episodes", type=int, default=200_000,
                        help="Episodes used to build the long-run MC reference V* for RMSE.")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack_prediction",
                        help="Directory inside AR1 where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from AR1.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction, TDnPrediction
    from AR1.envs.blackjack import BlackjackEnv
    from AR1.experiments.training import generate_episode, train_prediction_agent
    from AR1.plots.blackjack import (
        plot_methods_comparison,
        plot_methods_rmse,
        plot_value_difference,
        plot_value_function,
    )
    from AR1.policies.blackjack import ThresholdPolicy

    policy = ThresholdPolicy(threshold=args.threshold)

    try:
        sample_env = BlackjackEnv(seed=args.seed)
        sample_episode = generate_episode(sample_env, policy)
        print(f"Sample episode length: {len(sample_episode.transitions)}")
        print("First transitions:")
        for transition in sample_episode.transitions[:5]:
            print(transition)

        # ── Train MC and TD(0) (always) ─────────────────────────────────
        mc_env = BlackjackEnv(seed=args.seed)
        td_env = BlackjackEnv(seed=args.seed)
        mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
        td_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)

        checkpoints = sorted({cp for cp in (1000, 5000, args.episodes) if cp <= args.episodes})

        print(f"Training First-Visit Monte Carlo for {args.episodes} episodes...")
        mc_history = train_prediction_agent(mc_env, policy, mc_agent, args.episodes, checkpoints=checkpoints)

        print(f"Training TD(0) for {args.episodes} episodes...")
        td_history = train_prediction_agent(td_env, policy, td_agent, args.episodes, checkpoints=checkpoints)

        final_mc = mc_history[args.episodes]
        final_td = td_history[args.episodes]

        # ── Train TD(n) (optional) ──────────────────────────────────────
        method_values: dict[str, dict] = {
            "First-Visit MC": final_mc,
            "TD(0)": final_td,
        }

        final_tdn = None
        if args.n_step > 0:
            tdn_env = BlackjackEnv(seed=args.seed)
            tdn_agent = TDnPrediction(n=args.n_step, alpha=args.td_alpha, gamma=1.0)
            print(f"Training TD(n={args.n_step}) for {args.episodes} episodes...")
            tdn_history = train_prediction_agent(
                tdn_env, policy, tdn_agent, args.episodes, checkpoints=checkpoints,
            )
            final_tdn = tdn_history[args.episodes]
            method_values[f"TD(n={args.n_step})"] = final_tdn

        # ── Single-method plots (kept for backward compatibility) ───────
        fig_mc, _ = plot_value_function(final_mc, title=f"First-Visit MC after {args.episodes} episodes",
                                        vmin=-1.0, vmax=1.0)
        fig_td, _ = plot_value_function(final_td, title=f"TD(0) after {args.episodes} episodes",
                                        vmin=-1.0, vmax=1.0)
        fig_diff, _ = plot_value_difference(final_td, final_mc, title="TD(0) - First-Visit MC",
                                            vmin=-1.0, vmax=1.0)

        # ── New: side-by-side comparison and RMSE vs reference V* ───────
        fig_cmp_no_ace, _ = plot_methods_comparison(
            method_values,
            title=f"Blackjack value function comparison ({args.episodes} episodes, no usable ace)",
            usable_ace=False,
        )
        fig_cmp_ace, _ = plot_methods_comparison(
            method_values,
            title=f"Blackjack value function comparison ({args.episodes} episodes, usable ace)",
            usable_ace=True,
        )

        # Build a long-run MC reference treated as V* for RMSE.
        print(
            f"Training reference First-Visit MC for {args.reference_episodes} episodes "
            f"(treated as V* for RMSE)..."
        )
        ref_env = BlackjackEnv(seed=args.seed + 1)
        ref_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
        ref_history = train_prediction_agent(ref_env, policy, ref_agent, args.reference_episodes)
        reference_values = ref_history[args.reference_episodes]

        fig_rmse, _ = plot_methods_rmse(
            method_values, reference_values,
            title=f"RMSE vs long-run MC ({args.reference_episodes:,} episodes)",
        )

        # ── Save ────────────────────────────────────────────────────────
        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_mc.savefig(output_dir / "blackjack_mc.png", dpi=150, bbox_inches="tight")
        fig_td.savefig(output_dir / "blackjack_td0.png", dpi=150, bbox_inches="tight")
        fig_diff.savefig(output_dir / "blackjack_td_minus_mc.png", dpi=150, bbox_inches="tight")
        fig_cmp_no_ace.savefig(output_dir / "blackjack_comparison_no_ace.png", dpi=150, bbox_inches="tight")
        fig_cmp_ace.savefig(output_dir / "blackjack_comparison_ace.png", dpi=150, bbox_inches="tight")
        fig_rmse.savefig(output_dir / "blackjack_methods_rmse.png", dpi=150, bbox_inches="tight")
        if final_tdn is not None:
            fig_tdn, _ = plot_value_function(
                final_tdn, title=f"TD(n={args.n_step}) after {args.episodes} episodes",
                vmin=-1.0, vmax=1.0,
            )
            fig_tdn.savefig(output_dir / f"blackjack_tdn_n{args.n_step}.png", dpi=150, bbox_inches="tight")
        print(f"Saved plots to {output_dir}")

        if args.no_show:
            plt.close("all")
        else:
            plt.show()
    except NotImplementedError as exc:
        print("\nThis practical is not complete yet.")
        print("Please finish the TODOs in:")
        print("- envs/blackjack.py")
        print("- agents/prediction/monte_carlo.py")
        print("- agents/prediction/td.py")
        print(f"\nOriginal message: {exc}")
        return


if __name__ == "__main__":
    main()
