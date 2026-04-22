"""
PL3.2 — Run Jack's Car Rental experiments.

Plots produced:
  - Policy Iteration: final policy and V^pi
  - Value Iteration: greedy policy and V*

Usage:
    python -m AR1.scripts.run_car_rental
    python -m AR1.scripts.run_car_rental --no-show
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
    parser = argparse.ArgumentParser(description="Jack's Car Rental experiments.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/car_rental",
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

    from AR1.mdps.car_rental import CarRentalParams, CarRentalMDP
    from AR1.agents.dp import car_rental_policy_iteration, car_rental_value_iteration
    from AR1.plots.car_rental import plot_policy, plot_values

    params = CarRentalParams()
    mdp = CarRentalMDP(params)
    gamma = 0.9
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Policy Iteration
    print("Running Policy Iteration...")
    V_pi, pi_pi, hist = car_rental_policy_iteration(mdp, gamma=gamma, theta=1e-4, max_outer=20)
    print(f"Policy Iteration outer loops: {len(hist)}")
    fig, _ = plot_policy(mdp, pi_pi, title="Policy Iteration: final policy (cars moved 1->2)")
    fig.savefig(output_dir / "pi_policy.png", dpi=150, bbox_inches="tight")
    fig, _ = plot_values(mdp, V_pi, title="Policy Iteration: V^pi")
    fig.savefig(output_dir / "pi_values.png", dpi=150, bbox_inches="tight")

    # Value Iteration
    print("Running Value Iteration...")
    V_vi, pi_vi, it_vi = car_rental_value_iteration(mdp, gamma=gamma, theta=1e-4)
    print(f"Value Iteration iterations: {it_vi}")
    fig, _ = plot_policy(mdp, pi_vi, title="Value Iteration: greedy policy from V*")
    fig.savefig(output_dir / "vi_policy.png", dpi=150, bbox_inches="tight")
    fig, _ = plot_values(mdp, V_vi, title="Value Iteration: V*")
    fig.savefig(output_dir / "vi_values.png", dpi=150, bbox_inches="tight")

    print(f"Saved all plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
