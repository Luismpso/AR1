"""
PL2/PL3.1 — Run Gridworld experiments.

Plots produced:
  - Policy Evaluation (uniform random policy)
  - Value Iteration (optimal V* and greedy policy)
  - Varying gamma (0.5, 0.99)
  - Trap gridworld
  - Stochastic gridworld
  - Policy Iteration (convergence history)

Usage:
    python -m AR1.scripts.run_gridworld
    python -m AR1.scripts.run_gridworld --no-show
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
    parser = argparse.ArgumentParser(description="Gridworld DP experiments.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gridworld",
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

    from AR1.envs.gridworld import Gridworld, GridworldTrap, get_stochastic_transitions, ACTIONS
    from AR1.mdps.gridworld_mdp import zeros_V, uniform_random_policy
    from AR1.agents.dp import (
        policy_evaluation,
        value_iteration,
        greedy_policy_from_V,
        policy_evaluation_Q,
        policy_iteration,
        stochastic_value_iteration,
    )
    from AR1.plots.gridworld import plot_grid

    env = Gridworld()
    gamma = 0.9
    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── PL2: Policy Evaluation ──
    policy = uniform_random_policy(env)
    V_pi, iters = policy_evaluation(env, policy, gamma=gamma, theta=1e-8)
    print(f"Policy Evaluation converged in {iters} iterations")
    fig, _ = plot_grid(env, V_pi, None, title="Policy Evaluation: V^pi (uniform random pi)")
    fig.savefig(output_dir / "policy_evaluation.png", dpi=150, bbox_inches="tight")

    # ── PL2: Value Iteration ──
    V_star, iters_vi = value_iteration(env, gamma=gamma, theta=1e-8)
    pi_star = greedy_policy_from_V(env, V_star, gamma=gamma)
    print(f"Value Iteration converged in {iters_vi} iterations")
    fig, _ = plot_grid(env, V_star, pi_star, title="Value Iteration: V* and greedy policy")
    fig.savefig(output_dir / "value_iteration.png", dpi=150, bbox_inches="tight")

    # ── PL2: Q^pi ──
    Q_pi, itq = policy_evaluation_Q(env, policy, gamma=gamma, theta=1e-8)
    print(f"Q^pi converged in {itq} iterations")

    # ── Exercise A: Varying gamma ──
    for g in [0.5, 0.99]:
        V_g, it_g = value_iteration(env, gamma=g, theta=1e-8)
        pi_g = greedy_policy_from_V(env, V_g, gamma=g)
        print(f"gamma={g}: converged in {it_g} iterations")
        fig, _ = plot_grid(env, V_g, pi_g, title=f"V* with gamma = {g}")
        fig.savefig(output_dir / f"value_iteration_gamma_{g}.png", dpi=150, bbox_inches="tight")

    # ── Exercise B: Trap ──
    env_trap = GridworldTrap()
    V_trap, iters_trap = value_iteration(env_trap, gamma=0.9, theta=1e-8)
    pi_trap = greedy_policy_from_V(env_trap, V_trap, gamma=0.9)
    fig, _ = plot_grid(env_trap, V_trap, pi_trap, title="V* and policy with trap at (1,2)")
    fig.savefig(output_dir / "trap_gridworld.png", dpi=150, bbox_inches="tight")

    # ── Exercise C: Stochastic ──
    V_stoch, iters_stoch = stochastic_value_iteration(env, gamma=0.9)
    pi_stoch = {}
    for s in env.states():
        if env.is_terminal(s):
            pi_stoch[s] = "·"
            continue
        best_q, best_a = float("-inf"), None
        for a in ACTIONS:
            eq = sum(
                prob * (r + 0.9 * V_stoch[ns[0], ns[1]])
                for prob, ns, r in get_stochastic_transitions(env, s, a)
            )
            if eq > best_q:
                best_q = eq
                best_a = a
        pi_stoch[s] = best_a
    fig, _ = plot_grid(env, V_stoch, pi_stoch, title="V* in stochastic environment")
    fig.savefig(output_dir / "stochastic_gridworld.png", dpi=150, bbox_inches="tight")

    # ── PL3.1: Policy Iteration ──
    V_pi_star, pi_star_actions, hist = policy_iteration(env, gamma=gamma)
    print(f"Policy Iteration outer loops: {len(hist)}")
    fig, _ = plot_grid(env, V_pi_star, pi_star_actions, title="Policy Iteration: V* and pi*")
    fig.savefig(output_dir / "policy_iteration.png", dpi=150, bbox_inches="tight")

    # Show iteration history
    num_plots = len(hist)
    if num_plots > 0:
        fig_hist, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6), constrained_layout=True)
        if num_plots == 1:
            axes = [axes]
        for i, (outer_iter, pe_iters, V_hist, pi_actions_hist) in enumerate(hist):
            plot_grid(env, V_hist, pi_actions_hist,
                      title=f"Outer {outer_iter} (PE iters: {pe_iters})", ax=axes[i])
        fig_hist.savefig(output_dir / "policy_iteration_history.png", dpi=150, bbox_inches="tight")

    print(f"Saved all plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
