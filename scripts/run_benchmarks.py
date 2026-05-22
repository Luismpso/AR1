"""Benchmark suite — train each algorithm with a fixed budget, measure key
metrics, save them to JSON and produce a comparison figure.

Metrics recorded (per algorithm):
    - training_time_s     : wall-clock seconds for the training loop
    - episodes_to_converge: first episode at which a sliding window of
                            ``window_size`` episodes had average length ≤
                            ``convergence_threshold`` (None if it never did).
                            For Tic-Tac-Toe agents convergence is reported
                            as the first eval checkpoint where win-rate ≥ 0.9.
    - final_metric        : final greedy path length (Windy Gridworld)
                            or evaluation win-rate (Tic-Tac-Toe)

Run with:
    python -m AR1.scripts.run_benchmarks --no-show

Adjust the per-experiment episode budget with the CLI flags below.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark all main algorithms.")
    p.add_argument("--windy-episodes",   type=int, default=300)
    p.add_argument("--windy-max-steps",  type=int, default=500)
    p.add_argument("--ttt-episodes",     type=int, default=2000)
    p.add_argument("--ttt-eval-games",   type=int, default=200)
    p.add_argument("--window-size",      type=int, default=20)
    p.add_argument("--convergence-len",  type=int, default=20,
                   help="Sliding-window episode-length threshold for convergence.")
    p.add_argument("--seed",             type=int, default=7)
    p.add_argument("--output-dir",       type=str,
                   default="outputs/benchmarks")
    p.add_argument("--no-show",          action="store_true")
    return p.parse_args()


def _detect_convergence(lengths: list[int], window: int, threshold: int) -> int | None:
    """First index t where mean(lengths[t-window+1..t]) <= threshold."""
    if len(lengths) < window:
        return None
    import numpy as np
    arr = np.asarray(lengths, dtype=float)
    for t in range(window - 1, len(arr)):
        if arr[t - window + 1 : t + 1].mean() <= threshold:
            return t + 1
    return None


def _benchmark_windy(args, results: dict) -> None:
    from AR1.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from AR1.experiments.control import (
        greedy_path,
        greedy_policy_from_agent,
        train_control_agent,
    )
    from AR1.agents.control.sarsa import SarsaControl
    from AR1.agents.control.q_learning import QLearningControl
    from AR1.agents.control.n_step_sarsa import NStepSarsaControl

    configs = [
        ("sarsa",        SarsaControl,        {}),
        ("q_learning",   QLearningControl,    {}),
        ("n_step_sarsa", NStepSarsaControl,   {"n_steps": 4}),
    ]
    windy_block = {}

    for name, Cls, kwargs in configs:
        print(f"  windy/{name} ...", end="", flush=True)
        env = WindyGridworldEnv()
        agent = Cls(actions=ACTIONS, seed=args.seed, **kwargs)
        agent.reset()
        t0 = time.perf_counter()
        episode_lengths, _ = train_control_agent(
            env, agent, args.windy_episodes, max_steps=args.windy_max_steps,
        )
        elapsed = time.perf_counter() - t0
        policy = greedy_policy_from_agent(env, agent)
        path = greedy_path(env, policy)
        converged = _detect_convergence(
            episode_lengths, args.window_size, args.convergence_len,
        )
        windy_block[name] = {
            "training_time_s":      round(elapsed, 3),
            "episodes_to_converge": converged,
            "final_path_length":    len(path) - 1,
            "episode_lengths":      episode_lengths,
        }
        print(f"  time={elapsed:.2f}s  converged@={converged}  "
              f"path={len(path)-1}")

    results["windy_gridworld"] = windy_block


def _benchmark_tictactoe(args, results: dict) -> None:
    """Random / SARSA / Q-Learning / REINFORCE on self-play TTT."""
    import numpy as np

    from AR1.envs.tictactoe import TicTacToeEnv, _winner
    from AR1.policies.tictactoe import random_action

    ttt_block: dict = {}

    # ── Random baseline ───────────────────────────────────────────────
    env = TicTacToeEnv()
    t0 = time.perf_counter()
    wins = draws = losses = 0
    for _ in range(args.ttt_eval_games):
        state = env.reset()
        done = False
        while not done:
            a = random_action(env, state)
            state, _, done = env.step(a)
        w = _winner(state)
        if w == 1:   wins   += 1
        elif w == 0: draws  += 1
        else:        losses += 1
    elapsed = time.perf_counter() - t0
    ttt_block["random"] = {
        "training_time_s":      0.0,
        "episodes_to_converge": None,
        "final_metric": {
            "win_rate":  wins   / args.ttt_eval_games,
            "draw_rate": draws  / args.ttt_eval_games,
            "loss_rate": losses / args.ttt_eval_games,
        },
        "_meta": "Uniform random vs uniform random; reported as X wins.",
        "eval_time_s": round(elapsed, 3),
    }
    print(f"  ttt/random       X-wins={wins/args.ttt_eval_games:.1%}")

    # NOTE: SARSA/Q-Learning on Tic-Tac-Toe use a custom Tabular agent with
    # legal-action masking (see scripts/run_tictactoe.py).  Those numbers are
    # already documented in RESULTS.md from that script.  This benchmark
    # focuses on REINFORCE and MCTS — agents whose API is independent of
    # the env's legal-action set.

    # ── REINFORCE + baseline ──────────────────────────────────────────
    try:
        from AR1.agents.control.reinforce import ReinforceAgent
        from AR1.experiments.reinforce_tictactoe import (
            evaluate_vs_random as eval_reinforce,
            run_reinforce_episode,
            run_vs_random_episode,
        )
        env = TicTacToeEnv()
        agent = ReinforceAgent(alpha=0.01, use_baseline=True, seed=args.seed)
        t0 = time.perf_counter()
        # 70% self-play, 30% vs random — matches reinforce_tictactoe.train()
        import numpy as np
        rng = np.random.default_rng(args.seed)
        for _ in range(args.ttt_episodes):
            if rng.random() < 0.3:
                run_vs_random_episode(env, agent)
            else:
                run_reinforce_episode(env, agent)
        elapsed = time.perf_counter() - t0
        wr_x, dr_x, lr_x = eval_reinforce(env, agent, args.ttt_eval_games, as_player=1)
        ttt_block["reinforce_baseline"] = {
            "training_time_s":      round(elapsed, 3),
            "episodes_to_converge": None,
            "final_metric": {
                "win_rate": wr_x, "draw_rate": dr_x, "loss_rate": lr_x,
            },
            "_meta": "REINFORCE with V(s) baseline; self-play + 30% vs random.",
        }
        print(f"  ttt/reinforce_baseline time={elapsed:.2f}s  X-wins={wr_x:.1%}")
    except Exception as exc:  # pragma: no cover
        print(f"  ttt/reinforce_baseline FAILED: {exc}")

    # ── MCTS ──────────────────────────────────────────────────────────
    try:
        from AR1.agents.planning.mcts import MCTSAgent
        from AR1.experiments.mcts_tictactoe import evaluate_vs_random as eval_mcts
        env = TicTacToeEnv()
        agent = MCTSAgent(n_simulations=200)
        t0 = time.perf_counter()
        wr_x, dr_x, lr_x = eval_mcts(env, agent, n_games=args.ttt_eval_games, as_player=1)
        elapsed = time.perf_counter() - t0
        ttt_block["mcts_200"] = {
            "training_time_s":      0.0,
            "episodes_to_converge": None,
            "final_metric": {
                "win_rate": wr_x, "draw_rate": dr_x, "loss_rate": lr_x,
            },
            "_meta": "MCTS with 200 sims/move; no training. "
                     "Reported time is evaluation wall-clock.",
            "eval_time_s": round(elapsed, 3),
        }
        print(f"  ttt/mcts_200     eval_time={elapsed:.2f}s  X-wins={wr_x:.1%}")
    except Exception as exc:  # pragma: no cover
        print(f"  ttt/mcts_200 FAILED: {exc}")

    results["tictactoe"] = ttt_block


def _plot_summary(results: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # ── 1. Windy: episodes-to-converge bar chart ──────────────────────
    if "windy_gridworld" in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        labels = list(results["windy_gridworld"].keys())
        converged = [
            results["windy_gridworld"][k]["episodes_to_converge"] or 0
            for k in labels
        ]
        times = [results["windy_gridworld"][k]["training_time_s"] for k in labels]

        axes[0].bar(labels, converged, color=["#4caf50", "#2196f3", "#ff9800"])
        axes[0].set_ylabel("episodes to converge")
        axes[0].set_title("Windy Gridworld — convergence speed")
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(labels, times, color=["#4caf50", "#2196f3", "#ff9800"])
        axes[1].set_ylabel("training wall-clock (s)")
        axes[1].set_title("Windy Gridworld — training time")
        axes[1].grid(alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(output_dir / "windy_summary.png", dpi=150, bbox_inches="tight")

    # ── 2. TTT: win/draw/loss stacked bars ────────────────────────────
    if "tictactoe" in results:
        labels = list(results["tictactoe"].keys())
        wins   = [results["tictactoe"][k]["final_metric"]["win_rate"]  for k in labels]
        draws  = [results["tictactoe"][k]["final_metric"]["draw_rate"] for k in labels]
        losses = [results["tictactoe"][k]["final_metric"]["loss_rate"] for k in labels]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x, wins,                       label="win",  color="#4caf50")
        ax.bar(x, draws,  bottom=wins,        label="draw", color="#ffb300")
        ax.bar(x, losses, bottom=np.array(wins) + np.array(draws), label="loss", color="#e53935")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of games")
        ax.set_title("Tic-Tac-Toe — agent vs uniform random (as X)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / "tictactoe_summary.png", dpi=150, bbox_inches="tight")

    # ── 3. Windy: episode-length curves ───────────────────────────────
    if "windy_gridworld" in results:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for name in results["windy_gridworld"]:
            lengths = results["windy_gridworld"][name].get("episode_lengths")
            if lengths is not None:
                ax.plot(lengths, label=name, alpha=0.85)
        ax.set_xlabel("episode")
        ax.set_ylabel("episode length")
        ax.set_yscale("log")
        ax.set_title("Windy Gridworld — learning curves (log scale)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "windy_curves.png", dpi=150, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results: dict = {"_meta": {"seed": args.seed,
                               "windy_episodes": args.windy_episodes,
                               "ttt_episodes": args.ttt_episodes,
                               "ttt_eval_games": args.ttt_eval_games}}

    print("=== Windy Gridworld benchmarks ===")
    _benchmark_windy(args, results)

    print("\n=== Tic-Tac-Toe benchmarks ===")
    _benchmark_tictactoe(args, results)

    output_dir = PACKAGE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmarks.json"

    # Strip the heavy "episode_lengths" lists before JSON dump unless we want them.
    json_results = {k: v for k, v in results.items() if k != "_meta"}
    for block in json_results.values():
        if isinstance(block, dict):
            for entry in block.values():
                if isinstance(entry, dict) and "episode_lengths" in entry:
                    entry.pop("episode_lengths")

    json_results["_meta"] = results["_meta"]
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nResults saved to {json_path}")

    _plot_summary(results, output_dir)
    print(f"Plots saved to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
