"""PL6/PL7 — Tic-Tac-Toe: SARSA vs Q-Learning, evaluation, and interactive play.

Trains both a tabular SARSA (on-policy) and Q-Learning (off-policy) agent
playing as X against a random opponent O, compares their learning curves,
and optionally lets the user play against the trained agent.

Why both algorithms?
  - SARSA updates Q(s,a) using the action actually taken (on-policy),
    so its learned values reflect the epsilon-greedy behaviour policy.
  - Q-Learning updates Q(s,a) using max_a' Q(s',a') (off-policy),
    so it learns the optimal policy directly, regardless of exploration.
  In Tic-Tac-Toe the difference is visible: Q-Learning typically converges
  to a higher win rate faster because it always bootstraps from the best
  possible continuation.

Plots produced:
  - Rolling win-rate comparison (SARSA vs Q-Learning)
  - Greedy evaluation results (pie charts)

Interactive mode:
  python -m AR1.scripts.run_tictactoe --play
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AR1.envs.tictactoe import TicTacToeEnv, TicTacToeState, TicTacToeAction


# ── Agents ──────────────────────────────────────────────────────────────────

class TabularAgent:
    """Base class for tabular RL agents on Tic-Tac-Toe."""

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.Q: dict[tuple[TicTacToeState, TicTacToeAction], float] = {}

    def _q(self, state: TicTacToeState, action: TicTacToeAction) -> float:
        return self.Q.get((state, action), 0.0)

    def select_action(
        self, state: TicTacToeState, available: list[TicTacToeAction],
    ) -> TicTacToeAction:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available)
        best_q = max(self._q(state, a) for a in available)
        best = [a for a in available if self._q(state, a) == best_q]
        return self.rng.choice(best)

    def greedy_action(
        self, state: TicTacToeState, available: list[TicTacToeAction],
    ) -> TicTacToeAction:
        best_q = max(self._q(state, a) for a in available)
        best = [a for a in available if self._q(state, a) == best_q]
        return self.rng.choice(best)


class TicTacToeSarsa(TabularAgent):
    """On-policy SARSA: bootstraps from the action actually selected."""

    name = "SARSA"

    def update(
        self, state, action, reward, next_state, next_action, done,
    ) -> None:
        q_next = 0.0 if done or next_action is None else self._q(next_state, next_action)
        td_target = reward + self.gamma * q_next
        self.Q[(state, action)] = self._q(state, action) + self.alpha * (td_target - self._q(state, action))


class TicTacToeQLearning(TabularAgent):
    """Off-policy Q-Learning: bootstraps from max_a' Q(s', a')."""

    name = "Q-Learning"

    def update(
        self, state, action, reward, next_state, available_next, done,
    ) -> None:
        if done or available_next is None or len(available_next) == 0:
            q_next = 0.0
        else:
            q_next = max(self._q(next_state, a) for a in available_next)
        td_target = reward + self.gamma * q_next
        self.Q[(state, action)] = self._q(state, action) + self.alpha * (td_target - self._q(state, action))


def random_action(state, available, rng):
    return rng.choice(available)


# ── Training ────────────────────────────────────────────────────────────────

def train_agent(agent, num_episodes, seed=0):
    """Train agent (X) vs random (O). Works for both SARSA and Q-Learning."""
    env = TicTacToeEnv()
    opp_rng = random.Random(seed + 999)
    outcomes = []
    is_sarsa = isinstance(agent, TicTacToeSarsa)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        available = env.available_actions(state)
        action = agent.select_action(state, available)

        while not done:
            next_state, reward_x, done = env.step(action)

            if done:
                if is_sarsa:
                    agent.update(state, action, reward_x, None, None, True)
                else:
                    agent.update(state, action, reward_x, None, None, True)
                outcomes.append("win" if reward_x > 0 else "draw")
                break

            # Opponent moves
            opp_available = env.available_actions(next_state)
            opp_action = random_action(next_state, opp_available, opp_rng)
            state_after_opp, reward_o, done = env.step(opp_action)

            if done:
                x_reward = -1.0 if reward_o > 0 else 0.0
                if is_sarsa:
                    agent.update(state, action, x_reward, None, None, True)
                else:
                    agent.update(state, action, x_reward, None, None, True)
                outcomes.append("loss" if reward_o > 0 else "draw")
                break

            available = env.available_actions(state_after_opp)
            next_action = agent.select_action(state_after_opp, available)

            if is_sarsa:
                agent.update(state, action, 0.0, state_after_opp, next_action, False)
            else:
                agent.update(state, action, 0.0, state_after_opp, available, False)

            state = state_after_opp
            action = next_action

    return outcomes


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_greedy(agent, num_games=1_000, seed=42):
    env = TicTacToeEnv()
    opp_rng = random.Random(seed + 1234)
    results = {"win": 0, "loss": 0, "draw": 0}

    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            available = env.available_actions(state)
            action = agent.greedy_action(state, available)
            state, reward_x, done = env.step(action)
            if done:
                results["win" if reward_x > 0 else "draw"] += 1
                break
            opp_available = env.available_actions(state)
            opp_action = random_action(state, opp_available, opp_rng)
            state, reward_o, done = env.step(opp_action)
            if done:
                results["loss" if reward_o > 0 else "draw"] += 1
                break
    return results


# ── Interactive play ────────────────────────────────────────────────────────

def play_interactive(agent, agent_name: str = "Agent"):
    """Let the human play as O against the trained agent (X)."""
    env = TicTacToeEnv()
    state = env.reset()
    done = False

    print("\n" + "=" * 40)
    print(f"  You (O) vs {agent_name} (X)")
    print("  Enter cell number 0-8 to play.")
    print("  Board layout:")
    print("    0 | 1 | 2")
    print("    ---------")
    print("    3 | 4 | 5")
    print("    ---------")
    print("    6 | 7 | 8")
    print("=" * 40 + "\n")
    env.render(state)

    while not done:
        if env.current_player == 1:
            # Agent (X) moves
            available = env.available_actions(state)
            action = agent.greedy_action(state, available)
            state, reward, done = env.step(action)
            print(f"\n{agent_name} (X) plays cell {action}:")
            env.render(state)
            if done:
                if reward > 0:
                    print(f">>> {agent_name} wins!")
                else:
                    print(">>> Draw!")
        else:
            # Human (O) moves
            available = env.available_actions(state)
            while True:
                try:
                    cell = int(input(f"\nYour move (O) — available {available}: "))
                    if cell in available:
                        break
                    print(f"  Cell {cell} is not available. Try again.")
                except (ValueError, EOFError):
                    print("  Invalid input. Enter a number 0-8.")
            state, reward, done = env.step(cell)
            print(f"\nYou (O) play cell {cell}:")
            env.render(state)
            if done:
                if reward > 0:
                    print(">>> You win! Congratulations!")
                else:
                    print(">>> Draw!")

    again = input("\nPlay again? (y/n): ").strip().lower()
    if again == "y":
        play_interactive(agent, agent_name)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(outcomes_sarsa, outcomes_ql, window=500):
    import matplotlib.pyplot as plt

    def rolling_win_rate(outcomes, w):
        wins = np.array([1.0 if o == "win" else 0.0 for o in outcomes])
        return np.convolve(wins, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    n = min(len(outcomes_sarsa), len(outcomes_ql))
    xs = np.arange(window - 1, n)
    ax.plot(xs, rolling_win_rate(outcomes_sarsa, window), label="SARSA", linewidth=2, color="#1f77b4")
    ax.plot(xs, rolling_win_rate(outcomes_ql, window), label="Q-Learning", linewidth=2, color="#ff7f0e")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Win rate (rolling {window})")
    ax.set_title("Tic-Tac-Toe: SARSA vs Q-Learning (X vs Random O)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return fig, ax


def plot_eval_comparison(results_sarsa, results_ql):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for ax, results, name in zip(axes, [results_sarsa, results_ql], ["SARSA", "Q-Learning"]):
        sizes = [results["win"], results["draw"], results["loss"]]
        total = sum(sizes)
        ax.pie(
            sizes, labels=["Win", "Draw", "Loss"], colors=colors,
            autopct=lambda pct: f"{pct:.1f}%", startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(f"{name} ({total} games)")

    fig.suptitle("Greedy policy evaluation vs Random opponent", fontsize=13)
    return fig, axes


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe: SARSA vs Q-Learning.")
    parser.add_argument("--episodes", type=int, default=100_000, help="Training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Step-size.")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Exploration rate.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--eval-games", type=int, default=5_000, help="Greedy evaluation games.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--play", action="store_true", help="Play interactively against trained agent.")
    parser.add_argument("--play-algo", choices=["sarsa", "qlearning"], default="qlearning",
                        help="Which agent to play against (default: qlearning).")
    parser.add_argument("--output-dir", type=str, default="outputs/tictactoe")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_show and not args.play:
        import matplotlib
        matplotlib.use("Agg")

    common = dict(alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma, seed=args.seed)

    # Train both agents
    print(f"Training SARSA for {args.episodes:,} episodes...")
    agent_sarsa = TicTacToeSarsa(**common)
    outcomes_sarsa = train_agent(agent_sarsa, args.episodes, seed=args.seed)

    print(f"Training Q-Learning for {args.episodes:,} episodes...")
    agent_ql = TicTacToeQLearning(**common)
    outcomes_ql = train_agent(agent_ql, args.episodes, seed=args.seed)

    for agent, outcomes in [(agent_sarsa, outcomes_sarsa), (agent_ql, outcomes_ql)]:
        w = sum(1 for o in outcomes if o == "win")
        d = sum(1 for o in outcomes if o == "draw")
        l = sum(1 for o in outcomes if o == "loss")
        print(f"  {agent.name}: {w} wins, {d} draws, {l} losses | Q-table: {len(agent.Q):,} entries")

    # Evaluate
    print(f"\nEvaluating greedy policies over {args.eval_games:,} games...")
    res_sarsa = evaluate_greedy(agent_sarsa, args.eval_games, seed=args.seed + 100)
    res_ql = evaluate_greedy(agent_ql, args.eval_games, seed=args.seed + 100)
    print(f"  SARSA:      {res_sarsa}")
    print(f"  Q-Learning: {res_ql}")

    # Plots
    if not args.play:
        import matplotlib.pyplot as plt

        fig1, _ = plot_comparison(outcomes_sarsa, outcomes_ql, window=min(500, len(outcomes_sarsa) // 5))
        fig2, _ = plot_eval_comparison(res_sarsa, res_ql)

        output_dir = PACKAGE_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_dir / "training_comparison.png", dpi=150, bbox_inches="tight")
        fig2.savefig(output_dir / "eval_comparison.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved plots to {output_dir}")

        if args.no_show:
            plt.close("all")
        else:
            plt.show()

    # Interactive play
    if args.play:
        agent = agent_ql if args.play_algo == "qlearning" else agent_sarsa
        name = "Q-Learning" if args.play_algo == "qlearning" else "SARSA"
        play_interactive(agent, agent_name=name)


if __name__ == "__main__":
    main()
