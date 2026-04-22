"""PL1 — Bandit experiment runner."""
from __future__ import annotations

import numpy as np

from AR1.core.base import BanditAgent
from AR1.envs.bandit import KArmedBandit


def run_bandit_experiment(
    agent: BanditAgent,
    env: KArmedBandit,
    steps: int = 1000,
    runs: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (avg_rewards[steps], pct_optimal[steps])."""
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))

    for r in range(runs):
        env.reset()
        agent.reset()
        for t in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)

            rewards[r, t] = reward
            optimal[r, t] = (action == env.optimal_action)

    return rewards.mean(axis=0), optimal.mean(axis=0)
