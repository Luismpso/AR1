from __future__ import annotations

from typing import Callable

import numpy as np

from AR1.core.base import Transition
from AR1.envs.windy_gridworld import WindyGridworldAction, WindyGridworldEnv, WindyGridworldState
from AR1.experiments.control import run_control_episode


def train_fa_agent(
    env: WindyGridworldEnv,
    agent,
    num_episodes: int,
    max_steps: int = 1_000,
) -> tuple[list[int], list[float], list[float]]:
    """Train a function-approximation control agent (LinearSarsa or TorchSarsa).

    Returns episode_lengths, episode_rewards, and mean per-episode TD errors.
    The agent must expose a flush_td_errors() method.
    """
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    episode_mean_td_errors: list[float] = []

    for _ in range(num_episodes):
        length, reward = run_control_episode(env, agent, max_steps=max_steps)
        episode_lengths.append(length)
        episode_rewards.append(reward)
        errors = agent.flush_td_errors()    # get TD errors for this episode and reset the agent's internal TD error buffer
        episode_mean_td_errors.append(float(np.mean(errors)) if errors else 0.0)

    return episode_lengths, episode_rewards, episode_mean_td_errors


def run_linear_td_episode(
    env: WindyGridworldEnv,
    policy,
    agent,
    max_steps: int = 1_000,
) -> tuple[int, float]:
    """One episode of online TD(0) prediction with linear function approximation.

    `policy` must expose select_action(state) -> action.
    `agent` must expose update(transition) -> delta (LinearTD0).
    Returns (episode_length, mean_abs_td_error).
    """
    state = env.reset()
    done = False
    steps = 0
    td_errors: list[float] = []

    while not done and steps < max_steps:
        action = policy.select_action(state)
        next_state, reward, done = env.step(action)
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=None if done else next_state,
            done=done,
        )
        delta = agent.update(transition)    # updates weights and returns the TD error for this transition
        td_errors.append(abs(delta))    # delta is the TD error for this transition
        state = next_state
        steps += 1

    return steps, float(np.mean(td_errors)) if td_errors else 0.0


def train_linear_td_agent(
    env: WindyGridworldEnv,
    policy,
    agent,
    num_episodes: int,
    max_steps: int = 1_000,
) -> tuple[list[int], list[float]]:
    """Train a LinearTD0 agent over multiple episodes using a fixed policy."""
    episode_lengths: list[int] = []
    episode_mean_td_errors: list[float] = []

    for _ in range(num_episodes):
        length, mean_error = run_linear_td_episode(env, policy, agent, max_steps)
        episode_lengths.append(length)
        episode_mean_td_errors.append(mean_error)

    return episode_lengths, episode_mean_td_errors
