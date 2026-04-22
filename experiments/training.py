"""PL4 — Blackjack episode generation and prediction training."""
from __future__ import annotations

from AR1.core.base import Episode, Policy, PredictionAgent, Transition
from AR1.envs.blackjack import BlackjackAction, BlackjackEnv, BlackjackState


def generate_episode(
    env: BlackjackEnv,
    policy: Policy[BlackjackState, BlackjackAction],
) -> Episode[BlackjackState, BlackjackAction]:
    episode: Episode[BlackjackState, BlackjackAction] = Episode()
    state = env.reset()
    done = False

    while not done:
        action = policy.select_action(state)
        next_state, reward, done = env.step(action)
        episode.add(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=None if done else next_state,
                done=done,
            )
        )
        state = next_state

    return episode


def snapshot_blackjack_values(
    agent: PredictionAgent[BlackjackState, BlackjackAction],
) -> dict[BlackjackState, float]:
    snapshot: dict[BlackjackState, float] = {}
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            for usable in (False, True):
                state = (player_sum, dealer_showing, usable)
                snapshot[state] = agent.value_of(state)
    return snapshot


def train_prediction_agent(
    env: BlackjackEnv,
    policy: Policy[BlackjackState, BlackjackAction],
    agent: PredictionAgent[BlackjackState, BlackjackAction],
    num_episodes: int,
    checkpoints: list[int] | tuple[int, ...] | None = None,
) -> dict[int, dict[BlackjackState, float]]:
    if checkpoints is None:
        checkpoints = [num_episodes]

    ordered = sorted(set(int(cp) for cp in checkpoints if 0 < cp <= num_episodes))
    if num_episodes not in ordered:
        ordered.append(num_episodes)

    history: dict[int, dict[BlackjackState, float]] = {}

    for episode_idx in range(1, num_episodes + 1):
        episode = generate_episode(env, policy)
        agent.update_episode(episode)

        if episode_idx in ordered:
            history[episode_idx] = snapshot_blackjack_values(agent)

    return history
