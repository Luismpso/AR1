"""PL4 — Blackjack value function visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from AR1.envs.blackjack import BlackjackState

PLAYER_SUMS = tuple(range(12, 22))
DEALER_SHOWING = tuple(range(1, 11))


def values_to_array(values: dict[BlackjackState, float], usable_ace: bool) -> np.ndarray:
    arr = np.zeros((len(PLAYER_SUMS), len(DEALER_SHOWING)), dtype=float)
    for i, player_sum in enumerate(PLAYER_SUMS):
        for j, dealer_showing in enumerate(DEALER_SHOWING):
            arr[i, j] = values.get((player_sum, dealer_showing, usable_ace), 0.0)
    return arr


def plot_value_function(
    values: dict[BlackjackState, float],
    title: str = "",
    axes=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    created_figure = axes is None
    if created_figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    else:
        axes = np.asarray(axes).reshape(-1)
        fig = axes[0].figure

    axes = np.asarray(axes).reshape(-1)
    subtitles = [(False, "No usable ace"), (True, "Usable ace")]
    last_im = None

    for ax, (usable_ace, subtitle) in zip(axes, subtitles):
        arr = values_to_array(values, usable_ace)
        last_im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(subtitle)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_xticks(range(len(DEALER_SHOWING)), DEALER_SHOWING)
        ax.set_yticks(range(len(PLAYER_SUMS)), PLAYER_SUMS)

    if title:
        fig.suptitle(title)

    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes), shrink=0.85)

    return fig, axes


def plot_value_difference(
    values_a: dict[BlackjackState, float],
    values_b: dict[BlackjackState, float],
    title: str = "Value difference",
    vmin: float | None = None,
    vmax: float | None = None,
):
    diff_values: dict[BlackjackState, float] = {}
    all_states = set(values_a) | set(values_b)
    for state in all_states:
        diff_values[state] = values_a.get(state, 0.0) - values_b.get(state, 0.0)
    return plot_value_function(diff_values, title=title, cmap="coolwarm", vmin=vmin, vmax=vmax)


def plot_methods_comparison(
    method_values: dict[str, dict[BlackjackState, float]],
    title: str = "Prediction methods comparison",
    usable_ace: bool = False,
    vmin: float | None = -1.0,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
):
    """Side-by-side heatmaps of several prediction methods on Blackjack.

    Useful to visually compare First-Visit MC, TD(0) and TD(n) on the
    same colour scale. ``method_values`` maps a label (e.g. ``"MC"``,
    ``"TD(0)"``, ``"TD(n=4)"``) to a value-function snapshot.
    """
    n_methods = len(method_values)
    if n_methods == 0:
        raise ValueError("method_values must contain at least one method.")

    fig, axes = plt.subplots(1, n_methods, figsize=(4.2 * n_methods, 4.2), constrained_layout=True)
    if n_methods == 1:
        axes = [axes]

    last_im = None
    ace_label = "Usable ace" if usable_ace else "No usable ace"

    for ax, (label, values) in zip(axes, method_values.items()):
        arr = values_to_array(values, usable_ace)
        last_im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\n({ace_label})")
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_xticks(range(len(DEALER_SHOWING)), DEALER_SHOWING)
        ax.set_yticks(range(len(PLAYER_SUMS)), PLAYER_SUMS)

    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes), shrink=0.85)

    if title:
        fig.suptitle(title)

    return fig, axes


def plot_methods_rmse(
    method_values: dict[str, dict[BlackjackState, float]],
    reference_values: dict[BlackjackState, float],
    title: str = "RMSE vs reference value function",
):
    """Bar plot of RMSE between each method's value function and a reference.

    Typically the reference is a long-run First-Visit MC estimate
    treated as ground truth. Includes both ace flavours aggregated.
    """
    import numpy as np

    labels = list(method_values.keys())
    rmses: list[float] = []
    states = list(reference_values.keys())

    for label in labels:
        values = method_values[label]
        diffs = [values.get(s, 0.0) - reference_values[s] for s in states]
        rmse = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else 0.0
        rmses.append(rmse)

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 1.5, 4), constrained_layout=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bars = ax.bar(labels, rmses, color=[palette[i % len(palette)] for i in range(len(labels))])
    ax.set_ylabel("RMSE vs reference V*")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    return fig, ax
