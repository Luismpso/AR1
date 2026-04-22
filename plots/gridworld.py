"""PL2/PL3 — Gridworld visualization."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from AR1.envs.gridworld import Gridworld

ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→", "·": "·"}


def plot_grid(
    env: Gridworld,
    V: np.ndarray,
    policy: Optional[Dict[Tuple[int, int], str]] = None,
    title: str = "",
    ax=None,
):
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
    ax.set_title(title)

    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(env.n_cols + 1))
    ax.set_yticks(np.arange(env.n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for (r, c) in env.terminal_states:
        rect = plt.Rectangle((c, r), 1, 1, fill=True, alpha=0.15)
        ax.add_patch(rect)

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)
            ax.text(c + 0.5, r + 0.45, f"{V[r, c]:.2f}", ha="center", va="center", fontsize=12)
            if policy is not None:
                a = policy.get(s, "·") or "·"
                ax.text(c + 0.5, r + 0.78, ARROW.get(a, "·"), ha="center", va="center", fontsize=18)

    return fig, ax
