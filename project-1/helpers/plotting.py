"""Plotting helpers for Project 1.

All functions draw on a matplotlib Axes and return it, so you can combine them
in subplots and add titles or labels yourself.
"""

import matplotlib.pyplot as plt
import numpy as np

from gridworld import CLIFF, GOAL, N_COLS, N_ROWS, START, TERMINAL_STATES, to_cell

_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def plot_mean_std(ax, x, runs, label, color=None):
    """Plot the mean over seeds with a shaded band of ±1 standard deviation.

    runs : array of shape (n_seeds, len(x)), one row per seed.
    """
    runs = np.asarray(runs)
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    (line,) = ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, linewidth=0)
    return ax


def _grid(values):
    grid = np.full((N_ROWS, N_COLS), np.nan)
    for s, v in enumerate(values):
        r, c = to_cell(s)
        grid[r, c] = v
    return grid


def _label_special_cells(ax):
    for (r, c), text in [(START, "S"), (GOAL, "G")] + [(cell, "X") for cell in CLIFF]:
        ax.text(c - 0.42, r + 0.42, text, ha="left", va="top", fontsize=7, color="0.3")


def plot_values(ax, V, title=None, fmt=".2f", cmap="viridis"):
    """Heat map of a state-value function V (length 25), with values written in each cell."""
    grid = _grid(V)
    for s in TERMINAL_STATES:
        r, c = to_cell(s)
        grid[r, c] = np.nan
    im = ax.imshow(grid, origin="lower", cmap=cmap)
    for s, v in enumerate(V):
        if s in TERMINAL_STATES:
            continue
        r, c = to_cell(s)
        bright = im.norm(v) > 0.6  # dark text on the light end of the colour map
        ax.text(c, r, format(v, fmt), ha="center", va="center", fontsize=8,
                color="black" if bright else "white")
    _label_special_cells(ax)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_policy(ax, policy, title=None):
    """Draw a deterministic policy (length-25 array of actions) as arrows."""
    ax.imshow(np.zeros((N_ROWS, N_COLS)), origin="lower", cmap="Greys", vmin=0, vmax=1)
    for s, a in enumerate(policy):
        if s in TERMINAL_STATES:
            continue
        r, c = to_cell(s)
        ax.text(c, r, _ARROWS[int(a)], ha="center", va="center", fontsize=16)
    _label_special_cells(ax)
    ax.set_xticks(np.arange(-0.5, N_COLS), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS), minor=True)
    ax.grid(which="minor", color="0.7")
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)
    return ax
