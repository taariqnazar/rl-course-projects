"""Plotting helpers for Project 2.

All functions draw on a matplotlib Axes and return it, so you can combine them
in subplots and add titles or labels yourself.
"""

import matplotlib.pyplot as plt
import numpy as np

try:
    from .environments import CliffWalking, MountainCar
except ImportError:
    from environments import CliffWalking, MountainCar

_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def plot_mean_std(ax, x, runs, label, color=None):
    """Plot the mean over seeds with a shaded band of ±1 standard deviation.

    runs : array of shape (n_seeds, len(x)), one row per seed (a single run also works).
    """
    runs = np.atleast_2d(np.asarray(runs, dtype=float))
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    (line,) = ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, linewidth=0)
    return ax


def plot_cliff_policy(ax, policy, title=None):
    """Draw a greedy policy for CliffWalking (length-48 array of actions) as arrows."""
    env = CliffWalking
    grid = np.zeros((env.N_ROWS, env.N_COLS))
    grid[0, 1:-1] = 1  # cliff
    ax.imshow(grid, origin="lower", cmap="Reds", vmin=0, vmax=3)
    for s, a in enumerate(policy):
        row, col = env.to_cell(s)
        if row == 0 and 1 <= col <= env.N_COLS - 2:
            continue
        label = "G" if (row, col) == (0, env.N_COLS - 1) else _ARROWS[int(a)]
        ax.text(col, row, label, ha="center", va="center", fontsize=12)
    ax.set_xticks(np.arange(-0.5, env.N_COLS), minor=True)
    ax.set_yticks(np.arange(-0.5, env.N_ROWS), minor=True)
    ax.grid(which="minor", color="0.7")
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)
    return ax


def plot_cost_to_go(ax, value_fn, title=None, resolution=50):
    """Heat map of the cost-to-go -max_a q(s, a) over Mountain Car's state space.

    value_fn(x, v) must return the array of action values q((x, v), .).
    """
    xs = np.linspace(MountainCar.X_MIN, MountainCar.X_MAX, resolution)
    vs = np.linspace(MountainCar.V_MIN, MountainCar.V_MAX, resolution)
    cost = np.array([[-np.max(value_fn(x, v)) for x in xs] for v in vs])
    im = ax.imshow(cost, origin="lower", aspect="auto", cmap="viridis",
                   extent=[xs[0], xs[-1], vs[0], vs[-1]])
    ax.set_xlabel("Position x")
    ax.set_ylabel("Velocity v")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Cost-to-go")
    return ax
