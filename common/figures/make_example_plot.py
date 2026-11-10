"""Generate the example plot shared by all projects (README and instructions).

Synthetic data only: it shows the expected plot format (mean over seeds with
a +-1 std band) without revealing any results from the assignment.

Run: uv run --with numpy --with matplotlib python make_example_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

rng = np.random.default_rng(0)
episodes = np.arange(50, 2001, 50)
n_seeds = 10


def runs(start, floor, rate, noise):
    base = floor + (start - floor) * np.exp(-episodes / rate)
    return base * np.exp(noise * rng.standard_normal((n_seeds, len(episodes))))


curves = {"Method A": (runs(3.0, 0.3, 250, 0.10), BLUE),
          "Method B": (runs(3.0, 1.2, 500, 0.15), ORANGE)}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for name, (y, color) in curves.items():
    mean, std = y.mean(0), y.std(0)
    ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.plot(episodes, mean, color=color, linewidth=2, label=f"{name} (mean ± 1 std)")
    ax.annotate(name, (episodes[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)

ax.set_xlabel("Episodes")
ax.set_ylabel("RMSE")
ax.set_title(f"Example: RMSE vs. episodes ({n_seeds} seeds)", color=INK, fontsize=10, loc="left")
ax.set_xlim(0, 2250)
ax.set_ylim(bottom=0)
ax.grid(True, color=GRID, linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right", labelcolor=INK)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("example-plot.svg"))
