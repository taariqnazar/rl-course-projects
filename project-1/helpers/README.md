# Project 1 helpers

Optional Python helpers for Project 1. Requires NumPy and Matplotlib.

Copy this folder next to your own code, then:

```python
from gridworld import SlipperyCliff, UP, RIGHT, DOWN, LEFT, to_state, to_cell, GAMMA
from plotting import plot_mean_std, plot_values, plot_policy

env = SlipperyCliff(seed=0)
s = env.reset()                    # start state S = (0, 0)
s_next, r, done = env.step(s, UP)  # sample one transition
```

## `gridworld.py`

- `SlipperyCliff(seed)`: the environment. `reset(state=None)` returns the start state, or `state` if given. `step(s, a)` returns `(next_state, reward, done)`.
- States are integers `0..24`. Use `to_state(row, col)` and `to_cell(s)` to convert, with row 0 at the bottom.
- Actions: `UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3`.
- Constants: `GAMMA`, `T_MAX`, `SLIP`, `START`, `GOAL`, `CLIFF`, `TERMINAL_STATES`, `NON_TERMINAL_STATES`.
- `step` does not count steps, so stop episodes at `T_MAX` yourself. Calling `step` on a terminal state raises an error.

The model `P[s, a, s']` and `R[s, a]` is **not** provided, because building it is exercise A1.

## `plotting.py`

- `plot_mean_std(ax, x, runs, label, color=None)`: mean over seeds with a ±1 std band. `runs` has shape `(n_seeds, len(x))`.
- `plot_values(ax, V, title=None, fmt=".2f")`: heat map of any length-25 array, such as a value function. Use `fmt=".0f"` for counts.
- `plot_policy(ax, policy, title=None)`: a deterministic policy, given as a length-25 array of actions, drawn as arrows.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plot_values(axes[0], V, "v*")
plot_policy(axes[1], policy, "pi*")
fig.savefig("optimal.png")
```
