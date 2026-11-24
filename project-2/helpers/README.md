# Project 2 helpers

Optional Python helpers for Project 2. Requires NumPy and Matplotlib.

Copy this folder next to your own code, then:

```python
from environments import RandomWalk, CliffWalking, MountainCar
from tiles import TileCoder
from plotting import plot_mean_std, plot_cliff_policy, plot_cost_to_go

env = CliffWalking()
s = env.reset()                    # start state
s_next, r, done = env.step(s, CliffWalking.UP)
```

All environments use the same interface as in Project 1: `reset()` returns the start state and `step(s, a)` returns `(next_state, reward, done)`. `step` does not count steps, so stop long episodes yourself. All environments are undiscounted.

## `environments.py`

- `RandomWalk(n_states=19, max_jump=1, left_reward=-1, seed=None)`: states `1..n`, terminals `0` (reward `left_reward`) and `n + 1` (reward +1). There are no actions: call `step(s)`. `true_values()` returns the exact values of states `1..n`.
  - `RandomWalk(5, left_reward=0)`: the 5-state walk of Example 6.2.
  - `RandomWalk(19)`: the 19-state walk.
  - `RandomWalk(1000, max_jump=100)`: the 1000-state walk.
- `CliffWalking(seed=None)`: states `0..47`, actions `UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3`. Use `to_state(row, col)` and `to_cell(s)` to convert, with row 0 at the bottom. The cliff gives −100 and returns the agent to the start; the episode continues.
- `MountainCar(seed=None)`: the state is a tuple `(x, v)`; actions `0, 1, 2` are reverse, zero and forward throttle. Bounds: `X_MIN, X_MAX, V_MIN, V_MAX`.

## `tiles.py`

`TileCoder(lows, highs, n_tilings=8, tiles_per_dim=8)` maps a state to the indices of its active tiles, one per tiling. With linear function approximation the value is the sum of the weights at those indices:

```python
coder = TileCoder([MountainCar.X_MIN, MountainCar.V_MIN], [MountainCar.X_MAX, MountainCar.V_MAX])
w = np.zeros((MountainCar.N_ACTIONS, coder.n_features))   # one weight vector per action
idx = coder.active_tiles((x, v))
q = w[a, idx].sum()
w[a, idx] += alpha * (target - q)                          # semi-gradient update
```

For the 1000-state random walk, use a one-dimensional coder, e.g. `TileCoder([1], [1001], n_tilings=1, tiles_per_dim=10)` for state aggregation into 10 groups.

## `plotting.py`

- `plot_mean_std(ax, x, runs, label, color=None)`: mean over seeds with a ±1 std band. `runs` has shape `(n_seeds, len(x))`.
- `plot_cliff_policy(ax, policy, title=None)`: a greedy policy for cliff walking, given as a length-48 array of actions, drawn as arrows.
- `plot_cost_to_go(ax, value_fn, title=None)`: heat map of `-max_a q(s, a)` over Mountain Car's state space. `value_fn(x, v)` must return the array of action values.
