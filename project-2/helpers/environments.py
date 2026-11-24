"""Environments for Project 2.

All environments share the same interface as in Project 1:

    env = SomeEnvironment(seed=0)
    s = env.reset()                    # start state
    s_next, r, done = env.step(s, a)   # sample one transition

`step` does not count steps; stop episodes yourself if needed.
"""

import numpy as np


class RandomWalk:
    """Random walk on states 1..n with terminal states 0 (left) and n + 1 (right).

    Every episode starts in the middle state. There are no actions: pass
    `a=None` to `step`. Each step moves the agent left or right with equal
    probability, by a uniformly random distance between 1 and `max_jump`.
    Jumps past an end land in that end's terminal state. Entering the left
    terminal gives reward `left_reward` (default -1), entering the right
    terminal gives +1, and every other transition gives 0. There is no
    discounting (gamma = 1).

    RandomWalk(5, left_reward=0)     : the 5-state walk (Sutton & Barto, Example 6.2).
    RandomWalk(19)                   : the 19-state walk (Example 7.1).
    RandomWalk(1000, max_jump=100)   : the 1000-state walk (Example 9.1).
    """

    def __init__(self, n_states=19, max_jump=1, left_reward=-1.0, seed=None):
        self.n_states = n_states
        self.max_jump = max_jump
        self.left_reward = float(left_reward)
        self.rng = np.random.default_rng(seed)

    @property
    def start_state(self):
        return (self.n_states + 1) // 2

    def reset(self):
        return self.start_state

    def is_terminal(self, s):
        return s == 0 or s == self.n_states + 1

    def step(self, s, a=None):
        if self.is_terminal(s):
            raise ValueError(f"state {s} is terminal; call reset()")
        jump = self.rng.integers(1, self.max_jump + 1) * self.rng.choice((-1, 1))
        s_next = min(max(s + jump, 0), self.n_states + 1)
        if s_next == 0:
            return s_next, self.left_reward, True
        if s_next == self.n_states + 1:
            return s_next, 1.0, True
        return s_next, 0.0, False

    def true_values(self):
        """Exact state values for states 1..n (array of length n), computed by solving
        the Bellman equations. Use them to measure the error of your estimates."""
        n, k = self.n_states, self.max_jump
        p = np.zeros((n + 2, n + 2))
        for s in range(1, n + 1):
            for j in range(1, k + 1):
                for sign in (-1, 1):
                    p[s, min(max(s + sign * j, 0), n + 1)] += 1 / (2 * k)
        r = p[1:n + 1, n + 1] + self.left_reward * p[1:n + 1, 0]
        return np.linalg.solve(np.eye(n) - p[1:n + 1, 1:n + 1], r)


class CliffWalking:
    """The cliff-walking gridworld (Sutton & Barto, Example 6.6).

    A 4 x 12 grid. States are integers 0..47; state s is the cell
    (row, col) = divmod(s, 12), with row 0 at the bottom. The start is (0, 0)
    and the goal is (0, 11). Cells (0, 1)..(0, 10) are the cliff.

    Actions: UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3. Moves are deterministic;
    moving off the grid leaves the agent in place. Every step gives -1,
    except stepping into the cliff, which gives -100 and sends the agent back
    to the start (the episode continues). Entering the goal ends the episode.
    Undiscounted (gamma = 1).
    """

    N_ROWS, N_COLS = 4, 12
    N_STATES = N_ROWS * N_COLS
    N_ACTIONS = 4
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    _MOVES = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)  # unused: the dynamics are deterministic

    @staticmethod
    def to_state(row, col):
        return row * CliffWalking.N_COLS + col

    @staticmethod
    def to_cell(s):
        return divmod(int(s), CliffWalking.N_COLS)

    @property
    def start_state(self):
        return self.to_state(0, 0)

    @property
    def goal_state(self):
        return self.to_state(0, self.N_COLS - 1)

    def reset(self):
        return self.start_state

    def step(self, s, a):
        if s == self.goal_state:
            raise ValueError("the goal is terminal; call reset()")
        row, col = self.to_cell(s)
        dr, dc = self._MOVES[a]
        row = min(max(row + dr, 0), self.N_ROWS - 1)
        col = min(max(col + dc, 0), self.N_COLS - 1)
        if row == 0 and 1 <= col <= self.N_COLS - 2:
            return self.start_state, -100.0, False
        s_next = self.to_state(row, col)
        return s_next, -1.0, s_next == self.goal_state


class MountainCar:
    """Mountain Car (Sutton & Barto, Example 10.1).

    The state is a tuple (x, v) with position x in [-1.2, 0.5] and velocity
    v in [-0.07, 0.07]. Actions: 0 = full throttle reverse, 1 = zero throttle,
    2 = full throttle forward. Every step gives -1 until the car passes
    x = 0.5, which ends the episode. Each episode starts at a uniformly
    random x in [-0.6, -0.4] with v = 0. Undiscounted (gamma = 1).
    """

    X_MIN, X_MAX = -1.2, 0.5
    V_MIN, V_MAX = -0.07, 0.07
    N_ACTIONS = 3

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        return (self.rng.uniform(-0.6, -0.4), 0.0)

    def step(self, s, a):
        x, v = s
        v = min(max(v + 0.001 * (a - 1) - 0.0025 * np.cos(3 * x), self.V_MIN), self.V_MAX)
        x = x + v
        if x < self.X_MIN:
            x, v = self.X_MIN, 0.0
        if x >= self.X_MAX:
            return (x, v), -1.0, True
        return (x, v), -1.0, False
