"""The slippery cliff environment from Project 1.

States are integers 0..24. State s is the cell (row, col) = divmod(s, 5), with
row 0 at the bottom. Actions are integers: UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3.

Example
-------
>>> env = SlipperyCliff(seed=0)
>>> s = env.reset()
>>> s_next, r, done = env.step(s, UP)

In Part B you may only use `reset()` and `step()`. Building the model
P[s, a, s'] and R[s, a] is part of the exercise (A1); `env.step` gives you a
way to check it.
"""

import numpy as np

N_ROWS, N_COLS = 5, 5
N_STATES = N_ROWS * N_COLS
N_ACTIONS = 4
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]

START = (0, 0)
GOAL = (0, 4)
CLIFF = [(0, 1), (0, 2), (0, 3)]
SLIP = 0.1  # probability of moving perpendicular (split equally between the two sides)
GAMMA = 0.9
T_MAX = 200

R_STEP, R_GOAL, R_CLIFF = -1.0, 10.0, -10.0

_MOVES = {UP: (1, 0), RIGHT: (0, 1), DOWN: (-1, 0), LEFT: (0, -1)}


def to_state(row, col):
    """Cell (row, col) -> state index."""
    return row * N_COLS + col


def to_cell(s):
    """State index -> cell (row, col)."""
    return divmod(int(s), N_COLS)


TERMINAL_STATES = [to_state(*GOAL)] + [to_state(*c) for c in CLIFF]
NON_TERMINAL_STATES = [s for s in range(N_STATES) if s not in TERMINAL_STATES]


def is_terminal(s):
    return int(s) in TERMINAL_STATES


class SlipperyCliff:
    """Sample-based interface to the slippery cliff.

    Parameters
    ----------
    seed : int or None
        Seed for the environment's random number generator. Use different
        seeds for different runs.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, state=None):
        """Return the start state, or `state` if given (used for exploring starts)."""
        return to_state(*START) if state is None else int(state)

    def step(self, s, a):
        """Take action `a` in state `s`. Returns (next_state, reward, done)."""
        if is_terminal(s):
            raise ValueError(f"state {s} is terminal; call reset()")
        u = self.rng.random()
        if u < SLIP / 2:
            a = (a + 1) % N_ACTIONS
        elif u < SLIP:
            a = (a + 3) % N_ACTIONS
        row, col = to_cell(s)
        dr, dc = _MOVES[a]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < N_ROWS and 0 <= nc < N_COLS):
            nr, nc = row, col
        s_next = to_state(nr, nc)
        if (nr, nc) == GOAL:
            return s_next, R_GOAL, True
        if (nr, nc) in CLIFF:
            return s_next, R_CLIFF, True
        return s_next, R_STEP, False
