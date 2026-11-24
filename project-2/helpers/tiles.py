"""Tile coding (Sutton & Barto, Section 9.5.4).

A TileCoder maps a continuous state to the indices of its active tiles, one per
tiling. With linear function approximation, the value of a state is the sum of
the weights at those indices:

    coder = TileCoder(lows=[-1.2, -0.07], highs=[0.5, 0.07], n_tilings=8, tiles_per_dim=8)
    w = np.zeros(coder.n_features)
    idx = coder.active_tiles((x, v))   # array of n_tilings indices
    value = w[idx].sum()
    w[idx] += alpha * error            # semi-gradient update: the gradient is 1 at idx

For action values, keep one weight vector per action, e.g. `w = np.zeros((n_actions,
coder.n_features))` and `q = w[a, idx].sum()`.

The tilings are offset from each other by fractions of a tile width, using the
asymmetric displacement (1, 3, 5, ...) recommended in Section 9.5.4. As in the
book, `n_tilings` should be a power of 2 (and at least 4 times the number of
dimensions); otherwise some tilings would coincide.
"""

import numpy as np


class TileCoder:
    def __init__(self, lows, highs, n_tilings=8, tiles_per_dim=8):
        self.lows = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.n_dims = len(self.lows)
        # One extra tile per dimension so that offset tilings still cover the space.
        self._side = tiles_per_dim + 1
        self._tiles_per_tiling = self._side ** self.n_dims
        displacement = np.arange(1, 2 * self.n_dims, 2)  # 1, 3, 5, ...
        if n_tilings > 1 and np.any(np.gcd(displacement, n_tilings) > 1):
            raise ValueError("n_tilings should be a power of 2 (see Sutton & Barto, Section 9.5.4)")
        self._offsets = (np.arange(n_tilings)[:, None] * displacement[None, :] % n_tilings) / n_tilings
        self._strides = self._side ** np.arange(self.n_dims)

    @property
    def n_features(self):
        return self.n_tilings * self._tiles_per_tiling

    def active_tiles(self, state):
        """Indices of the active tiles for `state` (one per tiling)."""
        scaled = (np.asarray(state, dtype=float) - self.lows) / (self.highs - self.lows)
        scaled = np.clip(scaled, 0.0, 1.0) * self.tiles_per_dim
        coords = np.floor(scaled[None, :] + self._offsets).astype(int)
        coords = np.minimum(coords, self._side - 1)
        return np.arange(self.n_tilings) * self._tiles_per_tiling + coords @ self._strides
