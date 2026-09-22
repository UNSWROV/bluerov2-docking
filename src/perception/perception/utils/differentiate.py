"""Causal finite-difference velocity from a stamped position stream."""
from __future__ import annotations

from collections import deque

import numpy as np


class CausalDifferentiator:
    """Velocity as the least-squares slope over the last `window_s` seconds.

    A two-sample difference of a 50 Hz ground-truth pose is dominated by
    quantisation; a short least-squares window keeps the lag small (about half
    the window) while removing that noise. Returns None until two samples exist.
    """

    def __init__(self, window_s: float = 0.2):
        self.window_s = float(window_s)
        self._t: deque[float] = deque()
        self._p: deque[np.ndarray] = deque()

    def push(self, t: float, p) -> np.ndarray | None:
        p = np.asarray(p, dtype=float)
        if self._t and t <= self._t[-1]:
            return None
        self._t.append(float(t)); self._p.append(p)
        while self._t and self._t[-1] - self._t[0] > self.window_s:
            self._t.popleft(); self._p.popleft()
        if len(self._t) < 2:
            return None
        tt = np.array(self._t); tt = tt - tt.mean()
        pp = np.array(self._p); pp = pp - pp.mean(axis=0)
        return (tt @ pp) / float(tt @ tt)

    @property
    def lag_s(self) -> float:
        return 0.5 * (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0
