"""Correlated navigation-error model for the estimator experiments.

First-order Gauss-Markov velocity bias, integrated into a position error, plus
small white terms, as specified in the UT27 briefing:

    b_v[k+1] = rho b_v[k] + w_v,   rho = exp(-dt / tau_b),   w_v ~ N(0, sigma_b^2 (1 - rho^2))
    b_p[k+1] = b_p[k] + dt b_v[k] + w_p
    p_hat = p + b_p + n_p,   v_hat = v + b_v + n_v

sigma_b is the stationary standard deviation of the velocity bias, so the level
of navigation error is set independently of dt and tau_b. The same generator
drives the offline replays here and the injection node in simulation (T7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NavErrorParams:
    sigma_b: float          # stationary std of the velocity bias [m/s]
    tau_b: float = 30.0     # bias correlation time [s]
    sigma_wp: float = 0.0   # extra white position drift per step [m]
    sigma_np: float = 0.005 # white position noise [m]
    sigma_nv: float = 0.005 # white velocity noise [m/s]
    seed: int = 0


LEVELS = {
    "none": None,
    "low": NavErrorParams(sigma_b=0.01),
    "medium": NavErrorParams(sigma_b=0.03),
    "high": NavErrorParams(sigma_b=0.10),
}


def gauss_markov_bias(t: np.ndarray, params: NavErrorParams, dims: int = 3):
    """Return (b_v, b_p) sampled on the (possibly uneven) time grid t."""
    rng = np.random.default_rng(params.seed)
    n = len(t)
    b_v = np.zeros((n, dims)); b_p = np.zeros((n, dims))
    b_v[0] = rng.normal(0.0, params.sigma_b, dims)      # start in the stationary distribution
    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        rho = np.exp(-dt / params.tau_b)
        b_v[k] = rho * b_v[k - 1] + rng.normal(0.0, params.sigma_b * np.sqrt(1 - rho ** 2), dims)
        b_p[k] = b_p[k - 1] + dt * b_v[k - 1] + rng.normal(0.0, params.sigma_wp, dims)
    return b_v, b_p


def corrupt(t: np.ndarray, p: np.ndarray, v: np.ndarray, params: NavErrorParams | None):
    """Navigation signals the estimator sees: (p_hat, v_hat, b_v, b_p)."""
    if params is None:
        return p.copy(), v.copy(), np.zeros_like(p), np.zeros_like(p)
    rng = np.random.default_rng(params.seed + 1)
    b_v, b_p = gauss_markov_bias(t, params, p.shape[1])
    p_hat = p + b_p + rng.normal(0.0, params.sigma_np, p.shape)
    v_hat = v + b_v + rng.normal(0.0, params.sigma_nv, v.shape)
    return p_hat, v_hat, b_v, b_p
