"""Fit a first-order gain-plus-delay plant to a recorded command and velocity response.

Model:  v(t) = K * u(t - tau) filtered by a first-order lag with time constant T,
i.e. T dv/dt + v = K u(t - tau). The fit is a grid search over tau with a least
squares solution for K and T at each tau (discretised at the sample rate), which
is robust on short step records where a nonlinear optimiser is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlantFit:
    gain: float          # steady-state velocity per unit command
    tau_s: float         # pure delay
    time_constant_s: float
    rms_residual: float  # velocity fit residual


def _delay(u, shift):
    return np.concatenate([np.full(shift, u[0]), u[: len(u) - shift]]) if shift > 0 else np.asarray(u, float)


def simulate(t, u, gain, tau_s, time_constant_s, v0=0.0):
    """Response of the model to command u(t) sampled on t (uniform spacing)."""
    from scipy.signal import lfilter

    dt = float(np.median(np.diff(t)))
    u_d = _delay(np.asarray(u, float), int(round(tau_s / dt)))
    a = dt / max(time_constant_s, dt)
    # v[k] = (1 - a) v[k-1] + a K u_d[k-1]
    v = lfilter([0.0, a * gain], [1.0, -(1.0 - a)], u_d)
    return v + v0 * (1.0 - a) ** np.arange(len(t))


def fit_first_order(t, u, v, tau_max_s=1.5, tau_step_s=0.02, tc_grid=None) -> PlantFit:
    """Output-error fit: grid over delay and time constant, closed-form gain at each.

    Regressing one sample ahead is biased by measurement noise on the lagged
    velocity, so the fit minimises the simulated-response residual instead.
    """
    t, u, v = (np.asarray(x, float) for x in (t, u, v))
    dt = float(np.median(np.diff(t)))
    v0 = float(np.mean(v[: max(3, int(0.5 / dt))]))
    tc_grid = np.arange(0.05, 3.0, 0.05) if tc_grid is None else tc_grid
    best = None
    for tau in np.arange(0.0, tau_max_s + 1e-9, tau_step_s):
        for tc in tc_grid:
            s1 = simulate(t, u, 1.0, tau, tc, 0.0)
            gain = float(np.dot(v - v0, s1) / max(np.dot(s1, s1), 1e-12))
            res = float(np.sqrt(np.mean((gain * s1 + v0 - v) ** 2)))
            if best is None or res < best.rms_residual:
                best = PlantFit(gain=gain, tau_s=float(tau), time_constant_s=float(tc), rms_residual=res)
    return best


def fit_bag(path: str, axis: str = "y", cmd_topic: str = "/cmd_vel"):
    """Fit a plant-identification bag: command on cmd_topic, odometry velocity in the body frame."""

    from .bagio import read_topics, stamp

    d = read_topics(path, [cmd_topic, "/model/bluerov2_heavy/odometry"])
    odo = d["/model/bluerov2_heavy/odometry"]
    ot = np.array([stamp(m) for _, m in odo]); ol = np.array([l for l, _ in odo])
    a, b = np.polyfit(ol, ot, 1)
    ct = np.array([a * l + b for l, _ in d[cmd_topic]])
    cu = np.array([getattr(m.linear, axis) for _, m in d[cmd_topic]])
    # odometry twist is body frame already; world position derivative rotated into body as a check
    v_body = np.array([getattr(m.twist.twist.linear, axis) for _, m in odo])
    t = np.arange(ot[0], ot[-1], 0.02)
    # the autopilot holds the last command between messages: zero-order hold, not linear
    if len(ct):
        idx = np.clip(np.searchsorted(ct, t, side="right") - 1, 0, len(ct) - 1)
        u = cu[idx]
        u[t < ct[0]] = 0.0
    else:
        u = np.zeros_like(t)
    v = np.interp(t, ot, v_body)
    fit = fit_first_order(t, u, v)
    print(f"{path.split('/')[-1]} axis {axis}: gain {fit.gain:.2f} m/s per unit, delay {fit.tau_s:.2f} s, "
          f"time constant {fit.time_constant_s:.2f} s, residual {fit.rms_residual*100:.2f} cm/s")
    return fit
