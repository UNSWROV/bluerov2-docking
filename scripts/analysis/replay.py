"""Replay the dock-pose filter offline on a recorded measurement stream.

Runs the real filter class (perception.aruco.lib.kalman) on the world-frame
measurements rebuilt from a bag, with the node's parameters and timing: predict
at 30 Hz, update when the measurement arrives (stamp plus pipeline latency),
Mahalanobis gate, velocity clamp, covariance ceiling for the STALE health.

Variants let the estimator side of the fix be tested before any simulation run:
`stale_decay_s` decays the velocity state once no update has been accepted for
`stale_hold_s` seconds, so a stale filter cannot dead-reckon indefinitely.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "perception")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from perception.aruco.lib.kalman import DockPoseKalmanFilter, make_process_noise  # noqa: E402

from .tracks import Trial, world_pose_from_cam  # noqa: E402


@dataclass
class ReplayParams:
    regime: str = "sway"
    sigma_a: float = 0.16
    predict_rate_hz: float = 30.0
    gate_chi2: float = 18.548
    init_inflation: float = 100.0
    min_markers_for_init: int = 2
    max_dock_speed: float = 0.2
    stale_max_age_s: float = 3.0
    stale_max_position_std_m: float = 0.15
    latency_s: float = 0.04       # measured pipeline latency, stamp to arrival
    stale_hold_s: float | None = None    # variant: start decaying the velocity after this long without an update
    stale_decay_s: float = 1.0           # variant: velocity decay time constant


@dataclass
class ReplayResult:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    health: np.ndarray            # 1 healthy/degraded, 3 stale (age or covariance)
    accepted: np.ndarray          # per measurement: True if the update was applied
    n_init_deferred: int


def replay(trial: Trial, params: ReplayParams = ReplayParams()) -> ReplayResult:
    tm = trial.t_meas
    z_p, z_q = world_pose_from_cam(trial.p_cam, trial.q_cam, trial.tf, tm)
    arrival = tm + params.latency_s
    kf = DockPoseKalmanFilter(max_speed=params.max_dock_speed if params.max_dock_speed > 0 else None)
    dt_pred = 1.0 / params.predict_rate_hz
    ticks = np.arange(arrival[0] - dt_pred, tm[-1] + 1.0, dt_pred)
    out_t, out_p, out_v, out_h = [], [], [], []
    accepted = np.zeros(len(tm), bool)
    last_update = None
    n_deferred = 0
    j = 0
    for tk in ticks:
        # apply every measurement that has arrived by this tick, in order
        while j < len(tm) and arrival[j] <= tk:
            cov = trial.cov[j]
            if not kf.is_initialized:
                if trial.n_markers[j] >= params.min_markers_for_init:
                    kf.initialize(z_p[j].copy(), z_q[j].copy(), cov * params.init_inflation,
                                  velocity_std=0.2 if params.regime == "sway" else 0.0)
                    last_update = tk
                    accepted[j] = True
                else:
                    n_deferred += 1
            else:
                ok = kf.try_update(z_p[j].copy(), z_q[j].copy(), cov[:3, :3], cov[3:, 3:], gate_chi2=params.gate_chi2)
                accepted[j] = ok
                if ok:
                    last_update = tk
            j += 1
        if kf.is_initialized:
            kf.predict(dt=dt_pred, process_noise=make_process_noise(dt_pred, params.regime, params.sigma_a))
            age = tk - last_update
            if params.stale_hold_s is not None and age > params.stale_hold_s:
                kf._velocity *= math.exp(-dt_pred / params.stale_decay_s)   # variant under test
            pos_std = math.sqrt(max(kf.covariance[0, 0], kf.covariance[1, 1], kf.covariance[2, 2]))
            stale = age > params.stale_max_age_s or pos_std > params.stale_max_position_std_m
            out_t.append(tk); out_p.append(kf.position.copy()); out_v.append(kf.velocity.copy()); out_h.append(3 if stale else 1)
    return ReplayResult(np.array(out_t), np.array(out_p), np.array(out_v), np.array(out_h), accepted, n_deferred)


def compare(trial: Trial, res: ReplayResult, axis: int = 1) -> dict:
    """Replay against the recorded estimate and the truth."""
    t = res.t
    rec = trial.filt.pos(t)[:, axis]
    truth = trial.dock.pos(t)[:, axis]
    off = np.median(res.pos[:, axis] - truth)
    live = res.health == 1
    d = res.pos[:, axis] - rec
    return dict(
        replay_minus_recorded_rms_cm=float(np.sqrt(np.mean(d[live] ** 2))) * 100,
        replay_minus_recorded_max_cm=float(np.max(np.abs(d))) * 100,
        replay_minus_truth_max_cm=float(np.max(np.abs(res.pos[:, axis] - truth - off))) * 100,
        recorded_minus_truth_max_cm=float(np.max(np.abs(rec - truth - np.median(rec - truth)))) * 100,
        accepted_frac=float(res.accepted.mean()),
        stale_frac=float(np.mean(res.health == 3)),
    )


def report(trial: Trial, axis: int = 1, plot: str | None = None, hold_s: float = 1.0, decay_s: float = 1.0):
    base = replay(trial)
    var = replay(trial, ReplayParams(stale_hold_s=hold_s, stale_decay_s=decay_s))
    cb, cv = compare(trial, base, axis), compare(trial, var, axis)
    print(f"bag: {trial.path.split('/')[-1]}")
    print(f"replay of the node as recorded: replay minus recorded {cb['replay_minus_recorded_rms_cm']:.1f} cm RMS while live, "
          f"{cb['replay_minus_recorded_max_cm']:.0f} cm max; accepted {cb['accepted_frac']*100:.0f}% of measurements; "
          f"stale {cb['stale_frac']*100:.0f}% of time")
    print(f"max estimate excursion from truth: recorded {cb['recorded_minus_truth_max_cm']:.0f} cm, replay {cb['replay_minus_truth_max_cm']:.0f} cm, "
          f"replay with velocity decay after {hold_s:.1f} s (tau {decay_s:.1f} s) {cv['replay_minus_truth_max_cm']:.0f} cm")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t0 = trial.t0
        fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
        truth = trial.dock.pos(base.t)[:, axis]
        ax[0].plot(base.t - t0, truth * 100, "k", lw=0.8, label="dock, truth")
        ax[0].plot(base.t - t0, (trial.filt.pos(base.t)[:, axis]) * 100, lw=0.9, label="recorded estimate")
        ax[0].plot(base.t - t0, base.pos[:, axis] * 100, "--", lw=0.9, label="offline replay")
        ax[0].plot(var.t - t0, var.pos[:, axis] * 100, lw=0.9, label=f"replay + velocity decay ({hold_s:.0f} s hold, tau {decay_s:.0f} s)")
        ax[0].set_ylabel("lateral [cm]"); ax[0].legend(fontsize=7, ncol=2)
        ax[1].plot(base.t - t0, base.vel[:, axis] * 100, lw=0.9, label="replay velocity state")
        ax[1].plot(var.t - t0, var.vel[:, axis] * 100, lw=0.9, label="with decay")
        ax[1].plot(base.t - t0, trial.dock.vel(base.t)[:, axis] * 100, "k", lw=0.6, label="dock velocity, truth")
        ax[1].set_ylabel("lateral vel. [cm/s]"); ax[1].set_xlabel("time [s]"); ax[1].legend(fontsize=7, ncol=3)
        fig.tight_layout(); fig.savefig(plot, dpi=140); print("plot:", plot)
    return cb, cv
