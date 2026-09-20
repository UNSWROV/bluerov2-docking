"""Arms B and C replayed on recorded bags under injected navigation error.

The vehicle motion is fixed (the recorded trajectory), so this is the estimator
half of the navigation-robustness experiment: how much of a correlated
navigation error each estimator turns into dock-velocity error. Arm B rebuilds
the world-frame measurement through the corrupted vehicle pose; arm C uses the
camera-relative measurement rotated by attitude and the corrupted vehicle
velocity as its process input.
"""
from __future__ import annotations

import csv
import glob
import math
import os

import numpy as np

from .navsim import LEVELS, corrupt
from .relative_kf import RelativeCVFilter
from .replay import ReplayParams, ReplayResult, replay
from .tracks import PoseTrack, R_CAM, T_MOUNT, Trial, corr, load_trial


def corrupted_track(trial: Trial, level: str, seed: int = 0):
    """Vehicle pose and velocity as a navigation system with the given error would report."""
    params = LEVELS[level]
    if params is not None:
        params = type(params)(**{**params.__dict__, "seed": seed})
    t = trial.tf.t
    p_hat, v_hat, b_v, b_p = corrupt(t, trial.tf.p, trial.odom.vel(t), params)
    q = trial.tf.att(t).as_quat()
    return PoseTrack(t, p_hat, q), (t, v_hat), (t, b_v), (t, b_p)


def replay_c(trial: Trial, vel_input, params: ReplayParams = ReplayParams()) -> ReplayResult:
    tv, v_hat = vel_input
    tm = trial.t_meas
    z = trial.odom.att(tm).apply(T_MOUNT + R_CAM.apply(trial.p_cam))     # attitude only, no position
    arrival = tm + params.latency_s
    kf = RelativeCVFilter(max_speed=params.max_dock_speed, sigma_a=params.sigma_a)
    dt = 1.0 / params.predict_rate_hz
    ticks = np.arange(arrival[0] - dt, tm[-1] + 1.0, dt)
    out_t, out_r, out_v, out_h = [], [], [], []
    accepted = np.zeros(len(tm), bool)
    last_update = None; j = 0
    for tk in ticks:
        while j < len(tm) and arrival[j] <= tk:
            R = trial.cov[j][:3, :3]
            if not kf.is_initialized:
                if trial.n_markers[j] >= params.min_markers_for_init:
                    kf.initialize(z[j], R, params.init_inflation); last_update = tk; accepted[j] = True
            else:
                accepted[j] = kf.try_update(z[j], R)
                if accepted[j]:
                    last_update = tk
            j += 1
        if kf.is_initialized:
            vv = np.array([np.interp(tk, tv, v_hat[:, i]) for i in range(3)])
            kf.predict(dt, vv)
            age = tk - last_update
            if params.stale_hold_s is not None and age > params.stale_hold_s:
                kf.x[3:] *= math.exp(-dt / params.stale_decay_s)
            pos_std = math.sqrt(max(kf.P[0, 0], kf.P[1, 1], kf.P[2, 2]))
            stale = age > params.stale_max_age_s or pos_std > params.stale_max_position_std_m
            out_t.append(tk); out_r.append(kf.r.copy()); out_v.append(kf.v_d.copy()); out_h.append(3 if stale else 1)
    return ReplayResult(np.array(out_t), np.array(out_r), np.array(out_v), np.array(out_h), accepted, 0)


def score(trial: Trial, res: ReplayResult, arm: str, veh_hat: PoseTrack, bias, axis: int = 1) -> dict:
    """Velocity-state and relative-position error against the truth, while live."""
    t = res.t; live = res.health == 1
    truth_v = trial.dock.vel(t)[:, axis]
    r_true = trial.dock.pos(t) - trial.odom.pos(t)
    r_est = res.pos - veh_hat.pos(t) if arm == "B" else res.pos
    ev = res.vel[:, axis] - truth_v
    tb, b_v = bias
    bv = np.interp(t, tb, b_v[:, axis])
    # split the velocity error into a slow part (2 s moving average, where a bias
    # shows) and the jitter above it
    w = max(1, int(round(2.0 / np.median(np.diff(t)))))
    ev_slow = np.convolve(ev, np.ones(w) / w, mode="same")
    ev_fast = ev - ev_slow
    # the dock model origin sits a constant offset from the dock frame origin
    dr = r_est - r_true
    dr = dr - np.median(dr[live], axis=0)
    return dict(
        vel_err_rms_m_s=float(np.sqrt(np.mean(ev[live] ** 2))),
        vel_err_slow_rms_m_s=float(np.sqrt(np.mean(ev_slow[live] ** 2))),
        vel_err_fast_rms_m_s=float(np.sqrt(np.mean(ev_fast[live] ** 2))),
        vel_amp_ratio=float(np.std(res.vel[live, axis]) / (np.std(truth_v[live]) + 1e-9)),
        vel_err_corr_bias=corr(ev_slow[live], bv[live]),
        bias_rms_m_s=float(np.sqrt(np.mean(bv[live] ** 2))),
        rel_pos_err_rms_cm=float(np.sqrt(np.mean(np.sum(dr[live] ** 2, axis=1)))) * 100,
        accepted_frac=float(res.accepted.mean()), stale_frac=float(np.mean(~live)),
    )


def run(bags: list[str], out_csv: str, levels=("none", "low", "medium", "high"), seeds=(0,), plot: str | None = None):
    rows = []
    for bag in bags:
        mcap = sorted(glob.glob(os.path.join(bag, "*.mcap")))[0] if os.path.isdir(bag) else bag
        trial = load_trial(mcap)
        name = os.path.basename(bag.rstrip("/"))
        for level in levels:
            for seed in seeds:
                veh_hat, vel_in, bias, _ = corrupted_track(trial, level, seed)
                resB = replay(trial, veh_track=veh_hat)
                resC = replay_c(trial, vel_in)
                for arm, res in (("B", resB), ("C", resC)):
                    row = dict(bag=name, level=level, seed=seed, arm=arm, **score(trial, res, arm, veh_hat, bias))
                    rows.append(row)
                    print(f"{name[:32]:32s} {level:6s} seed {seed} arm {arm}: vel err {row['vel_err_rms_m_s']*100:5.2f} cm/s "
                          f"(slow {row['vel_err_slow_rms_m_s']*100:4.2f}, fast {row['vel_err_fast_rms_m_s']*100:4.2f}, bias {row['bias_rms_m_s']*100:4.2f}), "
                          f"amp ratio {row['vel_amp_ratio']:.2f}, slow corr with bias {row['vel_err_corr_bias']:+.2f}, "
                          f"rel pos err {row['rel_pos_err_rms_cm']:.1f} cm", flush=True)
                if plot and level == "high" and seed == seeds[0] and bag == bags[0]:
                    _plot(trial, resB, resC, bias, plot)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out_csv)
    return rows


def _plot(trial, resB, resC, bias, out, axis=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = trial.t0
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(resB.t - t0, trial.dock.vel(resB.t)[:, axis] * 100, "k", lw=0.8, label="dock velocity, truth")
    ax[0].plot(resB.t - t0, resB.vel[:, axis] * 100, lw=0.9, label="arm B (world frame)")
    ax[0].plot(resC.t - t0, resC.vel[:, axis] * 100, lw=0.9, label="arm C (relative)")
    ax[0].set_ylabel("lateral vel. [cm/s]"); ax[0].legend(fontsize=7, ncol=3)
    tb, b_v = bias
    ax[1].plot(tb - t0, b_v[:, axis] * 100, label="injected velocity bias b_v")
    ax[1].plot(resB.t - t0, (resB.vel[:, axis] - trial.dock.vel(resB.t)[:, axis]) * 100, lw=0.8, label="arm B velocity error")
    ax[1].plot(resC.t - t0, (resC.vel[:, axis] - trial.dock.vel(resC.t)[:, axis]) * 100, lw=0.8, label="arm C velocity error")
    ax[1].set_ylabel("[cm/s]"); ax[1].set_xlabel("time [s]"); ax[1].legend(fontsize=7, ncol=3)
    fig.suptitle(f"{os.path.basename(trial.path)}: navigation error level high"); fig.tight_layout(); fig.savefig(out, dpi=140); print("plot:", out)
