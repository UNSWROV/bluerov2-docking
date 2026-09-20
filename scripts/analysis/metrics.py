"""Per-trial metrics for the moving-dock experiments.

Outcome and capture geometry follow the thesis scoring (scripts/rescore_contacts.py):
the vehicle is inside the dock once the camera passes the dock plane, a trial is
CONTACT if it is inside, before the DOCKED latch, and more than the aperture radius
off the entry axis; CLEAN if it latched without such a crossing; NO_DOCK otherwise.
Estimation metrics are computed only while the filter reports HEALTHY or DEGRADED.
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np

from .labels import parse_label
from .tracks import Trial, corr, load_trial, measured_mask, measurement_error

CAM_X = 0.21                 # camera ahead of base_link along body x
ENTRY_OFFSET = (0.002, 0.176)  # entry axis relative to the dock model origin (y, z), from the static cell
APERTURE = 0.15
DOCKED, COARSE, FINE = 2, 0, 1
HEALTHY, DEGRADED, STALE = 1, 2, 3

COLUMNS = ["cell", "arm", "dock", "period", "phase", "nav", "outcome", "docked", "t_dock_s",
           "capture_lateral_m", "capture_vertical_m", "capture_offset_m", "closing_speed_m_s",
           "contact_offset_m", "contact_speed_m_s", "meas_err_rms_cm", "track_err_rms_cm",
           "vel_err_rms_m_s", "vel_amp_ratio", "vel_err_corr_vehicle", "vel_err_corr_nav",
           "veh_over_dock", "blackout_frac", "n_demotions", "n_stale", "n_gaps_gt1s"]


def outcome(trial: Trial, dt: float = 0.02) -> dict:
    """Outcome, capture offset and closing speed at the dock plane, contact details."""
    g = np.arange(trial.odom.t[0], trial.odom.t[-1], dt)
    veh, dock = trial.odom.pos(g), trial.dock.pos(g)
    gap = dock[:, 0] - (veh[:, 0] + CAM_X)
    lateral = veh[:, 1] - dock[:, 1] - ENTRY_OFFSET[0]
    vertical = veh[:, 2] - dock[:, 2] - ENTRY_OFFSET[1]
    off = np.hypot(lateral, vertical)
    inside = gap < 0
    t_latch = next((t for t, s in trial.states if s == DOCKED), None)
    prelatch = np.ones(len(g), bool) if t_latch is None else g < t_latch
    closing = np.gradient(-gap, g)
    contact_mask = inside & prelatch & (off > APERTURE)
    res = dict(docked=t_latch is not None, t_dock_s=(t_latch - trial.t0) if t_latch else float("nan"),
               capture_lateral_m=float("nan"), capture_vertical_m=float("nan"), capture_offset_m=float("nan"),
               closing_speed_m_s=float("nan"), contact_offset_m=float("nan"), contact_speed_m_s=float("nan"))
    if inside.any():
        j = int(np.argmax(inside))
        res.update(capture_lateral_m=float(lateral[j]), capture_vertical_m=float(vertical[j]),
                   capture_offset_m=float(off[j]), closing_speed_m_s=float(closing[j]))
    if contact_mask.any():
        i = int(np.argmax(contact_mask))
        res.update(contact_offset_m=float(off[i]), contact_speed_m_s=float(closing[i]))
        res["outcome"] = "CONTACT"
    else:
        res["outcome"] = "CLEAN" if res["docked"] else "NO_DOCK"
    return res


def estimation(trial: Trial, axis: int = 1, nav_error=None) -> dict:
    """Velocity-state error against the true dock velocity while the filter is usable."""
    if trial.filt is None or len(trial.t_vel) == 0:
        return dict(vel_err_rms_m_s=float("nan"), vel_amp_ratio=float("nan"),
                    vel_err_corr_vehicle=float("nan"), vel_err_corr_nav=float("nan"), track_err_rms_cm=float("nan"))
    t = trial.t_vel
    h = np.interp(t, trial.t_health, trial.health, left=STALE, right=STALE) if len(trial.t_health) else np.full(len(t), HEALTHY)
    ok = (np.round(h) == HEALTHY) | (np.round(h) == DEGRADED)
    truth = trial.dock.vel(t)[:, axis]
    err = trial.vel[:, axis] - truth
    vveh = trial.odom.vel(t)[:, axis]
    _, _, _, off = measurement_error(trial)
    track = trial.filt.pos(t)[:, axis] - trial.dock.pos(t)[:, axis] - off[axis]
    nav_corr = float("nan")
    if nav_error is not None and ok.any():
        nav_corr = corr(err[ok], np.interp(t[ok], nav_error[0], nav_error[1]))
    return dict(
        vel_err_rms_m_s=float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else float("nan"),
        vel_amp_ratio=float(np.std(trial.vel[ok, axis]) / (np.std(truth[ok]) + 1e-9)) if ok.any() else float("nan"),
        vel_err_corr_vehicle=corr(err[ok], vveh[ok]) if ok.any() else float("nan"),
        vel_err_corr_nav=nav_corr,
        track_err_rms_cm=float(np.sqrt(np.mean(track[ok] ** 2))) * 100 if ok.any() else float("nan"),
    )


def supervision(trial: Trial) -> dict:
    """Demotions (FINE to COARSE), stale episodes, blackout fraction, long gaps."""
    seq = [s for _, s in trial.states]
    n_dem = sum(1 for i in range(1, len(seq)) if seq[i] == COARSE and seq[i - 1] == FINE)
    h = np.round(trial.health).astype(int) if len(trial.health) else np.array([], int)
    n_stale = int(np.sum((h[1:] == STALE) & (h[:-1] != STALE))) if len(h) > 1 else 0
    g = np.arange(trial.t_meas[0], trial.t_meas[-1], 0.1)
    gaps = np.diff(trial.t_meas)
    return dict(n_demotions=n_dem, n_stale=n_stale, blackout_frac=float(1 - measured_mask(trial, g).mean()),
                n_gaps_gt1s=int(np.sum(gaps > 1)))


def overswing(trial: Trial, axis: int = 1) -> dict:
    g = np.arange(trial.t_meas[0], trial.t_meas[-1], 0.1)
    have = measured_mask(trial, g)
    veh, dock = trial.odom.pos(g)[:, axis], trial.dock.pos(g)[:, axis]
    if have.sum() < 10 or np.std(dock[have]) < 1e-6:
        return dict(veh_over_dock=float("nan"))
    return dict(veh_over_dock=float(np.std(veh[have] - np.median(veh[have])) / np.std(dock[have])))


def trial_metrics(trial: Trial, axis: int = 1, nav_error=None) -> dict:
    e_world, _, _, _ = measurement_error(trial)
    row = dict(meas_err_rms_cm=float(np.sqrt(np.mean(e_world[:, axis] ** 2))) * 100)
    row.update(outcome(trial)); row.update(estimation(trial, axis, nav_error))
    row.update(supervision(trial)); row.update(overswing(trial, axis))
    return row


def run(bag_dir: str, out_csv: str, axis: int = 1):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for bag in sorted(glob.glob(os.path.join(bag_dir, "*"))):
            lab = parse_label(bag)
            mcaps = glob.glob(os.path.join(bag, "*.mcap")) if os.path.isdir(bag) else []
            if lab is None or not mcaps:
                continue
            try:
                row = trial_metrics(load_trial(mcaps[0]), axis)
            except Exception as e:  # noqa: BLE001
                print("skip", os.path.basename(bag), e); continue
            row.update(cell=lab.cell, arm=lab.arm, dock=lab.dock, period=lab.period, phase=lab.phase, nav=lab.nav or "")
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}); f.flush()
            print("done", lab.cell, row["outcome"], flush=True)
