"""One row per sweep bag: measurement accuracy, tracking accuracy, over-swing, blackout, outcome."""
from __future__ import annotations

import csv
import glob
import os

import numpy as np

from .labels import parse_label
from .tracks import corr, load_trial, measured_mask, measurement_error

COLUMNS = ["cell", "arm", "dock", "period", "phase", "nav", "outcome", "n_meas", "meas_err_rms_cm",
           "track_err_rms_cm", "vel_ratio", "vel_corr", "veh_over_dock", "gap_frac", "max_est_err_cm", "n_gaps_gt1s"]


def read_outcomes(bag_dir: str) -> dict:
    """cell -> outcome from every *_results.csv the sweep runner wrote."""
    out = {}
    for path in glob.glob(os.path.join(bag_dir, "*_results.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                if "label" in row and "outcome" in row:
                    out[row["label"]] = row["outcome"]
    return out


def summarize_bag(bag: str, axis: int = 1) -> dict | None:
    lab = parse_label(bag)
    if lab is None or lab.dock != "sway":
        return None
    mcaps = glob.glob(os.path.join(bag, "*.mcap"))
    if not mcaps:
        return None
    tr = load_trial(mcaps[0])
    t = tr.t_meas
    e_world, _, _, _ = measurement_error(tr)
    g = np.arange(t[0], t[-1], 0.1)
    have = measured_mask(tr, g)
    est = tr.filt.pos(g)[:, axis] - tr.dock.pos(g)[:, axis]; est -= np.median(est[have])
    vh = measured_mask(tr, tr.t_vel)
    gv = tr.dock.vel(tr.t_vel)[:, axis]
    veh, dock = tr.odom.pos(g)[:, axis], tr.dock.pos(g)[:, axis]
    gaps = np.diff(t)
    return dict(
        cell=lab.cell, arm=lab.arm, dock=lab.dock, period=lab.period, phase=lab.phase, nav=lab.nav or "",
        n_meas=len(t), meas_err_rms_cm=round(float(np.sqrt(np.mean(e_world[:, axis] ** 2))) * 100, 2),
        track_err_rms_cm=round(float(np.sqrt(np.mean(est[have] ** 2))) * 100, 2),
        vel_ratio=round(float(np.std(tr.vel[vh, axis]) / (np.std(gv[vh]) + 1e-9)), 2),
        vel_corr=round(corr(tr.vel[vh, axis], gv[vh]), 2),
        veh_over_dock=round(float(np.std(veh[have] - np.median(veh[have])) / np.std(dock[have])), 2),
        gap_frac=round(float(1 - have.mean()), 2), max_est_err_cm=round(float(np.max(np.abs(est))) * 100, 1),
        n_gaps_gt1s=int(np.sum(gaps > 1)),
    )


def run(bag_dir: str, out_csv: str, axis: int = 1):
    outcomes = read_outcomes(bag_dir)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for bag in sorted(glob.glob(os.path.join(bag_dir, "*"))):
            if not os.path.isdir(bag):
                continue
            try:
                row = summarize_bag(bag, axis)
            except Exception as e:  # noqa: BLE001
                print("skip", os.path.basename(bag), e)
                continue
            if row is None:
                continue
            row["outcome"] = outcomes.get(row["cell"], "?")
            w.writerow(row); f.flush()
            print("done", row["cell"], row["outcome"], flush=True)
