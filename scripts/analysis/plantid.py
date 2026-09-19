"""Preliminary plant identification from the recorded closed-loop trials.

For every bag, every stretch of a mission phase with measurements and at least
two dock periods is a window; the loop analysis fits the vehicle velocity response
to the lateral command at the dock frequency. COARSE runs in ALT_HOLD and FINE in
STABILIZE, so the table separates the two flight modes. Closed-loop fits are
biased by the feedback path; T4 step tests replace them. This gives the priors.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import numpy as np

from . import loop
from .labels import parse_label
from .tracks import load_trial, measured_mask

MODE = {0: "COARSE (ALT_HOLD)", 1: "FINE (STABILIZE)"}


def windows(trial, min_periods=2.0):
    """(state, t_start, t_end) relative to t0 for measured stretches of one phase."""
    lab = parse_label(trial.path.rsplit("/", 1)[0]) or parse_label(trial.path)
    period = lab.period if lab and lab.period > 0 else 8.0
    st = trial.states + [(trial.odom.t[-1], -1)]
    out = []
    for (t_a, s), (t_b, _) in zip(st[:-1], st[1:]):
        if s not in MODE:
            continue
        g = np.arange(t_a, t_b, 0.1)
        if len(g) < 10:
            continue
        have = measured_mask(trial, g, 0.5)
        # split into measured runs
        edges = np.flatnonzero(np.diff(np.r_[0, have.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if (g[min(b, len(g) - 1)] - g[a]) >= min_periods * period:
                out.append((s, g[a] - trial.t0, g[min(b, len(g) - 1)] - trial.t0, period))
    return out


def run(bag_dir: str, out_csv: str):
    rows = []
    for bag in sorted(glob.glob(os.path.join(bag_dir, "*_sway_p*"))):
        lab = parse_label(bag)
        mcaps = glob.glob(os.path.join(bag, "*.mcap")) if os.path.isdir(bag) else []
        if lab is None or lab.dock != "sway" or not mcaps:
            continue
        try:
            tr = load_trial(mcaps[0])
            for s, a, b, period in windows(tr):
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    r = loop.report(tr, a, b)
                rows.append(dict(cell=lab.cell, arm=lab.arm, period=lab.period, mode=MODE[s], t_start=round(a, 1),
                                 t_end=round(b, 1), **{k: round(float(v), 3) for k, v in r.items()}))
                print("window", lab.cell, MODE[s], f"{a:.0f}..{b:.0f}", f"gain {r['plant_gain']:.2f}/{r['plant_gain_broadband']:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(bag), e)
    if not rows:
        print("no windows"); return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} windows -> {out_csv}\n")
    by = defaultdict(list)
    for r in rows:
        if r["cmd_amp"] > 0.01 and np.isfinite(r["plant_gain"]):
            by[(r["mode"], r["arm"])].append(r)
    print(f"{'mode':20s} {'arm':8s} {'n':>3s} {'gain@f med':>11s} {'IQR':>12s} {'broadband med':>14s} {'lag med [s]':>12s} {'lag IQR':>12s}")
    for (mode, arm), rs in sorted(by.items()):
        g = np.array([r["plant_gain"] for r in rs]); gb = np.array([r["plant_gain_broadband"] for r in rs]); lag = np.array([r["plant_lag_s"] for r in rs])
        q = lambda x: f"{np.percentile(x, 25):.2f} to {np.percentile(x, 75):.2f}"
        print(f"{mode:20s} {arm:8s} {len(rs):3d} {np.median(g):11.2f} {q(g):>12s} {np.median(gb):14.2f} {np.median(lag):12.2f} {q(lag):>12s}")
