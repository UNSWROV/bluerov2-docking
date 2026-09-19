"""Paper figures: mechanism (one trial), P(dock) versus period (four arms), trajectory."""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .labels import ARM_NAMES
from .tracks import Trial, measured_mask, measurement_error


def mechanism(trial: Trial, out: str, axis: int = 1, tmax: float | None = None, compact: bool = False):
    t0 = trial.t0
    t = trial.t_meas
    g = np.arange(t[0], (t0 + tmax) if tmax else t[-1], 0.05)
    _, e_cam, _, off = measurement_error(trial)
    dock, veh = trial.dock.pos(g)[:, axis], trial.odom.pos(g)[:, axis]
    est = trial.filt.pos(g)[:, axis] - off[axis]
    dv, vv = trial.dock.vel(g)[:, axis], trial.odom.vel(g)[:, axis]
    have = measured_mask(trial, g, 0.5)
    T = lambda x: x - t0
    if compact:
        plt.rcParams.update({"font.size": 7})
        fig, ax = plt.subplots(3, 1, figsize=(3.4, 3.3), sharex=True); fs = 5.5
    else:
        fig, ax = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True); fs = 8
    for a in ax:
        a.fill_between(T(g), 0, 1, where=~have, transform=a.get_xaxis_transform(), color="0.88", lw=0)
    ax[0].plot(T(g), dock * 100, lw=0.9, label="dock, truth"); ax[0].plot(T(g), est * 100, lw=0.9, label="estimate")
    ax[0].plot(T(g), veh * 100, lw=0.9, label="vehicle, truth"); ax[0].set_ylabel("lateral [cm]")
    ax[0].legend(loc="upper left", fontsize=fs, ncol=3, frameon=False)
    i = 1
    if not compact:
        ax[1].plot(T(t), e_cam[:, axis] * 100, ".", ms=2); ax[1].set_ylabel("measurement minus\ntruth [cm]"); ax[1].set_ylim(-5, 5); i = 2
    ax[i].plot(T(g), dv * 100, lw=0.9, label="dock, truth"); ax[i].plot(T(trial.t_vel), trial.vel[:, axis] * 100, lw=0.9, label="velocity state")
    ax[i].plot(T(g), vv * 100, lw=0.7, alpha=0.7, label="vehicle"); ax[i].set_ylabel("lat. vel. [cm/s]")
    ax[i].legend(loc="upper left", fontsize=fs, ncol=3, frameon=False)
    ax[i + 1].step(T(trial.t_health), trial.health, where="post", lw=0.9, label="health (1 ok, 3 stale)")
    ax[i + 1].plot(T(trial.t_cmd), trial.cmd[:, axis] * 10, lw=0.6, alpha=0.7, label="lateral cmd x10")
    ax[i + 1].set_ylabel("health / cmd"); ax[i + 1].set_xlabel("time [s]"); ax[i + 1].legend(loc="upper left", fontsize=fs, ncol=2, frameon=False)
    if tmax:
        ax[-1].set_xlim(0, tmax)
    if not compact:
        fig.suptitle(trial.path.split("/")[-1])
    fig.tight_layout(pad=0.3); fig.savefig(out, dpi=220 if compact else 130); print(out)


def pdock(results_csv: str, out: str, clean_col: str | None = "clean"):
    """P(dock) versus period per arm from a results CSV with columns arm, period, outcome[, clean]."""
    cells = defaultdict(list)
    with open(results_csv) as f:
        for r in csv.DictReader(f):
            if r.get("dock", "sway") != "sway":
                continue
            docked = r["outcome"].upper() == "DOCKED"
            clean = docked and (r.get(clean_col, "1") in ("1", "True", "true", "")) if clean_col else docked
            cells[(r["arm"], float(r["period"]))].append((docked, clean))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for arm in sorted({k[0] for k in cells}):
        per = sorted({k[1] for k in cells if k[0] == arm}, reverse=True)
        raw = [np.mean([d for d, _ in cells[(arm, p)]]) for p in per]
        cl = [np.mean([c for _, c in cells[(arm, p)]]) for p in per]
        line, = ax.plot(per, cl, "o-", label=ARM_NAMES.get(arm, arm))
        ax.plot(per, raw, "o:", color=line.get_color(), alpha=0.6)
    ax.set_xlabel("dock sway period [s]"); ax.set_ylabel("P(dock)"); ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis(); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=200); print(out)


def trajectory(trial: Trial, out: str):
    """Top view of the vehicle path coloured by phase, and range to the dock over time."""
    from .windows import STATE_NAMES
    st = trial.states
    g = np.arange(trial.odom.t[0], trial.odom.t[-1], 0.1)
    veh, dock = trial.odom.pos(g), trial.dock.pos(g)
    rng = np.linalg.norm(veh - dock, axis=1)
    st_t = np.array([s for s, _ in st]); st_v = np.array([v for _, v in st])
    phase = np.array([st_v[np.searchsorted(st_t, x, side="right") - 1] if x >= st_t[0] else 3 for x in g])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for s, name in STATE_NAMES.items():
        m = phase == s
        if m.any():
            ax[0].plot(veh[m, 0], veh[m, 1], ".", ms=2, label=name)
    ax[0].plot(dock[:, 0], dock[:, 1], "k-", lw=0.5, label="dock (GT)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]"); ax[0].axis("equal"); ax[0].legend(fontsize=7)
    ax[1].semilogy(g - g[0], rng); ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("range to dock [m]"); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(trial.path.split("/")[-1], fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=150); print(out)
