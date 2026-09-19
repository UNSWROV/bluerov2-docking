"""Windowed comparison of the filter against its own measurements and the truth."""
from __future__ import annotations

import numpy as np

from .tracks import Trial, corr, world_from_cam

STATE_NAMES = {0: "COARSE", 1: "FINE", 2: "DOCKED", 3: "IDLE"}


def report(trial: Trial, axis: int = 1, win: float = 5.0):
    t = trial.t_meas
    z = world_from_cam(trial.p_cam, trial.tf, t)
    off = np.median(z - trial.dock.pos(t), axis=0)
    gaps = np.diff(t)
    print(f"bag: {trial.path.split('/')[-1]}  RTF {trial.rtf:.3f}")
    print(f"gaps >0.3 s: {np.sum(gaps > 0.3)}, >1 s: {np.sum(gaps > 1)}, longest {gaps.max():.1f} s, "
          f"time in gaps>0.3: {gaps[gaps > 0.3].sum():.0f} of {t[-1]-t[0]:.0f} s")
    res = trial.filt.pos(t)[:, axis] - z[:, axis]
    st_t = np.array([s for s, _ in trial.states]); st_v = np.array([v for _, v in trial.states])
    print(" win[s]  n_meas  |filt-meas|RMS  filt-true RMS  veh-dock RMS  vel ratio  vel corr  state")
    rows = []
    for a in np.arange(t[0], t[-1], win):
        b = a + win
        m = (t >= a) & (t < b); g = np.linspace(a, b, 60)
        e = trial.filt.pos(g)[:, axis] - trial.dock.pos(g)[:, axis] - off[axis]
        v = trial.odom.pos(g)[:, axis] - trial.dock.pos(g)[:, axis]
        r = np.sqrt(np.mean(res[m] ** 2)) * 100 if m.any() else float("nan")
        vm = (trial.t_vel >= a) & (trial.t_vel < b)
        gv = trial.dock.vel(trial.t_vel[vm])[:, axis] if vm.any() else np.array([])
        ratio = np.std(trial.vel[vm, axis]) / (np.std(gv) + 1e-9) if vm.any() else float("nan")
        vc = corr(trial.vel[vm, axis], gv) if vm.sum() > 5 else float("nan")
        s = STATE_NAMES.get(int(st_v[np.searchsorted(st_t, a, side="right") - 1]), "?") if len(st_t) and a >= st_t[0] else "-"
        rows.append((a - t[0], int(m.sum()), r, np.sqrt(np.mean(e ** 2)) * 100, np.sqrt(np.mean((v - np.median(v)) ** 2)) * 100, ratio, vc, s))
        print(f"{rows[-1][0]:6.0f}  {rows[-1][1]:5d}   {r:8.1f} cm   {rows[-1][3]:8.1f} cm  {rows[-1][4]:8.1f} cm  {ratio:7.2f}  {vc:+6.2f}   {s}")
    return rows
