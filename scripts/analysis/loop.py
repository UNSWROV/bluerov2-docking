"""Loop analysis inside a tracking window: vehicle over-swing, command make-up, plant gain and lag."""
from __future__ import annotations

import numpy as np

from .tracks import Trial, corr


def _fit(g, y, w):
    A = np.c_[np.sin(w * g), np.cos(w * g)]
    c, *_ = np.linalg.lstsq(A, y - y.mean(), rcond=None)
    return float(np.hypot(*c)), float(np.degrees(np.arctan2(c[1], c[0])))


def report(trial: Trial, a: float, b: float, axis: int = 1, ff_gain: float = 0.6):
    """Window [a, b] in sim time relative to the first measurement."""
    t0 = trial.t0
    g = np.arange(t0 + a, t0 + b, 0.02)
    dock = trial.dock.pos(g)[:, axis]; veh = trial.odom.pos(g)[:, axis]
    dockc = dock - dock.mean()
    spec = np.abs(np.fft.rfft(dockc)); k = np.argmax(spec[1:]) + 1
    period = (b - a) / k; w = 2 * np.pi / period
    amp_d, ph_d = _fit(g, dock, w); amp_v, ph_v = _fit(g, veh, w)
    dv = trial.dock.vel(g)[:, axis]; ev = np.interp(g, trial.t_vel, trial.vel[:, axis])
    vv = trial.odom.vel(g)[:, axis]
    cy = np.interp(g, trial.t_cmd, trial.cmd[:, axis]) if len(trial.t_cmd) else np.zeros_like(g)
    amp_c, ph_c = _fit(g, cy, w); amp_vv, ph_vv = _fit(g, vv, w)
    lag_deg = (ph_c - ph_vv) % 360
    out = dict(period=period, dock_amp=amp_d, veh_amp=amp_v, ratio=amp_v / amp_d,
               veh_phase_deg=(ph_d - ph_v) % 360, dock_vel_amp=np.std(dv) * np.sqrt(2),
               est_vel_amp=np.std(ev) * np.sqrt(2), est_vel_corr=corr(dv, ev),
               veh_vel_amp=np.std(vv) * np.sqrt(2), cmd_amp=amp_c, ff_share=ff_gain * np.std(ev) * np.sqrt(2),
               plant_gain=amp_vv / amp_c if amp_c > 0 else float("nan"),
               plant_gain_broadband=float(np.std(vv) / np.std(cy)) if np.std(cy) > 0 else float("nan"),
               plant_lag_s=lag_deg / 360 * period)
    print(f"window {a:.0f}..{b:.0f} s, dock period fit {period:.2f} s")
    print(f"dock amp {amp_d*100:.1f} cm, vehicle amp {amp_v*100:.1f} cm, ratio {out['ratio']:.2f}, "
          f"vehicle lags dock by {out['veh_phase_deg']:.0f} deg")
    print(f"velocity amps: dock {out['dock_vel_amp']*100:.1f} cm/s, estimate {out['est_vel_amp']*100:.1f} cm/s "
          f"(corr {out['est_vel_corr']:+.2f}), vehicle {out['veh_vel_amp']*100:.1f} cm/s")
    print(f"lateral command amp {amp_c:.3f} effort; feedforward share at gain {ff_gain} would be {out['ff_share']:.3f}")
    # Two gain estimates: the sinusoid fit at the dock frequency isolates the
    # response to the feedforward component, the broadband std ratio folds in the
    # feedback jitter; the closed loop makes both approximate. T4 step tests replace them.
    print(f"plant: gain at dock frequency {out['plant_gain']:.2f}, broadband std ratio {out['plant_gain_broadband']:.2f} "
          f"m/s per effort; lag {lag_deg:.0f} deg = {out['plant_lag_s']:.2f} s")
    return out
