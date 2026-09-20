"""Leak diagnosis: where does vehicle motion enter the world-frame dock measurement?

Rebuilds the measurement the filter ingested (camera pose from /tf at the image
stamp applied to the camera-frame measurement), subtracts the true dock position
and splits the error into a camera-frame part (same measurement through the
ground-truth vehicle pose) and a TF-path part (the remainder). Correlates each
with the vehicle state, checks marker-subset bias, and scans a time shift on the
vehicle pose to test for a stamp offset.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .tracks import Trial, corr, measurement_error, world_from_cam


def report(trial: Trial, axis: int = 1, plot: str | None = None) -> dict:
    t, a = trial.t_meas, "xyz"[axis]
    e_world, e_cam, e_tf, off = measurement_error(trial)
    p_dock, veh, vveh = trial.dock.pos(t), trial.odom.pos(t), trial.odom.vel(t)
    rng = np.linalg.norm(trial.p_cam, axis=1)
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    out = {
        "n": len(t), "span_s": float(t[-1] - t[0]), "offset": off.tolist(),
        "rms_world_cm": rms(e_world[:, axis]) * 100, "rms_cam_cm": rms(e_cam[:, axis]) * 100,
        "rms_tf_cm": rms(e_tf[:, axis]) * 100,
    }
    print(f"bag: {trial.path.split('/')[-1]}")
    print(f"measurements {len(t)}, span {out['span_s']:.1f} s, median range {np.median(rng):.2f} m, "
          f"model-origin offset {np.round(off, 3)}")
    print(f"[{a}] RMS error: world {out['rms_world_cm']:.1f} cm, camera-frame part {out['rms_cam_cm']:.1f} cm, "
          f"TF-path part {out['rms_tf_cm']:.1f} cm")
    print(f"[{a}] correlations              world-err  cam-part  tf-part")
    for name, x in [("dock position", p_dock[:, axis]), ("vehicle position", veh[:, axis]),
                    ("vehicle velocity", vveh[:, axis]), ("range", rng), ("num_markers", trial.n_markers.astype(float))]:
        c = (corr(e_world[:, axis], x), corr(e_cam[:, axis], x), corr(e_tf[:, axis], x))
        out[f"corr_{name.replace(' ', '_')}"] = c
        print(f"  {name:22s} {c[0]:+8.2f} {c[1]:+9.2f} {c[2]:+9.2f}")
    print(f"[{a}] camera-frame error by marker subset (mean, sd, n):")
    by = defaultdict(list)
    for k, e in zip(trial.marker_ids, e_cam[:, axis]):
        by[k].append(e)
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"  {str(k):44s} {np.mean(v)*100:+6.1f} cm  sd {np.std(v)*100:4.1f}  n={len(v)}")
    print(f"[{a}] camera-frame RMS error vs vehicle-pose time shift:")
    scan = {}
    for sh in [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]:
        e = world_from_cam(trial.p_cam, trial.odom, t, shift=sh) - p_dock
        e = e - np.median(e, axis=0)
        scan[sh] = rms(e[:, axis]) * 100
        print(f"  shift {sh:+.2f} s: {scan[sh]:.2f} cm")
    out["shift_scan_cm"] = scan
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        t0 = t[0]
        axs[0].plot(t - t0, (p_dock[:, axis] - np.median(p_dock[:, axis])) * 100, label="dock (GT)")
        axs[0].plot(t - t0, (veh[:, axis] - np.median(veh[:, axis])) * 100, label="vehicle (GT)")
        axs[0].set_ylabel(f"{a} [cm]"); axs[0].legend(loc="upper right")
        axs[1].plot(t - t0, e_world[:, axis] * 100, ".", ms=2, label="world-frame measurement error")
        axs[1].plot(t - t0, e_cam[:, axis] * 100, lw=0.8, label="camera-frame part")
        axs[1].plot(t - t0, e_tf[:, axis] * 100, lw=0.8, label="TF-path part")
        axs[1].set_ylabel("error [cm]"); axs[1].legend(loc="upper right")
        axs[2].plot(t - t0, trial.n_markers, ".", ms=2); axs[2].set_ylabel("markers fused"); axs[2].set_xlabel("sim time [s]")
        fig.suptitle(trial.path.split("/")[-1]); fig.tight_layout(); fig.savefig(plot, dpi=130)
        print("plot:", plot)
    return out
