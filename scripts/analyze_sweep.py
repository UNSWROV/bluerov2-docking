#!/usr/bin/env python3
"""Analyze a docking-trial sweep (#36) into the thesis figure + metrics table.

Reads every bag from a sweep tag, extracts per-run outcome (docked?, time-to-dock,
demotes, seat range), aggregates P(dock) per (regime, period), and plots the two
curves that are the result: method (regime=sway, CV filter + feedforward) vs baseline
(regime=static, feedforward off) as a function of sway period.

Usage (in the container, ROS + workspace sourced):
    python3 scripts/analyze_sweep.py [SWEEP_TAG] [--bags DIR] [--out PNG]
If SWEEP_TAG is omitted, the newest sweep_*_results.csv in the bags dir is used.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from interfaces.msg import DockingState, FineAlignStatus

from analysis.labels import parse_label
_SN = {0: "COARSE", 1: "FINE", 2: "DOCKED", 3: "IDLE"}


def read_run(bag):
    """Return per-run metrics dict, or None if the bag can't be read."""
    if not os.path.exists(os.path.join(bag, "metadata.yaml")):
        subprocess.run(["ros2", "bag", "reindex", bag, "-s", "mcap"],
                       capture_output=True)
    try:
        r = SequentialReader()
        r.open(StorageOptions(uri=bag, storage_id="mcap"),
               ConverterOptions("cdr", "cdr"))
    except Exception as e:
        print(f"  ! unreadable {os.path.basename(bag)}: {e}", file=sys.stderr)
        return None
    st, seat = [], []
    while r.has_next():
        tp, d, t = r.read_next()
        ts = t * 1e-9
        if tp == "/docking/state":
            st.append((ts, deserialize_message(d, DockingState).state))
        elif tp == "/control/fine_align/status":
            seat.append(deserialize_message(d, FineAlignStatus).range_to_dock_m)
    if not st:
        return {"docked": False, "t_dock": np.nan, "n_demote": 0, "min_range": np.nan}
    t0 = st[0][0]
    labels = []
    prev = None
    for ts, s in st:
        if int(s) != prev:
            labels.append((ts - t0, _SN.get(int(s), s)))
            prev = int(s)
    seq = [l for _, l in labels]
    docked = "DOCKED" in seq
    t_dock = next((t for t, l in labels if l == "DOCKED"), np.nan)
    n_demote = sum(1 for i in range(1, len(seq))
                   if seq[i] == "COARSE" and seq[i - 1] == "FINE")
    return {"docked": docked, "t_dock": t_dock, "n_demote": n_demote,
            "min_range": (min(seat) if seat else np.nan)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--bags", default=os.path.expanduser(
        os.environ.get("WS", "/home/ubuntu/ws_docking") + "/bags"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.tag
    if tag is None:
        csvs = sorted(glob.glob(os.path.join(args.bags, "sweep_*_results.csv")))
        if not csvs:
            sys.exit("no sweep_*_results.csv found; pass a SWEEP_TAG")
        tag = os.path.basename(csvs[-1])[: -len("_results.csv")]
    print(f"sweep tag: {tag}")

    runs = []
    for bag in sorted(glob.glob(os.path.join(args.bags, f"{tag}_*"))):
        if not os.path.isdir(bag):
            continue
        lab = parse_label(bag)
        if lab is None:
            continue
        met = read_run(bag)
        if met is None:
            continue
        runs.append(dict(regime=lab.arm, dock=lab.dock, period=lab.period,
                         phase=lab.phase, **met))

    if not runs:
        sys.exit(f"no readable runs for tag {tag}")

    # per-run table
    print("\n regime  dock    period  phase   docked  t_dock  demotes  seat_range")
    for r in sorted(runs, key=lambda r: (r["regime"], r["dock"], -r["period"], r["phase"])):
        print(f" {r['regime']:6} {r['dock']:6}  {r['period']:5.0f}  {r['phase']:5.2f}   "
              f"{'YES ' if r['docked'] else 'no  '}  "
              f"{r['t_dock'] if not np.isnan(r['t_dock']) else 0:5.0f}s   "
              f"{r['n_demote']:5d}    {r['min_range'] if not np.isnan(r['min_range']) else -1:6.3f}")

    # aggregate P(dock) per (regime, period) over phases, moving-dock cells only
    print("\n P(dock) per (regime, period)  [moving dock]:")
    periods = sorted({r["period"] for r in runs if r["dock"] == "sway"}, reverse=True)
    curves = {}
    for regime in sorted({r["regime"] for r in runs}):
        xs, ps, ns = [], [], []
        for per in periods:
            cell = [r for r in runs if r["regime"] == regime
                    and r["dock"] == "sway" and r["period"] == per]
            if not cell:
                continue
            pd = np.mean([r["docked"] for r in cell])
            xs.append(per); ps.append(pd); ns.append(len(cell))
            print(f"   {regime:6} T={per:4.0f}s  P(dock)={pd:.2f}  (n={len(cell)})")
        curves[regime] = (xs, ps, ns)
    # static-dock sanity
    for regime in sorted({r["regime"] for r in runs}):
        sd = [r for r in runs if r["regime"] == regime and r["dock"] == "static"]
        if sd:
            print(f"   {regime:6} static-dock  P(dock)={np.mean([r['docked'] for r in sd]):.2f} (n={len(sd)})")

    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {"sway": dict(color="C0", marker="o", ls="-", label="method (CV filter + feedforward)"),
             "static": dict(color="C3", marker="s", ls="--", label="baseline (feedforward off)")}
    for regime, (xs, ps, ns) in curves.items():
        if not xs:
            continue
        s = style.get(regime, dict(marker="o", ls="-", label=regime))
        ax.plot(xs, ps, **s)
        for x, p, n in zip(xs, ps, ns):
            ax.annotate(f"{int(round(p*n))}/{n}", (x, p), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("dock sway period [s]  (shorter = faster, harder)")
    ax.set_ylabel("P(dock)")
    ax.set_ylim(-0.05, 1.1)
    ax.invert_xaxis()  # easy (long period) on the left, hard on the right
    ax.set_title(f"Moving-dock docking success vs sway period\n{tag}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    out = args.out or os.path.join(args.bags, f"{tag}_pdock.png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
