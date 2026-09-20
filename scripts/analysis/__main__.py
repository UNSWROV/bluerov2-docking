"""Command-line entry: python3 -m analysis <command> ..."""
from __future__ import annotations

import argparse
import glob
import os

from . import figures, leak, loop, metrics, navreplay, replay, sweep, windows
from . import figures, leak, loop, metrics, plantid, replay, sweep, windows
from .tracks import load_trial


def _bag(path: str) -> str:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.mcap")))[0]
    return path


def main():
    ap = argparse.ArgumentParser(prog="analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep"); p.add_argument("bag_dir"); p.add_argument("--out", default="sweep_summary.csv")
    p = sub.add_parser("metrics"); p.add_argument("bag_dir"); p.add_argument("--out", default="metrics.csv")
    p = sub.add_parser("replay"); p.add_argument("bag"); p.add_argument("--plot"); p.add_argument("--hold", type=float, default=1.0); p.add_argument("--decay", type=float, default=1.0); p.add_argument("--regime", choices=["static", "sway"])
    p = sub.add_parser("navreplay"); p.add_argument("bags", nargs="+"); p.add_argument("--out", default="navreplay.csv"); p.add_argument("--plot"); p.add_argument("--seeds", type=int, default=1)
    p = sub.add_parser("plantid"); p.add_argument("bag_dir"); p.add_argument("--out", default="plantid.csv")
    p = sub.add_parser("leak"); p.add_argument("bag"); p.add_argument("--plot"); p.add_argument("--axis", default="y")
    p = sub.add_parser("windows"); p.add_argument("bag"); p.add_argument("--axis", default="y")
    p = sub.add_parser("loop"); p.add_argument("bag"); p.add_argument("t_start", type=float); p.add_argument("t_end", type=float)
    p = sub.add_parser("figure"); p.add_argument("kind", choices=["mechanism", "pdock", "trajectory"])
    p.add_argument("src"); p.add_argument("out"); p.add_argument("--tmax", type=float); p.add_argument("--compact", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep.run(a.bag_dir, a.out)
    elif a.cmd == "metrics":
        metrics.run(a.bag_dir, a.out)
    elif a.cmd == "navreplay":
        navreplay.run(a.bags, a.out, seeds=tuple(range(a.seeds)), plot=a.plot)
    elif a.cmd == "plantid":
        plantid.run(a.bag_dir, a.out)
    elif a.cmd == "replay":
        replay.report(load_trial(_bag(a.bag)), plot=a.plot, hold_s=a.hold, decay_s=a.decay, regime=a.regime)
    elif a.cmd == "leak":
        leak.report(load_trial(_bag(a.bag)), "xyz".index(a.axis), a.plot)
    elif a.cmd == "windows":
        windows.report(load_trial(_bag(a.bag)), "xyz".index(a.axis))
    elif a.cmd == "loop":
        loop.report(load_trial(_bag(a.bag)), a.t_start, a.t_end)
    elif a.cmd == "figure":
        if a.kind == "pdock":
            figures.pdock(a.src, a.out)
        elif a.kind == "mechanism":
            figures.mechanism(load_trial(_bag(a.src)), a.out, tmax=a.tmax, compact=a.compact)
        else:
            figures.trajectory(load_trial(_bag(a.src)), a.out)


if __name__ == "__main__":
    main()
