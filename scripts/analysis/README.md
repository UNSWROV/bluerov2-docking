# Offline trial analysis

Pure Python (numpy, scipy, matplotlib, mcap, mcap-ros2-support), no ROS. Reads the
mcap bags from `scripts/record_docking_trial.sh`, including the truncated tails the
trial runner leaves when it kills the recorder, and rebuilds what the nodes saw.

Run from `scripts/`:

```bash
python3 -m analysis sweep BAG_DIR --out sweep_summary.csv   # one row per moving-dock bag
python3 -m analysis leak BAG [--plot out.png]               # measurement error decomposition
python3 -m analysis windows BAG                             # filter vs measurement vs truth, 5 s windows
python3 -m analysis loop BAG T_START T_END                  # over-swing, command make-up, plant gain and lag
python3 -m analysis figure mechanism BAG OUT.png [--tmax 130 --compact]
python3 -m analysis figure pdock RESULTS.csv OUT.png        # sweep-runner CSV (outcome DOCKED, clean) or metrics CSV
python3 -m analysis figure trajectory BAG OUT.png
python3 -m analysis metrics BAG_DIR --out metrics.csv          # per-trial paper metrics
python3 -m pytest analysis/tests
```

`BAG` is the bag directory or the mcap file inside it. Trial labels follow
`labels.py`; the 2026-07 sweeps use the filter regime names `static` and `sway` as
the arm, newer sweeps use `A`, `B`, `C`, `D` and an optional `_nav<level>`.

The leak decomposition, windows and loop outputs are the evidence behind
`bluerov2-docking-thesis/ut27/t1-leak-diagnosis.md`.
