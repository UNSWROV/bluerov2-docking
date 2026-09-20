"""Offline analysis of docking trial bags, no ROS required.

Reads the mcap bags written by scripts/record_docking_trial.sh (including the
truncated tails the trial runner leaves behind), rebuilds what the perception
and control nodes saw, and produces the sweep tables and figures for the moving
dock experiments. Run from the scripts directory:

    python3 -m analysis sweep BAG_DIR
    python3 -m analysis leak BAG [--plot out.png]
    python3 -m analysis windows BAG
    python3 -m analysis loop BAG T_START T_END
    python3 -m analysis figure mechanism BAG OUT.png
    python3 -m analysis figure pdock RESULTS.csv OUT.png
    python3 -m analysis figure trajectory BAG OUT.png
"""
