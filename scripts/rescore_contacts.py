import glob
import os

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

from interfaces.msg import DockingState

CAM_X = 0.21
DY0, DZ0 = (
    0.002,
    0.176,
)  # entry-axis offset from dock origin, calibrated on the static-dock cell
APERTURE = 0.15
from analysis.labels import parse_label


def score(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    gt = []
    odo = []
    tlatch = None
    while r.has_next():
        tp, d, t = r.read_next()
        ts = t * 1e-9
        if tp == "/dock/ground_truth/pose":
            p = deserialize_message(d, PoseStamped).pose.position
            gt.append((ts, p.x, p.y, p.z))
        elif tp == "/model/bluerov2_heavy/odometry":
            p = deserialize_message(d, Odometry).pose.pose.position
            odo.append((ts, p.x, p.y, p.z))
        elif tp == "/docking/state" and tlatch is None:
            if deserialize_message(d, DockingState).state == 2:
                tlatch = ts
    gt = np.array(gt)
    odo = np.array(odo)
    dx = np.interp(odo[:, 0], gt[:, 0], gt[:, 1])
    dy = np.interp(odo[:, 0], gt[:, 0], gt[:, 2])
    dz = np.interp(odo[:, 0], gt[:, 0], gt[:, 3])
    gap = dx - (odo[:, 1] + CAM_X)
    off = np.sqrt((odo[:, 2] - dy - DY0) ** 2 + (odo[:, 3] - dz - DZ0) ** 2)
    inside = gap < 0
    prelatch = np.ones(len(odo), bool) if tlatch is None else (odo[:, 0] < tlatch)
    cm = inside & prelatch & (off > APERTURE)
    contact = bool(cm.any())
    c_off = c_v = np.nan
    if contact:
        i = int(np.argmax(cm))
        c_off = off[i]
        c_v = np.gradient(-gap, odo[:, 0])[i]
    ins_v = np.nan
    ins_off = np.nan
    if inside.any():
        j = int(np.argmax(inside))
        ins_v = np.gradient(-gap, odo[:, 0])[j]
        ins_off = off[j]
    docked = tlatch is not None
    pre_cross = bool((inside & prelatch).any())
    out = "CONTACT" if contact else ("CLEAN" if docked else "NO_DOCK")
    return out, docked, ins_off, ins_v, c_off, c_v, pre_cross


rows = []
BAGS = os.environ.get("BAGS", "/home/ubuntu/ws_docking/bags")
TAG = os.environ.get("SWEEP_TAG", "clamped")
for b in sorted(glob.glob(os.path.join(BAGS, TAG + "_*"))):
    if not os.path.isdir(b):
        continue
    lab = parse_label(b)
    if lab is None or lab.dock != "sway" and lab.period != 0:
        continue
    try:
        out, docked, io, iv, co, cv, pc = score(b)
    except Exception as e:
        print("!", os.path.basename(b), e)
        continue
    rows.append(
        (
            lab.arm,
            lab.dock,
            lab.period,
            lab.phase,
            out,
            docked,
            io,
            iv,
            co,
            cv,
            pc,
        )
    )

print(
    "regime dock   per  ph   | outcome  raw | insert_off insert_v | contact_off contact_v | precross"
)
for r in sorted(rows, key=lambda r: (r[0], r[1], -r[2], r[3])):
    print(
        " %-6s%-6s%4.0f %4.2f | %-8s %-5s| %6.3fm  %6.3f  | %s %s | %s"
        % (
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            str(r[5]),
            r[6],
            r[7],
            ("%.2fm" % r[8]) if not np.isnan(r[8]) else "  -  ",
            ("@%.2f" % r[9]) if not np.isnan(r[9]) else "     ",
            str(r[10]),
        )
    )
print("\n=== P(clean) / contacts, moving dock ===")
for regime in sorted({r[0] for r in rows}):
    for per in sorted({r[2] for r in rows if r[1] == "sway"}, reverse=True):
        cell = [r for r in rows if r[0] == regime and r[1] == "sway" and r[2] == per]
        if not cell:
            continue
        print(
            "  %-6s T=%4.0fs  P(clean)=%.2f  P(raw)=%.2f  contacts=%d/%d"
            % (
                regime,
                per,
                np.mean([r[4] == "CLEAN" for r in cell]),
                np.mean([r[5] for r in cell]),
                sum(r[4] == "CONTACT" for r in cell),
                len(cell),
            )
        )
