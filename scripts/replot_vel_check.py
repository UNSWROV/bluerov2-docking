"""Regenerate the velocity-state check figure (report fig:vel-check) from a bag.

Reads GT dock pose, the filtered dock pose, the filter velocity state and
vehicle odometry (sim-time header stamps throughout), computes amplitude
ratio + correlation between estimated and true dock velocity, and saves a
clean 300 dpi figure.
"""
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcap_ros2.reader import read_ros2_messages

bag_dir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
mcap_file = glob.glob(f"{bag_dir}/*.mcap")[0]

gt_t, gt_y, gt_z = [], [], []
fp_t, fp_y = [], []
est_t, est_vy, est_vz = [], [], []
for m in read_ros2_messages(mcap_file, topics=[
        "/dock/ground_truth/pose",
        "/perception/dock_pose_filtered",
        "/perception/dock_pose_filtered/velocity"]):
    msg = m.ros_msg
    t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
    if m.channel.topic == "/dock/ground_truth/pose":
        gt_t.append(t); gt_y.append(msg.pose.position.y); gt_z.append(msg.pose.position.z)
    elif m.channel.topic == "/perception/dock_pose_filtered":
        fp_t.append(t); fp_y.append(msg.pose.pose.position.y)
    else:
        est_t.append(t); est_vy.append(msg.twist.twist.linear.y); est_vz.append(msg.twist.twist.linear.z)

gt_t, gt_y, gt_z = map(np.asarray, (gt_t, gt_y, gt_z))
fp_t, fp_y = map(np.asarray, (fp_t, fp_y))
est_t, est_vy, est_vz = map(np.asarray, (est_t, est_vy, est_vz))
t0 = gt_t[0]
gt_t -= t0; fp_t -= t0; est_t -= t0

tg = np.arange(gt_t[0], gt_t[-1], 0.05)
y_u = np.interp(tg, gt_t, gt_y - gt_y.mean())
z_u = np.interp(tg, gt_t, gt_z - gt_z.mean())
vy_true = np.gradient(y_u, tg)
vz_true = np.gradient(z_u, tg)
vy_est = np.interp(tg, est_t, est_vy)
vz_est = np.interp(tg, est_t, est_vz)
mask = tg > 5.0

def metrics(a, b):
    return np.std(a[mask]) / np.std(b[mask]), np.corrcoef(a[mask], b[mask])[0, 1]

ry, cy = metrics(vy_est, vy_true)
rz, cz = metrics(vz_est, vz_true)
print(f"{bag_dir.rstrip('/').split('/')[-1]}: "
      f"vy ratio={ry:.2f} corr={cy:.2f}  vz ratio={rz:.2f} corr={cz:.2f}  dur={tg[-1]:.0f}s")

if out:
    fig, axes = plt.subplots(3, 1, figsize=(8.25, 7.3), sharex=True)
    axes[0].plot(tg, y_u, color="C0", label="ground truth")
    axes[0].plot(fp_t, fp_y - gt_y.mean(), color="C1", label="filtered estimate")
    axes[0].set_ylabel("dock sway y [m]")
    axes[1].plot(tg, vy_true, color="C0", label="true $v_y$ (ground-truth derivative)")
    axes[1].plot(est_t, est_vy, color="C1", label="estimated $v_y$ (filter velocity state)")
    axes[1].set_ylabel("$v_y$ [m/s]")
    axes[2].plot(tg, vz_true, color="C0", label="true $v_z$")
    axes[2].plot(est_t, est_vz, color="C1", label="estimated $v_z$")
    axes[2].set_ylabel("$v_z$ [m/s]")
    axes[2].set_xlabel("t [s]")
    for ax in axes[1:]:
        ax.axhline(0, color="k", lw=0.5)
    for ax in axes:
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"figure -> {out}")
