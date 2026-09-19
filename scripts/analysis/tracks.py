"""Ground truth, TF and estimate tracks rebuilt from a trial bag."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .bagio import quat_xyzw, read_topics, stamp, vec3

# Static chain from /tf_static: base_link -> camera_mount_link -> camera_link.
T_MOUNT = np.array([0.21, 0.0, 0.067])
R_CAM = R.from_quat([-0.4999999896293064, 0.5001018366018476, -0.4999999896293064, 0.4998981633981524])

TOPICS = {
    "meas": "/perception/dock_pose_measured",
    "odom": "/model/bluerov2_heavy/odometry",
    "gt": "/dock/ground_truth/pose",
    "tf": "/tf",
    "filt": "/perception/dock_pose_filtered",
    "vel": "/perception/dock_pose_filtered/velocity",
    "health": "/perception/dock_pose_filtered/health",
    "cmd": "/cmd_vel",
    "state": "/docking/state",
}


class PoseTrack:
    """Piecewise-linear position and slerp attitude over sim time, as tf2 interpolates."""

    def __init__(self, t, p, q):
        order = np.argsort(t)
        t, p, q = np.asarray(t, float)[order], np.asarray(p, float)[order], np.asarray(q, float)[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        self.t, self.p = t[keep], p[keep]
        self.rot = Slerp(self.t, R.from_quat(q[keep]))

    def pos(self, t):
        t = np.clip(t, self.t[0], self.t[-1])
        return np.stack([np.interp(t, self.t, self.p[:, i]) for i in range(3)], axis=-1)

    def att(self, t):
        return self.rot(np.clip(t, self.t[0], self.t[-1]))

    def vel(self, t, dt=0.05):
        return (self.pos(t + dt) - self.pos(t - dt)) / (2 * dt)


@dataclass
class Trial:
    """Everything the analyses need from one bag, in sim time."""

    path: str
    t_meas: np.ndarray          # measurement stamps
    p_cam: np.ndarray           # fused dock position in camera_link
    q_cam: np.ndarray           # fused dock orientation in camera_link, xyzw
    cov: np.ndarray             # 6x6 measurement covariance per measurement
    n_markers: np.ndarray
    marker_ids: list
    odom: PoseTrack             # ground-truth vehicle pose
    tf: PoseTrack               # map -> base_link as relayed on /tf
    dock: PoseTrack             # ground-truth dock pose
    filt: PoseTrack | None      # filter estimate
    t_vel: np.ndarray
    vel: np.ndarray             # filter velocity state, world frame
    t_health: np.ndarray
    health: np.ndarray
    t_cmd: np.ndarray           # cmd_vel has no header: log time mapped to sim time
    cmd: np.ndarray
    states: list                # [(t, state_int)]
    rtf: float                  # sim seconds per wall second

    @property
    def t0(self) -> float:
        return float(self.t_meas[0])


def load_trial(path: str) -> Trial:
    d = read_topics(path, list(TOPICS.values()))
    meas = d[TOPICS["meas"]]
    odo = d[TOPICS["odom"]]
    if not meas or not odo:
        raise ValueError(f"{path}: no measurements or odometry")
    ot_s = np.array([stamp(m) for _, m in odo])
    ot_l = np.array([l for l, _ in odo])
    rtf, off = np.polyfit(ot_l, ot_s, 1)
    tfs = [(stamp(tr), tr) for _, m in d[TOPICS["tf"]] for tr in m.transforms
           if tr.header.frame_id == "map" and tr.child_frame_id == "base_link"]
    filt = d[TOPICS["filt"]]
    return Trial(
        path=path,
        t_meas=np.array([stamp(m) for _, m in meas]),
        p_cam=np.array([vec3(m.pose.pose.position) for _, m in meas]),
        q_cam=np.array([quat_xyzw(m.pose.pose.orientation) for _, m in meas]),
        cov=np.array([np.asarray(m.pose.covariance, float).reshape(6, 6) for _, m in meas]),
        n_markers=np.array([m.num_markers for _, m in meas]),
        marker_ids=[tuple(sorted(m.marker_ids)) for _, m in meas],
        odom=PoseTrack(ot_s, [vec3(m.pose.pose.position) for _, m in odo],
                       [quat_xyzw(m.pose.pose.orientation) for _, m in odo]),
        tf=PoseTrack([t for t, _ in tfs], [vec3(tr.transform.translation) for _, tr in tfs],
                     [quat_xyzw(tr.transform.rotation) for _, tr in tfs]),
        dock=PoseTrack([stamp(m) for _, m in d[TOPICS["gt"]]],
                       [vec3(m.pose.position) for _, m in d[TOPICS["gt"]]],
                       [quat_xyzw(m.pose.orientation) for _, m in d[TOPICS["gt"]]]),
        filt=PoseTrack([stamp(m) for _, m in filt], [vec3(m.pose.pose.position) for _, m in filt],
                       [quat_xyzw(m.pose.pose.orientation) for _, m in filt]) if filt else None,
        t_vel=np.array([stamp(m) for _, m in d[TOPICS["vel"]]]),
        vel=np.array([vec3(m.twist.twist.linear) for _, m in d[TOPICS["vel"]]]).reshape(-1, 3),
        t_health=np.array([stamp(m) for _, m in d[TOPICS["health"]]]),
        health=np.array([m.status for _, m in d[TOPICS["health"]]]),
        t_cmd=np.array([rtf * l + off for l, _ in d[TOPICS["cmd"]]]),
        cmd=np.array([[m.linear.x, m.linear.y, m.linear.z] for _, m in d[TOPICS["cmd"]]]).reshape(-1, 3),
        states=[(stamp(m), int(m.state)) for _, m in d[TOPICS["state"]]],
        rtf=float(rtf),
    )


def world_from_cam(p_cam, veh: PoseTrack, t, shift=0.0):
    """Camera-frame dock position to world through the vehicle pose at t (+ shift)."""
    p_base = T_MOUNT + R_CAM.apply(p_cam)
    return veh.pos(t + shift) + veh.att(t + shift).apply(p_base)


def world_pose_from_cam(p_cam, q_cam, veh: PoseTrack, t):
    """Camera-frame dock pose to world pose (position, quaternion xyzw) as tf2 does."""
    r_veh = veh.att(t)
    p = world_from_cam(p_cam, veh, t)
    q = (r_veh * R_CAM * R.from_quat(q_cam)).as_quat()
    return p, q


def measurement_error(trial: Trial):
    """World-frame measurement error split into camera-frame and TF-path parts.

    Returns (e_world, e_cam, e_tf, offset) with the constant model-origin offset
    removed from e_world and e_cam.
    """
    t = trial.t_meas
    z_tf = world_from_cam(trial.p_cam, trial.tf, t)
    z_gt = world_from_cam(trial.p_cam, trial.odom, t)
    p_dock = trial.dock.pos(t)
    e_cam = z_gt - p_dock
    off = np.median(e_cam, axis=0)
    return z_tf - p_dock - off, e_cam - off, z_tf - z_gt, off


def measured_mask(trial: Trial, t, window=0.3):
    """True where a measurement exists within `window` seconds of each t."""
    tm = trial.t_meas
    idx = np.searchsorted(tm, t)
    lo = np.abs(t - tm[np.clip(idx - 1, 0, len(tm) - 1)])
    hi = np.abs(tm[np.clip(idx, 0, len(tm) - 1)] - t)
    return np.minimum(lo, hi) < window


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
