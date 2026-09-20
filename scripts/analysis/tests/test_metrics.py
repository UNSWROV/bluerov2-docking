import numpy as np

from analysis import metrics
from analysis.tracks import PoseTrack, Trial


def _const_track(t, p, q=(0, 0, 0, 1)):
    return PoseTrack(t, np.tile(p, (len(t), 1)), np.tile(q, (len(t), 1)))


def _trial(veh_xyz, states, dock_xyz=(5.0, 0.0, -1.5), t=None, health=None, vel=None, dock_track=None, meas_t=None):
    t = np.linspace(0, 20, 401) if t is None else t
    veh = PoseTrack(t, np.asarray(veh_xyz), np.tile([0, 0, 0, 1], (len(t), 1)))
    dock = dock_track or _const_track(t, dock_xyz)
    n = 40 if meas_t is None else len(meas_t)
    tm = np.linspace(t[0], t[-1], n) if meas_t is None else meas_t
    p_cam = np.zeros((len(tm), 3))
    p_cam[:, 2] = np.linalg.norm(veh.pos(tm) - dock.pos(tm), axis=1)   # optical axis points at the dock, roughly
    return Trial(path="synthetic", t_meas=tm, p_cam=p_cam, q_cam=np.tile([0, 0, 0, 1], (len(tm), 1)), cov=np.tile(np.eye(6) * 1e-4, (len(tm), 1, 1)), n_markers=np.full(len(tm), 3), marker_ids=[(301,)] * len(tm),
                 odom=veh, tf=veh, dock=dock, filt=dock, t_vel=t, vel=np.zeros((len(t), 3)) if vel is None else vel,
                 t_health=t, health=np.full(len(t), 1) if health is None else health, t_cmd=t, cmd=np.zeros((len(t), 3)),
                 states=states, rtf=1.0)


def _straight_in(offset_y=0.0, end_x=5.05):
    t = np.linspace(0, 20, 401)
    x = np.linspace(0, end_x, len(t)) - metrics.CAM_X
    y = np.full(len(t), offset_y + metrics.ENTRY_OFFSET[0])
    z = np.full(len(t), -1.5 + metrics.ENTRY_OFFSET[1])
    return t, np.stack([x, y, z], axis=1)


def test_clean_insertion_records_capture_offset_and_closing_speed():
    t, veh = _straight_in(offset_y=0.02)
    tr = _trial(veh, states=[(0.0, 0), (10.0, 1), (19.9, 2)], t=t)
    r = metrics.outcome(tr)
    assert r["outcome"] == "CLEAN" and r["docked"]
    assert abs(r["capture_lateral_m"] - 0.02) < 1e-6 and abs(r["capture_offset_m"] - 0.02) < 1e-6
    assert abs(r["closing_speed_m_s"] - 5.05 / 20) < 0.02
    assert abs(r["t_dock_s"] - 19.9) < 1e-9


def test_off_axis_crossing_before_latch_is_contact():
    t, veh = _straight_in(offset_y=0.30)
    tr = _trial(veh, states=[(0.0, 0), (19.9, 2)], t=t)
    r = metrics.outcome(tr)
    assert r["outcome"] == "CONTACT"
    assert abs(r["contact_offset_m"] - 0.30) < 1e-6


def test_never_reaching_the_plane_is_no_dock():
    t, veh = _straight_in(offset_y=0.0, end_x=3.0)
    tr = _trial(veh, states=[(0.0, 0)], t=t)
    r = metrics.outcome(tr)
    assert r["outcome"] == "NO_DOCK" and not r["docked"] and np.isnan(r["capture_offset_m"])


def test_velocity_error_metrics_on_a_swaying_dock():
    t = np.linspace(0, 40, 2001)
    dock = PoseTrack(t, np.stack([np.full(len(t), 5.0), 0.1 * np.sin(2 * np.pi * t / 8), np.full(len(t), -1.5)], axis=1),
                     np.tile([0, 0, 0, 1], (len(t), 1)))
    true_v = 0.1 * 2 * np.pi / 8 * np.cos(2 * np.pi * t / 8)
    vel = np.zeros((len(t), 3)); vel[:, 1] = 1.5 * true_v          # estimate 1.5x too large
    veh = np.stack([np.zeros(len(t)), np.zeros(len(t)), np.full(len(t), -1.5)], axis=1)
    tr = _trial(veh, states=[(0.0, 0)], t=t, dock_track=dock, vel=vel)
    r = metrics.estimation(tr)
    assert abs(r["vel_amp_ratio"] - 1.5) < 0.02
    assert abs(r["vel_err_rms_m_s"] - 0.5 * np.std(true_v)) < 0.003


def test_stale_periods_are_excluded_from_estimation():
    t = np.linspace(0, 40, 2001)
    health = np.where(t < 20, 1, 3)
    vel = np.zeros((len(t), 3)); vel[t >= 20, 1] = 5.0                 # garbage only while stale
    veh = np.zeros((len(t), 3))
    tr = _trial(veh, states=[(0.0, 0)], t=t, vel=vel, health=health)
    assert metrics.estimation(tr)["vel_err_rms_m_s"] < 1e-9


def test_supervision_counts_demotions_stale_and_blackout():
    t = np.linspace(0, 20, 401)
    health = np.where((t > 5) & (t < 8), 3, 1); health[(t > 15) & (t < 16)] = 3
    veh = np.zeros((len(t), 3))
    tm = np.concatenate([np.linspace(0, 10, 101), np.linspace(15, 20, 51)])   # 5 s blackout
    tr = _trial(veh, states=[(0.0, 0), (2.0, 1), (6.0, 0), (9.0, 1), (16.0, 0)], t=t, health=health, meas_t=tm)
    r = metrics.supervision(tr)
    assert r["n_demotions"] == 2 and r["n_stale"] == 2 and r["n_gaps_gt1s"] == 1
    # the 0.3 s measured window trims both ends of the 5 s gap: (5 - 0.6) / 20
    assert abs(r["blackout_frac"] - 0.22) < 0.02
