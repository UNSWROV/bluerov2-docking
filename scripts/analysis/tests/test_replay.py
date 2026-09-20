import numpy as np
from scipy.spatial.transform import Rotation as R

from analysis import replay as rp
from analysis.tracks import R_CAM, T_MOUNT, PoseTrack, Trial


def _dock_trial(dock_y, t_meas, t_stop=None):
    """Vehicle at rest at the origin facing +x; dock 5 m ahead moving laterally.

    dock_y(t) gives the dock's world y. Measurements are the exact camera-frame
    dock position, so the replay's only error source is the filter itself.
    """
    t = np.linspace(0.0, 30.0, 3001)
    veh = PoseTrack(t, np.zeros((len(t), 3)), np.tile([0, 0, 0, 1], (len(t), 1)))
    dock_p = np.stack([np.full(len(t), 5.0), dock_y(t), np.full(len(t), -1.5)], axis=1)
    dock = PoseTrack(t, dock_p, np.tile([0, 0, 0, 1], (len(t), 1)))
    tm = np.asarray(t_meas, float)
    p_cam = R_CAM.inv().apply(dock.pos(tm) - veh.pos(tm) - T_MOUNT)
    q_cam = np.tile((R_CAM.inv()).as_quat(), (len(tm), 1))
    return Trial(path="synthetic", t_meas=tm, p_cam=p_cam, q_cam=q_cam,
                 cov=np.tile(np.eye(6) * 1e-4, (len(tm), 1, 1)),
                 n_markers=np.full(len(tm), 3), marker_ids=[(301,)] * len(tm),
                 odom=veh, tf=veh, dock=dock, filt=dock, t_vel=t, vel=np.zeros((len(t), 3)),
                 t_health=t, health=np.full(len(t), 1), t_cmd=t, cmd=np.zeros((len(t), 3)),
                 states=[(0.0, 1)], rtf=1.0)


def test_replay_tracks_a_constant_velocity_dock_to_under_a_centimetre():
    v = 0.05
    tr = _dock_trial(lambda t: v * t, np.arange(0.0, 30.0, 0.1))
    res = rp.replay(tr, rp.ReplayParams(regime="sway"))
    settled = res.t > 5.0
    err = res.pos[settled, 1] - tr.dock.pos(res.t[settled])[:, 1]
    assert np.sqrt(np.mean(err ** 2)) < 0.01
    assert abs(np.mean(res.vel[settled, 1]) - v) < 0.01
    assert res.accepted.all()
    assert (res.health[settled] == 1).all()


def test_stale_decay_applies_only_after_the_hold_time():
    v = 0.05
    dock_y = lambda t: v * np.minimum(t, 10.0)          # dock stops when the markers are lost
    # markers lost from 10 s to 29 s, then re-acquired (the replay ticks stop one
    # second after the last measurement, so the re-acquisition spans the blackout)
    tm = np.concatenate([np.arange(0.0, 10.0, 0.1), np.arange(29.0, 30.0, 0.1)])
    tr = _dock_trial(dock_y, tm)
    base = rp.replay(tr, rp.ReplayParams(regime="sway"))
    var = rp.replay(tr, rp.ReplayParams(regime="sway", stale_hold_s=1.0, stale_decay_s=1.0))
    last = 9.9 + rp.ReplayParams().latency_s
    # inside the hold window the variant is the production filter, tick for tick
    hold = (var.t > last) & (var.t <= last + 1.0)
    assert hold.sum() > 20
    assert np.allclose(var.vel[hold], base.vel[hold])
    assert np.allclose(var.pos[hold], base.pos[hold])
    # afterwards the velocity state decays with the requested time constant
    blackout = (var.t > last + 1.0 + 5.0) & (var.t < 29.0)
    assert blackout.sum() > 100
    assert np.all(np.abs(var.vel[blackout, 1]) < 0.05 * np.exp(-5.0) * 1.5)
    assert np.all(np.abs(base.vel[blackout, 1] - v) < 0.005)
    # and the dead-reckoning excursion is bounded instead of growing with the blackout
    k = np.flatnonzero(var.t < 29.0)[-1]
    truth = dock_y(var.t[k])
    assert abs(base.pos[k, 1] - truth) > 0.5
    assert abs(var.pos[k, 1] - truth) < 0.15
    # both flag the blackout as stale once the update age passes the limit
    stale = (base.t > last + 3.5) & (base.t < 29.0)
    assert (base.health[stale] == 3).all()
    assert (var.health[stale] == 3).all()


def test_regime_for_reads_the_arm_from_the_label():
    tr = _dock_trial(lambda t: 0.0 * t, np.arange(0.0, 1.0, 0.5))
    tr.path = "/bags/clamped_static_sway_p8_ph0.0_20260725_061913/x.mcap"
    assert rp.regime_for(tr) == "static"
    tr.path = "/bags/clamped_B_sway_p8_ph0.0_20260725_061913/x.mcap"
    assert rp.regime_for(tr) == "sway"
