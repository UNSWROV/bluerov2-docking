import numpy as np
from scipy.spatial.transform import Rotation as R

from analysis.tracks import PoseTrack, corr, world_from_cam, T_MOUNT


def _track(yaw_deg=0.0):
    t = np.linspace(0, 10, 101)
    p = np.stack([t, 2 * t, np.zeros_like(t)], axis=1)
    q = np.tile(R.from_euler("z", yaw_deg, degrees=True).as_quat(), (len(t), 1))
    return PoseTrack(t, p, q)


def test_position_interpolates_linearly_and_clamps():
    tr = _track()
    np.testing.assert_allclose(tr.pos(2.5), [2.5, 5.0, 0.0])
    np.testing.assert_allclose(tr.pos(-1.0), tr.pos(0.0))
    np.testing.assert_allclose(tr.vel(5.0), [1.0, 2.0, 0.0], atol=1e-9)


def test_duplicate_stamps_are_dropped():
    tr = PoseTrack([0, 1, 1, 2], [[0, 0, 0], [1, 0, 0], [5, 0, 0], [2, 0, 0]], [[0, 0, 0, 1]] * 4)
    assert len(tr.t) == 3
    np.testing.assert_allclose(tr.pos(1.5), [1.5, 0, 0])


def test_world_from_cam_uses_mount_offset_and_optical_rotation():
    tr = _track(yaw_deg=0.0)
    # a point 1 m ahead of the camera along the optical axis (camera +Z) lands
    # 1 m ahead of the mount along body +X
    p = world_from_cam(np.array([[0.0, 0.0, 1.0]]), tr, np.array([0.0]))
    np.testing.assert_allclose(p[0], T_MOUNT + [1.0, 0.0, 0.0], atol=1e-3)


def test_world_from_cam_rotates_with_vehicle_yaw():
    tr = _track(yaw_deg=90.0)
    p = world_from_cam(np.array([[0.0, 0.0, 1.0]]), tr, np.array([0.0]))
    np.testing.assert_allclose(p[0], [0.0, T_MOUNT[0] + 1.0, T_MOUNT[2]], atol=1e-3)


def test_corr_is_nan_for_constant_input():
    assert np.isnan(corr([1, 1, 1], [1, 2, 3]))
    assert abs(corr([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
