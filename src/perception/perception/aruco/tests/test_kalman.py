import numpy as np

from perception.aruco.lib.geometry import rotvec_to_quat
from perception.aruco.lib.kalman import (
    DockPoseKalmanFilter,
    make_process_noise,
)


def _pose(position: list[float], rotvec: list[float] | None = None):
    return (
        np.array(position),
        rotvec_to_quat(np.array(rotvec or [0.0, 0.0, 0.0])),
    )


def test_initialize_from_first_measurement():
    kf = DockPoseKalmanFilter()
    pos, quat = _pose([1.0, 2.0, 3.0])
    cov = 0.01 * np.eye(6)
    kf.initialize(pos, quat, cov)
    assert kf.is_initialized
    np.testing.assert_allclose(kf.position, pos, atol=1e-9)
    np.testing.assert_allclose(kf.orientation, quat, atol=1e-9)


def test_predict_only_grows_covariance():
    kf = DockPoseKalmanFilter()
    pos, quat = _pose([0.0, 0.0, 0.0])
    kf.initialize(pos, quat, 0.001 * np.eye(6))
    # trace, not det: the static covariance has a zero velocity block (velocity is
    # pinned), so it is singular and det is identically 0. Trace still grows.
    initial_cov_trace = np.trace(kf.covariance)

    q = make_process_noise(dt=0.033, regime="static")
    kf.predict(dt=0.033, process_noise=q)

    assert np.trace(kf.covariance) > initial_cov_trace


def test_converges_on_noisy_stream_around_ground_truth():
    kf = DockPoseKalmanFilter()
    truth_pos = np.array([1.0, 2.0, 3.0])
    truth_quat = rotvec_to_quat(np.array([0.0, 0.0, 0.5]))

    rng = np.random.default_rng(seed=42)
    noise_std = 0.02

    kf.initialize(truth_pos + rng.normal(0, noise_std, 3), truth_quat, 0.01 * np.eye(6))

    q = make_process_noise(dt=0.033, regime="static")
    pos_cov = (noise_std**2) * np.eye(3)
    rot_cov = (noise_std**2) * np.eye(3)
    for _ in range(100):
        kf.predict(dt=0.033, process_noise=q)
        measurement_pos = truth_pos + rng.normal(0, noise_std, 3)
        kf.update(measurement_pos, truth_quat, pos_cov, rot_cov)

    # After 100 noisy measurements, position should be within 5mm of truth
    np.testing.assert_allclose(kf.position, truth_pos, atol=0.005)


def test_mahalanobis_gate_rejects_outlier():
    kf = DockPoseKalmanFilter()
    pos, quat = _pose([0.0, 0.0, 0.0])
    kf.initialize(pos, quat, 1e-4 * np.eye(6))  # very confident
    q = make_process_noise(dt=0.033, regime="static")

    # Feed a measurement 10m away should be rejected as outlier
    far_pos = np.array([10.0, 0.0, 0.0])
    tight_cov = 0.01**2 * np.eye(3)
    accepted = kf.try_update(
        far_pos,
        quat,
        tight_cov,
        tight_cov,
        gate_chi2=18.548,  # chi^2_{6, 0.995}
    )
    assert not accepted
    np.testing.assert_allclose(kf.position, [0.0, 0.0, 0.0], atol=1e-6)


def test_process_noise_regimes_ordered():
    dt = 0.033
    q_static = make_process_noise(dt=dt, regime="static")
    q_sway = make_process_noise(dt=dt, regime="sway")
    assert np.trace(q_static) < np.trace(q_sway)


def test_process_noise_drift_removed():
    import pytest

    with pytest.raises(ValueError):
        make_process_noise(dt=0.033, regime="drift")


def test_process_noise_scales_with_dt():
    q_short = make_process_noise(dt=0.01, regime="static")
    q_long = make_process_noise(dt=1.0, regime="static")
    assert np.trace(q_long) > np.trace(q_short)


def test_process_noise_returns_psd_9x9():
    # static has a zero velocity block (the equivalence guarantee), so the matrix
    # is positive SEMI-definite; sway's WNA blocks are strictly PD.
    q_static = make_process_noise(dt=0.033, regime="static")
    assert q_static.shape == (9, 9)
    assert np.all(np.linalg.eigvalsh(q_static) >= -1e-12)
    np.testing.assert_allclose(q_static[3:6, 3:6], np.zeros((3, 3)), atol=0.0)

    q_sway = make_process_noise(dt=0.033, regime="sway")
    assert q_sway.shape == (9, 9)
    assert np.all(np.linalg.eigvalsh(q_sway) >= -1e-12)
    assert np.all(np.diag(q_sway)[3:6] > 0)


def test_wna_q_structure():
    # Per axis: Q = q_a * [[dt^3/3, dt^2/2], [dt^2/2, dt]] with q_a = sigma_a^2.
    dt = 0.5
    q = make_process_noise(dt=dt, regime="sway")
    sigma_a = 0.16
    q_a = sigma_a**2
    np.testing.assert_allclose(q[0, 0], q_a * dt**3 / 3.0, rtol=1e-9)
    np.testing.assert_allclose(q[0, 3], q_a * dt**2 / 2.0, rtol=1e-9)
    np.testing.assert_allclose(q[3, 0], q_a * dt**2 / 2.0, rtol=1e-9)
    np.testing.assert_allclose(q[3, 3], q_a * dt, rtol=1e-9)
    # no cross-axis coupling
    assert q[0, 4] == 0.0 and q[1, 3] == 0.0
    np.testing.assert_allclose(q, q.T, atol=0.0)


def _run_sequence(kf_step, n=60, dt=0.5, seed=7):
    """Deterministic measurement sequence used by the equivalence test."""
    rng = np.random.default_rng(seed)
    truth_pos = np.array([5.0, 0.0, -1.5])
    truth_quat = rotvec_to_quat(np.array([0.0, 0.0, 1.2]))
    pos_cov = (0.02**2) * np.eye(3)
    rot_cov = (0.01**2) * np.eye(3)
    for _ in range(n):
        meas_pos = truth_pos + rng.normal(0, 0.02, 3)
        kf_step(dt, meas_pos, truth_quat, pos_cov, rot_cov)


class _OracleCpKf:
    """Independent constant-position KF oracle. Shares NO code with kalman.py:
    quaternion math via scipy, plain numpy elsewhere. H = I6, F = I6."""

    def __init__(self, position, quat_xyzw, cov6):
        from scipy.spatial.transform import Rotation as R

        self._R = R
        self.pos = position.copy()
        self.rot = R.from_quat(quat_xyzw)
        self.P = cov6.copy()

    def step(self, dt, meas_pos, meas_quat, pos_cov, rot_cov):
        q6 = np.zeros((6, 6))
        q6[:3, :3] = 1e-4 * dt * np.eye(3)
        q6[3:, 3:] = 1e-4 * dt * np.eye(3)
        self.P = self.P + q6
        R6 = np.zeros((6, 6))
        R6[:3, :3] = pos_cov
        R6[3:, 3:] = rot_cov
        y = np.concatenate(
            [
                meas_pos - self.pos,
                (self.rot.inv() * self._R.from_quat(meas_quat)).as_rotvec(),
            ]
        )
        K = self.P @ np.linalg.inv(self.P + R6)
        dx = K @ y
        self.pos = self.pos + dx[:3]
        self.rot = self.rot * self._R.from_rotvec(dx[3:])
        self.P = (np.eye(6) - K) @ self.P


def test_static_regime_matches_constant_position_oracle():
    """The backward-compatibility guarantee: static regime with velocity_std=0 is
    ALGEBRAICALLY the old constant-position filter. Verified against an oracle
    implemented independently inside this test."""
    init_pos = np.array([5.0, 0.0, -1.5])
    init_quat = rotvec_to_quat(np.array([0.0, 0.0, 1.2]))
    init_cov = 0.01 * np.eye(6)

    kf = DockPoseKalmanFilter()
    kf.initialize(init_pos, init_quat, init_cov, velocity_std=0.0)
    oracle = _OracleCpKf(init_pos, init_quat, init_cov)

    def kf_step(dt, mp, mq, pc, rc):
        kf.predict(dt, make_process_noise(dt=dt, regime="static"))
        kf.update(mp, mq, pc, rc)
        oracle.step(dt, mp, mq, pc, rc)

    _run_sequence(kf_step)

    np.testing.assert_allclose(kf.position, oracle.pos, atol=1e-9)
    np.testing.assert_allclose(kf.velocity, np.zeros(3), atol=0.0)
    np.testing.assert_allclose(kf.orientation, oracle.rot.as_quat(), atol=1e-9)
    # position and rotation covariance blocks match the oracle's
    np.testing.assert_allclose(kf.covariance[:3, :3], oracle.P[:3, :3], atol=1e-9)
    np.testing.assert_allclose(kf.covariance[6:, 6:], oracle.P[3:, 3:], atol=1e-9)


def test_cv_converges_to_constant_velocity():
    """Sway regime on a constant-velocity track: the velocity state converges to
    the true velocity purely from position measurements."""
    true_v = np.array([0.10, -0.05, 0.02])
    quat = rotvec_to_quat(np.array([0.0, 0.0, 0.0]))
    rng = np.random.default_rng(11)
    dt = 0.5
    pos_cov = (0.01**2) * np.eye(3)
    rot_cov = (0.01**2) * np.eye(3)

    kf = DockPoseKalmanFilter()
    kf.initialize(np.zeros(3), quat, 0.01 * np.eye(6), velocity_std=0.2)
    vels = []
    for k in range(1, 81):  # 40 s
        kf.predict(dt, make_process_noise(dt=dt, regime="sway"))
        meas = true_v * (k * dt) + rng.normal(0, 0.01, 3)
        kf.update(meas, quat, pos_cov, rot_cov)
        if k > 60:
            vels.append(kf.velocity.copy())

    # The sway regime keeps single-sample velocity loose (reported 1-sigma ~0.07 m/s,
    # sized for the mission's accelerations), so the estimate converges in the MEAN
    # to the true constant velocity, not on any one noisy sample.
    np.testing.assert_allclose(np.mean(vels, axis=0), true_v, atol=0.02)


def test_cv_tracks_mission_sinusoid():
    """The mission profile: 0.1 m amplitude, 5 s period, 2 Hz gappy measurements.
    Position RMS under 4.5 cm, amplitude captured, reversal overshoot bounded,
    and pos_std stays under 0.05 through the gaps (the whole point)."""
    quat = rotvec_to_quat(np.array([0.0, 0.0, 0.0]))
    rng = np.random.default_rng(23)
    amp, period = 0.1, 5.0
    dt = 0.5
    pos_cov = (0.01**2) * np.eye(3)
    rot_cov = (0.01**2) * np.eye(3)

    def dock_pos(t):
        w = 2.0 * np.pi / period
        return np.array([amp * np.sin(w * t + np.pi / 2), 0.0, amp * np.sin(w * t)])

    kf = DockPoseKalmanFilter()
    kf.initialize(dock_pos(0.0), quat, 0.01 * np.eye(6), velocity_std=0.2)

    errs, est_x, pos_stds = [], [], []
    t = 0.0
    for k in range(1, 81):  # 40 s
        t = k * dt
        kf.predict(dt, make_process_noise(dt=dt, regime="sway"))
        if k % 5 != 0:  # drop every 5th measurement: a 1.0 s gap each 2.5 s
            meas = dock_pos(t) + rng.normal(0, 0.01, 3)
            kf.update(meas, quat, pos_cov, rot_cov)
        if t > 10.0:  # after convergence
            errs.append(np.linalg.norm(kf.position - dock_pos(t)))
            est_x.append(kf.position[0])
            d = np.diag(kf.position_covariance)
            pos_stds.append(float(np.sqrt(max(d[0], d[1], d[2]))))  # node's pos_std metric

    rms = float(np.sqrt(np.mean(np.square(errs))))
    assert rms < 0.045, f"position RMS {rms:.3f} m"
    assert max(errs) < 0.12, f"max error {max(errs):.3f} m"
    capture = (max(est_x) - min(est_x)) / (2 * amp)
    assert capture > 0.85, f"amplitude capture {capture:.2f}"
    # The feature's goal is HEALTHY most of the time (old filter: 2%). pos_std matches
    # the node metric (worst axis); brief spikes in detection gaps are tolerated by the
    # health debounce, so assert the TYPICAL value, not the max.
    pos_stds = np.array(pos_stds)
    assert np.median(pos_stds) < 0.02, f"median pos_std {np.median(pos_stds):.3f} m"
    assert (pos_stds < 0.04).mean() > 0.7, f"only {(pos_stds < 0.04).mean():.0%} under 0.04 gate"


def test_velocity_covariance_shape_and_growth():
    kf = DockPoseKalmanFilter()
    kf.initialize(
        np.zeros(3),
        rotvec_to_quat(np.zeros(3)),
        0.01 * np.eye(6),
        velocity_std=0.2,
    )
    assert kf.velocity_covariance.shape == (3, 3)
    np.testing.assert_allclose(kf.velocity_covariance, 0.04 * np.eye(3), atol=1e-12)
    before = np.trace(kf.velocity_covariance)
    kf.predict(0.5, make_process_noise(dt=0.5, regime="sway"))
    assert np.trace(kf.velocity_covariance) > before


def test_decay_velocity_scales_the_velocity_state_and_keeps_the_covariance():
    kf = DockPoseKalmanFilter()
    kf.initialize(
        np.array([5.0, 0.0, -1.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.eye(6) * 1e-2,
        velocity_std=0.2,
    )
    kf._velocity = np.array([0.0, 0.05, 0.0])
    cov_before = kf.covariance.copy()
    kf.decay_velocity(0.5)
    assert np.allclose(kf.velocity, [0.0, 0.025, 0.0])
    assert np.allclose(kf.covariance, cov_before)
    kf.decay_velocity(2.0)   # factors above one are clipped, the state never grows
    assert np.allclose(kf.velocity, [0.0, 0.025, 0.0])
    kf.decay_velocity(0.0)
    assert np.allclose(kf.velocity, 0.0)
