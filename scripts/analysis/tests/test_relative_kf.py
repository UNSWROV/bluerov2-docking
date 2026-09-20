import numpy as np

from analysis.relative_kf import RelativeCVFilter


def test_relative_filter_recovers_dock_velocity_from_moving_vehicle():
    # dock moves at +0.05 m/s in y, vehicle at +0.10 m/s in y: r shrinks at 0.05 m/s
    dt, n = 1 / 30, 900
    rng = np.random.default_rng(0)
    # a low acceleration density so the velocity state settles; the production
    # value 0.16 tracks a swaying dock but leaves centimetre-per-second jitter
    kf = RelativeCVFilter(max_speed=0.5, sigma_a=0.02)
    R = 1e-4 * np.eye(3)
    r_true = lambda k: np.array([2.0, 1.0 - 0.05 * k * dt, 0.0])
    kf.initialize(r_true(0) + rng.normal(0, 0.01, 3), R)
    v_hist = []
    for k in range(1, n):
        kf.predict(dt, np.array([0.0, 0.10, 0.0]))
        if k % 3 == 0:                                   # 10 Hz measurements
            kf.try_update(r_true(k) + rng.normal(0, 0.01, 3), R)
        if k > n - 300:
            v_hist.append(kf.v_d[1])
    assert abs(np.mean(v_hist) - 0.05) < 0.01
    assert np.linalg.norm(kf.r - r_true(n - 1)) < 0.03


def test_relative_filter_gate_rejects_outliers():
    kf = RelativeCVFilter()
    kf.initialize(np.zeros(3), 1e-4 * np.eye(3), inflation=1.0)
    kf.predict(0.1, np.zeros(3))
    assert not kf.try_update(np.array([1.0, 0, 0]), 1e-4 * np.eye(3))
    assert kf.try_update(np.array([0.005, 0, 0]), 1e-4 * np.eye(3))
