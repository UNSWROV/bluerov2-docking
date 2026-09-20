import numpy as np

from perception.utils.differentiate import CausalDifferentiator


def test_slope_of_a_sinusoid_matches_its_derivative_with_half_window_lag():
    d = CausalDifferentiator(window_s=0.2)
    T, A = 8.0, 0.1
    err = []
    for k in range(2000):
        t = k * 0.02
        v = d.push(t, [0.0, A * np.sin(2 * np.pi * t / T), 0.0])
        if v is not None and t > 1.0:
            truth = A * 2 * np.pi / T * np.cos(2 * np.pi * (t - d.lag_s) / T)
            err.append(v[1] - truth)
    assert np.sqrt(np.mean(np.square(err))) < 1e-4


def test_returns_none_until_two_samples_and_ignores_non_increasing_stamps():
    d = CausalDifferentiator(0.2)
    assert d.push(0.0, [0, 0, 0]) is None
    assert d.push(0.0, [1, 1, 1]) is None
    v = d.push(0.02, [0.02, 0, 0])
    np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-9)
