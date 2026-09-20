import numpy as np

from analysis.navsim import NavErrorParams, corrupt, gauss_markov_bias


def test_bias_has_the_requested_stationary_sigma_and_correlation_time():
    t = np.arange(0, 20000, 0.1)
    p = NavErrorParams(sigma_b=0.05, tau_b=30.0, seed=3)
    b_v, _ = gauss_markov_bias(t, p, dims=1)
    x = b_v[:, 0]
    assert abs(np.std(x) - 0.05) < 0.004
    lag = int(30.0 / 0.1)
    r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert abs(r - np.exp(-1)) < 0.05          # autocorrelation at one time constant


def test_position_error_is_the_integral_of_the_velocity_bias():
    t = np.arange(0, 100, 0.05)
    b_v, b_p = gauss_markov_bias(t, NavErrorParams(sigma_b=0.02, seed=1), dims=1)
    np.testing.assert_allclose(b_p[1:, 0], np.cumsum(b_v[:-1, 0]) * 0.05, atol=1e-12)


def test_none_level_returns_clean_signals():
    t = np.arange(0, 10, 0.1); p = np.random.default_rng(0).normal(size=(len(t), 3)); v = p * 2
    p_hat, v_hat, b_v, b_p = corrupt(t, p, v, None)
    np.testing.assert_array_equal(p_hat, p); np.testing.assert_array_equal(v_hat, v)
    assert not b_v.any() and not b_p.any()


def test_corruption_is_reproducible_by_seed():
    t = np.arange(0, 10, 0.1); p = np.zeros((len(t), 3)); v = np.zeros((len(t), 3))
    a = corrupt(t, p, v, NavErrorParams(sigma_b=0.03, seed=7))
    b = corrupt(t, p, v, NavErrorParams(sigma_b=0.03, seed=7))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
