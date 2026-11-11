import numpy as np

from perception.utils.nav_error import NavErrorParams, corrupt, gauss_markov_bias


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


def test_streaming_generator_matches_the_batch_generator():
    from perception.utils.nav_error import GaussMarkovBias
    t = np.arange(0, 50, 0.1)
    p = NavErrorParams(sigma_b=0.03, seed=5)
    b_v, b_p = gauss_markov_bias(t, p, dims=2)
    gen = GaussMarkovBias(p, dims=2)
    sv, sp = [gen.b_v.copy()], [gen.b_p.copy()]
    for k in range(1, len(t)):
        v, q = gen.step(0.1); sv.append(v); sp.append(q)
    np.testing.assert_allclose(np.array(sv), b_v); np.testing.assert_allclose(np.array(sp), b_p)


def test_white_level_has_no_bias_and_keeps_the_white_terms():
    from perception.utils.nav_error import LEVELS
    base = LEVELS["white"]
    assert base.sigma_b == 0.0
    t = np.arange(0, 200, 0.01); p = np.zeros((len(t), 3)); v = np.zeros((len(t), 3))
    params = NavErrorParams(sigma_b=0.0, sigma_np=0.001, sigma_nv=0.05, seed=2)
    p_hat, v_hat, b_v, b_p = corrupt(t, p, v, params)
    assert not b_v.any() and not b_p.any()
    assert abs(np.std(v_hat) - 0.05) < 0.002
    assert abs(np.std(p_hat) - 0.001) < 0.0001


def test_sample_hold_updates_at_the_requested_rate_and_holds_in_between():
    from perception.utils.nav_error import SampleHold
    hold = SampleHold(rate_hz=10.0)
    out, updates = [], 0
    for k in range(1000):                      # 10 s of 100 Hz odometry
        t = 5.0 + k * 0.01
        if hold.due(t):
            hold.set(t, float(k)); updates += 1
        out.append(hold.value)
    assert updates == 100
    out = np.array(out)
    np.testing.assert_array_equal(out[:10], 0.0)      # first value held for one period
    np.testing.assert_array_equal(out[10:20], 10.0)
    assert len(np.unique(out)) == 100


def test_sample_hold_keeps_the_mean_rate_when_the_step_does_not_divide_the_period():
    from perception.utils.nav_error import SampleHold
    hold = SampleHold(rate_hz=7.0)
    updates = 0
    for k in range(7000):                      # 70 s at 100 Hz
        t = k * 0.01
        if hold.due(t):
            hold.set(t, k); updates += 1
    assert abs(updates - 490) <= 1


def test_sample_hold_rate_zero_passes_every_sample_and_survives_a_time_reset():
    from perception.utils.nav_error import SampleHold
    hold = SampleHold(rate_hz=0.0)
    for k in range(5):
        assert hold.due(0.01 * k)
        hold.set(0.01 * k, k)
    assert hold.value == 4
    hold = SampleHold(rate_hz=10.0)
    hold.set(100.0, 1.0)
    assert not hold.due(100.05)
    assert hold.due(2.0)                       # time went backwards
    hold.set(2.0, 2.0)
    assert not hold.due(2.05) and hold.due(2.1)
