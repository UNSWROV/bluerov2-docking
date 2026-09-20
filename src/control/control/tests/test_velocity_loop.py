import numpy as np

from control.velocity_loop import VelocityLoop, VelocityLoopParams


def _plant_run(loop, v_ref_fn, t_end, dt=0.05, g=2.7, tau=0.4, v_bias=0.0):
    """Simulate the loop on the identified first-order plant. Returns (t, v_true, v_ref)."""
    v = 0.0
    ts, vs, rs = [], [], []
    t = 0.0
    while t < t_end:
        r = v_ref_fn(t)
        u = loop.step(np.array([0.0, r, 0.0]), np.array([0.0, v + v_bias, 0.0]), dt)[1]
        v += dt * (g * u - v) / tau
        ts.append(t); vs.append(v); rs.append(r)
        t += dt
    return np.array(ts), np.array(vs), np.array(rs)


def test_step_reaches_the_setpoint_without_overshoot():
    loop = VelocityLoop()
    t, v, r = _plant_run(loop, lambda t: 0.08, 6.0)
    assert abs(v[-1] - 0.08) < 0.002
    assert v.max() < 0.08 * 1.05
    # settles within a few time constants of the designed bandwidth (1.5 rad/s)
    assert abs(v[t > 3.0] - 0.08).max() < 0.005


def test_sinusoid_at_8s_is_followed_at_unit_amplitude():
    loop = VelocityLoop()
    w = 2 * np.pi / 8.0
    t, v, r = _plant_run(loop, lambda t: 0.08 * np.sin(w * t), 40.0)
    tail = t > 16.0
    A = np.c_[np.sin(w * t[tail]), np.cos(w * t[tail])]
    c = np.linalg.lstsq(A, v[tail], rcond=None)[0]
    amp_ratio = np.hypot(*c) / 0.08
    lag_deg = -np.degrees(np.arctan2(c[1], c[0]))
    # the open-loop law would have driven 2.7 times this at unit feedforward gain;
    # the designed 2 rad/s bandwidth gives 0.93 amplitude and 19 degrees of lag
    assert 0.88 < amp_ratio < 1.02
    assert 0.0 < lag_deg < 25.0


def test_slow_bias_shared_by_setpoint_and_measurement_cancels():
    # a navigation velocity bias b appears in the dock velocity estimate (setpoint)
    # and in the measured vehicle velocity alike: the true vehicle velocity must not
    # carry it
    loop = VelocityLoop()
    b = 0.03
    t, v, r = _plant_run(loop, lambda t: 0.08 + b, 8.0, v_bias=b)
    assert abs(v[-1] - 0.08) < 0.003


def test_integral_does_not_wind_up_under_saturation():
    loop = VelocityLoop(VelocityLoopParams(effort_max=0.02))
    for _ in range(400):
        loop.step(np.array([0.0, 1.0, 0.0]), np.zeros(3), 0.05)
    assert abs(loop.integral[1]) * 0.74 <= 0.3 + 1e-9
    # once the error reverses the output must respond immediately
    out = loop.step(np.array([0.0, -1.0, 0.0]), np.zeros(3), 0.05)[1]
    assert out < 0.0


def test_reset_clears_the_integral():
    loop = VelocityLoop()
    loop.step(np.array([0.1, 0.1, 0.1]), np.zeros(3), 0.1)
    loop.reset()
    assert np.all(loop.integral == 0.0)
