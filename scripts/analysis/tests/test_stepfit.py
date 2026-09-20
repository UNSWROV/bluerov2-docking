import numpy as np

from analysis.stepfit import fit_first_order, simulate


def test_recovers_gain_delay_and_time_constant_from_a_step():
    t = np.arange(0, 20, 0.02)
    u = np.where((t > 2) & (t < 12), 0.1, 0.0)
    v = simulate(t, u, gain=2.6, tau_s=0.3, time_constant_s=0.8)
    v += np.random.default_rng(0).normal(0, 0.005, len(t))
    fit = fit_first_order(t, u, v)
    assert abs(fit.gain - 2.6) < 0.1
    assert abs(fit.tau_s - 0.3) < 0.05
    assert abs(fit.time_constant_s - 0.8) < 0.1


def test_recovers_parameters_from_a_sinusoid_sweep():
    t = np.arange(0, 60, 0.02)
    u = 0.05 * np.sin(2 * np.pi * t / 8) + 0.03 * np.sin(2 * np.pi * t / 3)
    v = simulate(t, u, gain=3.0, tau_s=0.4, time_constant_s=0.5)
    fit = fit_first_order(t, u, v)
    assert abs(fit.gain - 3.0) < 0.1 and abs(fit.tau_s - 0.4) < 0.05
