"""Black-box tests for the pure health gate and phase decision."""

from control.health_gate import (
    APPROACHING,
    AT_STANDOFF,
    BLOCKED,
    DEGRADED,
    GateResult,
    HEALTHY,
    STALE,
    Tolerances,
    WARMING_UP,
    decide_phase,
    gate_for_health,
    within_tolerances,
    within_velocity,
)


def test_healthy_passes_full_gain():
    g = gate_for_health(HEALTHY, degraded_gain_scale=0.5)
    assert g == GateResult(blocked=False, gain_scale=1.0, dock_healthy=True)


def test_degraded_derates_and_flags():
    g = gate_for_health(DEGRADED, degraded_gain_scale=0.5)
    assert g == GateResult(blocked=False, gain_scale=0.5, dock_healthy=False)


def test_stale_blocks():
    g = gate_for_health(STALE, degraded_gain_scale=0.5)
    assert g.blocked is True and g.gain_scale == 0.0


def test_warming_up_blocks():
    g = gate_for_health(WARMING_UP, degraded_gain_scale=0.5)
    assert g.blocked is True and g.gain_scale == 0.0


_TOL = Tolerances(
    position_m=0.10, axis_offset_m=0.10, yaw_rad=0.10, debounce_cycles=3
)


def test_within_tolerances_true_when_all_small():
    wp, wh = within_tolerances(
        range_to_standoff_m=0.05, axis_offset_m=0.02, yaw_err_rad=0.03, tol=_TOL
    )
    assert wp is True and wh is True


def test_within_tolerances_false_when_off_axis():
    wp, wh = within_tolerances(
        range_to_standoff_m=0.05, axis_offset_m=0.30, yaw_err_rad=0.03, tol=_TOL
    )
    assert wp is False and wh is True


def test_within_velocity_true_when_slow():
    assert within_velocity(0.02, vel_match_m_s=0.05) is True


def test_within_velocity_false_when_fast():
    # sway zero-crossing: position may be in tolerance but the dock is moving fast
    assert within_velocity(0.12, vel_match_m_s=0.05) is False


def test_handoff_counter_decays_under_velocity_mismatch():
    # position, yaw, health all good, but within_vel False -> cannot accumulate toward
    # AT_STANDOFF; the debounce counter decays (2 -> 1) instead of climbing.
    phase, ready, counter = decide_phase(
        blocked=False, within_pos=True, within_yaw=True, healthy=True,
        ready_counter=2, was_ready=False, tol=_TOL, within_vel=False,
    )
    assert phase == APPROACHING and ready is False and counter == 1


def test_blocked_phase_overrides():
    phase, ready, counter = decide_phase(
        blocked=True, within_pos=True, within_yaw=True, healthy=True,
        ready_counter=3, was_ready=True, tol=_TOL,
    )
    assert phase == BLOCKED and ready is False and counter == 0


def test_phase_debounces_to_at_standoff():
    # Three consecutive in-tolerance cycles needed (debounce_cycles=3).
    counter, ready = 0, False
    for _ in range(2):
        phase, ready, counter = decide_phase(
            blocked=False, within_pos=True, within_yaw=True, healthy=True,
            ready_counter=counter, was_ready=ready, tol=_TOL,
        )
        assert phase == APPROACHING and ready is False
    phase, ready, counter = decide_phase(
        blocked=False, within_pos=True, within_yaw=True, healthy=True,
        ready_counter=counter, was_ready=ready, tol=_TOL,
    )
    assert phase == AT_STANDOFF and ready is True


def test_ready_is_sticky_under_single_out_of_tol():
    # Hysteresis: once latched, a single out-of-tolerance cycle must NOT drop ready.
    phase, ready, counter = decide_phase(
        blocked=False, within_pos=False, within_yaw=True, healthy=True,
        ready_counter=_TOL.debounce_cycles, was_ready=True, tol=_TOL,
    )
    assert ready is True and phase == AT_STANDOFF
    assert counter == _TOL.debounce_cycles - 1


def test_ready_drops_after_sustained_out_of_tol():
    counter, ready = _TOL.debounce_cycles, True
    for _ in range(_TOL.debounce_cycles):
        phase, ready, counter = decide_phase(
            blocked=False, within_pos=False, within_yaw=True, healthy=True,
            ready_counter=counter, was_ready=ready, tol=_TOL,
        )
    assert ready is False and counter == 0 and phase == APPROACHING
