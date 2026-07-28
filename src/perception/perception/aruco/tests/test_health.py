import math

from perception.aruco.lib.health import FilterHealth, classify_health, HealthThresholds


DEFAULT = HealthThresholds(
    healthy_max_age_s=0.5,
    healthy_max_position_std_m=0.02,
    stale_max_age_s=3.0,
)


def test_warming_up_when_never_initialized():
    result = classify_health(
        seconds_since_init=0.0,
        seconds_since_last_update=math.inf,
        position_std_m=math.inf,
        thresholds=DEFAULT,
    )
    assert result == FilterHealth.WARMING_UP


def test_healthy_when_recent_and_bounded():
    result = classify_health(
        seconds_since_init=5.0,
        seconds_since_last_update=0.1,
        position_std_m=0.005,
        thresholds=DEFAULT,
    )
    assert result == FilterHealth.HEALTHY


def test_degraded_when_no_recent_update_but_not_stale():
    result = classify_health(
        seconds_since_init=5.0,
        seconds_since_last_update=1.0,
        position_std_m=0.04,
        thresholds=DEFAULT,
    )
    assert result == FilterHealth.DEGRADED


def test_stale_when_long_silence():
    result = classify_health(
        seconds_since_init=10.0,
        seconds_since_last_update=5.0,
        position_std_m=0.5,
        thresholds=DEFAULT,
    )
    assert result == FilterHealth.STALE


def test_degraded_when_uncertainty_too_large_even_if_recent():
    # DEFAULT has no covariance ceiling (inf): high uncertainty stays DEGRADED.
    result = classify_health(
        seconds_since_init=5.0,
        seconds_since_last_update=0.1,
        position_std_m=0.5,
        thresholds=DEFAULT,
    )
    assert result == FilterHealth.DEGRADED


def test_stale_when_uncertainty_exceeds_covariance_ceiling():
    # With a finite stale_max_position_std_m, a blown-up covariance blocks (STALE) even
    # when the last update was recent, so the controller holds instead of driving on it
    # (markers out of frame during fine -> covariance blows up -> would clip the dock).
    thr = HealthThresholds(
        healthy_max_age_s=0.5,
        healthy_max_position_std_m=0.02,
        stale_max_age_s=3.0,
        stale_max_position_std_m=0.15,
    )
    result = classify_health(
        seconds_since_init=5.0,
        seconds_since_last_update=0.1,
        position_std_m=0.5,
        thresholds=thr,
    )
    assert result == FilterHealth.STALE
