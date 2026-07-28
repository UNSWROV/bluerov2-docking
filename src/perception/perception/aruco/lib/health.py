"""Filter health classification.

Publishes a semantic status signal to downstream consumers (controllers,
state machines) so they can decide whether to trust the filter output.
This decouples them from the filter's internal representation they
subscribe to a health enum, not covariance values they'd have to threshold
themselves.

Starting thresholds are populated below; tune them against the
fine-alignment controller's safety envelope
"""

from dataclasses import dataclass
from enum import Enum


class FilterHealth(Enum):
    WARMING_UP = 0  # not yet initialized
    HEALTHY = 1  # recent update, uncertainty bounded
    DEGRADED = 2  # stale or uncertain but still potentially useful
    STALE = 3  # too long silent, state is unreliable


@dataclass(frozen=True)
class HealthThresholds:
    """Thresholds for health classification. See classify_health."""

    healthy_max_age_s: float  # max seconds since last update to still be HEALTHY
    healthy_max_position_std_m: float  # max sigma on position for HEALTHY
    stale_max_age_s: float  # beyond this age, STALE regardless of everything else
    # Beyond this position sigma, STALE regardless of age: the estimate is too uncertain
    # to drive on (markers out of frame -> covariance blows up). Blocks the controller
    # from creeping forward blind and clipping the dock. Default inf = disabled.
    stale_max_position_std_m: float = float("inf")


def classify_health(
    seconds_since_init: float,
    seconds_since_last_update: float,
    position_std_m: float,
    thresholds: HealthThresholds,
) -> FilterHealth:
    """Classify filter health.

    Args:
        seconds_since_init: how long the filter has been running (0.0 if
            never initialized)
        seconds_since_last_update: seconds since last accepted measurement
            (math.inf if never)
        position_std_m: sqrt of max diagonal of position covariance block
        thresholds: HealthThresholds instance

    Returns:
        FilterHealth enum value
    """
    if seconds_since_init == 0.0:
        return FilterHealth.WARMING_UP
    if seconds_since_last_update > thresholds.stale_max_age_s:
        return FilterHealth.STALE
    if position_std_m > thresholds.stale_max_position_std_m:
        # Uncertainty too high (e.g. markers out of frame) -> unreliable -> block.
        return FilterHealth.STALE
    if (
        seconds_since_last_update <= thresholds.healthy_max_age_s
        and position_std_m <= thresholds.healthy_max_position_std_m
    ):
        return FilterHealth.HEALTHY
    return FilterHealth.DEGRADED
