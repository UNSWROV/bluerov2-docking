"""PBVS control law for BlueROV2 docking (shared by the coarse and fine nodes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PbvsParams:
    """Tunable gains + fixed limits."""

    kp_surge: float
    kd_surge: float
    kp_sway: float
    kd_sway: float
    kp_heave: float
    kd_heave: float
    kp_yaw: float
    kd_yaw: float

    v_max_surge: float  # m/s
    v_max_sway: float  # m/s
    v_max_heave: float  # m/s
    v_max_yaw: float  # rad/s
    ff_vel_max: float = 0.25  # rad/s


@dataclass(frozen=True)
class CmdVel:
    """Body-frame velocity command."""

    surge: float
    sway: float
    heave: float
    yaw_rate: float


def clamp(value: float, limit: float) -> float:
    """Symmetric saturation to [-limit, +limit]."""
    return float(np.clip(value, -limit, limit))


def rate(curr: float, prev: float | None) -> float:
    """Finite difference rate (curr - prev). If prev is None, return 0."""
    if prev is None:
        return 0.0
    return curr - prev


def approach_speed_limit(
    range_to_dock_m: float, slope_per_s: float, v_floor: float, v_ceiling: float
) -> float:
    """Distance-gated surge speed cap: allow slope * range_to_dock, floored at
    v_floor (final creep) and ceilinged at v_ceiling. Slows the approach the
    closer the ROV gets to the dock, bounding the worst-case collision speed."""
    return float(np.clip(slope_per_s * range_to_dock_m, v_floor, v_ceiling))


def ff_authority_ramp(range_to_standoff_m: float, far_m: float, near_m: float) -> float:
    """Distance-gated feedforward authority in [0, 1] (rev 2026-07-22).

    Returns 0 when range_to_standoff_m >= far_m, ramps LINEARLY to 1 by the time
    range_to_standoff_m <= near_m, and clips to [0, 1] outside that band. The
    endpoints are inverted vs approach_speed_limit: feedforward GROWS as the ROV
    closes on the standoff. Far out, the dock-velocity estimate is noisy (few, small
    markers) and the ~0.13 m/s sway is second-order while the ROV is still closing
    fast; near the standoff, velocity sync is exactly what lets the position error
    settle enough to hand off to fine. Assumes far_m > near_m.
    """
    return float(np.clip((far_m - range_to_standoff_m) / (far_m - near_m), 0.0, 1.0))


class PbvsController:
    """Decoupled P/PD regulator: body-frame error to body velocity command."""

    def __init__(self, params: PbvsParams) -> None:
        self._p = params
        self.reset()

    def reset(self) -> None:
        """Clear derivative state before a fresh approach."""
        self._prev_forward: float | None = None
        self._prev_lateral: float | None = None
        self._prev_vertical: float | None = None
        self._prev_yaw_err: float | None = None

    def step(
        self,
        rel_pos_body: np.ndarray,
        yaw_err: float,
        dt: float,
        ff_vel_body: float | None = None,
    ) -> CmdVel:
        """Compute one velocity command from the current relative dock pose."""

        range_ahead, lateral_left, vertical_up = rel_pos_body

        ff = (
            np.zeros(3) if ff_vel_body is None else np.asarray(ff_vel_body, dtype=float)
        )

        # PD on the forward axis: kd_surge damps the closing velocity so the
        # vehicle brakes as it nears the target instead of coasting through it
        # (there is no physical stop at the dock). kd_surge=0 -> pure P.
        surge = clamp(
            self._p.kp_surge * range_ahead
            + self._p.kd_surge * rate(range_ahead, self._prev_forward) / dt,
            self._p.v_max_surge,
        ) + clamp(ff[0], self._p.ff_vel_max)

        sway = clamp(
            self._p.kp_sway * lateral_left
            + self._p.kd_sway * rate(lateral_left, self._prev_lateral) / dt,
            self._p.v_max_sway,
        ) + clamp(ff[1], self._p.ff_vel_max)

        heave = clamp(
            self._p.kp_heave * vertical_up
            + self._p.kd_heave * rate(vertical_up, self._prev_vertical) / dt,
            self._p.v_max_heave,
        ) + clamp(ff[2], self._p.ff_vel_max)

        yaw_rate = clamp(
            self._p.kp_yaw * yaw_err
            + self._p.kd_yaw * rate(yaw_err, self._prev_yaw_err) / dt,
            self._p.v_max_yaw,
        )

        self._prev_forward = range_ahead
        self._prev_lateral = lateral_left
        self._prev_vertical = vertical_up
        self._prev_yaw_err = yaw_err

        return CmdVel(surge, sway, heave, yaw_rate)
