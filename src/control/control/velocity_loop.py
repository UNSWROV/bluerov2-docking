"""Body-velocity loop: closes a velocity setpoint on the navigation velocity.

The original feedforward adds the estimated dock velocity, scaled by a fixed
gain, to an effort-like command, so the vehicle's response is (gain x plant
gain) times the dock velocity for whatever plant gain the flight mode produces
(2.7 m/s per unit command with a 0.4 s time constant in the step tests, against
the 1.3 to 1.6 the gain was tuned to). This loop treats the sum of the position
feedback and the rotated dock velocity as a body-frame velocity setpoint and
drives the measured body velocity to it with a PI per axis, so the plant gain
no longer sets the vehicle's amplitude and a slow bias shared by the dock
velocity estimate and the navigation velocity cancels in the error.

Gains follow from the identified first-order plant g / (1 + s T): with the
integral time equal to T the open loop is kp g / (T s), a first-order closed
loop with bandwidth kp g / T, placed below the plant corner 1 / T so the loop
stays well damped. For g = 2.7, T = 0.4 s and a 2.0 rad/s bandwidth: kp = 0.30
effort per m/s and ki = kp / T = 0.74 effort per m. On that plant the loop follows
the 8 s sway at 0.93 amplitude with 19 degrees of lag and no step overshoot; the
position feedback in the setpoint trims the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityLoopParams:
    kp: float = 0.30          # effort per m/s of velocity error
    ki: float = 0.74          # effort per m of integrated velocity error
    integral_max: float = 0.3  # bound on |ki * integral| per axis (anti-windup)
    effort_max: float = 0.5   # bound on the effort output per axis


class VelocityLoop:
    """Three-axis PI on body velocity (surge, sway, heave)."""

    def __init__(self, params: VelocityLoopParams = VelocityLoopParams()) -> None:
        self._p = params
        self.reset()

    def reset(self) -> None:
        self._integral = np.zeros(3)

    @property
    def integral(self) -> np.ndarray:
        return self._integral.copy()

    def step(
        self,
        v_ref: np.ndarray,
        v_meas: np.ndarray,
        dt: float,
        effort_max: float | None = None,
    ) -> np.ndarray:
        """Effort for one control cycle.

        v_ref and v_meas are body-frame linear velocities in m/s. The integral is
        clamped in its effort contribution and is not advanced while the output is
        saturated in the direction of the error (conditional integration), so a
        long saturation does not wind it up. effort_max overrides the configured
        bound for this cycle: the caller passes the authority it will actually
        apply (the health gate halves the command while DEGRADED), so the
        anti-windup sees the real saturation.
        """
        v_ref = np.asarray(v_ref, dtype=float)
        v_meas = np.asarray(v_meas, dtype=float)
        err = v_ref - v_meas
        p = self._p
        e_max = p.effort_max if effort_max is None else max(float(effort_max), 1e-6)
        proposed = self._integral + err * dt
        i_term = np.clip(p.ki * proposed, -p.integral_max, p.integral_max)
        raw = p.kp * err + i_term
        out = np.clip(raw, -e_max, e_max)
        saturated = (np.abs(raw) > e_max) & (np.sign(raw) == np.sign(err))
        self._integral = np.where(saturated, self._integral, i_term / p.ki if p.ki > 0 else 0.0)
        return out


OPEN_LOOP = "open_loop"
VELOCITY_LOOP = "velocity_loop"
MODES = (OPEN_LOOP, VELOCITY_LOOP)


# Effort per m/s of setpoint when the loop cannot close: the inverse of the identified
# plant gain (2.7 m/s per unit command), so a navigation dropout degrades to a
# correctly scaled open-loop command instead of driving the plant at unit gain.
FALLBACK_GAIN = 1.0 / 2.7


def effort_from_setpoint(
    loop: VelocityLoop,
    v_ref: np.ndarray,
    v_meas: np.ndarray | None,
    dt: float,
    effort_max: float | None = None,
    fallback_gain: float = FALLBACK_GAIN,
) -> np.ndarray:
    """One velocity-loop step, or, when no navigation velocity is available, the
    setpoint converted to effort through the plant gain (the open-loop law at the
    right scale) so the vehicle keeps regulating rather than stopping or
    over-driving. The loop's integral is reset so it does not resume stale."""
    if v_meas is None:
        loop.reset()
        return np.asarray(v_ref, dtype=float) * fallback_gain
    return loop.step(v_ref, v_meas, dt, effort_max=effort_max)
