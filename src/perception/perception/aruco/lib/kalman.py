"""9-state error-state Kalman filter for dock pose in odom frame.

State: [x_dock, y_dock, z_dock, vx_dock, vy_dock, vz_dock, deltarx, deltary, deltarz]
  - position is tracked directly in metres
  - orientation is tracked as a small-angle perturbation (deltar) about a
    reference quaternion q_ref which is carried alongside the state.
    After each update, the perturbation is absorbed into q_ref and the
    deltar part of the state is zeroed.

Process model: F = [   I3      0      0   ]
                   [   0    dt * I3   0   ]
                   [   0       0      I3  ]
Measurement model: identity (measurement IS the pose, in the same frame).

Reference: Ligorio & Sabatini 2013, Sensors 13:1919 error-state EKF template.
"""

import numpy as np
from scipy.linalg import block_diag

from perception.aruco.lib.geometry import (
    quat_inverse,
    quat_multiply,
    quat_to_rotvec,
    rotvec_to_quat,
)


class DockPoseKalmanFilter:
    """Error-state Kalman filter for a static dock in odom frame."""

    def __init__(self, max_speed: float | None = None) -> None:
        self._position: np.ndarray | None = None
        self._velocity: np.ndarray = np.zeros(3)  # m/s
        self._q_ref: np.ndarray | None = None  # reference quaternion
        self._error_state: np.ndarray = np.zeros(
            9
        )  # [deltax, deltay, deltaz, deltarx, deltary, deltarz]
        self._covariance: np.ndarray | None = None
        # Physical bound on the dock's speed. The dock cannot move faster than this;
        # clamping the velocity STATE to it (default None = off) stops the ego-motion
        # feedback loop from extrapolating the estimate to metres/s under vehicle
        # motion. See the moving-dock docking analysis (#36).
        self._max_speed = max_speed

    @property
    def is_initialized(self) -> bool:
        return self._position is not None

    @property
    def position(self) -> np.ndarray:
        assert self._position is not None
        return self._position + self._error_state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity + self._error_state[3:6]

    @property
    def orientation(self) -> np.ndarray:
        assert self._q_ref is not None
        delta = self._error_state[6:9]
        return quat_multiply(self._q_ref, rotvec_to_quat(delta))

    @property
    def position_covariance(self) -> np.ndarray:
        assert self._covariance is not None
        return self._covariance[:3, :3]

    @property
    def velocity_covariance(self) -> np.ndarray:
        assert self._covariance is not None
        return self._covariance[3:6, 3:6]

    @property
    def rotation_covariance(self) -> np.ndarray:
        assert self._covariance is not None
        return self._covariance[6:9, 6:9]

    @property
    def covariance(self) -> np.ndarray:
        assert self._covariance is not None
        return self._covariance

    def initialize(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        covariance: np.ndarray,
        velocity_std: float = 0.0,
    ) -> None:
        assert covariance.shape == (6, 6)
        self._position = position.copy()
        self._velocity = np.zeros(3)
        self._q_ref = orientation / np.linalg.norm(orientation)
        self._error_state = np.zeros(9)
        position_covariance = covariance[:3, :3]
        velocity_covariance = velocity_std**2 * np.eye(3)
        rotation_covariance = covariance[3:, 3:]
        self._covariance = block_diag(
            position_covariance, velocity_covariance, rotation_covariance
        )

    def predict(self, dt: float, process_noise: np.ndarray) -> None:
        assert self._covariance is not None
        assert self._position is not None
        self._position += self._velocity * dt
        F = np.eye(9)
        F[:3, 3:6] = dt * np.eye(3)
        self._covariance = F @ self._covariance @ F.T + process_noise

    def update(
        self,
        measurement_position: np.ndarray,
        measurement_orientation: np.ndarray,
        measurement_position_covariance: np.ndarray,
        measurement_rotation_covariance: np.ndarray,
    ) -> None:
        """Unconditional update caller is responsible for gating.

        Measurement covariances for position and rotation are passed as
        separate 3x3 blocks with different physical units (m^2 and rad^2).
        They're combined here into a block-diagonal 6x6 R (ArUco PnP corner
        noise has no meaningful position/rotation cross-correlation).
        """
        assert self.is_initialized
        assert self._position is not None
        assert self._q_ref is not None
        assert self._covariance is not None
        assert measurement_position_covariance.shape == (3, 3)
        assert measurement_rotation_covariance.shape == (3, 3)

        R = block_diag(measurement_position_covariance, measurement_rotation_covariance)

        y_pos = measurement_position - self.position
        delta_q = quat_multiply(quat_inverse(self._q_ref), measurement_orientation)
        y_rot = quat_to_rotvec(delta_q) - self._error_state[6:]
        y = np.concatenate([y_pos, y_rot])

        P = self._covariance
        H = np.zeros((6, 9))
        H[:3, :3] = np.eye(3)
        H[-3:, -3:] = np.eye(3)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        self._error_state += K @ y
        I9 = np.eye(9)
        self._covariance = (I9 - K @ H) @ P

        # Absorb error-state into nominal: shift position and velocity, compose rotation.
        self._position += self._error_state[:3]
        self._velocity += self._error_state[3:6]
        # Clamp to the physical dock-speed bound (isotropic). A measurement that implies
        # a faster dock is ego-motion leaking through the transform, not real dock
        # motion; capping the state breaks the extrapolation runaway.
        if self._max_speed is not None:
            speed = float(np.linalg.norm(self._velocity))
            if speed > self._max_speed:
                self._velocity *= self._max_speed / speed
        self._q_ref = quat_multiply(self._q_ref, rotvec_to_quat(self._error_state[6:9]))
        self._q_ref = self._q_ref / np.linalg.norm(self._q_ref)
        self._error_state = np.zeros(9)

    def try_update(
        self,
        measurement_position: np.ndarray,
        measurement_orientation: np.ndarray,
        measurement_position_covariance: np.ndarray,
        measurement_rotation_covariance: np.ndarray,
        gate_chi2: float,
    ) -> bool:
        """Apply Mahalanobis gating. Returns True iff the update was applied.

        Also stores split diagnostics on self for caller inspection:
            self.last_d_sq_total, last_d_sq_pos, last_d_sq_rot, last_innovation
        """
        assert self.is_initialized
        assert self._position is not None
        assert self._q_ref is not None
        assert self._covariance is not None
        R6 = np.zeros((6, 6))
        R6[:3, :3] = measurement_position_covariance
        R6[3:, 3:] = measurement_rotation_covariance

        y_pos = measurement_position - self.position
        delta_q = quat_multiply(quat_inverse(self._q_ref), measurement_orientation)
        y_rot = quat_to_rotvec(delta_q)
        y = np.concatenate([y_pos, y_rot])

        H = np.zeros((6, 9))
        H[:3, :3] = np.eye(3)
        H[-3:, -3:] = np.eye(3)

        S = H @ self._covariance @ H.T + R6
        S_inv = np.linalg.inv(S)
        d_sq = float(y @ S_inv @ y)
        d_sq_pos = float(y_pos @ np.linalg.inv(S[:3, :3]) @ y_pos)
        d_sq_rot = float(y_rot @ np.linalg.inv(S[3:, 3:]) @ y_rot)

        self.last_d_sq_total = d_sq
        self.last_d_sq_pos = d_sq_pos
        self.last_d_sq_rot = d_sq_rot
        self.last_innovation = y.copy()

        if d_sq > gate_chi2:
            return False
        self.update(
            measurement_position,
            measurement_orientation,
            measurement_position_covariance,
            measurement_rotation_covariance,
        )
        return True


def make_process_noise(dt: float, regime: str, sigma_a: float = 0.16) -> np.ndarray:
    """Process noise covariance Q for the 9-state constant-pose filter.

    Q encodes how much we expect the dock's pose to drift between predict
    steps. Larger Q -> filter responds faster to new measurements (less
    smoothing). Smaller Q -> filter resists noise harder (more smoothing,
    more lag).

    Args:
        dt: timestep since last predict (seconds)
        regime: "static" | "sway"
        sigma_a: sway-regime white-noise-acceleration density (m/s^2). Sets the
            velocity-state gain: higher tracks a faster dock but also amplifies
            ego-motion leakage into the velocity state (#36); tunable for the
            stability-vs-tracking trade study. Ignored in the static regime.

    Returns:
        9x9 positive-definite Q matrix (m^2, (m/s)^2 and rad^2 on diagonal).
    """

    if regime == "static":
        q_pos = 1e-4 * dt
        q_vel = 0.0
        q_rot = 1e-4 * dt
        q = np.zeros((9, 9))
        q = block_diag(q_pos * np.eye(3), q_vel * np.eye(3), q_rot * np.eye(3))
    elif regime == "sway":
        """
        Builds:
        Q =  I3 * [ q_a * dt^3/3   q_a * dt^2/2      0     ]
                  [ q_a * dt^2/2     q_a * dt        0     ]
                  [       0              0      q_rot * dt ]
        """
        q_a = sigma_a**2
        q_rot = 7.6e-3
        q_axis = np.zeros((6, 6))
        for i in range(3):
            q_axis[i, i] = q_a * dt**3 / 3.0
            q_axis[i, i + 3] = q_a * dt**2 / 2.0
            q_axis[i + 3, i] = q_a * dt**2 / 2.0
            q_axis[i + 3, i + 3] = q_a * dt
        q = block_diag(q_axis, q_rot * dt * np.eye(3))
    else:
        raise ValueError(f"unknown regime: {regime!r} (expected 'static'|'sway')")

    return q
