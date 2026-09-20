"""Arm C: ego-motion-aware relative dock estimator (offline reference implementation).

Translational state x = [r, v_d] with r = p_d - p_v the dock position relative to
the vehicle in world-aligned axes. The vehicle velocity estimate enters the
process model as a known input; the vehicle position is never added:

    r[k+1]   = r[k] + (v_d[k] - v_v_hat[k]) dt + w_r
    v_d[k+1] = v_d[k] + w_v
    z[k]     = r[k] + nu      (camera-relative translation rotated by attitude only)

Process noise is the same white-acceleration model as the production filter
(perception.aruco.lib.kalman.make_process_noise, translational block) plus
B R_vv B^T for the uncertainty of the velocity input. Gate, velocity clamp and
covariance ceiling mirror the production node. Orientation is not estimated
here; the comparison with arm B is about translation and velocity.
"""
from __future__ import annotations

import numpy as np

from .replay import make_process_noise

CHI2_3_995 = 12.838


class RelativeCVFilter:
    def __init__(self, max_speed: float | None = 0.2, sigma_a: float = 0.16, sigma_vv: float = 0.005):
        self.x = None
        self.P = None
        self.max_speed = max_speed
        self.sigma_a = sigma_a
        self.sigma_vv = sigma_vv
        self.last_d_sq = float("nan")

    @property
    def is_initialized(self):
        return self.x is not None

    @property
    def r(self):
        return self.x[:3]

    @property
    def v_d(self):
        return self.x[3:]

    def initialize(self, z, R, inflation=100.0, velocity_std=0.2):
        self.x = np.concatenate([np.asarray(z, float), np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = R * inflation
        self.P[3:, 3:] = velocity_std ** 2 * np.eye(3)

    def predict(self, dt, v_v_hat):
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        B = np.zeros((6, 3)); B[:3, :] = -dt * np.eye(3)
        self.x = F @ self.x + B @ np.asarray(v_v_hat, float)
        Q = make_process_noise(dt, "sway", self.sigma_a)[:6, :6]
        self.P = F @ self.P @ F.T + Q + B @ (self.sigma_vv ** 2 * np.eye(3)) @ B.T

    def try_update(self, z, R, gate=CHI2_3_995):
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        y = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        self.last_d_sq = float(y @ S_inv @ y)
        if self.last_d_sq > gate:
            return False
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        if self.max_speed is not None:
            sp = float(np.linalg.norm(self.x[3:]))
            if sp > self.max_speed:
                self.x[3:] *= self.max_speed / sp
        return True
