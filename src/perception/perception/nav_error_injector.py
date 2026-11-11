#!/usr/bin/env python3
"""Navigation-error injector (#69).

Republishes the vehicle odometry with a first-order Gauss-Markov velocity bias
integrated into the position, plus white noise, so every estimator and controller
sees the same imperfect navigation while the ground truth stays available for
scoring on the original topic. The bias is expressed in the world frame and
rotated into the body frame for the odometry twist. Publishes the injected bias
so the analysis can correlate estimator error with it.
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R

from perception.utils.nav_error import LEVELS, GaussMarkovBias, NavErrorParams, SampleHold


class NavErrorInjector(Node):
    def __init__(self) -> None:
        super().__init__("nav_error_injector")
        self.declare_parameter("level", "none")
        self.declare_parameter("sigma_b", 0.0)      # overrides the level's sigma_b when > 0
        self.declare_parameter("tau_b", 30.0)
        # white position and velocity noise on the corrupted solution; negative keeps
        # the level's default. The 5 mm default defeats the coarse handoff gate, which
        # differentiates the relayed pose, so the navigation sweep sets it explicitly.
        self.declare_parameter("sigma_np", -1.0)
        self.declare_parameter("sigma_nv", -1.0)
        # update rate of the velocity measurement [Hz]: the reported velocity (truth plus
        # bias plus one noise draw) is sampled at this rate and held in between, as a
        # Doppler velocity log reports it; 0 draws new noise on every odometry sample (#97)
        self.declare_parameter("velocity_rate_hz", 0.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("input_topic", "/model/bluerov2_heavy/odometry")
        self.declare_parameter("output_topic", "/nav/odometry")
        self.declare_parameter("bias_topic", "/nav/error_bias")
        level = self.get_parameter("level").get_parameter_value().string_value
        base = LEVELS.get(level)
        sigma_b = self.get_parameter("sigma_b").get_parameter_value().double_value
        if base is None and sigma_b <= 0:
            self._params = None
        else:
            default = base if base is not None else NavErrorParams(sigma_b=sigma_b)
            s_np = self.get_parameter("sigma_np").get_parameter_value().double_value
            s_nv = self.get_parameter("sigma_nv").get_parameter_value().double_value
            self._params = NavErrorParams(
                sigma_b=sigma_b if sigma_b > 0 else base.sigma_b,
                tau_b=self.get_parameter("tau_b").get_parameter_value().double_value,
                sigma_np=s_np if s_np >= 0 else default.sigma_np,
                sigma_nv=s_nv if s_nv >= 0 else default.sigma_nv,
                seed=self.get_parameter("seed").get_parameter_value().integer_value,
            )
        self._gen = GaussMarkovBias(self._params) if self._params else None
        self._rng = np.random.default_rng((self._params.seed if self._params else 0) + 1)
        self._last_t = None
        rate = self.get_parameter("velocity_rate_hz").get_parameter_value().double_value
        self._vel_hold = SampleHold(rate)
        # RELIABLE so the TF relay's default (reliable) subscription accepts it; a
        # reliable publisher also serves best-effort subscribers, the reverse does not.
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._pub = self.create_publisher(
            Odometry, self.get_parameter("output_topic").get_parameter_value().string_value, pub_qos
        )
        self._pub_bias = self.create_publisher(
            Vector3Stamped, self.get_parameter("bias_topic").get_parameter_value().string_value, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter("input_topic").get_parameter_value().string_value, self._on_odom, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"nav_error_injector level={level} "
            + (
                f"sigma_b={self._params.sigma_b} tau_b={self._params.tau_b} "
                f"sigma_np={self._params.sigma_np} sigma_nv={self._params.sigma_nv} velocity_rate_hz={rate}"
                if self._params
                else "(pass-through)"
            )
        )

    def _on_odom(self, msg: Odometry) -> None:
        if self._gen is None:
            self._pub.publish(msg)
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        b_v, b_p = self._gen.step(dt)
        p = self._params
        out = Odometry()
        out.header = msg.header; out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose; out.twist = msg.twist
        n_p = self._rng.normal(0.0, p.sigma_np, 3)
        out.pose.pose.position.x = msg.pose.pose.position.x + b_p[0] + n_p[0]
        out.pose.pose.position.y = msg.pose.pose.position.y + b_p[1] + n_p[1]
        out.pose.pose.position.z = msg.pose.pose.position.z + b_p[2] + n_p[2]
        q = msg.pose.pose.orientation
        if self._vel_hold.due(t):
            b_v_body = R.from_quat([q.x, q.y, q.z, q.w]).inv().apply(b_v)
            lin = msg.twist.twist.linear
            v_meas = np.array([lin.x, lin.y, lin.z]) + b_v_body + self._rng.normal(0.0, p.sigma_nv, 3)
            self._vel_hold.set(t, v_meas)
        v_meas = self._vel_hold.value
        out.twist.twist.linear.x = float(v_meas[0])
        out.twist.twist.linear.y = float(v_meas[1])
        out.twist.twist.linear.z = float(v_meas[2])
        self._pub.publish(out)
        bias = Vector3Stamped()
        bias.header = msg.header
        bias.vector.x, bias.vector.y, bias.vector.z = (float(x) for x in b_v)
        self._pub_bias.publish(bias)


def main(args=None):
    rclpy.init(args=args)
    node = NavErrorInjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
