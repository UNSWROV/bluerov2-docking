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

from perception.utils.nav_error import LEVELS, GaussMarkovBias, NavErrorParams


class NavErrorInjector(Node):
    def __init__(self) -> None:
        super().__init__("nav_error_injector")
        self.declare_parameter("level", "none")
        self.declare_parameter("sigma_b", 0.0)      # overrides the level's sigma_b when > 0
        self.declare_parameter("tau_b", 30.0)
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
            self._params = NavErrorParams(
                sigma_b=sigma_b if sigma_b > 0 else base.sigma_b,
                tau_b=self.get_parameter("tau_b").get_parameter_value().double_value,
                seed=self.get_parameter("seed").get_parameter_value().integer_value,
            )
        self._gen = GaussMarkovBias(self._params) if self._params else None
        self._rng = np.random.default_rng((self._params.seed if self._params else 0) + 1)
        self._last_t = None
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
            f"nav_error_injector level={level} " + (f"sigma_b={self._params.sigma_b} tau_b={self._params.tau_b}" if self._params else "(pass-through)")
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
        b_v_body = R.from_quat([q.x, q.y, q.z, q.w]).inv().apply(b_v)
        n_v = self._rng.normal(0.0, p.sigma_nv, 3)
        out.twist.twist.linear.x = msg.twist.twist.linear.x + b_v_body[0] + n_v[0]
        out.twist.twist.linear.y = msg.twist.twist.linear.y + b_v_body[1] + n_v[1]
        out.twist.twist.linear.z = msg.twist.twist.linear.z + b_v_body[2] + n_v[2]
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
