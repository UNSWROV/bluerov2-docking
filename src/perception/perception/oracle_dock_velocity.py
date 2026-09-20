#!/usr/bin/env python3
"""Oracle dock velocity for the diagnostic arm D (#68).

Differentiates the ground-truth dock pose bridged from Gazebo and publishes it
in the twist message the controllers already consume, so the same feedforward
law can be driven by the true dock velocity. Ground truth is used only in this
diagnostic arm, never in a proposed method: if the oracle over-swings as the
original method does, the plant is the limit; if the fix approaches the oracle,
the mechanism and the fix are confirmed.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, TwistWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from perception.utils.differentiate import CausalDifferentiator


class OracleDockVelocity(Node):
    def __init__(self) -> None:
        super().__init__("oracle_dock_velocity")
        self.declare_parameter("pose_topic", "/dock/ground_truth/pose")
        self.declare_parameter("velocity_topic", "/dock/oracle_velocity")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("window_s", 0.2)
        self.declare_parameter("velocity_variance", 1e-6)
        self._diff = CausalDifferentiator(
            self.get_parameter("window_s").get_parameter_value().double_value
        )
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._pub = self.create_publisher(
            TwistWithCovarianceStamped,
            self.get_parameter("velocity_topic").get_parameter_value().string_value,
            qos,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("pose_topic").get_parameter_value().string_value,
            self._on_pose,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10),
        )
        self.get_logger().info("oracle_dock_velocity ready (diagnostic arm D)")

    def _on_pose(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.position
        v = self._diff.push(t, [p.x, p.y, p.z])
        if v is None:
            return
        out = TwistWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = (float(x) for x in v)
        var = self.get_parameter("velocity_variance").get_parameter_value().double_value
        cov = [0.0] * 36
        for i in range(3):
            cov[i * 7] = var
        for i in range(3, 6):
            cov[i * 7] = 1e6  # angular: not estimated
        out.twist.covariance = cov
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OracleDockVelocity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
