#!/usr/bin/env python3
"""dock_pose_filter ROS2 node temporal KF on fused dock pose."""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    TransformStamped,
    TwistWithCovarianceStamped,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_geometry_msgs  # noqa: F401  # registers do_transform_pose
from tf2_ros import (
    Buffer,
    ConnectivityException,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)

from perception.aruco.lib.health import (
    FilterHealth as HealthEnum,
    HealthThresholds,
    classify_health,
)
from perception.aruco.lib.init_policy import is_initialization_eligible
from perception.aruco.lib.kalman import DockPoseKalmanFilter, make_process_noise
from interfaces.msg import DockPoseMeasurement
from interfaces.msg import FilterHealth as FilterHealthMsg


def _enum_to_msg_field(h: HealthEnum) -> int:
    return {
        HealthEnum.WARMING_UP: FilterHealthMsg.WARMING_UP,
        HealthEnum.HEALTHY: FilterHealthMsg.HEALTHY,
        HealthEnum.DEGRADED: FilterHealthMsg.DEGRADED,
        HealthEnum.STALE: FilterHealthMsg.STALE,
    }[h]


class DockPoseFilter(Node):
    def __init__(self):
        super().__init__("dock_pose_filter")

        self.declare_parameter("predict_rate_hz", 30.0)
        self.declare_parameter("process_noise_regime", "static")
        self.declare_parameter("mahalanobis_gate_chi2", 18.548)
        self.declare_parameter("initial_covariance_inflation", 100.0)
        # Minimum markers backing a fused measurement before the filter is
        # allowed to initialize on it. A single marker suffers the planar PnP
        # flip/tilt ambiguity; >=2 non-collinear markers break it. See
        # lib/init_policy.py for the rationale. (Default is a placeholder -
        # tune to the dock's geometry / approach range.)
        self.declare_parameter("min_markers_for_init", 2)
        self.declare_parameter("tf_connectivity_grace_period_s", 5.0)
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("child_frame", "dock_filtered")
        self.declare_parameter("healthy_max_age_s", 0.5)
        self.declare_parameter("healthy_max_position_std_m", 0.02)
        self.declare_parameter("stale_max_age_s", 3.0)
        # Covariance ceiling: above this position sigma the estimate is too uncertain to
        # drive on (markers out of frame during fine -> covariance blows up), so report
        # STALE and let the controller hold instead of creeping into the dock blind.
        self.declare_parameter("stale_max_position_std_m", 0.15)
        # Physical cap on the estimated dock speed (m/s); <=0 disables. Bounds the CV
        # velocity state so ego-motion leaking through the transform can't be
        # extrapolated into a runaway estimate under vehicle motion (#36).
        self.declare_parameter("max_dock_speed_m_s", 0.2)
        # Sway-regime WNA density (m/s^2). Velocity-state gain knob for the
        # stability-vs-tracking trade study (#36): high tracks a fast dock but
        # amplifies ego-motion leakage; low stiffens toward the static filter.
        self.declare_parameter("sway_sigma_a", 0.16)

        _max_speed = (
            self.get_parameter("max_dock_speed_m_s").get_parameter_value().double_value
        )
        self._kf = DockPoseKalmanFilter(max_speed=_max_speed if _max_speed > 0 else None)
        self._last_update_wall_t: float | None = None
        self._init_wall_t: float = time.monotonic()
        self._node_start_t: float = time.monotonic()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publisher QoS: RELIABLE so RViz + other RELIABLE consumers can see the
        # filtered pose. RELIABLE publishers serve both RELIABLE and BEST_EFFORT
        # subscribers (the reverse isn't true). Subscribe QoS stays BEST_EFFORT
        # since aruco_fusion publishes that way.
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, "/perception/dock_pose_filtered", pub_qos
        )
        self._pub_twist = self.create_publisher(
            TwistWithCovarianceStamped,
            "/perception/dock_pose_filtered/velocity",
            pub_qos,
        )
        self._pub_health = self.create_publisher(
            FilterHealthMsg, "/perception/dock_pose_filtered/health", pub_qos
        )

        self.create_subscription(
            DockPoseMeasurement,
            "/perception/dock_pose_measured",
            self._on_fused,
            sub_qos,
        )

        rate = self.get_parameter("predict_rate_hz").get_parameter_value().double_value
        self._last_predict_t: float | None = None
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info("dock_pose_filter ready")

    def _thresholds(self) -> HealthThresholds:
        return HealthThresholds(
            healthy_max_age_s=self.get_parameter("healthy_max_age_s")
            .get_parameter_value()
            .double_value,
            healthy_max_position_std_m=self.get_parameter("healthy_max_position_std_m")
            .get_parameter_value()
            .double_value,
            stale_max_age_s=self.get_parameter("stale_max_age_s")
            .get_parameter_value()
            .double_value,
            stale_max_position_std_m=self.get_parameter("stale_max_position_std_m")
            .get_parameter_value()
            .double_value,
        )

    def _on_fused(self, msg: DockPoseMeasurement) -> None:
        grace = (
            self.get_parameter("tf_connectivity_grace_period_s")
            .get_parameter_value()
            .double_value
        )
        target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                msg.header.stamp,
                timeout=Duration(seconds=0.05),
            )
        except ConnectivityException:
            if time.monotonic() - self._node_start_t > grace:
                self.get_logger().error(
                    f"TF {target_frame} -> {msg.header.frame_id} unreachable after grace period. "
                    "Configuration error: the TF tree is disconnected."
                )
                raise
            self.get_logger().warn(
                "TF not yet connected; still in startup grace period",
                throttle_duration_sec=2.0,
            )
            return
        except TransformException as e:
            self.get_logger().warn(
                f"Could not look up transform; skipping this update: {e}",
                throttle_duration_sec=1.0,
            )
            return

        pose_odom = self._transform_pose(msg, tf)
        pos = np.array(
            [
                pose_odom.pose.pose.position.x,
                pose_odom.pose.pose.position.y,
                pose_odom.pose.pose.position.z,
            ]
        )
        quat = np.array(
            [
                pose_odom.pose.pose.orientation.x,
                pose_odom.pose.pose.orientation.y,
                pose_odom.pose.pose.orientation.z,
                pose_odom.pose.pose.orientation.w,
            ]
        )
        cov = np.array(pose_odom.pose.covariance).reshape(6, 6)
        meas_cov_pos = cov[:3, :3]
        meas_cov_rot = cov[3:, 3:]

        if not self._kf.is_initialized:
            min_markers = (
                self.get_parameter("min_markers_for_init")
                .get_parameter_value()
                .integer_value
            )
            if not is_initialization_eligible(msg.num_markers, min_markers):
                self.get_logger().info(
                    f"Deferring init: measurement has {msg.num_markers} marker(s), "
                    f"need >= {min_markers} to constrain orientation; staying WARMING_UP",
                    throttle_duration_sec=2.0,
                )
                return
            inflation = (
                self.get_parameter("initial_covariance_inflation")
                .get_parameter_value()
                .double_value
            )
            regime = (
                self.get_parameter("process_noise_regime")
                .get_parameter_value()
                .string_value
            )
            velocity_std = 0.2 if regime == "sway" else 0.0
            self._kf.initialize(pos, quat, cov * inflation, velocity_std)
            self._last_update_wall_t = time.monotonic()
            return

        chi2 = (
            self.get_parameter("mahalanobis_gate_chi2")
            .get_parameter_value()
            .double_value
        )
        accepted = self._kf.try_update(
            pos, quat, meas_cov_pos, meas_cov_rot, gate_chi2=chi2
        )
        if accepted:
            self._last_update_wall_t = time.monotonic()
        else:
            innov = self._kf.last_innovation
            self.get_logger().warn(
                f"Mahalanobis gate rejected: "
                f"d^2_total={self._kf.last_d_sq_total:.2f} "
                f"(pos={self._kf.last_d_sq_pos:.2f}, "
                f"rot={self._kf.last_d_sq_rot:.2f}), "
                f"gate={chi2:.2f}, "
                f"y_pos=[{innov[0]:+.4f},{innov[1]:+.4f},{innov[2]:+.4f}]m, "
                f"y_rot=[{innov[3]:+.4f},{innov[4]:+.4f},{innov[5]:+.4f}]rad",
                throttle_duration_sec=1.0,
            )

    def _transform_pose(
        self, msg: DockPoseMeasurement, tf: TransformStamped
    ) -> PoseWithCovarianceStamped:
        target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        transformed = tf2_geometry_msgs.do_transform_pose(msg.pose.pose, tf)
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = target_frame
        out.pose.pose = transformed
        out.pose.covariance = msg.pose.covariance
        return out

    def _tick(self) -> None:
        now = time.monotonic()
        if self._last_predict_t is None:
            self._last_predict_t = now
            self._publish_health()
            return
        dt = now - self._last_predict_t
        self._last_predict_t = now

        if self._kf.is_initialized:
            q = make_process_noise(
                dt=dt,
                regime=self.get_parameter("process_noise_regime")
                .get_parameter_value()
                .string_value,
                sigma_a=self.get_parameter("sway_sigma_a")
                .get_parameter_value()
                .double_value,
            )
            self._kf.predict(dt=dt, process_noise=q)

            target_frame = (
                self.get_parameter("target_frame").get_parameter_value().string_value
            )
            child_frame = (
                self.get_parameter("child_frame").get_parameter_value().string_value
            )
            stamp = self.get_clock().now().to_msg()
            p = self._kf.position
            q_out = self._kf.orientation

            msg = PoseWithCovarianceStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = target_frame
            msg.pose.pose.position.x = float(p[0])
            msg.pose.pose.position.y = float(p[1])
            msg.pose.pose.position.z = float(p[2])
            msg.pose.pose.orientation.x = float(q_out[0])
            msg.pose.pose.orientation.y = float(q_out[1])
            msg.pose.pose.orientation.z = float(q_out[2])
            msg.pose.pose.orientation.w = float(q_out[3])
            pose_cov = np.zeros((6, 6))
            pose_cov[:3, :3] = self._kf.position_covariance
            pose_cov[3:, 3:] = self._kf.rotation_covariance
            msg.pose.covariance = pose_cov.flatten().tolist()
            self._pub_pose.publish(msg)

            tw = TwistWithCovarianceStamped()
            tw.header = msg.header
            tw.twist.twist.linear.x = float(self._kf.velocity[0])
            tw.twist.twist.linear.y = float(self._kf.velocity[1])
            tw.twist.twist.linear.z = float(self._kf.velocity[2])
            tw_cov = np.zeros((6, 6))
            tw_cov[:3, :3] = self._kf.velocity_covariance
            tw_cov[3:, 3:] = 1e6 * np.eye(3)  # angular: no estimate
            tw.twist.covariance = tw_cov.flatten().tolist()
            self._pub_twist.publish(tw)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = target_frame
            tf_msg.child_frame_id = child_frame
            tf_msg.transform.translation.x = float(p[0])
            tf_msg.transform.translation.y = float(p[1])
            tf_msg.transform.translation.z = float(p[2])
            tf_msg.transform.rotation.x = float(q_out[0])
            tf_msg.transform.rotation.y = float(q_out[1])
            tf_msg.transform.rotation.z = float(q_out[2])
            tf_msg.transform.rotation.w = float(q_out[3])
            self._tf_broadcaster.sendTransform(tf_msg)

        self._publish_health()

    def _publish_health(self) -> None:
        now = time.monotonic()
        since_init = 0.0 if not self._kf.is_initialized else (now - self._init_wall_t)
        if self._last_update_wall_t is None:
            since_update = math.inf
        else:
            since_update = now - self._last_update_wall_t
        if self._kf.is_initialized:
            pos_std = float(
                math.sqrt(
                    max(
                        self._kf.covariance[0, 0],
                        self._kf.covariance[1, 1],
                        self._kf.covariance[2, 2],
                    )
                )
            )
        else:
            pos_std = math.inf

        health = classify_health(
            seconds_since_init=since_init,
            seconds_since_last_update=since_update,
            position_std_m=pos_std,
            thresholds=self._thresholds(),
        )

        msg = FilterHealthMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = _enum_to_msg_field(health)
        msg.seconds_since_last_update = (
            0.0 if math.isinf(since_update) else since_update
        )
        msg.position_std_m = 0.0 if math.isinf(pos_std) else pos_std
        self._pub_health.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DockPoseFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
