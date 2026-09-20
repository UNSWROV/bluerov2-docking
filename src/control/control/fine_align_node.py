#!/usr/bin/env python3
"""fine_align: align-then-advance PBVS controller for terminal docking.

Takes over from coarse at the standoff point and drives the BlueROV2 onto the
dock entry. Reuses the coarse PBVS regulator and health gate; the new behaviour
is the align-then-advance guidance law (do not surge forward until laterally,
vertically and angularly centred). Self-gates /cmd_vel on /docking/state so it
only drives while the FSM has FINE active. Publishes FineAlignStatus telemetry,
including a debounced `seated` flag the FSM reads to advance to DOCKED.
"""

import rclpy
from geometry_msgs.msg import (
    Twist,
    PoseWithCovarianceStamped,
    TwistWithCovarianceStamped,
)
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

from control.pbvs import PbvsController, PbvsParams, approach_speed_limit
from control import guidance as guidance_lib
from control import health_gate as hg
from control import fine_guidance as fg
from interfaces.msg import FilterHealth, FineAlignStatus, DockingState
import numpy as np
from nav_msgs.msg import Odometry

from control.velocity_loop import (
    MODES,
    VELOCITY_LOOP,
    VelocityLoop,
    VelocityLoopParams,
    effort_from_setpoint,
)


class FineAlign(Node):
    def __init__(self, **kwargs):
        super().__init__("fine_align", **kwargs)

        # Fail fast if the pure mirrors drift from the generated messages.
        assert hg.WARMING_UP == FilterHealth.WARMING_UP
        assert hg.HEALTHY == FilterHealth.HEALTHY
        assert hg.DEGRADED == FilterHealth.DEGRADED
        assert hg.STALE == FilterHealth.STALE
        assert fg.ALIGNING == FineAlignStatus.ALIGNING
        assert fg.SEATED == FineAlignStatus.SEATED
        assert fg.BLOCKED == FineAlignStatus.BLOCKED

        # all values come from fine_pbvs.yaml; declared by type, no defaults
        ptype = Parameter.Type
        self.declare_parameter("target_frame", ptype.STRING)
        self.declare_parameter("aim_offset_in_dock", ptype.DOUBLE_ARRAY)
        self.declare_parameter("seated_debounce_cycles", ptype.INTEGER)
        for name in (
            "standoff_distance_m",
            "align_lateral_tol_m",
            "align_vertical_tol_m",
            "align_yaw_tol_rad",
            "seated_range_m",
            "seated_lateral_tol_m",
            "seated_vertical_tol_m",
            "seated_yaw_tol_rad",
            "degraded_gain_scale",
            "control_rate_hz",
            "max_pose_age_s",
            "max_twist_age_s",
            "kp_surge",
            "kd_surge",
            "kp_sway",
            "kd_sway",
            "kp_heave",
            "kd_heave",
            "kp_yaw",
            "kd_yaw",
            "v_max_surge",
            "v_max_sway",
            "v_max_heave",
            "v_max_yaw",
            "approach_speed_slope",
            "approach_speed_floor",
            "ff_vel_max",
        ):
            self.declare_parameter(name, ptype.DOUBLE)
        # Feedforward mode (the fix arm, rev 2026-09-21); see coarse_approach_node.
        self.declare_parameter("feedforward_mode", "open_loop")
        self.declare_parameter("vloop_kp", 0.30)
        self.declare_parameter("vloop_ki", 0.74)
        self.declare_parameter("robot_odom_topic", "/model/bluerov2_heavy/odometry")
        mode = self.get_parameter("feedforward_mode").get_parameter_value().string_value
        if mode not in MODES:
            raise ValueError(f"feedforward_mode must be one of {MODES}, got {mode!r}")
        self._velocity_loop_on = mode == VELOCITY_LOOP
        self._vloop = VelocityLoop(
            VelocityLoopParams(
                kp=self.get_parameter("vloop_kp").get_parameter_value().double_value,
                ki=self.get_parameter("vloop_ki").get_parameter_value().double_value,
            )
        )
        self._latest_odom: Odometry | None = None
        self._latest_odom_t: float | None = None

        self._controller = PbvsController(self._params())
        self._seated_counter = 0
        self._seated = False
        self._latest_pose: PoseWithCovarianceStamped | None = None
        self._latest_pose_t: float | None = None
        self._latest_twist: TwistWithCovarianceStamped | None = None
        self._latest_twist_t: float | None = None
        self._latest_health: int | None = None
        self._latest_state: int | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", qos)
        self._pub_status = self.create_publisher(
            FineAlignStatus, "/control/fine_align/status", qos
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/perception/dock_pose_filtered",
            self._on_pose,
            qos,
        )
        self.create_subscription(
            TwistWithCovarianceStamped,
            "/perception/dock_pose_filtered/velocity",
            self._on_twist,
            qos,
        )
        self.create_subscription(
            FilterHealth, "/perception/dock_pose_filtered/health", self._on_health, qos
        )
        self.create_subscription(DockingState, "/docking/state", self._on_state, qos)
        self.create_subscription(
            Odometry,
            self.get_parameter("robot_odom_topic").get_parameter_value().string_value,
            self._on_odom,
            qos,
        )

        rate = self.get_parameter("control_rate_hz").get_parameter_value().double_value
        self._dt = 1.0 / rate
        self.create_timer(self._dt, self._tick)
        self.get_logger().info("fine_align ready")

    def _params(self) -> PbvsParams:
        g = lambda n: self.get_parameter(n).get_parameter_value().double_value
        return PbvsParams(
            kp_surge=g("kp_surge"),
            kd_surge=g("kd_surge"),
            kp_sway=g("kp_sway"),
            kd_sway=g("kd_sway"),
            kp_heave=g("kp_heave"),
            kd_heave=g("kd_heave"),
            kp_yaw=g("kp_yaw"),
            kd_yaw=g("kd_yaw"),
            v_max_surge=g("v_max_surge"),
            v_max_sway=g("v_max_sway"),
            v_max_heave=g("v_max_heave"),
            v_max_yaw=g("v_max_yaw"),
            ff_vel_max=g("ff_vel_max"),
        )

    def _align_tol(self) -> fg.AlignTol:
        gd = lambda n: self.get_parameter(n).get_parameter_value().double_value
        return fg.AlignTol(
            lateral_m=gd("align_lateral_tol_m"),
            vertical_m=gd("align_vertical_tol_m"),
            yaw_rad=gd("align_yaw_tol_rad"),
        )

    def _seated_tol(self) -> fg.SeatedTol:
        gd = lambda n: self.get_parameter(n).get_parameter_value().double_value
        gi = lambda n: self.get_parameter(n).get_parameter_value().integer_value
        return fg.SeatedTol(
            range_m=gd("seated_range_m"),
            lateral_m=gd("seated_lateral_tol_m"),
            vertical_m=gd("seated_vertical_tol_m"),
            yaw_rad=gd("seated_yaw_tol_rad"),
            debounce_cycles=gi("seated_debounce_cycles"),
        )

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._latest_pose = msg
        self._latest_pose_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_twist(self, msg: TwistWithCovarianceStamped) -> None:
        self._latest_twist = msg
        self._latest_twist_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_health(self, msg: FilterHealth) -> None:
        self._latest_health = int(msg.status)

    def _on_state(self, msg: DockingState) -> None:
        self._latest_state = int(msg.state)

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg
        self._latest_odom_t = self.get_clock().now().nanoseconds * 1e-9

    def _body_velocity(self) -> np.ndarray | None:
        """Navigation linear velocity in the body frame (Odometry twist is in
        child_frame_id = base_link), or None when missing or older than the twist
        age limit."""
        if self._latest_odom is None or self._latest_odom_t is None:
            return None
        age = self.get_clock().now().nanoseconds * 1e-9 - self._latest_odom_t
        max_age = (
            self.get_parameter("max_twist_age_s").get_parameter_value().double_value
        )
        if age < 0.0 or age > max_age:
            return None
        lin = self._latest_odom.twist.twist.linear
        return np.array([lin.x, lin.y, lin.z])

    def _pose_too_old(self) -> bool:
        if self._latest_pose_t is None:
            return True
        age = self.get_clock().now().nanoseconds * 1e-9 - self._latest_pose_t
        max_age = (
            self.get_parameter("max_pose_age_s").get_parameter_value().double_value
        )
        return age < 0.0 or age > max_age

    def _twist_too_old(self) -> bool:
        if self._latest_twist_t is None:
            return True
        age = self.get_clock().now().nanoseconds * 1e-9 - self._latest_twist_t
        max_age = (
            self.get_parameter("max_twist_age_s").get_parameter_value().double_value
        )
        return age < 0.0 or age > max_age

    def _publish_zero(self, phase: int) -> None:
        self._pub_cmd.publish(Twist())
        st = FineAlignStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.dock_healthy = False
        st.aligned = False
        st.seated = False
        self._pub_status.publish(st)

    def _block(self) -> None:
        # reset clears controller state so a resumed approach has no stale jump
        self._controller.reset()
        self._vloop.reset()
        self._seated_counter = 0
        self._seated = False
        self._publish_zero(FineAlignStatus.BLOCKED)

    def _tick(self) -> None:
        # Fail-safe active-phase gate: drive only when the FSM has explicitly
        # asserted FINE (unknown state counts as not-FINE), so the terminal
        # controller stays silent until the phase is confirmed. Coarse is permissive.
        if self._latest_state != DockingState.FINE:
            self._controller.reset()
            self._seated_counter = 0
            self._seated = False
            return

        if (
            self._latest_pose is None
            or self._latest_health is None
            or self._pose_too_old()
        ):
            self._block()
            return

        scale = (
            self.get_parameter("degraded_gain_scale").get_parameter_value().double_value
        )
        gate = hg.gate_for_health(self._latest_health, scale)
        if gate.blocked:
            self._block()
            return

        target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        try:
            # Time() = latest transform; staleness is fine for coarse approach
            tf = self._tf_buffer.lookup_transform(target_frame, "base_link", Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {target_frame}->base_link unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            self._block()
            return

        p = self._latest_pose.pose.pose
        rov = tf.transform
        aim_offset = (
            self.get_parameter("aim_offset_in_dock")
            .get_parameter_value()
            .double_array_value
        )
        standoff = (
            self.get_parameter("standoff_distance_m").get_parameter_value().double_value
        )

        g = guidance_lib.compute_guidance(
            dock_pos=(p.position.x, p.position.y, p.position.z),
            dock_quat_xyzw=(
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ),
            rov_pos=(rov.translation.x, rov.translation.y, rov.translation.z),
            rov_quat_xyzw=(
                rov.rotation.x,
                rov.rotation.y,
                rov.rotation.z,
                rov.rotation.w,
            ),
            aim_offset_in_dock=list(aim_offset),
            standoff_distance_m=standoff,
            yaw_to_boresight=True,
        )

        # Dock-velocity feedforward. Missing or stale twist -> None (pure feedback).
        # Fine passes full authority (it operates at close range where a coarse-style
        # distance ramp would already be ~1). Rotate world -> body with the SAME rov
        # quaternion compute_guidance used. Health scaling rides the gain_scale multiply
        # below, which now scales feedback + feedforward together.
        if self._latest_twist is None or self._twist_too_old():
            ff_vel_body = None
        else:
            lin = self._latest_twist.twist.twist.linear
            ff_vel_body = guidance_lib.world_to_body(
                (lin.x, lin.y, lin.z),
                (rov.rotation.x, rov.rotation.y, rov.rotation.z, rov.rotation.w),
            )

        cmd = self._controller.step(
            g.rel_pos_body, g.yaw_err, self._dt, ff_vel_body=ff_vel_body
        )
        is_aligned = fg.aligned(g.rel_pos_body, g.yaw_err, self._align_tol())
        cmd = fg.advance_command(cmd, is_aligned)

        # distance-gated surge cap: slows the approach as it nears the dock
        gd = lambda n: self.get_parameter(n).get_parameter_value().double_value
        surge_cap = approach_speed_limit(
            g.range_to_dock_m,
            gd("approach_speed_slope"),
            gd("approach_speed_floor"),
            gd("v_max_surge"),
        )
        surge = max(-surge_cap, min(cmd.surge, surge_cap))

        lin = np.array([surge, cmd.sway, cmd.heave])
        if self._velocity_loop_on:
            v_meas = self._body_velocity()
            if v_meas is None:
                self.get_logger().warn(
                    "velocity loop: no navigation velocity, passing the setpoint "
                    "through as effort",
                    throttle_duration_sec=2.0,
                )
            lin = effort_from_setpoint(self._vloop, lin, v_meas, self._dt)

        twist = Twist()
        twist.linear.x = float(lin[0]) * gate.gain_scale
        twist.linear.y = float(lin[1]) * gate.gain_scale
        twist.linear.z = float(lin[2]) * gate.gain_scale
        twist.angular.z = cmd.yaw_rate * gate.gain_scale
        self._pub_cmd.publish(twist)

        stol = self._seated_tol()
        within = fg.within_seated(g.range_to_dock_m, g.rel_pos_body, g.yaw_err, stol)
        phase, self._seated, self._seated_counter = fg.decide_seated(
            within,
            gate.dock_healthy,
            self._seated_counter,
            self._seated,
            stol.debounce_cycles,
        )

        st = FineAlignStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.range_to_dock_m = g.range_to_dock_m
        st.lateral_error_m = g.rel_pos_body[1]
        st.vertical_error_m = g.rel_pos_body[2]
        st.yaw_error_rad = g.yaw_err
        st.aligned = is_aligned
        st.seated = self._seated
        st.dock_healthy = gate.dock_healthy
        self._pub_status.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = FineAlign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
