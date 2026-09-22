#!/usr/bin/env python3
"""coarse_approach: PBVS coarse-approach controller node.

Drives the BlueROV2 to a standoff point on the dock entry axis using the
filtered dock pose. Publishes body-frame cmd_vel + CoarseApproachStatus.
Fixed-rate timer always emits a command (zero when BLOCKED) so ardusub_bridge
never re-sends a stale command."""

import numpy as np
import rclpy
from geometry_msgs.msg import (
    Twist,
    PoseStamped,
    PoseWithCovarianceStamped,
    TwistWithCovarianceStamped,
)
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

from control.pbvs import (
    PbvsController,
    PbvsParams,
    approach_speed_limit,
    ff_authority_ramp,
)
from control import guidance as guidance_lib
from control import health_gate as hg
from interfaces.msg import FilterHealth, CoarseApproachStatus, DockingState
from nav_msgs.msg import Odometry

from control.velocity_loop import (
    MODES,
    VELOCITY_LOOP,
    VelocityLoop,
    VelocityLoopParams,
    effort_from_setpoint,
)

# Low-pass on the standoff-relative speed feeding the velocity-match handoff gate. The
# single-step derivative of even the smoothed dock pose is noisy at 20 Hz; this EMA
# (~3-4 frame memory) plus the ready debounce keeps within_vel from chattering.
_REL_SPEED_EMA_ALPHA = 0.3


class CoarseApproach(Node):
    def __init__(self, **kwargs):
        super().__init__("coarse_approach", **kwargs)

        # Fail fast if the pure mirrors drift from the generated message.
        assert hg.WARMING_UP == FilterHealth.WARMING_UP
        assert hg.HEALTHY == FilterHealth.HEALTHY
        assert hg.DEGRADED == FilterHealth.DEGRADED
        assert hg.STALE == FilterHealth.STALE
        assert hg.APPROACHING == CoarseApproachStatus.APPROACHING
        assert hg.AT_STANDOFF == CoarseApproachStatus.AT_STANDOFF
        assert hg.BLOCKED == CoarseApproachStatus.BLOCKED

        # all values come from coarse_pbvs.yaml; declared by type, no defaults
        ptype = Parameter.Type
        self.declare_parameter("target_frame", ptype.STRING)
        self.declare_parameter("aim_offset_in_dock", ptype.DOUBLE_ARRAY)
        self.declare_parameter("ready_debounce_cycles", ptype.INTEGER)
        for name in (
            "standoff_distance_m",
            "position_tol_m",
            "axis_offset_tol_m",
            "yaw_tol_rad",
            "degraded_gain_scale",
            "control_rate_hz",
            "max_pose_age_s",
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
            "max_twist_age_s",
            "ff_vel_max",
            "ff_ramp_far_m",
            "ff_ramp_near_m",
            "ff_velocity_gain",
            "vel_match_m_s",
        ):
            self.declare_parameter(name, ptype.DOUBLE)
        # Feedforward mode (the fix arm, rev 2026-09-21): open_loop adds the scaled
        # dock velocity to the effort command; velocity_loop treats feedback plus
        # dock velocity as a body-velocity setpoint and closes it on the navigation
        # velocity with a PI, so the plant gain no longer sets the vehicle amplitude.
        self.declare_parameter("feedforward_mode", "open_loop")
        self.declare_parameter("vloop_kp", 0.30)
        self.declare_parameter("vloop_ki", 0.74)
        # effort per m/s applied when navigation velocity is missing (1 / plant gain)
        self.declare_parameter("vloop_fallback_gain", 1.0 / 2.7)
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
        self._vloop_effort_max = VelocityLoopParams().effort_max
        self._latest_odom: Odometry | None = None
        self._latest_odom_t: float | None = None

        # gains are read once here; tolerances are read live each tick
        self._controller = PbvsController(self._params())
        self._ready_counter = 0
        self._ready = False
        self._latest_pose: PoseWithCovarianceStamped | None = None
        self._latest_pose_t: float | None = None
        self._latest_twist: TwistWithCovarianceStamped | None = None
        self._latest_twist_t: float | None = None
        self._latest_health: int | None = None
        self._latest_state: int | None = None
        # standoff-relative velocity estimate for the handoff velocity-match gate
        self._prev_rel_pos_body = None
        self._rel_speed_ema: float | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", qos)
        self._pub_status = self.create_publisher(
            CoarseApproachStatus, "/control/coarse_approach/status", qos
        )
        self._pub_standoff = self.create_publisher(
            PoseStamped, "/control/coarse_approach/standoff_pose", qos
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
        self.create_subscription(
            DockingState, "/docking/state", self._on_state, qos
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("robot_odom_topic").get_parameter_value().string_value,
            self._on_odom,
            qos,
        )

        rate = self.get_parameter("control_rate_hz").get_parameter_value().double_value
        self._dt = 1.0 / rate
        self.create_timer(self._dt, self._tick)
        self.get_logger().info("coarse_approach ready")
        # Assumes the FSM has set ALT_HOLD; in POSHOLD ArduSub fights cmd_vel.
        self.get_logger().warn(
            "coarse_approach assumes ALT_HOLD; horizontal control is undefined "
            "in POSHOLD (ArduSub position-hold fights cmd_vel)"
        )

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

    def _tolerances(self) -> hg.Tolerances:
        gi = lambda n: self.get_parameter(n).get_parameter_value().integer_value
        gd = lambda n: self.get_parameter(n).get_parameter_value().double_value
        return hg.Tolerances(
            position_m=gd("position_tol_m"),
            axis_offset_m=gd("axis_offset_tol_m"),
            yaw_rad=gd("yaw_tol_rad"),
            debounce_cycles=gi("ready_debounce_cycles"),
            vel_match_m_s=gd("vel_match_m_s"),
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
        # negative age = clock jumped back (sim reset); treat as stale
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
        st = CoarseApproachStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.dock_healthy = False
        st.ready_for_handoff = False
        self._pub_status.publish(st)

    def _block(self) -> None:
        # reset clears controller state so a resumed approach has no stale jump
        self._controller.reset()
        self._vloop.reset()
        self._ready_counter = 0
        self._ready = False
        self._prev_rel_pos_body = None
        self._rel_speed_ema = None
        self._publish_zero(CoarseApproachStatus.BLOCKED)

    def _publish_standoff(self) -> None:
        # visualization only: the target standoff pose in the target frame
        p = self._latest_pose.pose.pose
        aim_offset = (
            self.get_parameter("aim_offset_in_dock")
            .get_parameter_value()
            .double_array_value
        )
        standoff = (
            self.get_parameter("standoff_distance_m").get_parameter_value().double_value
        )
        pos, quat = guidance_lib.standoff_pose_in_target(
            dock_pos=(p.position.x, p.position.y, p.position.z),
            dock_quat_xyzw=(
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ),
            aim_offset_in_dock=list(aim_offset),
            standoff_distance_m=standoff,
        )
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._latest_pose.header.frame_id
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._pub_standoff.publish(msg)

    def _tick(self) -> None:
        # Active-phase gate: silent while another phase is active (don't fight it on
        # the shared /cmd_vel). Permissive until the FSM first asserts a state.
        if self._latest_state is not None and self._latest_state != DockingState.COARSE:
            self._controller.reset()
            self._vloop.reset()
            self._ready_counter = 0
            self._ready = False
            self._prev_rel_pos_body = None
            self._rel_speed_ema = None
            return
        if self._latest_pose is not None:
            self._publish_standoff()
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
        )

        gd = lambda n: self.get_parameter(n).get_parameter_value().double_value

        # Dock-velocity feedforward, distance-gated. Far from the standoff the velocity
        # estimate is noisy and the sway is second-order, so authority ramps 0 -> 1 as
        # the ROV closes in (opposite the surge cap). Missing/stale twist -> pure
        # feedback. Rotated world -> body with the SAME rov quaternion as the guidance.
        if self._latest_twist is None or self._twist_too_old():
            ff_vel_body = None
        else:
            lin = self._latest_twist.twist.twist.linear
            v_body = guidance_lib.world_to_body(
                (lin.x, lin.y, lin.z),
                (rov.rotation.x, rov.rotation.y, rov.rotation.z, rov.rotation.w),
            )
            # ff_velocity_gain converts the dock's physical velocity into cmd_vel units
            # (cmd_vel is effort-like, not a velocity servo), so the ff produces the dock
            # velocity at the plant output instead of over-driving by the plant gain.
            # In velocity_loop mode the dock velocity is a physical setpoint, so the
            # gain conversion does not apply; only the distance ramp remains.
            ff_gain = 1.0 if self._velocity_loop_on else gd("ff_velocity_gain")
            ff_scale = ff_gain * ff_authority_ramp(
                g.range_to_standoff_m, gd("ff_ramp_far_m"), gd("ff_ramp_near_m")
            )
            ff_vel_body = ff_scale * v_body

        cmd = self._controller.step(
            g.rel_pos_body, g.yaw_err, self._dt, ff_vel_body=ff_vel_body
        )

        # distance-gated surge cap: slows the approach as it nears the dock
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
            lin = effort_from_setpoint(
                self._vloop,
                lin,
                v_meas,
                self._dt,
                effort_max=self._vloop_effort_max * gate.gain_scale,
                fallback_gain=self.get_parameter("vloop_fallback_gain").get_parameter_value().double_value,
            )

        twist = Twist()
        twist.linear.x = float(lin[0]) * gate.gain_scale
        twist.linear.y = float(lin[1]) * gate.gain_scale
        twist.linear.z = float(lin[2]) * gate.gain_scale
        twist.angular.z = cmd.yaw_rate * gate.gain_scale
        self._pub_cmd.publish(twist)

        tol = self._tolerances()
        within_pos, within_yaw = hg.within_tolerances(
            g.range_to_standoff_m, g.axis_offset_m, g.yaw_err, tol
        )

        # Velocity-match term: |d(rel_pos_body)/dt|, EMA-smoothed. Needs two samples,
        # so the first cycle after (re)acquiring the pose cannot hand off. Guards the
        # handoff against a sway zero-crossing where position is briefly in tolerance
        # but the dock is moving fastest under the vehicle.
        if self._prev_rel_pos_body is None:
            within_vel = False
        else:
            inst = float(
                np.linalg.norm(g.rel_pos_body - self._prev_rel_pos_body)
            ) / self._dt
            self._rel_speed_ema = (
                inst
                if self._rel_speed_ema is None
                else _REL_SPEED_EMA_ALPHA * inst
                + (1.0 - _REL_SPEED_EMA_ALPHA) * self._rel_speed_ema
            )
            within_vel = hg.within_velocity(self._rel_speed_ema, tol.vel_match_m_s)
        self._prev_rel_pos_body = g.rel_pos_body

        phase, ready, self._ready_counter = hg.decide_phase(
            blocked=False,
            within_pos=within_pos,
            within_yaw=within_yaw,
            healthy=gate.dock_healthy,
            ready_counter=self._ready_counter,
            was_ready=self._ready,
            tol=tol,
            within_vel=within_vel,
        )
        self._ready = ready

        st = CoarseApproachStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.range_to_standoff_m = g.range_to_standoff_m
        st.axis_offset_m = g.axis_offset_m
        st.vertical_error_m = g.vertical_error_m
        st.yaw_error_rad = g.yaw_err
        st.within_position_tol = within_pos
        st.within_yaw_tol = within_yaw
        st.dock_healthy = gate.dock_healthy
        st.ready_for_handoff = ready
        self._pub_status.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = CoarseApproach()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
