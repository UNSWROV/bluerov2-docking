"""Integration test: coarse_approach node.

Publishes synthetic dock_pose_filtered + health + a static TF map->base_link,
asserts cmd_vel and CoarseApproachStatus behave per phase."""

import math
import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Twist,
    TransformStamped,
    TwistWithCovarianceStamped,
)
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from interfaces.msg import FilterHealth, CoarseApproachStatus

_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Harness(Node):
    def __init__(self):
        super().__init__("coarse_test_harness")
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/perception/dock_pose_filtered", _RELIABLE
        )
        self._health_pub = self.create_publisher(
            FilterHealth, "/perception/dock_pose_filtered/health", _RELIABLE
        )
        self._twist_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            "/perception/dock_pose_filtered/velocity",
            _RELIABLE,
        )
        self.cmds: list[Twist] = []
        self.status: list[CoarseApproachStatus] = []
        self.create_subscription(Twist, "/cmd_vel", self.cmds.append, _RELIABLE)
        self.create_subscription(
            CoarseApproachStatus,
            "/control/coarse_approach/status",
            self.status.append,
            _RELIABLE,
        )
        self._tf = StaticTransformBroadcaster(self)

    def send_tf(self, x, y, z, qx, qy, qz, qw):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = float(z)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        self._tf.sendTransform(tf)

    def publish_dock(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        m = PoseWithCovarianceStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.pose.pose.position.x = float(x)
        m.pose.pose.position.y = float(y)
        m.pose.pose.position.z = float(z)
        m.pose.pose.orientation.x = float(qx)
        m.pose.pose.orientation.y = float(qy)
        m.pose.pose.orientation.z = float(qz)
        m.pose.pose.orientation.w = float(qw)
        self._pose_pub.publish(m)

    def publish_health(self, status):
        h = FilterHealth()
        h.header.stamp = self.get_clock().now().to_msg()
        h.status = status
        self._health_pub.publish(h)

    def publish_twist(self, vx, vy, vz):
        m = TwistWithCovarianceStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.twist.twist.linear.x = float(vx)
        m.twist.twist.linear.y = float(vy)
        m.twist.twist.linear.z = float(vz)
        self._twist_pub.publish(m)


def _spin(*nodes, seconds):
    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [
            ex.spin_once(timeout_sec=0.02)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    t.start()
    time.sleep(seconds)
    stop.set()
    t.join(timeout=1.0)


def _load_params():
    """Load coarse_pbvs.yaml (the single source of truth) as parameter overrides
    so the node -- which declares parameters by type with no in-code defaults --
    can be constructed in tests."""
    import os

    import yaml
    from ament_index_python.packages import get_package_share_directory
    from rclpy.parameter import Parameter

    cfg = os.path.join(
        get_package_share_directory("control"), "config", "coarse_pbvs.yaml"
    )
    with open(cfg) as f:
        values = yaml.safe_load(f)["coarse_approach"]["ros__parameters"]
    return [
        Parameter(name=k, value=v) for k, v in values.items() if k != "use_sim_time"
    ]


def test_blocked_and_zero_cmd_before_any_pose(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    _spin(node, harness, seconds=1.0)

    assert harness.cmds, "expected cmd_vel to be published on the timer"
    last = harness.cmds[-1]
    assert last.linear.x == 0.0 and last.angular.z == 0.0
    assert harness.status and harness.status[-1].phase == CoarseApproachStatus.BLOCKED

    node.destroy_node()
    harness.destroy_node()


def _spin_while_feeding(node, harness, feed, *, iterations, period):
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(harness)
    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [
            ex.spin_once(timeout_sec=0.02)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    t.start()
    for _ in range(iterations):
        feed()
        time.sleep(period)
    stop.set()
    t.join(timeout=1.0)


def test_approaches_when_healthy_and_off_target(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)

    def feed():
        harness.publish_dock(3.0, 0.0, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)

    _spin_while_feeding(node, harness, feed, iterations=20, period=0.05)

    moving = [c for c in harness.cmds if c.linear.x > 0.0]
    assert moving, "expected positive surge toward the dock"
    assert any(s.phase == CoarseApproachStatus.APPROACHING for s in harness.status)
    node.destroy_node()
    harness.destroy_node()


def test_handoff_latches_when_dock_settles(ros_context):
    """Rev 2026-07-22: a settled, in-tolerance dock latches the handoff. ROV faces
    world +Y; dock at (0,1,0) puts the standoff (dock + (0,-1,0)) on the ROV, so
    position + yaw are in tolerance and, with a static dock, the error-derivative
    decays to zero -> within_vel True -> ready_for_handoff after the debounce."""
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    qz, qw = math.sin(math.pi / 4), math.cos(math.pi / 4)  # ROV faces world +Y
    harness.send_tf(0, 0, 0, 0, 0, qz, qw)

    def feed():
        harness.publish_dock(0.0, 1.0, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)
        harness.publish_twist(0.0, 0.0, 0.0)  # settled

    _spin_while_feeding(node, harness, feed, iterations=40, period=0.05)

    assert any(s.ready_for_handoff for s in harness.status), \
        "a settled, in-tolerance dock must eventually latch the handoff"
    node.destroy_node()
    harness.destroy_node()


def test_velocity_mismatch_blocks_handoff(ros_context):
    """Rev 2026-07-22: the velocity-match term keeps the handoff from latching while
    the dock still moves under the vehicle, even though position is in tolerance. The
    dock jitters +/-5 cm about (0,1,0): range stays inside the 10 cm position tol, but
    the standoff-relative speed stays high, so ready_for_handoff must never latch."""
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    qz, qw = math.sin(math.pi / 4), math.cos(math.pi / 4)
    harness.send_tf(0, 0, 0, 0, 0, qz, qw)

    toggle = [0]

    def feed():
        toggle[0] ^= 1
        dy = 0.05 if toggle[0] else -0.05
        harness.publish_dock(0.0, 1.0 + dy, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)
        harness.publish_twist(0.0, 0.0, 0.0)

    _spin_while_feeding(node, harness, feed, iterations=40, period=0.05)

    assert harness.status, "expected status telemetry"
    assert not any(s.ready_for_handoff for s in harness.status), \
        "a dock still moving under the vehicle must NOT latch the handoff"
    # position IS in tolerance -> proves the block is the velocity gate, not position
    assert any(s.within_position_tol for s in harness.status)
    node.destroy_node()
    harness.destroy_node()


def test_feedforward_biases_command_toward_dock_velocity(ros_context):
    """Rev 2026-07-22: coarse feeds the dock velocity forward. ROV sits at the standoff
    (feedback ~0) facing world +Y; the dock slides along world +X at 0.1 m/s. For a
    ROV facing +Y, world +X is body -Y, so the wired feedforward should command a
    steady negative sway ~ -0.1 (distance ramp is saturated at the standoff)."""
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    qz, qw = math.sin(math.pi / 4), math.cos(math.pi / 4)  # ROV faces world +Y
    harness.send_tf(0, 0, 0, 0, 0, qz, qw)

    def feed():
        harness.publish_dock(0.0, 1.0, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)
        harness.publish_twist(0.1, 0.0, 0.0)  # dock slides along world +X

    _spin_while_feeding(node, harness, feed, iterations=30, period=0.05)

    sways = sorted(c.linear.y for c in harness.cmds[-10:])
    assert sways, "expected cmd_vel"
    median_sway = sways[len(sways) // 2]
    # ROV faces +Y, dock slides world +X -> ff drives body -Y (sway), scaled by
    # ff_velocity_gain (cmd_vel is effort-like, so ff is converted to command units).
    # Assert the wiring + sign + order of magnitude; the exact scale is a tuning param.
    assert -0.15 < median_sway < -0.02, \
        f"feedforward should drive a modest negative sway (got {median_sway})"
    node.destroy_node()
    harness.destroy_node()


def test_blocks_on_stale_health(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)

    def feed():
        harness.publish_dock(3.0, 0.0, 0.0)
        harness.publish_health(FilterHealth.STALE)

    _spin_while_feeding(node, harness, feed, iterations=15, period=0.05)

    assert harness.cmds, "expected cmd_vel to be published on the timer"
    assert harness.cmds[-1].linear.x == 0.0
    assert harness.status[-1].phase == CoarseApproachStatus.BLOCKED
    node.destroy_node()
    harness.destroy_node()
