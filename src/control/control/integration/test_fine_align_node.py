"""Integration test: fine_align node (align-then-advance + gate + seated)."""

import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from interfaces.msg import FilterHealth, FineAlignStatus, DockingState

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
        super().__init__("fine_test_harness")
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/perception/dock_pose_filtered", _RELIABLE
        )
        self._health_pub = self.create_publisher(
            FilterHealth, "/perception/dock_pose_filtered/health", _RELIABLE
        )
        self._state_pub = self.create_publisher(DockingState, "/docking/state", _RELIABLE)
        self.cmds: list[Twist] = []
        self.status: list[FineAlignStatus] = []
        self.create_subscription(Twist, "/cmd_vel", self.cmds.append, _RELIABLE)
        self.create_subscription(
            FineAlignStatus, "/control/fine_align/status", self.status.append, _RELIABLE
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

    def feed(self, dock_x, dock_y, state=DockingState.FINE, health=FilterHealth.HEALTHY,
             dock_qz=0.0, dock_qw=1.0):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.pose.pose.position.x = float(dock_x)
        m.pose.pose.position.y = float(dock_y)
        m.pose.pose.orientation.z = float(dock_qz)
        m.pose.pose.orientation.w = float(dock_qw)
        self._pose_pub.publish(m)
        h = FilterHealth()
        h.status = health
        self._health_pub.publish(h)
        s = DockingState()
        s.state = state
        self._state_pub.publish(s)


def _load_params():
    import os
    import yaml
    from ament_index_python.packages import get_package_share_directory
    from rclpy.parameter import Parameter

    cfg = os.path.join(
        get_package_share_directory("control"), "config", "fine_pbvs.yaml"
    )
    with open(cfg) as f:
        values = yaml.safe_load(f)["fine_align"]["ros__parameters"]
    return [Parameter(name=k, value=v) for k, v in values.items() if k != "use_sim_time"]


def _run(node, harness, feed_fn, *, iterations, period=0.05):
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
        feed_fn()
        time.sleep(period)
    stop.set()
    t.join(timeout=1.0)


def test_silent_when_not_fine(ros_context):
    from control.fine_align_node import FineAlign

    node = FineAlign(parameter_overrides=_load_params())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)
    _run(node, harness, lambda: harness.feed(0.5, 0.0, state=DockingState.COARSE),
         iterations=12)
    assert harness.cmds == [], "fine must be silent unless FINE is active"
    node.destroy_node()
    harness.destroy_node()


def test_no_surge_while_laterally_misaligned(ros_context):
    from control.fine_align_node import FineAlign

    node = FineAlign(parameter_overrides=_load_params())
    harness = _Harness()
    # ROV at origin facing +x; dock 0.5 ahead but 0.4 to the side -> misaligned
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)
    _run(node, harness, lambda: harness.feed(0.5, 0.4), iterations=15)
    assert harness.cmds, "expected cmd_vel on the timer"
    assert all(abs(c.linear.x) < 1e-9 for c in harness.cmds), \
        "surge must stay zero while misaligned (align-then-advance)"
    assert any(abs(c.linear.y) > 0.0 for c in harness.cmds), \
        "expected sway to centre the vehicle"
    node.destroy_node()
    harness.destroy_node()


def test_surge_when_aligned(ros_context):
    from control.fine_align_node import FineAlign

    node = FineAlign(parameter_overrides=_load_params())
    harness = _Harness()
    # ROV at origin facing +x. Dock 0.5 ahead, oriented -90deg about Z so its
    # boresight (dock +Y) points along world +X -> the ROV heading is aligned
    # with the dock entry axis (boresight yaw ~ 0) -> surge allowed.
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)
    qz, qw = -0.7071067811865476, 0.7071067811865476
    _run(node, harness, lambda: harness.feed(0.5, 0.0, dock_qz=qz, dock_qw=qw),
         iterations=15)
    assert any(c.linear.x > 0.0 for c in harness.cmds), \
        "expected forward surge once aligned"
    node.destroy_node()
    harness.destroy_node()


def test_blocks_on_stale(ros_context):
    from control.fine_align_node import FineAlign

    node = FineAlign(parameter_overrides=_load_params())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)
    _run(node, harness, lambda: harness.feed(0.5, 0.0, health=FilterHealth.STALE),
         iterations=12)
    assert harness.cmds and harness.cmds[-1].linear.x == 0.0
    assert harness.status and harness.status[-1].phase == FineAlignStatus.BLOCKED
    node.destroy_node()
    harness.destroy_node()


def test_ff_absent_falls_back_to_feedback(ros_context):
    """With FINE active, HEALTHY, a fresh pose, and NO velocity topic ever published,
    the node must degrade to pure feedback: still commands (nonzero sway to centre a
    lateral offset) and never crashes on the missing feedforward. Guards the
    'missing/stale twist -> ff = None' contract."""
    from control.fine_align_node import FineAlign

    node = FineAlign(parameter_overrides=_load_params())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)
    # laterally offset dock; the harness never publishes the velocity twist
    _run(node, harness, lambda: harness.feed(0.5, 0.4), iterations=15)
    assert harness.cmds, "expected cmd_vel on the timer"
    assert any(abs(c.linear.y) > 0.0 for c in harness.cmds), \
        "feedback must still centre the vehicle with no velocity topic (ff -> None)"
    node.destroy_node()
    harness.destroy_node()
