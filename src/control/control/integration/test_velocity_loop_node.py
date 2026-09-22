"""Integration test: the velocity-closed feedforward mode of coarse_approach.
Reuses the coarse harness and adds a navigation odometry publisher and a docking
state publisher, and checks the three node-level behaviours the pure loop tests
cannot see: the loop drives effort against the measured velocity, a missing
odometry degrades to the plant-scaled open-loop command, and leaving the coarse
phase resets the loop's integral."""
import math
import time

import pytest
import rclpy
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from interfaces.msg import DockingState, FilterHealth

from control.integration.test_coarse_approach_node import (
    _RELIABLE,
    _Harness,
    _load_params,
    _spin_while_feeding,
)
from control.velocity_loop import FALLBACK_GAIN


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _VloopHarness(_Harness):
    def __init__(self):
        super().__init__()
        self._odom_pub = self.create_publisher(
            Odometry, "/model/bluerov2_heavy/odometry", _RELIABLE
        )
        self._state_pub = self.create_publisher(DockingState, "/docking/state", _RELIABLE)

    def publish_odom(self, vx, vy, vz):
        m = Odometry()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.child_frame_id = "base_link"
        m.twist.twist.linear.x = float(vx)
        m.twist.twist.linear.y = float(vy)
        m.twist.twist.linear.z = float(vz)
        self._odom_pub.publish(m)

    def publish_state(self, state):
        m = DockingState()
        m.state = int(state)
        self._state_pub.publish(m)


def _vloop_params():
    return _load_params() + [Parameter(name="feedforward_mode", value="velocity_loop")]


def _standoff_setup(harness):
    """ROV at the standoff facing world +Y; the dock slides along world +X at 0.1 m/s,
    which is body -Y, so the setpoint is a steady negative sway of about 0.1 m/s."""
    qz, qw = math.sin(math.pi / 4), math.cos(math.pi / 4)
    harness.send_tf(0, 0, 0, 0, 0, qz, qw)


def _feed(harness, odom):
    def feed():
        harness.publish_dock(0.0, 1.0, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)
        harness.publish_twist(0.1, 0.0, 0.0)
        harness.publish_state(DockingState.COARSE)
        if odom is not None:
            harness.publish_odom(*odom)
    return feed


def test_loop_drives_effort_against_the_measured_velocity(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_vloop_params())
    harness = _VloopHarness()
    _standoff_setup(harness)
    # the vehicle reports zero velocity, so the loop's error stays at the full setpoint
    _spin_while_feeding(node, harness, _feed(harness, (0.0, 0.0, 0.0)), iterations=40, period=0.05)
    sways = [c.linear.y for c in harness.cmds]
    assert len(sways) > 10
    early = sorted(sways[5:10])[2]
    late = sorted(sways[-5:])[2]
    # negative (toward body -Y), proportional part alone is kp * 0.1 = 0.03, and the
    # integral makes it grow while the error persists, up to the effort bound
    assert late < -0.03, f"loop should drive negative sway beyond the proportional term (got {late})"
    assert late < early, f"integral should grow the effort while the error persists ({early} -> {late})"
    assert late >= -0.5
    node.destroy_node()
    harness.destroy_node()


def test_missing_odometry_degrades_to_the_plant_scaled_open_loop_command(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_vloop_params())
    harness = _VloopHarness()
    _standoff_setup(harness)
    _spin_while_feeding(node, harness, _feed(harness, None), iterations=30, period=0.05)
    sways = sorted(c.linear.y for c in harness.cmds[-10:])
    median = sways[len(sways) // 2]
    expected = -0.1 * FALLBACK_GAIN
    assert abs(median - expected) < 0.02, f"fallback should be setpoint x 1/plant gain ({expected}), got {median}"
    node.destroy_node()
    harness.destroy_node()


def test_leaving_the_coarse_phase_resets_the_integral(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_vloop_params())
    harness = _VloopHarness()
    _standoff_setup(harness)
    _spin_while_feeding(node, harness, _feed(harness, (0.0, 0.0, 0.0)), iterations=40, period=0.05)
    before = sorted(c.linear.y for c in harness.cmds[-5:])[2]
    harness.cmds.clear()

    def fine_feed():
        harness.publish_dock(0.0, 1.0, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)
        harness.publish_twist(0.1, 0.0, 0.0)
        harness.publish_state(DockingState.FINE)
        harness.publish_odom(0.0, 0.0, 0.0)

    _spin_while_feeding(node, harness, fine_feed, iterations=10, period=0.05)
    harness.cmds.clear()
    _spin_while_feeding(node, harness, _feed(harness, (0.0, 0.0, 0.0)), iterations=6, period=0.05)
    first_back = [c.linear.y for c in harness.cmds if c.linear.y != 0.0][:2]
    assert first_back, "expected commands after re-entering COARSE"
    # a wound-up integral would resume at `before`; a reset loop restarts near the
    # proportional term alone
    assert abs(first_back[0]) < abs(before) - 0.01, f"integral not reset on phase exit ({before} -> {first_back[0]})"
    node.destroy_node()
    harness.destroy_node()
