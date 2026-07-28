#!/usr/bin/env python3
"""docking_fsm: YASMIN orchestrator for the BlueROV2 docking mission.

Sequences IDLE -> COARSE -> FINE -> DOCKED. Has no velocity authority: it only
publishes the active phase on /docking/state (which the phase controllers
self-gate on) and drives flight mode / arming via VehicleIO. Transition policy
lives in the pure orchestrator.transitions helpers; this node is the ROS + YASMIN
wiring around them.
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool
from yasmin import State, StateMachine, Blackboard
from yasmin_viewer import YasminViewerPub

from interfaces.msg import (
    CoarseApproachStatus,
    FineAlignStatus,
    FilterHealth,
    DockingState as DockingStateMsg,
)
from orchestrator import transitions as tr
from orchestrator.vehicle_io import VehicleIO


class Outcome:
    ENGAGE = "engage"
    DISENGAGE = "disengage"
    HANDOFF = "handoff"
    SEATED = "seated"
    DEMOTE = "demote"
    FINISHED = "finished"


class StateName:
    IDLE = "idle"
    COARSE = "coarse"
    FINE = "fine"
    DOCKED = "docked"


class DockingFSM(Node):
    def __init__(self, vehicle_io=None, **kwargs):
        super().__init__("docking_fsm", **kwargs)

        # Fail fast if the pure mirrors drift from the generated message.
        assert tr.WARMING_UP == FilterHealth.WARMING_UP
        assert tr.HEALTHY == FilterHealth.HEALTHY
        assert tr.DEGRADED == FilterHealth.DEGRADED
        assert tr.STALE == FilterHealth.STALE

        ptype = Parameter.Type
        self.declare_parameter("tick_rate_hz", ptype.DOUBLE)
        self.declare_parameter("loss_timeout_cycles", ptype.INTEGER)
        self.declare_parameter("drift_timeout_cycles", ptype.INTEGER)
        self.declare_parameter("demote_range_m", ptype.DOUBLE)
        self.declare_parameter("coarse_mode", ptype.STRING)
        self.declare_parameter("fine_mode", ptype.STRING)
        self.declare_parameter("idle_mode", ptype.STRING)
        # Visualization only. Disabled in tests, where the viewer's timer and
        # publisher race node teardown and segfault.
        self.declare_parameter("enable_viewer", True)

        # Injected in tests (a fake); a real MAVROS adapter otherwise.
        self.vehicle_io = vehicle_io if vehicle_io is not None else VehicleIO(self)

        # Latest telemetry, written by subscriptions, read by the FSM states.
        self._engaged = False
        self._coarse_ready = False
        self._fine_seated = False
        self._fine_range = 0.0
        self._health = FilterHealth.HEALTHY

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_state = self.create_publisher(DockingStateMsg, "/docking/state", qos)
        self.create_subscription(
            CoarseApproachStatus,
            "/control/coarse_approach/status",
            self._on_coarse,
            qos,
        )
        self.create_subscription(
            FineAlignStatus, "/control/fine_align/status", self._on_fine, qos
        )
        self.create_subscription(
            FilterHealth, "/perception/dock_pose_filtered/health", self._on_health, qos
        )
        self.create_subscription(Bool, "/docking/engaged", self._on_engaged, qos)

        self.get_logger().info("docking_fsm ready")
        self._start_fsm()

    def publish_state(self, state: int, label: str) -> None:
        msg = DockingStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.label = label
        self._pub_state.publish(msg)

    def param_int(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def param_double(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def param_str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _on_coarse(self, msg: CoarseApproachStatus) -> None:
        self._coarse_ready = bool(msg.ready_for_handoff)

    def _on_fine(self, msg: FineAlignStatus) -> None:
        self._fine_seated = bool(msg.seated)
        self._fine_range = float(msg.range_to_dock_m)

    def _on_health(self, msg: FilterHealth) -> None:
        self._health = int(msg.status)

    def _on_engaged(self, msg: Bool) -> None:
        self._engaged = bool(msg.data)

    def _start_fsm(self) -> None:
        self._stop = threading.Event()
        sm = StateMachine(outcomes=[Outcome.FINISHED])  # nominal, unreachable
        sm.add_state(StateName.IDLE, IdleState(self),
                     transitions={Outcome.ENGAGE: StateName.COARSE})
        sm.add_state(StateName.COARSE, CoarseState(self),
                     transitions={Outcome.DISENGAGE: StateName.IDLE,
                                  Outcome.HANDOFF: StateName.FINE})
        sm.add_state(StateName.FINE, FineState(self),
                     transitions={Outcome.DISENGAGE: StateName.IDLE,
                                  Outcome.SEATED: StateName.DOCKED,
                                  Outcome.DEMOTE: StateName.COARSE})
        sm.add_state(StateName.DOCKED, DockedState(self),
                     transitions={Outcome.DISENGAGE: StateName.IDLE})
        sm.set_start_state(StateName.IDLE)

        def _run():
            # A state loop that exits on shutdown falls off to a None outcome;
            # swallow that here so the daemon thread ends quietly during teardown.
            try:
                sm(Blackboard())
            except Exception as exc:
                if self._stop.is_set() or not rclpy.ok():
                    self.get_logger().debug(f"FSM thread ended during shutdown: {exc}")
                else:
                    self.get_logger().error(f"FSM supervisor thread died: {exc}")

        self._fsm_thread = threading.Thread(target=_run, daemon=True)
        self._fsm_thread.start()
        if self.get_parameter("enable_viewer").get_parameter_value().bool_value:
            YasminViewerPub(sm, "DOCKING", node=self)

    def destroy_node(self):
        # Stop the FSM thread before tearing down ROS objects: otherwise a state
        # loop can publish on a destroyed node during shutdown and segfault.
        if getattr(self, "_stop", None) is not None:
            self._stop.set()
        if getattr(self, "_fsm_thread", None) is not None:
            self._fsm_thread.join(timeout=2.0)
        super().destroy_node()


class IdleState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=[Outcome.ENGAGE])
        self._node = node

    def execute(self, blackboard) -> str:  # type: ignore
        # Re-arm on idle so manual control works after a DOCKED disarm. A single
        # arming call can fail transiently (mavros command service ack timeouts),
        # which left the vehicle disarmed and deaf to the sticks, so IDLE retries
        # periodically instead of fire-and-forget. Re-arming an armed vehicle is a
        # no-op, so the retry is also self-healing against a later auto-disarm.
        self._node.vehicle_io.set_mode(self._node.param_str("idle_mode"))
        period = 1.0 / self._node.param_double("tick_rate_hz")
        arm_retry_s = 3.0
        next_arm_t = 0.0
        while rclpy.ok() and not self._node._stop.is_set():
            now = time.monotonic()
            if now >= next_arm_t:
                self._node.vehicle_io.set_arm(True)
                next_arm_t = now + arm_retry_s
            self._node.publish_state(DockingStateMsg.IDLE, "IDLE")
            if self._node._engaged:
                return Outcome.ENGAGE
            time.sleep(period)


class CoarseState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=[Outcome.DISENGAGE, Outcome.HANDOFF])
        self._node = node

    def execute(self, blackboard) -> str:  # type: ignore
        self._node.vehicle_io.set_mode(self._node.param_str("coarse_mode"))
        self._node._coarse_ready = False   # invalidate stale flags on entry
        self._node._fine_seated = False
        period = 1.0 / self._node.param_double("tick_rate_hz")
        while rclpy.ok() and not self._node._stop.is_set():
            self._node.publish_state(DockingStateMsg.COARSE, "COARSE")
            if not self._node._engaged:
                return Outcome.DISENGAGE
            if self._node._coarse_ready:
                return Outcome.HANDOFF
            time.sleep(period)


class FineState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=[Outcome.DISENGAGE, Outcome.SEATED, Outcome.DEMOTE])
        self._node = node

    def execute(self, blackboard) -> str:  # type: ignore
        # STABILIZE: hand depth to the controller for the terminal descent (ALT_HOLD
        # fights small heave commands). CoarseState restores ALT_HOLD on a DEMOTE.
        self._node.vehicle_io.set_mode(self._node.param_str("fine_mode"))
        self._node._coarse_ready = False   # invalidate stale flags on entry
        self._node._fine_seated = False
        self._node._fine_range = 0.0
        loss_counter = 0
        drift_counter = 0
        loss_to = self._node.param_int("loss_timeout_cycles")
        drift_to = self._node.param_int("drift_timeout_cycles")
        demote_range = self._node.param_double("demote_range_m")
        period = 1.0 / self._node.param_double("tick_rate_hz")
        while rclpy.ok() and not self._node._stop.is_set():
            self._node.publish_state(DockingStateMsg.FINE, "FINE")
            if not self._node._engaged:
                return Outcome.DISENGAGE
            if self._node._fine_seated:
                return Outcome.SEATED
            loss_counter, lost = tr.sustained(
                loss_counter, tr.is_loss(self._node._health), loss_to)
            drift_counter, drifted = tr.sustained(
                drift_counter, tr.is_drift(self._node._fine_range, demote_range), drift_to)
            if lost or drifted:
                return Outcome.DEMOTE
            time.sleep(period)


class DockedState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=[Outcome.DISENGAGE])
        self._node = node

    def execute(self, blackboard) -> str:  # type: ignore
        self._node.vehicle_io.set_arm(False)
        period = 1.0 / self._node.param_double("tick_rate_hz")
        while rclpy.ok() and not self._node._stop.is_set():
            self._node.publish_state(DockingStateMsg.DOCKED, "DOCKED")
            if not self._node._engaged:
                return Outcome.DISENGAGE
            time.sleep(period)


def main(args=None):
    rclpy.init(args=args)
    node = DockingFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
