#!/usr/bin/env bash
# T4: plant identification step and sinusoid tests (#66). Runs inside the container
# with the sim up and the dock static; the vehicle must be armed in the mode under
# test. Publishes a command sequence on /cmd_vel_auto (or /cmd_vel when no deadman
# relay is running) and records odometry and the command so scripts/analysis can
# fit gain, delay and time constant per axis and mode.
#
# Usage: ./scripts/plantid/record_step_tests.sh <ALT_HOLD|STABILIZE> [axis: y|z] [label]
# Env:   WS (workspace), CMD_TOPIC (default /cmd_vel), STEP (default 0.1), HOLD (s, default 8)
set -uo pipefail
MODE="${1:?mode}"; AXIS="${2:-y}"; LABEL="${3:-plantid_${MODE}_${AXIS}}"
WS="${WS:-/home/ubuntu/ws_docking}"; CMD_TOPIC="${CMD_TOPIC:-/cmd_vel}"
STEP="${STEP:-0.1}"; HOLD="${HOLD:-8}"
STAMP=$(date +%Y%m%d_%H%M%S); OUT="$WS/bags/${LABEL}_${STAMP}"
log(){ echo "[plantid $(date +%H:%M:%S)] $*"; }

field(){ case "$AXIS" in y) echo "{linear: {x: 0.0, y: $1, z: 0.0}}";; z) echo "{linear: {x: 0.0, y: 0.0, z: $1}}";; *) echo "{linear: {x: $1, y: 0.0, z: 0.0}}";; esac; }

log "mode $MODE axis $AXIS step $STEP hold ${HOLD}s -> $OUT"
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: '$MODE'}" >/dev/null 2>&1 || log "set_mode call failed, continuing"
sleep 2
ros2 bag record -o "$OUT" /clock "$CMD_TOPIC" /model/bluerov2_heavy/odometry /mavros/state >/dev/null 2>&1 &
REC=$!; sleep 2

pub(){ timeout "$2" ros2 topic pub -r 20 "$CMD_TOPIC" geometry_msgs/msg/Twist "$(field "$1")" >/dev/null 2>&1; }
# rest, positive step, rest, negative step, rest: two steps give gain and delay
pub 0.0 4; pub "$STEP" "$HOLD"; pub 0.0 "$HOLD"; pub "-$STEP" "$HOLD"; pub 0.0 "$HOLD"
# sinusoid sweep at the dock periods used in the campaign (8 and 6 s), 3 cycles each
for PER in 8 6; do
  log "sinusoid period ${PER}s"
  python3 - "$CMD_TOPIC" "$AXIS" "$STEP" "$PER" <<'PY'
import math, sys, time, rclpy
from geometry_msgs.msg import Twist
topic, axis, amp, per = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rclpy.init(); n = rclpy.create_node("plantid_sine"); p = n.create_publisher(Twist, topic, 10)
t0 = time.time()
while time.time() - t0 < 3 * per:
    m = Twist(); setattr(m.linear, axis, amp * math.sin(2 * math.pi * (time.time() - t0) / per)); p.publish(m); time.sleep(0.05)
p.publish(Twist()); n.destroy_node(); rclpy.shutdown()
PY
done
pub 0.0 3
kill -INT $REC; wait $REC 2>/dev/null
log "done: $OUT  (fit with: python3 -m analysis stepfit $OUT)"
