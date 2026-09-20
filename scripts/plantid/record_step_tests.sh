#!/usr/bin/env bash
# T4: plant identification step and sinusoid tests (#66). Runs inside the container
# with the sim up and the dock static; the vehicle must be armed in the mode under
# test. One publisher process drives the whole command sequence on CMD_TOPIC so
# there are no gaps between segments (ArduSub holds the last RC override through
# a gap, which would corrupt the delay estimate), and records odometry and the
# command so scripts/analysis can fit gain, delay and time constant.
#
# Usage: ./scripts/plantid/record_step_tests.sh <ALT_HOLD|STABILIZE> [axis: x|y|z] [label]
# Env:   WS (workspace), CMD_TOPIC (default /cmd_vel), STEP (default 0.1), HOLD (s, default 8)
set -uo pipefail
MODE="${1:?mode}"; AXIS="${2:-y}"; LABEL="${3:-plantid_${MODE}_${AXIS}}"
WS="${WS:-/home/ubuntu/ws_docking}"; CMD_TOPIC="${CMD_TOPIC:-/cmd_vel}"
STEP="${STEP:-0.1}"; HOLD="${HOLD:-8}"
STAMP=$(date +%Y%m%d_%H%M%S); OUT="$WS/bags/${LABEL}_${STAMP}"
log(){ echo "[plantid $(date +%H:%M:%S)] $*"; }

log "mode $MODE axis $AXIS step $STEP hold ${HOLD}s -> $OUT"
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: '$MODE'}" >/dev/null 2>&1 || log "set_mode call failed, continuing"
sleep 2
ros2 bag record -o "$OUT" /clock "$CMD_TOPIC" /model/bluerov2_heavy/odometry /mavros/state >/dev/null 2>&1 &
REC=$!; sleep 2

# rest, +step, rest, -step, rest, then three cycles of sinusoid at 8 s and 6 s, then rest
python3 - "$CMD_TOPIC" "$AXIS" "$STEP" "$HOLD" <<'PY'
import math, sys, time, rclpy
from geometry_msgs.msg import Twist
topic, axis, step, hold = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rclpy.init(); n = rclpy.create_node("plantid_cmd"); p = n.create_publisher(Twist, topic, 10)
def run(fn, dur):
    t0 = time.time()
    while time.time() - t0 < dur:
        m = Twist(); setattr(m.linear, axis, float(fn(time.time() - t0))); p.publish(m); time.sleep(0.05)
run(lambda t: 0.0, 4)
run(lambda t: step, hold); run(lambda t: 0.0, hold); run(lambda t: -step, hold); run(lambda t: 0.0, hold)
for per in (8.0, 6.0):
    run(lambda t, per=per: step * math.sin(2 * math.pi * t / per), 3 * per)
run(lambda t: 0.0, 3)
p.publish(Twist()); n.destroy_node(); rclpy.shutdown()
PY
kill -INT $REC; wait $REC 2>/dev/null
log "done: $OUT  (fit with: python3 -m analysis stepfit $OUT --axis $AXIS --cmd-topic $CMD_TOPIC)"
