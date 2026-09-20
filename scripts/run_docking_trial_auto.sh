#!/usr/bin/env bash
# One self-driving docking trial (#36, #70). Launches the full sim, waits for the filter
# to go HEALTHY, auto-engages the FSM, records a --low-impact bag, waits for DOCKED
# (or times out), then tears the whole stack down cleanly. This is the building block
# the sweep loop wraps; run it standalone first to prove the loop on one cell.
#
# Usage:
#   ./scripts/run_docking_trial_auto.sh [period_s] [arm] [sway_phase_rad] [label] [nav_level]
#     period_s   dock sway period; "0" (or DOCK_STATIC=1) -> static dock (no sway)
#     arm        A reactive | B original feedforward | C fix | D oracle
#                (legacy names accepted: static = A, sway = B)
#     phase_rad  sway start phase (vary across reps for an honest success rate)
#     label      bag label prefix (default built from the cell)
#     nav_level  navigation-error level: none | low | medium | high (default none)
# Env knobs: SWAY_AMP, READY_TIMEOUT, DOCK_TIMEOUT (simulation seconds), WALL_GUARD
#            (wall seconds per simulation second before a stalled sim is abandoned),
#            WS, EXTRA_LAUNCH_ARGS, DRY_RUN=1
#
# Bag directory names follow scripts/analysis/labels.py:
#   <label>_<arm>_<dock>_p<period>_ph<phase>[_nav<level>]_<YYYYMMDD>_<HHMMSS>
# so the analysis package parses every cell without a per-sweep regex.
#
# WARNING: teardown pkills gazebo/ardusub/mavros, so do not run other sims meanwhile.
set -uo pipefail   # not -e: a failed step should still trigger teardown, not abort

PERIOD="${1:-18}"
ARM="${2:-B}"
PHASE="${3:-0.0}"
NAV="${5:-none}"
case "$ARM" in static) ARM=A;; sway) ARM=B;; esac
DOCK="sway"; { [ "$PERIOD" = "0" ] || [ "${DOCK_STATIC:-0}" = "1" ]; } && DOCK="static"
NAVTAG=""; [ "$NAV" != "none" ] && NAVTAG="_nav${NAV}"
LABEL="${4:-auto}_${ARM}_${DOCK}_p${PERIOD}_ph${PHASE}${NAVTAG}"

# Arm -> launch arguments. A, B and D exist (D needs feedforward_source from #68);
# every arm passes the navigation-error level that #69 adds. Arm C (the fix, #67) has
# no launch arguments yet. ros2 launch silently ignores unknown key:=value pairs, so
# passing made-up ones would run arm B under a C label; refuse instead until the
# fix arm lands, then put its arguments here (expected: a velocity-closed feedforward
# switch and a stale velocity decay time constant).
case "$ARM" in
  A) ARM_ARGS="process_noise_regime:=static" ;;
  B) ARM_ARGS="process_noise_regime:=sway" ;;
  C) echo "arm C is not implemented yet: no launch arguments exist for the fix (#67), refusing to run it as arm B"; exit 2 ;;
  D) ARM_ARGS="process_noise_regime:=sway feedforward_source:=oracle" ;;
  *) echo "unknown arm: $ARM (A|B|C|D)"; exit 2 ;;
esac
[ "$NAV" != "none" ] && ARM_ARGS="$ARM_ARGS nav_error_level:=$NAV"

SWAY_AMP="${SWAY_AMP:-0.1}"
# Timeouts count SIMULATION seconds read from /clock, so a throttled host (real-time
# factor 0.4 to 0.7 when hot) runs the same trial, only slower; with the filter on the
# node clock the outcome does not depend on the real-time factor. WALL_GUARD bounds
# the wall time per simulation second so a simulator that never advances is abandoned.
READY_TIMEOUT="${READY_TIMEOUT:-150}"   # sim-seconds for the filter to initialise
DOCK_TIMEOUT="${DOCK_TIMEOUT:-200}"     # sim-seconds to reach DOCKED before giving up
WALL_GUARD="${WALL_GUARD:-4}"
# A launch occasionally stalls before the simulator starts stepping (no /clock at all,
# the GUI waiting for the world list, ArduSub waiting for the physics backend); seen
# once in ten launches on 2026-09-20. Relaunch the stack rather than lose the cell.
STARTUP_WALL="${STARTUP_WALL:-60}"          # wall seconds for /clock to appear and advance
LAUNCH_ATTEMPTS="${LAUNCH_ATTEMPTS:-2}"
WS="${WS:-/home/ubuntu/ws_docking}"
REC="$WS/src/bluerov2-docking/scripts/record_docking_trial.sh"
LAUNCH_PID=""
ENGAGE_PID=""
REC_PID=""

log(){ echo "[auto $(date +%H:%M:%S)] $*"; }
# Integer simulation seconds from /clock, empty until the simulator publishes it.
sim_now(){ timeout 3 ros2 topic echo /clock --once 2>/dev/null | awk '/^  sec:/{print $2; exit}'; }
# Elapsed simulation seconds since $1 (a sim_now reading); falls back to wall seconds
# since $2 while /clock is not available yet.
sim_elapsed(){ local now; now=$(sim_now); if [ -n "$now" ] && [ -n "$1" ]; then echo $((now - $1)); else echo $((SECONDS - $2)); fi; }
pgid_of(){ ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '; }
killgrp(){ local g; g=$(pgid_of "${1:-}"); [ -n "$g" ] && kill -"${2:-TERM}" -- "-$g" 2>/dev/null; return 0; }
# ArduSub SITL binds TCP 5760 (= hex 1680). `ss`/`netstat` aren't in the container,
# so read /proc/net/tcp{,6} directly; column 2 is LOCALADDR:PORT in hex.
port_5760_busy(){ awk 'NR>1{print $2}' /proc/net/tcp /proc/net/tcp6 2>/dev/null | grep -qiE ':1680$'; }
stop_recorder(){
  # SIGINT the ros2 bag record process itself (only one runs per trial) so it writes the
  # mcap footer + metadata.yaml, then wait for it to exit. Match by the recorder command,
  # not the (long, brittle) label; a group SIGINT killed bash before the bag finalized.
  if pkill -INT -f "ros2 bag record" 2>/dev/null; then
    log "stopping recorder, waiting for finalize"
    for _ in $(seq 1 25); do pgrep -f "ros2 bag record" >/dev/null 2>&1 || break; sleep 1; done
  fi
  killgrp "$REC_PID" KILL
}

teardown(){
  log "teardown"
  stop_recorder
  killgrp "$ENGAGE_PID" KILL
  killgrp "$LAUNCH_PID" TERM; sleep 6; killgrp "$LAUNCH_PID" KILL
  pkill -9 -f "gz sim"          2>/dev/null
  pkill -9 -f "ardusub"         2>/dev/null
  pkill -9 -f "parameter_bridge" 2>/dev/null
  pkill -9 -f "mavros_node"     2>/dev/null
  pkill -9 -f "ruby.*gz"        2>/dev/null
  # wait for ArduSub's TCP 5760 to release, else the next launch collides
  for _ in $(seq 1 30); do port_5760_busy || break; sleep 1; done
  sleep 3
}
trap 'log "interrupted"; teardown; exit 130' INT TERM

# --- dock motion via env (read by the DockSway plugin at world load) ---
if [ "$PERIOD" = "0" ] || [ "${DOCK_STATIC:-0}" = "1" ]; then
  export DOCK_SWAY_ENABLED=false
  log "dock: STATIC (no sway)"
else
  # Orbital motion = sway and heave 90deg out of phase. PHASE is the per-rep START
  # offset applied to BOTH axes, preserving their pi/2 relative offset (the circle).
  SWAY_PH=$(awk "BEGIN{printf \"%.5f\", $PHASE + 1.5707963}")
  export DOCK_SWAY_ENABLED=true DOCK_SWAY_PERIOD="$PERIOD" \
         DOCK_SWAY_SWAY_AMPLITUDE="$SWAY_AMP" DOCK_SWAY_HEAVE_AMPLITUDE="$SWAY_AMP" \
         DOCK_SWAY_HEAVE_PHASE="$PHASE" DOCK_SWAY_SWAY_PHASE="$SWAY_PH"
  log "dock: orbital period=${PERIOD}s amp=${SWAY_AMP} start_phase=${PHASE} (heave_ph=$PHASE sway_ph=$SWAY_PH)"
fi

# --- launch the whole stack in its own process group ---
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY $LABEL: ros2 launch sim sim.launch.py use_control:=true $ARM_ARGS ${EXTRA_LAUNCH_ARGS:-} (dock ${DOCK_SWAY_ENABLED:-} period ${DOCK_SWAY_PERIOD:-})"
  echo "RESULT $LABEL DRYRUN"; exit 0
fi
launch_stack(){
  setsid ros2 launch sim sim.launch.py \
      use_control:=true use_deadman:=false use_aruco:=true use_mock_led:=true \
      use_docking_rviz:=false $ARM_ARGS ${EXTRA_LAUNCH_ARGS:-} \
      > "/tmp/${LABEL}.log" 2>&1 &
  LAUNCH_PID=$!
}
# true once /clock exists and has advanced, false if it has not within STARTUP_WALL
sim_stepping(){
  local t0=$SECONDS first="" now
  while [ $((SECONDS - t0)) -lt "$STARTUP_WALL" ]; do
    now=$(sim_now)
    if [ -n "$now" ]; then
      [ -z "$first" ] && first=$now
      [ "$now" -gt "$first" ] && return 0
    fi
    sleep 3
  done
  return 1
}
attempt=1
while :; do
  log "launching sim (arm=$ARM nav=$NAV, attempt $attempt of $LAUNCH_ATTEMPTS) -> /tmp/${LABEL}.log"
  launch_stack
  sim_stepping && break
  log "simulator never started stepping within ${STARTUP_WALL}s wall"
  if [ "$attempt" -ge "$LAUNCH_ATTEMPTS" ]; then
    log "giving up on this cell"; teardown; echo "RESULT $LABEL STALLED"; exit 2
  fi
  cp "/tmp/${LABEL}.log" "/tmp/${LABEL}.stalled${attempt}.log"
  teardown; attempt=$((attempt + 1))
done

# --- wait for readiness: the filter has INITIALIZED, i.e. it is publishing a filtered
# pose (it stays silent while WARMING_UP). Coarse drives on DEGRADED too, so waiting
# for HEALTHY would deadlock at the ~5m spawn range where the filter is only DEGRADED.
# Heartbeat logs the health status (0 WARMING_UP, 1 HEALTHY, 2 DEGRADED, 3 STALE) so a
# genuine stuck-at-1-marker case (status pinned 0) is distinguishable from slow warmup.
log "waiting for filter to initialize (filtered pose flowing; <=${READY_TIMEOUT}s sim, wall guard $((READY_TIMEOUT * WALL_GUARD))s)"
ready=0; t0=$SECONDS; last=0; s0=""
while [ $((SECONDS - t0)) -lt $((READY_TIMEOUT * WALL_GUARD)) ]; do
  [ -z "$s0" ] && s0=$(sim_now)   # the clock starts with the simulator, some seconds after launch
  [ "$(sim_elapsed "$s0" "$t0")" -ge "$READY_TIMEOUT" ] && break
  if timeout 4 ros2 topic echo /perception/dock_pose_filtered --once \
       --qos-reliability best_effort 2>/dev/null | grep -q "position:"; then
    ready=1; break
  fi
  if [ $((SECONDS - last)) -ge 15 ]; then
    h=$(timeout 3 ros2 topic echo /perception/dock_pose_filtered/health --once \
         --qos-reliability best_effort 2>/dev/null | grep -oP "status: \K\d+")
    log "  ...warming up (health=${h:-?}; 0=WARMING 1=HEALTHY 2=DEGRADED 3=STALE) sim t+$(sim_elapsed "$s0" "$t0")s, wall t+$((SECONDS - t0))s"
    last=$SECONDS
  fi
  sleep 3
done
if [ "$ready" != "1" ]; then log "NOT INITIALIZED within ${READY_TIMEOUT}s sim (wall $((SECONDS - t0))s) -> abort"; teardown; exit 2; fi
log "filter initialized at sim t+$(sim_elapsed "$s0" "$t0")s, wall t+$((SECONDS - t0))s"

# --- wait for ardusub_init to finish applying the default flight mode BEFORE engaging.
# It retries flight_mode:=POSHOLD until it sticks; if the FSM's COARSE-entry ALT_HOLD
# beats that retry, ardusub_init clobbers it back to POSHOLD, whose position-hold then
# fights the small sway commands near the standoff -> "couldn't sway at end of coarse".
# Engaging after the mode settles makes the FSM's ALT_HOLD the last word.
log "waiting for flight mode to settle (ardusub_init done)"
t0=$SECONDS
while [ $((SECONDS - t0)) -lt 60 ]; do
  grep -q "Flight mode set to" "/tmp/${LABEL}.log" 2>/dev/null && break
  sleep 2
done
log "flight mode settled at t+$((SECONDS - t0))s (+3s guard)"
sleep 3

# --- start the recorder (own group so we can SIGINT just it) ---
# record straight to bags/ (no --low-impact tmpfs): finalizing + moving a tmpfs bag on
# an abrupt stop is what corrupted the earlier runs. Reliability over the I/O saving here.
log "recording ($LABEL -> bags/)"
setsid "$REC" "$LABEL" > "/tmp/${LABEL}_rec.log" 2>&1 &
REC_PID=$!
sleep 2

# --- auto-engage: publish /docking/engaged=true continuously (FSM IDLE -> COARSE) ---
log "engage"
setsid ros2 topic pub -r 2 /docking/engaged std_msgs/msg/Bool "{data: true}" > /dev/null 2>&1 &
ENGAGE_PID=$!

# --- wait for DOCKED (or timeout) ---
log "waiting for DOCKED (<=${DOCK_TIMEOUT}s sim, wall guard $((DOCK_TIMEOUT * WALL_GUARD))s)"
outcome="TIMEOUT"; t0=$SECONDS; s0=$(sim_now)
while [ $((SECONDS - t0)) -lt $((DOCK_TIMEOUT * WALL_GUARD)) ]; do
  [ "$(sim_elapsed "$s0" "$t0")" -ge "$DOCK_TIMEOUT" ] && break
  s=$(timeout 4 ros2 topic echo /docking/state --once 2>/dev/null | grep -oP "label: \K\w+")
  if [ "$s" = "DOCKED" ]; then outcome="DOCKED"; break; fi
  sleep 2
done
el=$(sim_elapsed "$s0" "$t0"); wall=$((SECONDS - t0))
[ "$wall" -ge $((DOCK_TIMEOUT * WALL_GUARD)) ] && [ "$el" -lt "$DOCK_TIMEOUT" ] && outcome="STALLED"
log "outcome: $outcome at sim t+${el}s, wall t+${wall}s (real-time factor $(awk "BEGIN{printf \"%.2f\", ($wall>0)?$el/$wall:0}"))"

trap - INT TERM
teardown

# safety net: if finalize was cut short, rebuild the index so the bag is readable now
# (the mcap footer is present; only metadata.yaml is missing).
bag=$(ls -dt "$WS"/bags/"${LABEL}"_* 2>/dev/null | head -1)
if [ -n "$bag" ] && [ ! -f "$bag/metadata.yaml" ]; then
  log "reindexing bag (metadata.yaml missing)"
  ros2 bag reindex "$bag" -s mcap >/dev/null 2>&1
fi

log "done: $LABEL -> $outcome  (bag in $WS/bags/, log /tmp/${LABEL}.log)"
echo "RESULT $LABEL $outcome"
