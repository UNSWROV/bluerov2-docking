#!/usr/bin/env bash
# Record a moving-dock docking trial for offline analysis (#36).
#
# Usage:
#   ./scripts/record_docking_trial.sh [label] [-l|--low-impact]
# Start it just before you engage the deadman; Ctrl+C to stop.
# Writes ./bags/<label>_<timestamp>/ (rosbag2). Share that path for analysis.
#
# -l/--low-impact: use when the sim is CPU-bound (RTF < 1). Records into a tmpfs
# (/dev/shm, so no disk I/O during the run) under nice+ionice so the recorder yields
# CPU/IO to the camera + perception threads, then moves the bag to bags/ on stop.
# Without it, recording can starve the pipeline and drop camera frames, degrading the
# very estimate you are trying to capture. /dev/shm is RAM-backed and size-limited;
# for very long runs, record without -l.

set -euo pipefail

LOW_IMPACT=0
LABEL="dock_trial"
for arg in "$@"; do
  case "$arg" in
    -l|--low-impact) LOW_IMPACT=1 ;;
    -h|--help) echo "usage: $0 [label] [-l|--low-impact]"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) LABEL="$arg" ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${LABEL}_${STAMP}"
FINAL="bags/${NAME}"

# Curated: FSM state, filter estimate + health + raw measurement, ROV pose,
# commands, frames, clock. No camera/image topics (large, not needed here).
TOPICS=(
  /clock                                  # sim time, needed to replay with use_sim_time
  /docking/state                          # FSM phase: IDLE/COARSE/FINE/DOCKED (convergence vs flapping)
  /docking/engaged                        # deadman engage signal
  /cmd_vel                                # commanded body velocity (post deadman gate)
  /cmd_vel_auto                           # controller output (pre gate)
  /control/coarse_approach/status         # coarse gate: within_*_tol, ready_for_handoff, errors
  /control/fine_align/status              # fine gate: aligned, seated, phase
  /perception/dock_pose_filtered          # Kalman dock-pose estimate
  /perception/dock_pose_filtered/velocity # CV filter dock-velocity state (#36); needed to check velocity estimation
  /dock/oracle_velocity                   # ground-truth dock velocity when feedforward_source:=oracle (#68)
  /perception/dock_pose_filtered/health   # filter health (WARMING_UP/HEALTHY/DEGRADED/STALE)
  /perception/dock_pose_measured          # raw fused measurement (pre-filter)
  /model/bluerov2_heavy/odometry          # ROV pose
  /tf
  /tf_static
  /dock/ground_truth/pose                 # true dock pose; empty until the ros_gz bridge is added
)

mkdir -p bags

REC_DIR="${FINAL}"
RECORD_PREFIX=()

finalize() {
  # Move the tmpfs recording back to bags/ once recording has stopped (no contention
  # left). No-op when recording straight to bags/ or if the bag was never created.
  [ "${REC_DIR}" = "${FINAL}" ] && return 0
  [ -d "${REC_DIR}" ] || return 0
  mv "${REC_DIR}" "${FINAL}" && echo "saved ${FINAL}" \
    || echo "WARN: recording left in ${REC_DIR}" >&2
}

if [ "${LOW_IMPACT}" = "1" ]; then
  command -v nice   >/dev/null 2>&1 && RECORD_PREFIX+=(nice -n 19)
  command -v ionice >/dev/null 2>&1 && RECORD_PREFIX+=(ionice -c3)
  if [ -d /dev/shm ] && [ -w /dev/shm ]; then
    REC_DIR="/dev/shm/${NAME}"
    trap finalize EXIT
    echo "low-impact: recording to tmpfs ${REC_DIR}; will move to ${FINAL} on stop"
  else
    echo "low-impact: /dev/shm unavailable; recording to ${FINAL} with nice/ionice only" >&2
  fi
fi

echo "Recording ${#TOPICS[@]} topics -> ${REC_DIR}"
echo "Engage the deadman and run the docking attempt. Ctrl+C to stop."
"${RECORD_PREFIX[@]}" ros2 bag record -o "${REC_DIR}" "${TOPICS[@]}"
