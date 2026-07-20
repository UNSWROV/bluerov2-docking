#!/usr/bin/env bash
# Record a moving-dock docking trial for offline analysis (#36).
#
# Usage:
#   ./scripts/record_docking_trial.sh [label]
# Start it just before you engage the deadman; Ctrl+C to stop.
# Writes ./bags/<label>_<timestamp>/ (rosbag2). Share that path for analysis.

set -euo pipefail

LABEL="${1:-dock_trial}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="bags/${LABEL}_${STAMP}"

# Curated: FSM state, filter estimate + health + raw measurement, ROV pose,
# commands, frames, clock. No camera/image topics (large, not needed here).
TOPICS=(
  /clock                                  # sim time, needed to replay with use_sim_time
  /docking/state                          # FSM phase: IDLE/COARSE/FINE/DOCKED (convergence vs flapping)
  /docking/engaged                        # deadman engage signal
  /cmd_vel                                # commanded body velocity (post deadman gate)
  /cmd_vel_auto                           # controller output (pre gate)
  /perception/dock_pose_filtered          # Kalman dock-pose estimate
  /perception/dock_pose_filtered/health   # filter health (WARMING_UP/HEALTHY/DEGRADED/STALE)
  /perception/dock_pose_measured          # raw fused measurement (pre-filter)
  /model/bluerov2_heavy/odometry          # ROV pose
  /tf
  /tf_static
  /dock/ground_truth/pose                 # true dock pose; empty until the ros_gz bridge is added
)

mkdir -p bags
echo "Recording ${#TOPICS[@]} topics -> ${OUT}"
echo "Engage the deadman and run the docking attempt. Ctrl+C to stop."
ros2 bag record -o "${OUT}" "${TOPICS[@]}"
