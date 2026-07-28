#!/usr/bin/env bash
# Docking-trial sweep (#36). Runs the proven single-cell auto trial over a grid of
# (regime x period x sway-phase) and collects a results CSV. Fully unattended.
#
# The two curves this generates are the thesis result: P(dock) vs sway period for
#   regime=sway   -> method   (CV filter estimates dock velocity + feeds it forward)
#   regime=static -> baseline (velocity pinned to 0 -> feedforward effectively off)
# plus a static-dock sanity cell. Analyze with scripts/analyze_sweep.py.
#
# Grid via env (defaults shown). Periods start HIGH (easy) and descend to locate the
# breakdown, so a partial run still covers the interesting high-period end first:
#   REGIMES="sway static"
#   PERIODS="20 16 12 10 8 6"      sway periods (s)
#   PHASES="0.0 2.094 4.189"       start phases (rad, = 0, 2pi/3, 4pi/3) -> reps
#   STATIC_DOCK=1                  prepend one static-dock (no sway) sanity cell
#   SWEEP_TAG=<tag>               reuse a tag to RESUME (cells with an existing bag skip)
# Per-cell timing passes through: READY_TIMEOUT, DOCK_TIMEOUT, SWAY_AMP.
#
# WARNING: each cell pkills gazebo/ardusub on teardown. Do not run other sims meanwhile.
set -uo pipefail

WS="${WS:-/home/ubuntu/ws_docking}"
AUTO="$WS/src/bluerov2-docking/scripts/run_docking_trial_auto.sh"
TAG="${SWEEP_TAG:-sweep_$(date +%Y%m%d_%H%M%S)}"
CSV="$WS/bags/${TAG}_results.csv"

read -r -a REGIMES <<< "${REGIMES:-sway static}"
read -r -a PERIODS <<< "${PERIODS:-20 16 12 10 8 6}"
read -r -a PHASES  <<< "${PHASES:-0.0 2.094 4.189}"
STATIC_DOCK="${STATIC_DOCK:-1}"

log(){ echo "[sweep $(date +%H:%M:%S)] $*"; }
[ -f "$CSV" ] || echo "label,regime,dock,period,phase,outcome,bag" > "$CSV"

run_one(){  # regime dock period phase
  local regime=$1 dock=$2 period=$3 phase=$4
  local label="${TAG}_${regime}_${dock}_p${period}_ph${phase}"
  # resume: an existing bag for this exact cell means it already ran -> skip
  local existing; existing=$(ls -dt "$WS"/bags/"${label}"_* 2>/dev/null | head -1)
  if [ -n "$existing" ]; then log "SKIP $label (have $(basename "$existing"))"; return; fi
  log "=== RUN $label ==="
  local extra=(); [ "$dock" = "static" ] && extra=(DOCK_STATIC=1)
  local out; out=$(env "${extra[@]}" "$AUTO" "$period" "$regime" "$phase" "$label" 2>&1 \
                   | tee /dev/stderr | grep "^RESULT" | tail -1)
  local outcome; outcome=$(awk '{print $3}' <<< "$out"); outcome=${outcome:-NORUN}
  local bag; bag=$(ls -dt "$WS"/bags/"${label}"_* 2>/dev/null | head -1)
  echo "${label},${regime},${dock},${period},${phase},${outcome},$(basename "${bag:-none}")" >> "$CSV"
  log "=== $label -> ${outcome} ==="
  sleep 5   # margin for teardown to fully release before the next launch
}

log "sweep tag=$TAG  regimes=[${REGIMES[*]}] periods=[${PERIODS[*]}] phases=[${PHASES[*]}]"
n=0
for regime in "${REGIMES[@]}"; do
  [ "$STATIC_DOCK" = "1" ] && { run_one "$regime" static 0 0.0; n=$((n+1)); }
  for period in "${PERIODS[@]}"; do
    for phase in "${PHASES[@]}"; do run_one "$regime" sway "$period" "$phase"; n=$((n+1)); done
  done
done
log "sweep done: $n cells -> $CSV"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
