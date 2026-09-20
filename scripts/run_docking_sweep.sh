#!/usr/bin/env bash
# Docking-trial sweep (#36, #70). Runs the single-cell auto trial over a grid of
# (arm x navigation level x period x sway phase) and collects a results CSV. Fully
# unattended and resumable: a cell whose bag was finalised (metadata.yaml present)
# is skipped, so a sweep that died mid-week is relaunched with the same SWEEP_TAG
# and continues; the cell it died in is rerun.
#
# Grid via env (defaults shown). Periods start HIGH (easy) and descend so a partial
# run covers the interesting long-period end first:
#   ARMS="A B"                    A reactive | B original ff | C fix | D oracle
#   NAV_LEVELS="none"             none | low | medium | high (#69)
#   PERIODS="20 16 12 10 8 6"     sway periods (s)
#   PHASES="0.0 2.094 4.189"      explicit start phases (rad), or
#   PHASE_COUNT=8                 evenly spaced phases in [0, 2pi) (overrides PHASES)
#   PHASE_COUNT_SHORT=8 SHORT_PERIOD_MAX=12   more phases at and below this period
#   STATIC_DOCK=1                 prepend one static-dock sanity cell per arm
#   SWEEP_TAG=<tag>               reuse a tag to RESUME
#   DRY_RUN=1                     list the cells and exit without launching anything
# Per-cell timing passes through: READY_TIMEOUT, DOCK_TIMEOUT, SWAY_AMP.
#
# WARNING: each cell pkills gazebo/ardusub on teardown. Do not run other sims meanwhile.
set -uo pipefail

WS="${WS:-/home/ubuntu/ws_docking}"
AUTO="$WS/src/bluerov2-docking/scripts/run_docking_trial_auto.sh"
TAG="${SWEEP_TAG:-sweep_$(date +%Y%m%d_%H%M%S)}"
CSV="$WS/bags/${TAG}_results.csv"

read -r -a ARMS <<< "${ARMS:-A B}"
read -r -a NAV_LEVELS <<< "${NAV_LEVELS:-none}"
read -r -a PERIODS <<< "${PERIODS:-20 16 12 10 8 6}"
STATIC_DOCK="${STATIC_DOCK:-1}"
SHORT_PERIOD_MAX="${SHORT_PERIOD_MAX:-12}"

phases_for(){  # period -> space-separated start phases
  local n="${PHASE_COUNT:-}"
  if [ -n "${PHASE_COUNT_SHORT:-}" ] && awk "BEGIN{exit !($1 <= $SHORT_PERIOD_MAX)}"; then n="$PHASE_COUNT_SHORT"; fi
  if [ -n "$n" ]; then awk -v n="$n" 'BEGIN{for(k=0;k<n;k++) printf "%.4f ", k*6.283185307/n}'; else echo "${PHASES:-0.0 2.094 4.189}"; fi
}

log(){ echo "[sweep $(date +%H:%M:%S)] $*"; }
if [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p "$WS/bags"
  [ -f "$CSV" ] || echo "label,arm,dock,period,phase,nav,outcome,bag" > "$CSV"
fi

run_one(){  # arm nav dock period phase
  local arm=$1 nav=$2 dock=$3 period=$4 phase=$5
  local navtag=""; [ "$nav" != "none" ] && navtag="_nav${nav}"
  local label="${TAG}_${arm}_${dock}_p${period}_ph${phase}${navtag}"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "  $label"; return; fi
  # A bag directory appears as soon as the recorder starts, so its existence alone
  # does not mean the cell finished: a trial killed mid-recording (thermal crash,
  # power loss) leaves a directory without metadata.yaml. Only a bag the recorder
  # finalised counts as done; anything else is rerun and logged loudly.
  local existing; existing=$(ls -dt "$WS"/bags/"${label}"_* 2>/dev/null | head -1)
  if [ -n "$existing" ] && [ -f "$existing/metadata.yaml" ]; then
    log "SKIP $label (have $(basename "$existing"))"; return
  elif [ -n "$existing" ]; then
    log "RERUN $label (incomplete bag $(basename "$existing"), no metadata.yaml)"
  fi
  log "=== RUN $label ==="
  local extra=(); [ "$dock" = "static" ] && extra=(DOCK_STATIC=1)
  local out; out=$(env "${extra[@]}" "$AUTO" "$period" "$arm" "$phase" "$TAG" "$nav" 2>&1 \
                   | tee /dev/stderr | grep "^RESULT" | tail -1)
  local outcome; outcome=$(awk '{print $3}' <<< "$out"); outcome=${outcome:-NORUN}
  local bag; bag=$(ls -dt "$WS"/bags/"${label}"_* 2>/dev/null | head -1)
  echo "${label},${arm},${dock},${period},${phase},${nav},${outcome},$(basename "${bag:-none}")" >> "$CSV"
  log "=== $label -> ${outcome} ==="
  sleep 5   # margin for teardown to fully release before the next launch
}

log "sweep tag=$TAG arms=[${ARMS[*]}] nav=[${NAV_LEVELS[*]}] periods=[${PERIODS[*]}]"
n=0
for arm in "${ARMS[@]}"; do
  for nav in "${NAV_LEVELS[@]}"; do
    [ "$STATIC_DOCK" = "1" ] && { run_one "$arm" "$nav" static 0 0.0; n=$((n+1)); }
    for period in "${PERIODS[@]}"; do
      for phase in $(phases_for "$period"); do run_one "$arm" "$nav" sway "$period" "$phase"; n=$((n+1)); done
    done
  done
done
log "sweep planned/done: $n cells -> $CSV"
[ "${DRY_RUN:-0}" = "1" ] || { column -t -s, "$CSV" 2>/dev/null || cat "$CSV"; }
