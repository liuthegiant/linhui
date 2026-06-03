#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SUPERVISOR="${ROOT_DIR}/run_3ds_geotopo_global5seed_geofeat2to5_until_done.sh"
LOGROOT="${EST_3DS_GLOBAL5_GEOF2TO5_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_3ds_virtualnode_splitmask_global5seed_geotopo_geoF2to5}"
WD_LOG="${LOGROOT}/watchdog_geotopo_global5.log"
mkdir -p "$LOGROOT"

log() { printf '%s %s\n' "[$(date '+%F %T')]" "$*" | tee -a "$WD_LOG"; }

LOCK_FILE="${LOGROOT}/.watchdog.lock"
exec 7>"$LOCK_FILE"
if ! flock -n 7; then
  log "watchdog already running, exiting."
  exit 0
fi

log "watchdog started."

while true; do
  if pgrep -f "$SUPERVISOR" >/dev/null 2>&1; then
    :
  else
    log "supervisor missing -> restart."
    bash "$SUPERVISOR" >/dev/null 2>&1 &
  fi
  sleep 60
done

