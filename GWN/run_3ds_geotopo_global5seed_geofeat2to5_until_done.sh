#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNNER="${ROOT_DIR}/run_3ds_geotopo_global5seed_geofeat2to5_gpu0235.sh"
LOGROOT="${EST_3DS_GLOBAL5_GEOF2TO5_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_3ds_virtualnode_splitmask_global5seed_geotopo_geoF2to5}"
SUP_LOG="${LOGROOT}/supervisor_geotopo_global5_geoF2to5.log"
mkdir -p "$LOGROOT"

log() { printf '%s %s\n' "[$(date '+%F %T')]" "$*" | tee -a "$SUP_LOG"; }

LOCK_FILE="${LOGROOT}/.supervisor.lock"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
  log "another supervisor is active, exiting."
  exit 0
fi

count_state() {
  python - "$LOGROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
datasets = ["METRLA", "PEMSBAY", "PEMSD7M"]
seeds = [42, 88, 250, 555, 999]
feats = [2, 3, 4, 5]
done = running = pending = 0
for ds in datasets:
    for seed in seeds:
        for feat in feats:
            p = root / ds / str(seed) / "est" / f"A_geo_topo_f{feat}_s{seed}.log"
            if not p.exists():
                pending += 1
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "SCRIPT DURATION" in txt:
                done += 1
            else:
                running += 1
print(done, running, pending)
PY
}

total=60
round=0

log "supervisor started, target=${total} tasks"

while true; do
  round=$((round + 1))
  read -r done running pending <<<"$(count_state)"
  log "round=${round} precheck done=${done}/${total} running=${running} pending=${pending}"
  if [[ "$done" -ge "$total" ]]; then
    log "all tasks completed."
    break
  fi

  bash "$RUNNER" || true

  read -r done2 running2 pending2 <<<"$(count_state)"
  log "round=${round} post-run done=${done2}/${total} running=${running2} pending=${pending2}"
  if [[ "$done2" -ge "$total" ]]; then
    log "all tasks completed."
    break
  fi

  sleep 20
done

