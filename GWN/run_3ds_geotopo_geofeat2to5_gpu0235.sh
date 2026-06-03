#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SCRIPT="pred_maskpredition_GWN_scpt_geo_topomoe_virtualnode_splitmask.py"
GPUS=(0 2 3 5)
FEATURES_LIST=(2 3 4 5)
export FIXED_MASK_FRAC="${FIXED_MASK_FRAC:-0.05}"

declare -A SEEDS
SEEDS[METRLA]="42 88 100 250 432"
SEEDS[PEMSBAY]="42 66 88 233 999"
SEEDS[PEMSD7M]="42 100 432 555 999"

LOGROOT="${EST_3DS_GEOF2TO5_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_3ds_virtualnode_splitmask_5seed_geotopo_geoF2to5}"
SCHED_LOG="${LOGROOT}/scheduler_geotopo_geoF2to5_gpu0235.log"
mkdir -p "$LOGROOT"

log() { printf '%s\n' "$*" | tee -a "$SCHED_LOG"; }

is_done() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  "$PYBIN" - "$f" <<'PY'
import sys
from pathlib import Path
txt = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
raise SystemExit(0 if "SCRIPT DURATION" in txt else 1)
PY
}

run_one() {
  local gpu="$1" ds="$2" seed="$3" feat="$4"
  local logf="${LOGROOT}/${ds}/${seed}/est/A_geo_topo_f${feat}_s${seed}.log"
  mkdir -p "$(dirname "$logf")"
  if is_done "$logf"; then
    log "[skip] ${ds} seed=${seed} feat=${feat}"
    return 0
  fi

  read -r -a BASE <<<"1 0.7 0 ${seed} 1.0 ${ds} -1 1 0.0 1 1 ${feat} 64 0.01 100 100 0 0.001 1 320"
  local EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

  log "[launch] gpu=${gpu} ds=${ds} seed=${seed} feat=${feat}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="geo,topo" \
    MOE_TOP_K="2" \
    MOE_RUN_TAG="est_vnode_split_${ds,,}_geo_topo_f${feat}_s${seed}" \
    "$PYBIN" "$SCRIPT" "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "$logf"
}

tasks=()
for ds in METRLA PEMSBAY PEMSD7M; do
  for seed in ${SEEDS[$ds]}; do
    for feat in "${FEATURES_LIST[@]}"; do
      tasks+=("${ds}|${seed}|${feat}")
    done
  done
done

log "[start] total_tasks=${#tasks[@]} gpus=${GPUS[*]} logroot=${LOGROOT}"

max_parallel=${#GPUS[@]}
i=0
pids=()
for t in "${tasks[@]}"; do
  IFS='|' read -r ds seed feat <<<"$t"
  gpu="${GPUS[$((i % max_parallel))]}"
  i=$((i + 1))

  run_one "$gpu" "$ds" "$seed" "$feat" &
  pids+=($!)
  if [[ ${#pids[@]} -ge $max_parallel ]]; then
    wait "${pids[0]}" || true
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

log "[done] all tasks finished."
