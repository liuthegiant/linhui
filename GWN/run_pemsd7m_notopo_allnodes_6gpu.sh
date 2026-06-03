#!/usr/bin/env bash
# PEMSD7M extended estimation on GPUs 0,1,3,4,6,7:
#   Batch A: no_pretrain + topo_only (5 seeds each) -> update main report
#   Batch B: geo_only + geo_topo with GEO pretrain on ALL nodes (5 seeds) -> new report
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(100 42 999 555 250)
GPUS=(0 1 3 4 6 7)
MAX_PARALLEL=${#GPUS[@]}
LOGROOT_MAIN="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
LOGROOT_ALLNOD="${EST_GEOTOPO_ALLNOD_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5}"
SCHED_LOG="${LOGROOT_MAIN}/pemsd7m_extended_batches.log"
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

mkdir -p "$LOGROOT_MAIN" "$LOGROOT_ALLNOD"
log() { printf '%s\n' "$*" | tee -a "$SCHED_LOG"; }
log "[pemsd7m-extended] scheduler_pid=$$ log=${SCHED_LOG}"

is_done () {
  local log="$1"
  [[ -f "$log" ]] || return 1
  "$PYBIN" - "$log" <<'PY'
import sys
from pathlib import Path
raise SystemExit(0 if "SCRIPT DURATION" in Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore") else 1)
PY
}

cfg_info () {
  case "$1" in
    no_pretrain) echo "geo 1 0" ;;
    topo_only)   echo "topo 1 1" ;;
    geo_only)    echo "geo 1 1" ;;
    geo_topo)    echo "geo,topo 2 1" ;;
    *) return 1 ;;
  esac
}

base_args () {
  local seed="$1"
  echo "1 0.7 0 ${seed} 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

run_one () {
  local gpu="$1" seed="$2" cfg="$3" logroot="$4" tag_suffix="$5" geo_train_only="$6"
  local log="${logroot}/PEMSD7M/${seed}/est/A_${cfg}_s${seed}.log"
  mkdir -p "$(dirname "$log")"
  if is_done "$log"; then
    log "[skip] PEMSD7M ${cfg} seed=${seed} (${logroot})"
    return 0
  fi
  local experts topk pretrain_flag
  read -r experts topk pretrain_flag <<<"$(cfg_info "$cfg")"
  read -r -a BASE <<<"$(base_args "$seed")"
  BASE[0]="${pretrain_flag}"
  local tag="est_geotopo_pemsd7m_${cfg}_s${seed}"
  if [[ -n "$tag_suffix" ]]; then
    tag="${tag}_${tag_suffix}"
  fi
  log "[launch] gpu=${gpu} PEMSD7M cfg=${cfg} seed=${seed} GEO_PRETRAIN_TRAIN_ONLY=${geo_train_only} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="${tag}" \
    GEO_PRETRAIN_TRAIN_ONLY="${geo_train_only}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
}

run_batch () {
  local batch_name="$1"
  shift
  local -a jobs=("$@")
  local round=0
  while true; do
    round=$((round + 1))
    local -a pending=()
    for j in "${jobs[@]}"; do
      IFS='|' read -r seed cfg logroot tag_suffix geo_train_only <<<"$j"
      local log="${logroot}/PEMSD7M/${seed}/est/A_${cfg}_s${seed}.log"
      if ! is_done "$log"; then
        pending+=("$j")
      fi
    done
    log "======== ${batch_name} round=${round} pending=${#pending[@]} ========"
    [[ ${#pending[@]} -eq 0 ]] && break
    local i=0
    local -a pids=()
    for j in "${pending[@]}"; do
      IFS='|' read -r seed cfg logroot tag_suffix geo_train_only <<<"$j"
      local gpu="${GPUS[$((i % MAX_PARALLEL))]}"
      i=$((i + 1))
      run_one "$gpu" "$seed" "$cfg" "$logroot" "$tag_suffix" "$geo_train_only" &
      pids+=($!)
      if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
        wait "${pids[0]}" || true
        pids=("${pids[@]:1}")
      fi
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done
  done
}

log "[pemsd7m-extended] GPUs=${GPUS[*]} LOGROOT_MAIN=${LOGROOT_MAIN}"
log "[pemsd7m-extended] LOGROOT_ALLNOD=${LOGROOT_ALLNOD}"

JOBS_A=()
for seed in "${SEEDS[@]}"; do
  JOBS_A+=("${seed}|no_pretrain|${LOGROOT_MAIN}||1")
  JOBS_A+=("${seed}|topo_only|${LOGROOT_MAIN}||1")
done

run_batch "batch-A-notopo" "${JOBS_A[@]}"
log "[aggregate] main report..."
"$PYBIN" aggregate_topomoe_est_geotopo_3ds_5seed.py \
  --logroot "${LOGROOT_MAIN}" \
  --out "PEMSBAY_PEMSD7M_METRLA_geo_topo_Estimation.md"

JOBS_B=()
for seed in "${SEEDS[@]}"; do
  JOBS_B+=("${seed}|geo_only|${LOGROOT_ALLNOD}|allnod|0")
  JOBS_B+=("${seed}|geo_topo|${LOGROOT_ALLNOD}|allnod|0")
done

run_batch "batch-B-geo-allnodes" "${JOBS_B[@]}"
log "[aggregate] all-nodes geo report..."
"$PYBIN" aggregate_pemsd7m_geo_allnodes_5seed.py \
  --logroot "${LOGROOT_ALLNOD}" \
  --out "PEMSD7M_geo_allnodes_pretrain_Estimation.md"

log "[all done] PEMSD7M extended batches finished."
