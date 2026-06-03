#!/usr/bin/env bash
# PEMSBAY + PEMSD7M split virtual-node evaluation (same settings as METRLA splitmask).
# train=random point masks only; test=tst_u / tst_v / tst_a
# Seeds: 100 42 999 555 250 88 66 233 38 432 | GPUs: 0 1 2 3 4 6 7
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SCRIPT="pred_maskpredition_GWN_scpt_geo_topomoe_virtualnode_splitmask.py"
SEEDS=(100 42 999 555 250 88 66 233 38 432)
CONFIGS=(no_pretrain topo_only geo_only geo_topo)
DATASETS=(PEMSBAY PEMSD7M)
GPUS=(0 1 2 3 4 6 7)
MAX_PARALLEL=${#GPUS[@]}
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

export FIXED_MASK_FRAC="${FIXED_MASK_FRAC:-0.05}"

logroot_for () {
  local ds="$1"
  case "$ds" in
    PEMSBAY) echo "${EST_PEMSBAY_SPLITMASK_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10}" ;;
    PEMSD7M) echo "${EST_PEMSD7M_SPLITMASK_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10}" ;;
    *) return 1 ;;
  esac
}

SCHED_LOG="${ROOT_DIR}/logs_topomoe/pems2ds_virtualnode_splitmask_7gpu.log"
mkdir -p "$(dirname "$SCHED_LOG")"
log() { printf '%s\n' "$*" | tee -a "$SCHED_LOG"; }
log "[pems2ds-vnode-splitmask] pid=$$ GPUs=${GPUS[*]} FIXED_MASK_FRAC=${FIXED_MASK_FRAC}"

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

run_one () {
  local gpu="$1" ds="$2" seed="$3" cfg="$4"
  local logroot
  logroot="$(logroot_for "$ds")"
  local logf="${logroot}/${ds}/${seed}/est/A_${cfg}_s${seed}.log"
  mkdir -p "$(dirname "$logf")"
  if is_done "$logf"; then
    log "[skip] ${ds} ${cfg} seed=${seed}"
    return 0
  fi
  local experts topk pretrain_flag
  read -r experts topk pretrain_flag <<<"$(cfg_info "$cfg")"
  read -r -a BASE <<<"1 0.7 0 ${seed} 1.0 ${ds} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
  BASE[0]="${pretrain_flag}"
  log "[launch] gpu=${gpu} ${ds} cfg=${cfg} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="est_vnode_split_${ds,,}_${cfg}_s${seed}" \
    "$PYBIN" "$SCRIPT" \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "$logf"
}

round=0
while true; do
  round=$((round + 1))
  pending=()
  for ds in "${DATASETS[@]}"; do
    logroot="$(logroot_for "$ds")"
    for seed in "${SEEDS[@]}"; do
      for cfg in "${CONFIGS[@]}"; do
        logf="${logroot}/${ds}/${seed}/est/A_${cfg}_s${seed}.log"
        if ! is_done "$logf"; then
          pending+=("${ds}|${seed}|${cfg}")
        fi
      done
    done
  done
  log "======== round=${round} pending=${#pending[@]} ========"
  [[ ${#pending[@]} -eq 0 ]] && break

  i=0
  pids=()
  for item in "${pending[@]}"; do
    IFS='|' read -r ds seed cfg <<<"$item"
    gpu="${GPUS[$((i % MAX_PARALLEL))]}"
    i=$((i + 1))
    run_one "$gpu" "$ds" "$seed" "$cfg" &
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

for ds in "${DATASETS[@]}"; do
  logroot="$(logroot_for "$ds")"
  out="${ds}_virtualnode_splitmask_Estimation.md"
  log "[aggregate] ${ds} -> ${out}"
  "$PYBIN" aggregate_virtualnode_splitmask_10seed.py \
    --dataset "$ds" \
    --logroot "$logroot" \
    --out "$out"
done
log "[all done] PEMSBAY + PEMSD7M virtual-node splitmask finished."
