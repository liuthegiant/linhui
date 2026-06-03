#!/usr/bin/env bash
# PEMSD7M: 4 configs x 5 new seeds -> update main geo/topo estimation report.
# Seeds: 88 66 233 38 432  |  GPUs: 0 1 2 3 4 6 7
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(88 66 233 38 432)
CONFIGS=(no_pretrain geo_only topo_only geo_topo)
GPUS=(0 1 2 3 4 6 7)
MAX_PARALLEL=${#GPUS[@]}
LOGROOT="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
SCHED_LOG="${LOGROOT}/pemsd7m_newseeds_8gpu.log"
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

mkdir -p "$LOGROOT"
log() { printf '%s\n' "$*" | tee -a "$SCHED_LOG"; }
log "[pemsd7m-newseeds] pid=$$ GPUs=${GPUS[*]} seeds=${SEEDS[*]}"

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
    geo_only)    echo "geo 1 1" ;;
    topo_only)   echo "topo 1 1" ;;
    geo_topo)    echo "geo,topo 2 1" ;;
    *) return 1 ;;
  esac
}

run_one () {
  local gpu="$1" seed="$2" cfg="$3"
  local log="${LOGROOT}/PEMSD7M/${seed}/est/A_${cfg}_s${seed}.log"
  mkdir -p "$(dirname "$log")"
  if is_done "$log"; then
    log "[skip] PEMSD7M ${cfg} seed=${seed}"
    return 0
  fi
  local experts topk pretrain_flag
  read -r experts topk pretrain_flag <<<"$(cfg_info "$cfg")"
  read -r -a BASE <<<"1 0.7 0 ${seed} 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
  BASE[0]="${pretrain_flag}"
  log "[launch] gpu=${gpu} PEMSD7M cfg=${cfg} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="est_geotopo_pemsd7m_${cfg}_s${seed}" \
    GEO_PRETRAIN_TRAIN_ONLY=1 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
}

round=0
while true; do
  round=$((round + 1))
  pending=()
  for seed in "${SEEDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      logf="${LOGROOT}/PEMSD7M/${seed}/est/A_${cfg}_s${seed}.log"
      if ! is_done "$logf"; then
        pending+=("${seed}|${cfg}")
      fi
    done
  done
  log "======== round=${round} pending=${#pending[@]} ========"
  [[ ${#pending[@]} -eq 0 ]] && break
  i=0
  pids=()
  for item in "${pending[@]}"; do
    IFS='|' read -r seed cfg <<<"$item"
    gpu="${GPUS[$((i % MAX_PARALLEL))]}"
    i=$((i + 1))
    run_one "$gpu" "$seed" "$cfg" &
    pids+=($!)
    if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
      wait "${pids[0]}" || true
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done

log "[aggregate] updating PEMSBAY_PEMSD7M_METRLA_geo_topo_Estimation.md ..."
"$PYBIN" aggregate_topomoe_est_geotopo_3ds_5seed.py \
  --logroot "${LOGROOT}" \
  --out "PEMSBAY_PEMSD7M_METRLA_geo_topo_Estimation.md"
log "[all done] PEMSD7M new-seed batch finished."
