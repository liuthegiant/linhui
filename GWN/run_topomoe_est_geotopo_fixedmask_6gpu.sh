#!/usr/bin/env bash
# Geo/Topo estimation + fixed full-node masks (simulate virtual nodes with no history).
# Uses: pred_maskpredition_GWN_scpt_geo_topomoe_fixedmask.py
#
# Mask policy (see pred_maskpredition_GWN_scpt_geo_fixedmask.py):
#   - Selected nodes: 100% masked at ALL timesteps (train/val/test)
#   - Other nodes: random point mask each TRAIN epoch (MISS_RATIO); fixed at setup for val/test
#
# Env (optional):
#   FIXED_MASK_FRAC=0.05     sample fraction from train/tst pools
#   FIXED_MASK_N=10          override count per pool
#   FIXED_MASK_NODES=1,2,3   explicit node ids (global)
#   VIRTUAL_NODES=207,208    merged into fixed set
#   MASK_RANDOM_ON_OTHER=1   0 = only fixed-node masking, no random points
#
# Datasets: PEMSBAY PEMSD7M METRLA | Configs: no_pretrain geo_only topo_only geo_topo
# Seeds: 100 42 999 555 250 | GPUs: 0 1 3 4 6 7
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SCRIPT="pred_maskpredition_GWN_scpt_geo_topomoe_fixedmask.py"
SEEDS=(100 42 999 555 250)
GPUS=(0 1 3 4 6 7)
MAX_PARALLEL=${#GPUS[@]}
LOGROOT="${EST_GEOTOPO_FIXED_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_fixedmask_seed5}"

export FIXED_MASK_FRAC="${FIXED_MASK_FRAC:-0.05}"
export MASK_RANDOM_ON_OTHER="${MASK_RANDOM_ON_OTHER:-1}"

mkdir -p "$LOGROOT"

base_args_for () {
  local seed="$1"
  local dname="$2"
  echo "1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

cfg_info () {
  local cfg="$1"
  case "$cfg" in
    no_pretrain) echo "geo 1 0" ;;
    geo_only)    echo "geo 1 1" ;;
    topo_only)   echo "topo 1 1" ;;
    geo_topo)    echo "geo,topo 2 1" ;;
    *) return 1 ;;
  esac
}

EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

is_done () {
  local log="$1"
  [[ -f "$log" ]] || return 1
  "$PYBIN" - "$log" <<'PY'
import sys
from pathlib import Path
t = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
raise SystemExit(0 if "SCRIPT DURATION" in t else 1)
PY
}

build_pending () {
  local out_file="$1"
  : > "$out_file"
  local -a datasets=(PEMSBAY PEMSD7M METRLA)
  local -a configs=(no_pretrain geo_only topo_only geo_topo)
  for seed in "${SEEDS[@]}"; do
    for ds in "${datasets[@]}"; do
      for cfg in "${configs[@]}"; do
        local ddir="${LOGROOT}/${ds}/${seed}/est"
        mkdir -p "$ddir"
        local log="${ddir}/A_${cfg}_s${seed}.log"
        if ! is_done "$log"; then
          echo "${seed}|${ds}|${cfg}|${log}" >> "$out_file"
        fi
      done
    done
  done
}

run_one () {
  local gpu="$1"
  local seed="$2"
  local ds="$3"
  local cfg="$4"
  local log="$5"
  local experts topk pretrain_flag
  read -r experts topk pretrain_flag <<<"$(cfg_info "$cfg")"
  read -r -a BASE <<<"$(base_args_for "$seed" "$ds")"
  BASE[0]="${pretrain_flag}"

  echo "[launch-fixedmask] seed=${seed} ds=${ds} cfg=${cfg} gpu=${gpu} FIXED_MASK_FRAC=${FIXED_MASK_FRAC}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="est_geotopo_fm_${ds,,}_${cfg}_s${seed}" \
    "$PYBIN" "$SCRIPT" \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}" >/dev/null &
}

echo "[geotopo-fixedmask] LOGROOT=${LOGROOT} FIXED_MASK_FRAC=${FIXED_MASK_FRAC}"

round=0
while true; do
  round=$((round + 1))
  pending_file="$(mktemp)"
  build_pending "$pending_file"
  pending_n=$(wc -l < "$pending_file")
  echo "======== round=${round} pending=${pending_n} ========"
  if [[ "$pending_n" -eq 0 ]]; then
    rm -f "$pending_file"
    break
  fi
  i=0
  while IFS='|' read -r seed ds cfg log; do
    gpu="${GPUS[$((i % MAX_PARALLEL))]}"
    i=$((i + 1))
    run_one "$gpu" "$seed" "$ds" "$cfg" "$log"
    if [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; then
      wait
    fi
  done < "$pending_file"
  rm -f "$pending_file"
  wait
done

echo "[all done] aggregate..."
"$PYBIN" aggregate_topomoe_est_geotopo_3ds_5seed.py \
  --logroot "${LOGROOT}" \
  --out "PEMSBAY_PEMSD7M_METRLA_geo_topo_fixedmask_Estimation.md"
echo "[done] mask_policy.json under each run dir in ${LOGROOT}"
