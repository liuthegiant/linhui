#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(100 42 999 555 250)
DATASETS=(PEMSBAY PEMSD7M)
GPUS=(0 1 3 4 6 7)
LOGROOT="${EST2DS_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_2ds_seed5_imgbase}"
mkdir -p "$LOGROOT"

base_args_for () {
  local seed="$1"
  local dname="$2"
  # Keep aligned with TOPOMOe_STAGE1_REPORT_imgbase.md
  echo "1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

is_done () {
  local log="$1"
  [[ -f "$log" ]] || return 1
  python - "$log" <<'PY'
import sys
from pathlib import Path
t = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
sys.exit(0 if "SCRIPT DURATION" in t else 1)
PY
}

launch_one () {
  local gpu="$1"
  local dataset="$2"
  local seed="$3"
  local cfg="$4"
  local experts="$5"
  local topk="$6"
  local pretrain_flag="$7"
  local log="$8"

  read -r -a BASE <<<"$(base_args_for "$seed" "$dataset")"
  BASE[0]="${pretrain_flag}"

  echo "[launch] ds=${dataset} seed=${seed} cfg=${cfg} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="est2ds_${dataset,,}_${cfg}_s${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}" >/dev/null &
}

cfg_info () {
  local cfg="$1"
  case "$cfg" in
    no_pretrain) echo "scpt 1 0" ;;
    scpt_only)   echo "scpt 1 1" ;;
    topo_only)   echo "topo 1 1" ;;
    scpt_topo)   echo "scpt,topo 2 1" ;;
    *) return 1 ;;
  esac
}

wait_for_slot () {
  local max_jobs="$1"
  while true; do
    local n
    n=$(jobs -pr | wc -l)
    if [[ "$n" -lt "$max_jobs" ]]; then
      break
    fi
    sleep 2
  done
}

gpu_idx=0
max_parallel=${#GPUS[@]}

for seed in "${SEEDS[@]}"; do
  echo "======== fixed-gpu resume seed=${seed} ========"
  for ds in "${DATASETS[@]}"; do
    for cfg in no_pretrain scpt_only topo_only scpt_topo; do
      ddir="${LOGROOT}/${ds}/${seed}/est"
      mkdir -p "$ddir"
      log="${ddir}/A_${cfg}_s${seed}.log"
      if is_done "$log"; then
        echo "[skip] done ds=${ds} seed=${seed} cfg=${cfg}"
        continue
      fi

      read -r experts topk pretrain_flag <<<"$(cfg_info "$cfg")"
      wait_for_slot "$max_parallel"
      gpu="${GPUS[$((gpu_idx % max_parallel))]}"
      gpu_idx=$((gpu_idx + 1))
      launch_one "$gpu" "$ds" "$seed" "$cfg" "$experts" "$topk" "$pretrain_flag" "$log"
    done
  done
done

wait
echo "[all tasks finished] generating report..."
"$PYBIN" aggregate_topomoe_est_2ds_5seed.py --logroot "${LOGROOT}" --out "PEMSBAY_PEMSD7M_topo_Estimation.md"
echo "[done] report: ${ROOT_DIR}/PEMSBAY_PEMSD7M_topo_Estimation.md"
