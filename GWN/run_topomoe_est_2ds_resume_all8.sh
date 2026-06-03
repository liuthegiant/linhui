#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(100 42 999 555 250)
DATASETS=(PEMSBAY PEMSD7M)
CONFIGS=(no_pretrain scpt_only topo_only scpt_topo)
GPUS=(0 1 2 3 4 5 6 7)
MAX_PARALLEL=${#GPUS[@]}
LOGROOT="${EST2DS_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_2ds_seed5_imgbase}"
mkdir -p "$LOGROOT"

base_args_for () {
  local seed="$1"
  local dname="$2"
  echo "1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
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

build_pending () {
  local out_file="$1"
  : > "$out_file"
  for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for cfg in "${CONFIGS[@]}"; do
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

  echo "[launch] seed=${seed} ds=${ds} cfg=${cfg} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="est2ds_${ds,,}_${cfg}_s${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}" >/dev/null &
}

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

echo "[all done] generating report..."
"$PYBIN" aggregate_topomoe_est_2ds_5seed.py --logroot "${LOGROOT}" --out "PEMSBAY_PEMSD7M_topo_Estimation.md"
echo "[done] report: ${ROOT_DIR}/PEMSBAY_PEMSD7M_topo_Estimation.md"
