#!/usr/bin/env bash
# Estimation mask prediction — no SCPT; geo / topo experts only.
# Datasets: PEMSBAY, PEMSD7M, METRLA (METRLA last in scheduling order).
# Configs: no_pretrain, geo_only, topo_only, geo_topo
# Seeds: 100 42 999 555 250  |  GPUs: 0 1 3 4 6 7
#
# Default RUN_SCOPE=supplement (only jobs listed as 待补):
#   PEMSBAY  : geo_only, geo_topo
#   PEMSD7M  : geo_only, geo_topo
#   METRLA   : no_pretrain
# Set RUN_SCOPE=full for 3 datasets x 4 configs x 5 seeds.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(100 42 999 555 250)
GPUS=(0 1 3 4 6 7)
MAX_PARALLEL=${#GPUS[@]}
LOGROOT="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
RUN_SCOPE="${RUN_SCOPE:-supplement}"

mkdir -p "$LOGROOT"

base_args_for () {
  local seed="$1"
  local dname="$2"
  echo "1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

# cfg -> "MOE_EXPERTS MOE_TOP_K IS_PRETRN"
cfg_info () {
  local cfg="$1"
  case "$cfg" in
    no_pretrain) echo "geo 1 0" ;;      # IS_PRETRN=0 disables MoE; experts unused
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

should_run () {
  local ds="$1"
  local cfg="$2"
  if [[ "$RUN_SCOPE" == "full" ]]; then
    return 0
  fi
  case "${ds}|${cfg}" in
    PEMSBAY\|geo_only|PEMSBAY\|geo_topo|PEMSD7M\|geo_only|PEMSD7M\|geo_topo|METRLA\|no_pretrain)
      return 0 ;;
    *) return 1 ;;
  esac
}

build_pending () {
  local out_file="$1"
  : > "$out_file"
  local -a datasets=(PEMSBAY PEMSD7M METRLA)
  local -a configs=(no_pretrain geo_only topo_only geo_topo)
  for seed in "${SEEDS[@]}"; do
    for ds in "${datasets[@]}"; do
      for cfg in "${configs[@]}"; do
        should_run "$ds" "$cfg" || continue
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

  echo "[launch] seed=${seed} ds=${ds} cfg=${cfg} experts=${experts} topk=${topk} pretrain=${pretrain_flag} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
    MOE_RUN_TAG="est_geotopo_${ds,,}_${cfg}_s${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
    "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}" >/dev/null &
}

echo "[geotopo-est] LOGROOT=${LOGROOT} RUN_SCOPE=${RUN_SCOPE} GPUs=${GPUS[*]}"

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

echo "[all done] aggregate report..."
"$PYBIN" aggregate_topomoe_est_geotopo_3ds_5seed.py \
  --logroot "${LOGROOT}" \
  --out "PEMSBAY_PEMSD7M_METRLA_geo_topo_Estimation.md"
echo "[done] ${ROOT_DIR}/PEMSBAY_PEMSD7M_METRLA_geo_topo_Estimation.md"
