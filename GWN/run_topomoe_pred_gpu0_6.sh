#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python"
LOGDIR="logs_topomoe/pred"
mkdir -p "$LOGDIR"

# Base args (README_RUN_TOPO_MOE.md)
BASE=(1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320)

# B1：scpt+geo 用 sparse_moe；其余 B2–B4 用 topo_moe
FORE_TOPO=(1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0)
FORE_SPARSE=(1 0.5 0 encoder encoderg "" dual sparse_moe 1.0 0.0 64 1 0.0 0.0)
EXTRA=(16 2 1.0 0.001 0.001 0.0 1)

run_one () {
  local gpu="$1"; shift
  local experts="$1"; shift
  local topk="$1"; shift
  local log="$1"; shift
  local fore=("$@")
  echo "[launch] gpu=$gpu experts=$experts topk=$topk log=$LOGDIR/$log"
  CUDA_VISIBLE_DEVICES="$gpu" MOE_EXPERTS="$experts" MOE_TOP_K="$topk" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${fore[@]}" "${EXTRA[@]}" \
    2>&1 | tee "$LOGDIR/$log" &
}

run_one 0 "scpt" 1 "B_scpt_only.log" "${FORE_TOPO[@]}"
run_one 1 "geo" 1 "B_geo_only.log" "${FORE_TOPO[@]}"
run_one 2 "topo" 1 "B_topo_only.log" "${FORE_TOPO[@]}"
run_one 3 "scpt,geo" 2 "B_scpt_geo.log" "${FORE_SPARSE[@]}"
run_one 4 "scpt,topo" 2 "B_scpt_topo.log" "${FORE_TOPO[@]}"
run_one 5 "geo,topo" 2 "B_geo_topo.log" "${FORE_TOPO[@]}"
run_one 6 "scpt,geo,topo" 2 "B_scpt_geo_topo.log" "${FORE_TOPO[@]}"

wait
echo "[done] all forecasting ablations finished"
