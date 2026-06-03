#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python"
LOGDIR="logs_topomoe/est_rerun"
mkdir -p "$LOGDIR"

# Base args (README_RUN_TOPO_MOE.md)
BASE=(1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320)
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)
EXTRA_SPARSE=(sparse_moe 64 16 2 1.0 0.001 0.001 0.0 1)

run_one () {
  local gpu="$1"; shift
  local experts="$1"; shift
  local topk="$1"; shift
  local log="$1"; shift
  echo "[launch] gpu=$gpu experts=$experts topk=$topk log=$LOGDIR/$log"
  CUDA_VISIBLE_DEVICES="$gpu" MOE_EXPERTS="$experts" MOE_TOP_K="$topk" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "$@" 2>&1 | tee "$LOGDIR/$log" &
}

# NOTE: `A_scpt_only` 已在 logs_topomoe/est 里跑完，不在此处重复。
run_one 0 "geo" 1 "A_geo_only.log"         "${BASE[@]}" "${EXTRA_TOPO[@]}"
run_one 1 "topo" 1 "A_topo_only.log"       "${BASE[@]}" "${EXTRA_TOPO[@]}"
run_one 2 "scpt,geo" 2 "A_scpt_geo.log"    "${BASE[@]}" "${EXTRA_SPARSE[@]}"
run_one 3 "scpt,topo" 2 "A_scpt_topo.log"  "${BASE[@]}" "${EXTRA_TOPO[@]}"
run_one 4 "geo,topo" 2 "A_geo_topo.log"    "${BASE[@]}" "${EXTRA_TOPO[@]}"
run_one 5 "scpt,geo,topo" 2 "A_scpt_geo_topo.log" "${BASE[@]}" "${EXTRA_TOPO[@]}"

wait
echo "[done] all estimation ablations finished"

