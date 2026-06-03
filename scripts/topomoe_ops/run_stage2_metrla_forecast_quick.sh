#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "GWN" ]]; then
  GWN_DIR="$REPO_ROOT"
else
  GWN_DIR="$REPO_ROOT/GWN"
fi
cd "$GWN_DIR"

SEEDS_STR="${SEEDS:-100 42 999}"
DATASET="${DATASET:-METRLA}"
NODE_ARG="${NODE_ARG:--1}"
PRE_EPOCH="${PRE_EPOCH:-100}"
MAIN_EPOCH="${MAIN_EPOCH:-100}"
BATCHSIZE="${BATCHSIZE:-64}"
TOPO_LAP_K="${TOPO_LAP_K:-16}"
SCRIPT="${SCRIPT:-pred_GWN_16_adpAdj_topomoe.py}"

run_one () {
  local tag="$1" experts="$2" topk="$3" tau="$4" lb="$5" smooth="$6" delta="$7" usectx="$8" seed="$9"
  echo ""
  echo "========== FCST $DATASET seed=$seed tag=$tag experts=$experts topk=$topk tau=$tau lb=$lb smooth=$smooth =========="
  # BASE argv[1]..[20] (keep everything else unchanged):
  local base_args=(1 0.7 0 "$seed" 1.0 "$DATASET" "$NODE_ARG" 1 0.0 1 1 2 "$BATCHSIZE" 0.01 "$PRE_EPOCH" "$MAIN_EPOCH" 0 0.001 1 320)
  local forecast_args=(1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0)
  local moe_tail=("$TOPO_LAP_K" "$topk" "$tau" "$lb" "$smooth" "$delta" "$usectx")
  MOE_EXPERTS="$experts" \
  MOE_RUN_TAG="fcst_${tag}_seed${seed}" \
  python "$SCRIPT" "${base_args[@]}" "${forecast_args[@]}" "${moe_tail[@]}"
}

for seed in $SEEDS_STR; do
  run_one "sctopo_tau05_noreg" "scpt,topo"     2 0.5 0.0 0.0 0.0 1 "$seed"
  run_one "full_tau05_noreg"   "scpt,geo,topo" 2 0.5 0.0 0.0 0.0 1 "$seed"
done

cd "$REPO_ROOT"
python scripts/topomoe_ops/collect_scores.py --root "${LOG_ROOT:-logs_topomoe}" --csv "${CSV_OUT:-topomoe_stage2_metrla_forecast.csv}" || true
