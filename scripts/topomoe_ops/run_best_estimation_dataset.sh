#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   DATASET=PEMSBAY BEST_TAG=sctopo_tau05_noreg BEST_EXPERTS=scpt,topo BEST_TAU=0.5 ./scripts/topomoe_ops/run_best_estimation_dataset.sh
#   DATASET=PEMSD7M ./scripts/topomoe_ops/run_best_estimation_dataset.sh

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
GATE_HIDDEN="${GATE_HIDDEN:-64}"
TOPO_LAP_K="${TOPO_LAP_K:-16}"
BEST_TAG="${BEST_TAG:-sctopo_tau05_noreg}"
BEST_EXPERTS="${BEST_EXPERTS:-scpt,topo}"
BEST_TOPK="${BEST_TOPK:-2}"
BEST_TAU="${BEST_TAU:-0.5}"
BEST_LB="${BEST_LB:-0.0}"
BEST_SMOOTH="${BEST_SMOOTH:-0.0}"
BEST_DELTA="${BEST_DELTA:-0.0}"
BEST_USECTX="${BEST_USECTX:-1}"
FUSION="${FUSION:-topo_moe}"
SCRIPT="${SCRIPT:-pred_maskpredition_GWN_scpt_geo_topomoe.py}"

for seed in $SEEDS_STR; do
  echo ""
  echo "========== BEST EST $DATASET seed=$seed tag=$BEST_TAG experts=$BEST_EXPERTS =========="
  base_args=(1 0.7 0 "$seed" 1.0 "$DATASET" "$NODE_ARG" 1 0.0 1 1 2 "$BATCHSIZE" 0.01 "$PRE_EPOCH" "$MAIN_EPOCH" 0 0.001 1 320)
  MOE_EXPERTS="$BEST_EXPERTS" \
  MOE_RUN_TAG="est_${BEST_TAG}_${DATASET}_seed${seed}" \
  python "$SCRIPT" "${base_args[@]}" "$FUSION" "$GATE_HIDDEN" "$TOPO_LAP_K" "$BEST_TOPK" "$BEST_TAU" "$BEST_LB" "$BEST_SMOOTH" "$BEST_DELTA" "$BEST_USECTX"
done

cd "$REPO_ROOT"
python scripts/topomoe_ops/collect_scores.py --root "${LOG_ROOT:-logs_topomoe}" --csv "${CSV_OUT:-topomoe_stage2_${DATASET}_best_estimation.csv}" || true
python scripts/topomoe_ops/show_alpha.py --root "${LOG_ROOT:-logs_topomoe}" || true
