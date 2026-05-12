#!/usr/bin/env bash
set -euo pipefail

# Run from repo root or from GWN/. This script will cd into GWN/.
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
HIDDEN="${HIDDEN:-64}"
TOPO_LAP_K="${TOPO_LAP_K:-16}"
GATE_HIDDEN="${GATE_HIDDEN:-64}"
SCRIPT="${SCRIPT:-pred_maskpredition_GWN_scpt_geo_topomoe.py}"

run_one () {
  local tag="$1" experts="$2" topk="$3" tau="$4" lb="$5" smooth="$6" delta="$7" usectx="$8" fusion="$9" seed="${10}"
  echo ""
  echo "========== EST $DATASET seed=$seed tag=$tag experts=$experts topk=$topk tau=$tau lb=$lb smooth=$smooth =========="
  # BASE argv[1]..[20] (keep everything else unchanged):
  # 1 0.7 0 <seed> 1.0 <DATASET> -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320
  local base_args=(1 0.7 0 "$seed" 1.0 "$DATASET" "$NODE_ARG" 1 0.0 1 1 2 "$BATCHSIZE" 0.01 "$PRE_EPOCH" "$MAIN_EPOCH" 0 0.001 1 320)
  MOE_EXPERTS="$experts" \
  MOE_RUN_TAG="est_${tag}_seed${seed}" \
  python "$SCRIPT" "${base_args[@]}" "$fusion" "$GATE_HIDDEN" "$TOPO_LAP_K" "$topk" "$tau" "$lb" "$smooth" "$delta" "$usectx"
}

for seed in $SEEDS_STR; do
  # Step 2: SCPT + TOPO tau/reg sweep
  run_one "sctopo_tau1_noreg"       "scpt,topo"     2 1.0 0.0 0.0    0.0 1 topo_moe "$seed"
  run_one "sctopo_tau05_noreg"      "scpt,topo"     2 0.5 0.0 0.0    0.0 1 topo_moe "$seed"
  run_one "sctopo_tau05_smooth1e4"  "scpt,topo"     2 0.5 0.0 0.0001 0.0 1 topo_moe "$seed"

  # Step 3: full SCPT + GEO + TOPO without load-balance
  run_one "full_tau1_noreg"         "scpt,geo,topo" 2 1.0 0.0 0.0    0.0 1 topo_moe "$seed"
  run_one "full_tau05_noreg"        "scpt,geo,topo" 2 0.5 0.0 0.0    0.0 1 topo_moe "$seed"
done

if [[ "${RUN_TOPK1_DIAG:-0}" == "1" ]]; then
  for seed in $SEEDS_STR; do
    run_one "sctopo_topk1_diag" "scpt,topo" 1 1.0 0.0 0.0 0.0 1 topo_moe "$seed"
  done
fi

cd "$REPO_ROOT"
python scripts/topomoe_ops/collect_scores.py --root "${LOG_ROOT:-logs_topomoe}" --csv "${CSV_OUT:-topomoe_stage2_metrla_estimation.csv}" || true
python scripts/topomoe_ops/show_alpha.py --root "${LOG_ROOT:-logs_topomoe}" || true
