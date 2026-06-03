#!/usr/bin/env bash
# Stage-1 (README_RUN_TOPO_MOE.md) 5-seed sweep: A(Estimation) 7-way parallel -> B(Forecasting) 7-way parallel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"

# Seeds (METRLA docs convention)
SEEDS=(100 42 999 555 250)

# Logs (keep separate from any prior runs)
LOGROOT="${STAGE1_LOGROOT:-${ROOT_DIR}/logs_topomoe/stage1_seed5_imgbase_rerun}"
mkdir -p "$LOGROOT"

# BASE argv[1]..[20] (per user screenshot; keep other args unchanged)
base_args_for_seed () {
  local seed="$1"
  # 1 0.7 0 <seed> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320
  echo "1 0.7 0 ${seed} 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)
EXTRA_SPARSE=(sparse_moe 64 16 2 1.0 0.001 0.001 0.0 1)

run_est_round () {
  local seed="$1"
  local edir="${LOGROOT}/${seed}/est"
  mkdir -p "$edir"
  read -r -a BASE <<<"$(base_args_for_seed "$seed")"

  echo "[seed=${seed}] A estimation launch -> ${edir}"

  CUDA_VISIBLE_DEVICES=0 MOE_EXPERTS=scpt MOE_TOP_K=1 MOE_RUN_TAG="stage1_est_scpt_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_scpt_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=1 MOE_EXPERTS=geo MOE_TOP_K=1 MOE_RUN_TAG="stage1_est_geo_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_geo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=2 MOE_EXPERTS=topo MOE_TOP_K=1 MOE_RUN_TAG="stage1_est_topo_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_topo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=3 MOE_EXPERTS=scpt,geo MOE_TOP_K=2 MOE_RUN_TAG="stage1_est_scptgeo_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_SPARSE[@]}" \
    2>&1 | tee "${edir}/A_scpt_geo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=4 MOE_EXPERTS=scpt,topo MOE_TOP_K=2 MOE_RUN_TAG="stage1_est_scpttopo_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_scpt_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=5 MOE_EXPERTS=geo,topo MOE_TOP_K=2 MOE_RUN_TAG="stage1_est_geotopo_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_geo_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=6 MOE_EXPERTS=scpt,geo,topo MOE_TOP_K=2 MOE_RUN_TAG="stage1_est_full_seed${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_scpt_geo_topo_s${seed}.log" >/dev/null &

  wait
  echo "[seed=${seed}] A estimation DONE"
}

run_pred_round () {
  local seed="$1"
  local pdir="${LOGROOT}/${seed}/pred"
  mkdir -p "$pdir"
  read -r -a BASE <<<"$(base_args_for_seed "$seed")"

  local FORE_TOPO=(1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0)
  local FORE_SPARSE=(1 0.5 0 encoder encoderg "" dual sparse_moe 1.0 0.0 64 1 0.0 0.0)
  local EXTRA_PRED=(16 2 1.0 0.001 0.001 0.0 1)

  echo "[seed=${seed}] B forecasting launch -> ${pdir}"

  CUDA_VISIBLE_DEVICES=0 MOE_EXPERTS=scpt MOE_TOP_K=1 MOE_RUN_TAG="stage1_fcst_scpt_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=1 MOE_EXPERTS=geo MOE_TOP_K=1 MOE_RUN_TAG="stage1_fcst_geo_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_geo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=2 MOE_EXPERTS=topo MOE_TOP_K=1 MOE_RUN_TAG="stage1_fcst_topo_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_topo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=3 MOE_EXPERTS=scpt,geo MOE_TOP_K=2 MOE_RUN_TAG="stage1_fcst_scptgeo_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_SPARSE[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_geo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=4 MOE_EXPERTS=scpt,topo MOE_TOP_K=2 MOE_RUN_TAG="stage1_fcst_scpttopo_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=5 MOE_EXPERTS=geo,topo MOE_TOP_K=2 MOE_RUN_TAG="stage1_fcst_geotopo_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_geo_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=6 MOE_EXPERTS=scpt,geo,topo MOE_TOP_K=2 MOE_RUN_TAG="stage1_fcst_full_seed${seed}" \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_geo_topo_s${seed}.log" >/dev/null &

  wait
  echo "[seed=${seed}] B forecasting DONE"
}

for SEED in "${SEEDS[@]}"; do
  echo "======== seed=${SEED} ========"
  run_est_round "${SEED}"
  run_pred_round "${SEED}"
done

printf '%s\n' "[ALL DONE] stage1 5-seed finished. Logs: ${LOGROOT}/<seed>/{est,pred}/"
