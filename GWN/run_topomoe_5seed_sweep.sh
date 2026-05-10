#!/usr/bin/env bash
# TopoMoE METRLA 五随机种子扫种：估计 A（7 并行）→ 预测 B（7 并行）。
# 种子与仓库内 METRLA 文档一致：100, 42, 999, 555, 250（n=5）。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python"

# 与 GWN_METRLA_predict12_5seeds.md 等文档对齐
SEEDS=(100 42 999 555 250)

# New base argv (METRLA -1, 100/100 epochs, …) — separate tree from older seed5 runs
LOGROOT="${ROOT_DIR}/logs_topomoe/seed5_metrla_neg1"
mkdir -p "$LOGROOT"

EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)
EXTRA_SPARSE=(sparse_moe 64 16 2 1.0 0.001 0.001 0.0 1)

run_est_round () {
  local seed="$1"
  local edir="${LOGROOT}/${seed}/est"
  mkdir -p "$edir"

  local BASE=(
    1 0.7 0 "${seed}" 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320
  )

  echo "[seed=${seed}] A estimation launch -> ${edir}"

  CUDA_VISIBLE_DEVICES=0 MOE_EXPERTS=scpt MOE_TOP_K=1 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_scpt_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=1 MOE_EXPERTS=geo MOE_TOP_K=1 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_geo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=2 MOE_EXPERTS=topo MOE_TOP_K=1 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_topo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=3 MOE_EXPERTS=scpt,geo MOE_TOP_K=2 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_SPARSE[@]}" \
    2>&1 | tee "${edir}/A_scpt_geo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=4 MOE_EXPERTS=scpt,topo MOE_TOP_K=2 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_scpt_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=5 MOE_EXPERTS=geo,topo MOE_TOP_K=2 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_geo_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=6 MOE_EXPERTS=scpt,geo,topo MOE_TOP_K=2 \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${edir}/A_scpt_geo_topo_s${seed}.log" >/dev/null &

  wait
  echo "[seed=${seed}] A estimation DONE"
}

run_pred_round () {
  local seed="$1"
  local pdir="${LOGROOT}/${seed}/pred"
  mkdir -p "$pdir"

  local BASE=(
    1 0.7 0 "${seed}" 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320
  )
  # README B1：`scpt,geo` 用 `sparse_moe`；其余 B2–B4 用 `topo_moe`
  local FORE_TOPO=(1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0)
  local FORE_SPARSE=(1 0.5 0 encoder encoderg "" dual sparse_moe 1.0 0.0 64 1 0.0 0.0)
  local EXTRA_PRED=(16 2 1.0 0.001 0.001 0.0 1)

  echo "[seed=${seed}] B forecasting launch -> ${pdir}"

  CUDA_VISIBLE_DEVICES=0 MOE_EXPERTS=scpt MOE_TOP_K=1 \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=1 MOE_EXPERTS=geo MOE_TOP_K=1 \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_geo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=2 MOE_EXPERTS=topo MOE_TOP_K=1 \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_topo_only_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=3 MOE_EXPERTS=scpt,geo MOE_TOP_K=2 \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_SPARSE[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_geo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=4 MOE_EXPERTS=scpt,topo MOE_TOP_K=2 \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_scpt_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=5 MOE_EXPERTS=geo,topo MOE_TOP_K=2 \
    "$PYBIN" pred_GWN_16_adpAdj_topomoe.py "${BASE[@]}" "${FORE_TOPO[@]}" "${EXTRA_PRED[@]}" \
    2>&1 | tee "${pdir}/B_geo_topo_s${seed}.log" >/dev/null &

  CUDA_VISIBLE_DEVICES=6 MOE_EXPERTS=scpt,geo,topo MOE_TOP_K=2 \
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

printf '%s\n' "[ALL DONE] topo_moe seed sweep finished, n=5. Logs under: ${LOGROOT}/<seed>/est|pred/"
cd "$ROOT_DIR" && python3 aggregate_topomoe_5seed.py --out TOPOMOe_RUN_REPORT_METRLA_neg1.md
printf '%s\n' "[ALL DONE] wrote TOPOMOe_RUN_REPORT_METRLA_neg1.md (aggregate_topomoe_5seed.py)"
