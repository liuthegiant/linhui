#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(100 42 999 555 250)
DATASETS=(PEMSBAY PEMSD7M)
LOGROOT="${EST2DS_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_2ds_seed5_imgbase}"
mkdir -p "$LOGROOT"

# BASE argv[1]..[20] aligned with TOPOMOe_STAGE1_REPORT_imgbase.md
base_args_for () {
  local seed="$1"
  local dname="$2"
  # NOTE:
  # - no_pretrain baseline switches argv[1] to 0 in run_one.
  # - other args remain consistent with the METRLA stage-1 settings.
  echo "1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

run_one () {
  local gpu="$1"
  local dataset="$2"
  local seed="$3"
  local cfg="$4"
  local experts="$5"
  local topk="$6"
  local pretrain_flag="$7"

  local ddir="${LOGROOT}/${dataset}/${seed}/est"
  mkdir -p "$ddir"
  read -r -a BASE <<<"$(base_args_for "$seed" "$dataset")"
  BASE[0]="${pretrain_flag}"

  local log="${ddir}/A_${cfg}_s${seed}.log"
  echo "[launch] ds=${dataset} seed=${seed} cfg=${cfg} gpu=${gpu} pretrain=${pretrain_flag} experts=${experts} topk=${topk}"
  CUDA_VISIBLE_DEVICES="${gpu}" MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" MOE_RUN_TAG="est2ds_${dataset,,}_${cfg}_s${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py "${BASE[@]}" "${EXTRA_TOPO[@]}" \
    2>&1 | tee "${log}" >/dev/null &
}

run_seed_round () {
  local seed="$1"
  echo "======== seed=${seed} ========"

  # 8 GPUs in parallel: 2 datasets x 4 configs
  # Configs: no_pretrain, scpt, topo, scpt_topo
  run_one 0 PEMSBAY "${seed}" "no_pretrain" "scpt" 1 0
  run_one 1 PEMSBAY "${seed}" "scpt_only" "scpt" 1 1
  run_one 2 PEMSBAY "${seed}" "topo_only" "topo" 1 1
  run_one 3 PEMSBAY "${seed}" "scpt_topo" "scpt,topo" 2 1

  run_one 4 PEMSD7M "${seed}" "no_pretrain" "scpt" 1 0
  run_one 5 PEMSD7M "${seed}" "scpt_only" "scpt" 1 1
  run_one 6 PEMSD7M "${seed}" "topo_only" "topo" 1 1
  run_one 7 PEMSD7M "${seed}" "scpt_topo" "scpt,topo" 2 1

  wait
  echo "[seed=${seed}] all 8 runs DONE"
}

for SEED in "${SEEDS[@]}"; do
  run_seed_round "${SEED}"
done

echo "[ALL DONE] estimation 2-dataset 5-seed finished: ${LOGROOT}"
