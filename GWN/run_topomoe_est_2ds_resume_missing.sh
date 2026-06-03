#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SEEDS=(100 42 999 555 250)
DATASETS=(PEMSBAY PEMSD7M)
LOGROOT="${EST2DS_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_2ds_seed5_imgbase}"
mkdir -p "$LOGROOT"

# Keep args aligned with TOPOMOe_STAGE1_REPORT_imgbase.md
base_args_for () {
  local seed="$1"
  local dname="$2"
  echo "1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
}

EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

is_done () {
  local log="$1"
  if [[ ! -f "$log" ]]; then
    return 1
  fi
  python - "$log" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    t = p.read_text(encoding="utf-8", errors="ignore")
except Exception:
    sys.exit(1)
sys.exit(0 if "SCRIPT DURATION" in t else 1)
PY
}

pick_gpu_order () {
  # Sort by current memory usage (ascending) and print indices.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1" "$2}' \
    | sort -k2,2n \
    | awk '{print $1}'
}

run_with_retry () {
  local dataset="$1"
  local seed="$2"
  local cfg="$3"
  local experts="$4"
  local topk="$5"
  local pretrain_flag="$6"
  local log="$7"

  read -r -a BASE <<<"$(base_args_for "$seed" "$dataset")"
  BASE[0]="${pretrain_flag}"

  while true; do
    local tried=0
    while read -r gpu; do
      [[ -z "${gpu}" ]] && continue
      tried=1
      echo "[retry] ds=${dataset} seed=${seed} cfg=${cfg} gpu=${gpu}"
      if CUDA_VISIBLE_DEVICES="${gpu}" MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
           MOE_RUN_TAG="est2ds_${dataset,,}_${cfg}_s${seed}" \
           "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
           "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}" >/dev/null; then
        echo "[done] ds=${dataset} seed=${seed} cfg=${cfg}"
        return 0
      fi
      # If finished successfully by log check (rare race), treat as done.
      if is_done "${log}"; then
        echo "[done-after-check] ds=${dataset} seed=${seed} cfg=${cfg}"
        return 0
      fi
      # If OOM, try next GPU; otherwise also continue retry cycle.
      if python - "${log}" <<'PY'
import sys
from pathlib import Path
t = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
sys.exit(0 if ("OutOfMemoryError" in t or "CUDA error: out of memory" in t) else 1)
PY
      then
        echo "[oom] ds=${dataset} seed=${seed} cfg=${cfg} gpu=${gpu}, trying next gpu..."
      else
        echo "[fail] ds=${dataset} seed=${seed} cfg=${cfg} gpu=${gpu}, retrying on other gpu..."
      fi
    done < <(pick_gpu_order)

    if [[ "${tried}" -eq 0 ]]; then
      echo "[wait] no GPUs listed; sleep 60s"
      sleep 60
    else
      echo "[retry-cycle] ds=${dataset} seed=${seed} cfg=${cfg} sleep 30s"
      sleep 30
    fi
  done
}

run_missing () {
  local dataset="$1"
  local seed="$2"
  local cfg="$3"
  local experts="$4"
  local topk="$5"
  local pretrain_flag="$6"
  local ddir="${LOGROOT}/${dataset}/${seed}/est"
  mkdir -p "$ddir"
  local log="${ddir}/A_${cfg}_s${seed}.log"

  if is_done "${log}"; then
    echo "[skip] done ds=${dataset} seed=${seed} cfg=${cfg}"
    return 0
  fi
  echo "[run]  ds=${dataset} seed=${seed} cfg=${cfg}"
  run_with_retry "${dataset}" "${seed}" "${cfg}" "${experts}" "${topk}" "${pretrain_flag}" "${log}"
}

for SEED in "${SEEDS[@]}"; do
  echo "======== resume seed=${SEED} ========"
  for DS in "${DATASETS[@]}"; do
    run_missing "${DS}" "${SEED}" "no_pretrain" "scpt" 1 0
    run_missing "${DS}" "${SEED}" "scpt_only" "scpt" 1 1
    run_missing "${DS}" "${SEED}" "topo_only" "topo" 1 1
    run_missing "${DS}" "${SEED}" "scpt_topo" "scpt,topo" 2 1
  done
done

echo "[all done] all missing jobs finished; generating markdown..."
"$PYBIN" aggregate_topomoe_est_2ds_5seed.py --logroot "${LOGROOT}" --out "PEMSBAY_PEMSD7M_topo_Estimation.md"
echo "[done] report written: ${ROOT_DIR}/PEMSBAY_PEMSD7M_topo_Estimation.md"
