#!/usr/bin/env bash
# Remaining supplement jobs on GPU 0 (skip done / already running elsewhere).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
LOGROOT="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
GPU="${SUPPLEMENT_GPU:-0}"
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

is_done () {
  local log="$1"
  [[ -f "$log" ]] || return 1
  "$PYBIN" - "$log" <<'PY'
import sys
from pathlib import Path
raise SystemExit(0 if "SCRIPT DURATION" in Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore") else 1)
PY
}

run_pemsbay () {
  local cfg="$1" experts="$2" topk="$3"
  shift 3
  for seed in "$@"; do
    local log="${LOGROOT}/PEMSBAY/${seed}/est/A_${cfg}_s${seed}.log"
    mkdir -p "$(dirname "$log")"
    if is_done "$log"; then
      echo "[skip] GPU${GPU} PEMSBAY ${cfg} seed=${seed}"
      continue
    fi
    if [[ -f "$log" ]] && pgrep -af "PEMSBAY.*${seed}.*${cfg}" >/dev/null 2>&1; then
      echo "[skip] GPU${GPU} PEMSBAY ${cfg} seed=${seed} (already running)"
      continue
    fi
    echo "[run] GPU${GPU} PEMSBAY ${cfg} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
      MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
      MOE_RUN_TAG="est_geotopo_pemsbay_${cfg}_s${seed}" \
      "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
      1 0.7 0 "${seed}" 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
      "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
  done
}

echo "[gpu0-supplement] start $(date) GPU=${GPU}"
# 250 geo_only 通常在 GPU1；此处只补 geo_topo
run_pemsbay geo_topo geo,topo 2 250
echo "[done] GPU${GPU} $(date)"
