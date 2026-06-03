#!/usr/bin/env bash
# Run remaining supplement jobs on idle GPU 1, 3, 4 (skip in-progress / done).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
LOGROOT="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
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
  local gpu="$1" cfg="$2" experts="$3" topk="$4"
  shift 4
  for seed in "$@"; do
    local log="${LOGROOT}/PEMSBAY/${seed}/est/A_${cfg}_s${seed}.log"
    mkdir -p "$(dirname "$log")"
    if is_done "$log"; then
      echo "[skip] GPU${gpu} PEMSBAY ${cfg} seed=${seed}"
      continue
    fi
    echo "[run] GPU${gpu} PEMSBAY ${cfg} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
      MOE_RUN_TAG="est_geotopo_pemsbay_${cfg}_s${seed}" \
      "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
      1 0.7 0 "${seed}" 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
      "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
  done
  echo "[done] GPU${gpu} PEMSBAY ${cfg}"
}

run_metrla () {
  local gpu="$1"
  shift
  for seed in "$@"; do
    local log="${LOGROOT}/METRLA/${seed}/est/A_no_pretrain_s${seed}.log"
    mkdir -p "$(dirname "$log")"
    if is_done "$log"; then
      echo "[skip] GPU${gpu} METRLA no_pretrain seed=${seed}"
      continue
    fi
    echo "[run] GPU${gpu} METRLA no_pretrain seed=${seed}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      MOE_EXPERTS=geo MOE_TOP_K=1 \
      MOE_RUN_TAG="est_geotopo_metrla_no_pretrain_s${seed}" \
      "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
      0 0.7 0 "${seed}" 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
      "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
  done
  echo "[done] GPU${gpu} METRLA no_pretrain"
}

echo "[remaining-gpu134] start $(date)"
run_pemsbay 1 geo_only geo 1 555 250 &
run_pemsbay 3 geo_topo geo,topo 2 555 250 &
run_metrla 4 250 &
wait
echo "[all done] remaining supplement on GPU 1,3,4 $(date)"
