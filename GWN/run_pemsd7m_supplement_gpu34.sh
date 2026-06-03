#!/usr/bin/env bash
# PEMSD7M supplement: geo_only on GPU3, geo_topo on GPU4 (seeds 42 999 555 250).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
LOGROOT="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)
SEEDS=(42 999 555 250)

is_done () {
  local log="$1"
  [[ -f "$log" ]] || return 1
  "$PYBIN" - "$log" <<'PY'
import sys
from pathlib import Path
raise SystemExit(0 if "SCRIPT DURATION" in Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore") else 1)
PY
}

run_cfg () {
  local gpu="$1" cfg="$2" experts="$3" topk="$4"
  for seed in "${SEEDS[@]}"; do
    local log="${LOGROOT}/PEMSD7M/${seed}/est/A_${cfg}_s${seed}.log"
    mkdir -p "$(dirname "$log")"
    if is_done "$log"; then
      echo "[skip] GPU${gpu} PEMSD7M ${cfg} seed=${seed}"
      continue
    fi
    echo "[run] GPU${gpu} PEMSD7M ${cfg} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      MOE_EXPERTS="${experts}" MOE_TOP_K="${topk}" \
      MOE_RUN_TAG="est_geotopo_pemsd7m_${cfg}_s${seed}" \
      "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
      1 0.7 0 "${seed}" 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
      "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
  done
  echo "[done] GPU${gpu} PEMSD7M ${cfg}"
}

echo "[pemsd7m-gpu34] LOGROOT=${LOGROOT} seeds=${SEEDS[*]}"
run_cfg 3 geo_only geo 1 &
run_cfg 4 geo_topo geo,topo 2 &
wait
echo "[all done] PEMSD7M supplement on GPU 3+4"
