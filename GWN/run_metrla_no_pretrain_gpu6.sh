#!/usr/bin/env bash
# METRLA supplement: no_pretrain x 5 seeds on GPU 6 only.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
LOGROOT="${EST_GEOTOPO_LOGROOT:-${ROOT_DIR}/logs_topomoe/est_geotopo_3ds_seed5}"
GPU="${METRLA_GPU:-6}"
EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

for seed in 100 42 999 555 250; do
  log="${LOGROOT}/METRLA/${seed}/est/A_no_pretrain_s${seed}.log"
  mkdir -p "$(dirname "$log")"
  if "$PYBIN" - "$log" <<'PY'; then
import sys
from pathlib import Path
raise SystemExit(0 if "SCRIPT DURATION" in Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore") else 1)
PY
    echo "[skip] seed=${seed} already done"
    continue
  fi
  echo "[run] METRLA no_pretrain seed=${seed} gpu=${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    MOE_EXPERTS=geo MOE_TOP_K=1 \
    MOE_RUN_TAG="est_geotopo_metrla_no_pretrain_s${seed}" \
    "$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
    0 0.7 0 "${seed}" 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
    "${EXTRA_TOPO[@]}" 2>&1 | tee "${log}"
done
echo "[done] METRLA no_pretrain all seeds"
