#!/usr/bin/env bash
# 6-GPU HPO: topo_only + scpt_topo | 2 seeds | PEMSBAY, PEMSD7M, METRLA (last)
# GPUs 0,1,3,4,6,7 — leaves 2 cards free
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
N_TRIALS="${N_TRIALS:-24}"
HPO_SEED="${HPO_SEED:-20260516}"
LOGDIR="${LOGDIR:-logs_hpo_est}"
NOHUP_LOG="${NOHUP_LOG:-${LOGDIR}/hpo_runner.nohup.log}"

mkdir -p "$LOGDIR"

echo "[HPO] root=$ROOT_DIR"
echo "[HPO] n_trials=$N_TRIALS sampler_seed=$HPO_SEED"
echo "[HPO] gpus=0,1,3,4,6,7"
echo "[HPO] live report -> HPO_ESTIMATION_LIVE.md"
echo "[HPO] state -> ${LOGDIR}/hpo_state.json"

nohup "$PYBIN" hpo_estimation_runner.py \
  --n-trials "$N_TRIALS" \
  --seed "$HPO_SEED" \
  --state "${LOGDIR}/hpo_state.json" \
  --live-md HPO_ESTIMATION_LIVE.md \
  --pybin "$PYBIN" \
  >> "$NOHUP_LOG" 2>&1 &

echo "[HPO] started pid=$! nohup_log=$NOHUP_LOG"
echo "[HPO] tail -f $NOHUP_LOG"
echo "[HPO] watch ranking: less +F HPO_ESTIMATION_LIVE.md"
