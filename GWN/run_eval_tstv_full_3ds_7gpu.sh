#!/usr/bin/env bash
# Re-run tst_v_full (full-graph forward, V nodes fully masked) using existing checkpoints.
# Uses logs to locate each run's save directory, then runs eval-only.
#
# GPUs: 0 1 2 3 4 6 7
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
# User request: keep GPU0 free; use 1-7 (including 5).
GPUS=(1 2 3 4 5 6 7)
MAX_PARALLEL=${#GPUS[@]}

OUTROOT="${EVAL_TSTVFULL_LOGROOT:-${ROOT_DIR}/logs_topomoe/eval_tstvfull_3ds_splitmask_seed10}"
SCHED_LOG="${OUTROOT}/eval_tstvfull_3ds_7gpu.log"
mkdir -p "$OUTROOT"
log() { printf '%s\n' "$*" | tee -a "$SCHED_LOG"; }

SEEDS=(100 42 999 555 250 88 66 233 38 432)
CONFIGS=(no_pretrain topo_only geo_only geo_topo)
DATASETS=(METRLA PEMSBAY PEMSD7M)

is_done () {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  ls "$dir"/*tst_v_full*_prediction.npy >/dev/null 2>&1
}

run_one () {
  local gpu="$1" ds="$2" seed="$3" cfg="$4"
  log "[launch-eval] gpu=${gpu} ${ds} cfg=${cfg} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "$PYBIN" eval_tst_v_full_from_logs.py \
      --datasets "${ds}" \
      --seeds "${seed}" \
      --configs "${cfg}" \
      --only-missing \
    >> "${OUTROOT}/${ds}_${cfg}_s${seed}.log" 2>&1
}

log "[eval-tstvfull] pid=$$ GPUs=${GPUS[*]} out=${OUTROOT}"

pending=()
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      pending+=("${ds}|${seed}|${cfg}")
    done
  done
done

log "pending=${#pending[@]} (3 ds * 10 seeds * 4 cfg = 120)"

i=0
pids=()
for item in "${pending[@]}"; do
  IFS='|' read -r ds seed cfg <<<"$item"
  gpu="${GPUS[$((i % MAX_PARALLEL))]}"
  i=$((i + 1))
  run_one "$gpu" "$ds" "$seed" "$cfg" &
  pids+=($!)
  if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
    wait "${pids[0]}" || true
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

log "[all done] eval tst_v_full finished."

