# Virtual-Node SplitMask Estimation — GEO+TOPO（Geo Feature=2..5）

- **任务**：`geo+topo` 的 Estimation（splitmask：`tst_u` / `tst_v` / `tst_a`）
- **目标**：固定你之前筛出的 5-seed，调 `geo` 的 `FEATURES`（`argv[12]`）从 **2 到 5**
- **数据集**：`METRLA` / `PEMSBAY` / `PEMSD7M`
- **GPU**：`0,2,3,5`
- **总任务数**：`3 datasets x 5 seeds x 4 features = 60` runs

---

## 固定 5-seed（沿用之前结论）

| 数据集 | 种子 |
| --- | --- |
| METRLA | `42, 88, 100, 250, 432` |
| PEMSBAY | `42, 66, 88, 233, 999` |
| PEMSD7M | `42, 100, 432, 555, 999` |

---

## 变体定义（仅 GEO 参数）

- 专家配置固定为 **GEO+TOPO**：`MOE_EXPERTS=geo,topo`，`MOE_TOP_K=2`
- 预训练标志固定：`argv[1]=1`
- `FEATURES`（`argv[12]`）扫描：`2, 3, 4, 5`
- 其余参数与 splitmask 基线保持一致：
  - `1 0.7 0 <seed> 1.0 <dataset> -1 1 0.0 1 1 <FEATURES> 64 0.01 100 100 0 0.001 1 320`
  - TopoMoE 扩展：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`

---

## 批量执行（GPU=0,2,3,5）

已提供同目录脚本：`run_3ds_geotopo_geofeat2to5_gpu0235.sh`

```bash
cd /mnt/data728/linhui/9991/5_11/GWN
bash ./run_3ds_geotopo_geofeat2to5_gpu0235.sh
```

或直接使用内联版本：

```bash
cd /mnt/data728/linhui/9991/5_11/GWN

bash <<'BASH'
set -euo pipefail

PYBIN="${PYBIN:-/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python}"
SCRIPT="pred_maskpredition_GWN_scpt_geo_topomoe_virtualnode_splitmask.py"
GPUS=(0 2 3 5)
FEATURES_LIST=(2 3 4 5)
export FIXED_MASK_FRAC="${FIXED_MASK_FRAC:-0.05}"

declare -A SEEDS
SEEDS[METRLA]="42 88 100 250 432"
SEEDS[PEMSBAY]="42 66 88 233 999"
SEEDS[PEMSD7M]="42 100 432 555 999"

LOGROOT="${EST_3DS_GEOF2TO5_LOGROOT:-/mnt/data728/linhui/9991/5_11/GWN/logs_topomoe/est_3ds_virtualnode_splitmask_5seed_geotopo_geoF2to5}"
SCHED_LOG="${LOGROOT}/scheduler_geotopo_geoF2to5_gpu0235.log"
mkdir -p "$LOGROOT"

log() { printf '%s\n' "$*" | tee -a "$SCHED_LOG"; }
is_done() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  "$PYBIN" - "$f" <<'PY'
import sys
from pathlib import Path
txt = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
raise SystemExit(0 if "SCRIPT DURATION" in txt else 1)
PY
}

run_one() {
  local gpu="$1" ds="$2" seed="$3" feat="$4"
  local logf="${LOGROOT}/${ds}/${seed}/est/A_geo_topo_f${feat}_s${seed}.log"
  mkdir -p "$(dirname "$logf")"
  if is_done "$logf"; then
    log "[skip] ${ds} seed=${seed} feat=${feat}"
    return 0
  fi

  read -r -a BASE <<<"1 0.7 0 ${seed} 1.0 ${ds} -1 1 0.0 1 1 ${feat} 64 0.01 100 100 0 0.001 1 320"
  local EXTRA_TOPO=(topo_moe 64 16 2 1.0 0.001 0.001 0.0 1)

  log "[launch] gpu=${gpu} ds=${ds} seed=${seed} feat=${feat}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MOE_EXPERTS="geo,topo" \
    MOE_TOP_K="2" \
    MOE_RUN_TAG="est_vnode_split_${ds,,}_geo_topo_f${feat}_s${seed}" \
    "$PYBIN" "$SCRIPT" "${BASE[@]}" "${EXTRA_TOPO[@]}" 2>&1 | tee "$logf"
}

tasks=()
for ds in METRLA PEMSBAY PEMSD7M; do
  for seed in ${SEEDS[$ds]}; do
    for feat in "${FEATURES_LIST[@]}"; do
      tasks+=("${ds}|${seed}|${feat}")
    done
  done
done

log "[start] total_tasks=${#tasks[@]} gpus=${GPUS[*]} logroot=${LOGROOT}"

max_parallel=${#GPUS[@]}
i=0
pids=()
for t in "${tasks[@]}"; do
  IFS='|' read -r ds seed feat <<<"$t"
  gpu="${GPUS[$((i % max_parallel))]}"
  i=$((i + 1))

  run_one "$gpu" "$ds" "$seed" "$feat" &
  pids+=($!)
  if [[ ${#pids[@]} -ge $max_parallel ]]; then
    wait "${pids[0]}" || true
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

log "[done] all tasks finished."
BASH
```

---

## 结果汇总建议

- 日志目录：`logs_topomoe/est_3ds_virtualnode_splitmask_5seed_geotopo_geoF2to5`
- 每条日志命名：`A_geo_topo_f<feature>_s<seed>.log`
- 建议优先比较 `tst_v` 的 MAE（同时看 `tst_u` / `tst_a` 是否稳定）
- 汇总粒度：
  1. 每数据集每 feature 的 5-seed 均值 ± std  
  2. 每数据集最佳 feature（按 `tst_v` MAE 最小）  
  3. 三数据集整体最佳 feature（可做简单 rank 或归一化平均）

---

## 结果表（待填）

### METRLA（5 seeds）

| Geo Feature | tst_u MAE | tst_v MAE | tst_a MAE | 备注 |
| --- | --- | --- | --- | --- |
| 2 | TBD | TBD | TBD | |
| 3 | TBD | TBD | TBD | |
| 4 | TBD | TBD | TBD | |
| 5 | TBD | TBD | TBD | |

### PEMSBAY（5 seeds）

| Geo Feature | tst_u MAE | tst_v MAE | tst_a MAE | 备注 |
| --- | --- | --- | --- | --- |
| 2 | TBD | TBD | TBD | |
| 3 | TBD | TBD | TBD | |
| 4 | TBD | TBD | TBD | |
| 5 | TBD | TBD | TBD | |

### PEMSD7M（5 seeds）

| Geo Feature | tst_u MAE | tst_v MAE | tst_a MAE | 备注 |
| --- | --- | --- | --- | --- |
| 2 | TBD | TBD | TBD | |
| 3 | TBD | TBD | TBD | |
| 4 | TBD | TBD | TBD | |
| 5 | TBD | TBD | TBD | |

