# TopoMoE Estimation 运行与超参搜索规格（HPO Spec）

> 本文档供 AI / AutoML 工具（如 GPT Pro、Optuna、Ray Tune）直接解析，描述当前仓库中 **Estimation（缺失值估计）** 任务的入口脚本、参数表、批量跑法、日志解析与建议搜索空间。

---

## 1. 任务标识

```yaml
task_id: topomoe_estimation_2ds
task_type: estimation          # 缺失值估计 / imputation，不是 forecasting
workspace_root: /mnt/data728/linhui/9991/5_11/GWN
python_executable: /mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python
main_script: pred_maskpredition_GWN_scpt_geo_topomoe.py
base_script_wrapped: pred_maskpredition_GWN_scpt_geo.py   # 被 topomoe 包装，勿直接改训练逻辑
utility_module: topo_moe_utils.py
batch_runner_primary: run_topomoe_est_2ds_resume_all8.sh
batch_runner_alternative: run_topomoe_est_2ds_5seed_8gpu.sh
report_aggregator: aggregate_topomoe_est_2ds_5seed.py
report_output_default: PEMSBAY_PEMSD7M_topo_Estimation.md
```

---

## 2. 单次运行模板（最小可执行单元）

```bash
cd /mnt/data728/linhui/9991/5_11/GWN

export CUDA_VISIBLE_DEVICES=<GPU_ID>          # 例如 0
export MOE_EXPERTS=<EXPERTS>                  # 见第 5 节
export MOE_TOP_K=<TOPK>                       # 整数
export MOE_RUN_TAG=<UNIQUE_TAG>               # 强烈建议每次试验唯一，避免覆盖 checkpoint

/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python \
  pred_maskpredition_GWN_scpt_geo_topomoe.py \
  <ARGV_1..20> \
  <ARGV_21..29>
```

### 当前默认参数（PEMSBAY / PEMSD7M Estimation）

```text
ARGV_1..20:
  1 0.7 0 <SEED> 1.0 <DATASET> -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320

ARGV_21..29:
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

### 四种实验配置（当前 sweep 固定组合）

| config_name | argv[1] IS_PRETRN | MOE_EXPERTS | MOE_TOP_K | 说明 |
|---|---:|---|---:|---|
| no_pretrain | 0 | scpt | 1 | 无预训练基线 |
| scpt_only | 1 | scpt | 1 | 仅 temporal/scpt 专家 |
| topo_only | 1 | topo | 1 | 仅 topology 专家 |
| scpt_topo | 1 | scpt,topo | 2 | temporal + topology |

---

## 3. argv 参数表（位置参数，1-based，不含脚本名）

> 规则：可用**同名环境变量覆盖** argv（见 `pred_maskpredition_GWN_scpt_geo_topomoe.py` 的 `_arg()`）。

### argv[1]..argv[20]（来自 `pred_maskpredition_GWN_scpt_geo.py:get_argv`）

| index | param_name | type | current_default | meaning |
|---:|---|---|---|---|
| 1 | IS_PRETRN | bool(0/1) | 1 | 是否做预训练；no_pretrain 设为 0 |
| 2 | R_TRN | float | 0.7 | 空间划分训练节点比例 |
| 3 | IS_EPOCH_1 | bool(0/1) | 0 | 是否只跑 1 epoch（调试） |
| 4 | seed | int | 100/42/... | 随机种子 |
| 5 | TEMPERATURE | float | 1.0 | 对比学习温度 |
| 6 | DATANAME | str | PEMSBAY / PEMSD7M | 数据集名 |
| 7 | seed_SS | int | -1 | 子采样种子（-1 表示默认） |
| 8 | IS_DESEASONED | bool(0/1) | 1 | 是否去季节 |
| 9 | weight_decay | float | 0.0 | 权重衰减 |
| 10 | adp_adj | bool(0/1) | 1 | 自适应邻接 |
| 11 | is_SGA | bool(0/1) | 1 | SGA 开关 |
| 12 | FEATURES | int | 2 | 输入特征维 |
| 13 | SUBGRAPH_SIZE | int | 64 | 子图大小 |
| 14 | QUOTIENT_GRAPH_RADIUS | float | 0.01 | 商图半径 |
| 15 | PRETRN_EPOCH | int | 100 | 预训练 epoch |
| 16 | EPOCH | int | 100 | 主训练 epoch |
| 17 | NETWORK_CALLS | bool(0/1) | 0 | 网络调用开关 |
| 18 | PRE_LEARN | float | 0.001 | 预训练学习率 |
| 19 | GRAPH_NORM | bool(0/1) | 1 | 图归一化 |
| 20 | HIDDEN | int | 320 | GWN 隐层维度 |

### argv[21]..argv[29]（TopoMoE 扩展，`get_argv_topomoe_estimation`）

| index | param_name | env_override | type | current_default | meaning |
|---:|---|---|---|---|---|
| 21 | FUSION_MODE | FUSION_MODE | str | topo_moe | 融合模式：`topo_moe` 或 `sparse_moe` |
| 22 | GATE_HIDDEN | GATE_HIDDEN | int | 64 | MoE gate MLP 隐层 |
| 23 | TOPO_LAP_K | TOPO_LAP_K | int | 16 | 拓扑拉普拉斯特征数 |
| 24 | MOE_TOP_K | MOE_TOP_K | int | 2（脚本里常设 1） | 每节点激活专家数 |
| 25 | MOE_TAU | MOE_TAU | float | 1.0 | router 温度 |
| 26 | MOE_LB_REG | MOE_LB_REG | float | 0.001 | load-balance 正则 |
| 27 | MOE_SMOOTH_REG | MOE_SMOOTH_REG | float | 0.001 | 平滑正则 |
| 28 | MOE_DELTA_REG | MOE_DELTA_REG | float | 0.0 | delta L2 正则 |
| 29 | MOE_USE_CTX | MOE_USE_CTX | bool(0/1) | 1 | 是否使用 MoE context 特征 |

### argv[30]（可选）

| index | param_name | env_override | meaning |
|---:|---|---|---|
| 30 | MOE_EXPERTS | MOE_EXPERTS | 专家列表，逗号分隔；**推荐用环境变量设置** |

---

## 4. 环境变量

```yaml
required_for_sweep:
  CUDA_VISIBLE_DEVICES: "<int>"     # GPU
  MOE_EXPERTS: "<csv>"              # scpt | geo | topo 及别名
  MOE_TOP_K: "<int>"
  MOE_RUN_TAG: "<unique_string>"    # 输出子目录标签，防覆盖

optional_overrides:
  FUSION_MODE: "topo_moe"
  GATE_HIDDEN: "64"
  TOPO_LAP_K: "16"
  MOE_TAU: "1.0"
  MOE_LB_REG: "0.001"
  MOE_SMOOTH_REG: "0.001"
  MOE_DELTA_REG: "0.0"
  MOE_USE_CTX: "1"
  MOE_FORCE_EXPERT: ""            # 诊断用，强制单专家
  MOE_INIT_TEMPORAL_BIAS: "1.0"
  TOPO_FORCE_RECOMPUTE: "0"       # 1=强制重算 topology embedding
  MOE_AUTO_TAG: "1"               # 0=关闭自动子目录命名
  FORECASTING_EVAL_TST_A_DIR: ""  # 仅特殊 eval，Estimation 通常不用
  EST2DS_LOGROOT: ""              # 批量脚本日志根目录，默认 logs_topomoe/est_2ds_seed5_imgbase
  PYBIN: ""                       # Python 路径，默认 AutoTSenv
```

### MOE_EXPERTS 别名映射

```text
scpt, temp, tmp, temporal      -> temporal
geo, geom, geometric           -> geometric
topo, lap, laplacian, topology -> topology
```

---

## 5. 四种配置的完整命令示例

### 5.1 SCPT only（有预训练）

```bash
cd /mnt/data728/linhui/9991/5_11/GWN
CUDA_VISIBLE_DEVICES=0 \
MOE_EXPERTS=scpt MOE_TOP_K=1 MOE_RUN_TAG=hp_scpt_s100 \
/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python \
  pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

### 5.2 TOPO only

```bash
CUDA_VISIBLE_DEVICES=0 \
MOE_EXPERTS=topo MOE_TOP_K=1 MOE_RUN_TAG=hp_topo_s100 \
/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python \
  pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

### 5.3 SCPT + TOPO

```bash
CUDA_VISIBLE_DEVICES=0 \
MOE_EXPERTS=scpt,topo MOE_TOP_K=2 MOE_RUN_TAG=hp_scpt_topo_s100 \
/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python \
  pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

### 5.4 No pretrain（注意 argv[1]=0）

```bash
CUDA_VISIBLE_DEVICES=0 \
MOE_EXPERTS=scpt MOE_TOP_K=1 MOE_RUN_TAG=hp_nopre_s100 \
/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python \
  pred_maskpredition_GWN_scpt_geo_topomoe.py \
  0 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

---

## 6. 批量跑法

### 6.1 8 卡并行 + 断点续跑（resume，推荐）

```bash
cd /mnt/data728/linhui/9991/5_11/GWN
bash ./run_topomoe_est_2ds_resume_all8.sh
```

行为：

- 数据集：`PEMSBAY`, `PEMSD7M`
- 种子：`100, 42, 999, 555, 250`
- 配置：`no_pretrain`, `scpt_only`, `topo_only`, `scpt_topo`
- 日志根目录：`logs_topomoe/est_2ds_seed5_imgbase`（可用 `EST2DS_LOGROOT` 覆盖）
- 完成判定：日志含 `SCRIPT DURATION`
- 结束后自动汇总：`aggregate_topomoe_est_2ds_5seed.py` → `PEMSBAY_PEMSD7M_topo_Estimation.md`

### 6.2 按 seed 一轮 8 任务（非 resume）

```bash
cd /mnt/data728/linhui/9991/5_11/GWN
bash ./run_topomoe_est_2ds_5seed_8gpu.sh
```

### 6.3 批量脚本内部 launch 逻辑（摘自 `run_topomoe_est_2ds_resume_all8.sh`）

```bash
CUDA_VISIBLE_DEVICES="${gpu}" \
MOE_EXPERTS="${experts}" \
MOE_TOP_K="${topk}" \
MOE_RUN_TAG="est2ds_${ds}_${cfg}_s${seed}" \
"$PYBIN" pred_maskpredition_GWN_scpt_geo_topomoe.py \
  "${BASE[@]}" "${EXTRA_TOPO[@]}" \
  2>&1 | tee "${log}" >/dev/null &
```

其中 `BASE` 默认 `1 0.7 0 ${seed} 1.0 ${dname} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`，`no_pretrain` 时将 `BASE[0]` 改为 `0`。

---

## 7. 日志与产物路径

```yaml
log_path_pattern: logs_topomoe/est_2ds_seed5_imgbase/<DATASET>/<SEED>/est/A_<config>_s<SEED>.log

config_to_log_prefix:
  no_pretrain: A_no_pretrain
  scpt_only:   A_scpt_only
  topo_only:   A_topo_only
  scpt_topo:   A_scpt_topo

examples:
  - logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/100/est/A_scpt_topo_s100.log
  - logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/42/est/A_topo_only_s42.log

checkpoint_dir_pattern: ../save/<KEYWORD>/<MOE_RUN_TAG>/
  # KEYWORD 由脚本根据时间戳/参数生成，日志里可搜 "KEYWORD"
```

---

## 8. 成功判定与指标解析

### 8.1 Run 成功条件

```text
log file exists AND contains line matching:
  ^SCRIPT DURATION
```

### 8.2 指标正则（来自 `aggregate_topomoe_est_2ds_5seed.py`）

```regex
^SCRIPT DURATION\s+(.+?)\s*$
GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)
```

| split | 含义 | 用途 |
|---|---|---|
| `tst_u` | unseen 节点测试 | **主指标，建议 optimize** |
| `tst_a` | all-node transductive 测试 | 辅助（训练节点会重叠，易偏乐观） |

### 8.3 建议优化目标

```yaml
primary_objective: minimize tst_u_MAE
secondary_objectives:
  - minimize tst_u_RMSE
  - minimize tst_u_MAPE
constraint_metrics:
  - tst_a_MAE    # 仅作参考
early_stop_signal_in_log: "Best val_u loss"
model_selection_checkpoint: "{name}_best.pt"   # 按 val_u 选最优
```

### 8.4 指标解读说明

- **`tst_u`**：对应 `spatialSplit_unseen`，检验未见节点上的泛化，更贴合 Estimation 核心问题。
- **`tst_a`**：对应 `spatialSplit_allNod`（transductive），测试集包含训练见过的节点，反映全节点平均重建效果，不宜单独作为主结论。

---

## 9. 超参搜索建议空间

```yaml
search_space:
  config_name: [no_pretrain, scpt_only, topo_only, scpt_topo]
  DATANAME: [PEMSBAY, PEMSD7M]
  seed: [100, 42, 999, 555, 250]

  PRETRN_EPOCH: {type: choice, values: [50, 100]}
  EPOCH: {type: choice, values: [50, 100]}

  GATE_HIDDEN: {type: choice, values: [32, 64, 128]}
  TOPO_LAP_K: {type: choice, values: [8, 16, 32]}
  MOE_TAU: {type: loguniform, low: 0.3, high: 2.0}
  MOE_LB_REG: {type: loguniform, low: 1e-5, high: 1e-2}
  MOE_SMOOTH_REG: {type: loguniform, low: 1e-5, high: 1e-2}
  MOE_DELTA_REG: {type: choice, values: [0.0, 1e-4, 1e-3]}
  MOE_USE_CTX: {type: choice, values: [0, 1]}

  PRE_LEARN: {type: loguniform, low: 1e-4, high: 5e-3}
  weight_decay: {type: choice, values: [0.0, 1e-4, 1e-3]}
  HIDDEN: {type: choice, values: [256, 320, 384]}
  SUBGRAPH_SIZE: {type: choice, values: [32, 64]}

fixed_by_config_mapping:
  no_pretrain: {IS_PRETRN: 0, MOE_EXPERTS: "scpt", MOE_TOP_K: 1}
  scpt_only:   {IS_PRETRN: 1, MOE_EXPERTS: "scpt", MOE_TOP_K: 1}
  topo_only:   {IS_PRETRN: 1, MOE_EXPERTS: "topo", MOE_TOP_K: 1}
  scpt_topo:   {IS_PRETRN: 1, MOE_EXPERTS: "scpt,topo", MOE_TOP_K: 2}
```

---

## 10. Trial 生成伪代码（Optuna / Ray Tune）

```python
def build_estimation_command(trial, gpu_id):
    cfg = trial.suggest_categorical(
        "config_name", ["no_pretrain", "scpt_only", "topo_only", "scpt_topo"]
    )
    seed = trial.suggest_categorical("seed", [100, 42, 999, 555, 250])
    dataset = trial.suggest_categorical("DATANAME", ["PEMSBAY", "PEMSD7M"])

    mapping = {
        "no_pretrain": dict(IS_PRETRN=0, experts="scpt", topk=1),
        "scpt_only":   dict(IS_PRETRN=1, experts="scpt", topk=1),
        "topo_only":   dict(IS_PRETRN=1, experts="topo", topk=1),
        "scpt_topo":   dict(IS_PRETRN=1, experts="scpt,topo", topk=2),
    }[cfg]

    gate_hidden = trial.suggest_categorical("GATE_HIDDEN", [32, 64, 128])
    topo_lap_k = trial.suggest_categorical("TOPO_LAP_K", [8, 16, 32])
    moe_tau = trial.suggest_float("MOE_TAU", 0.3, 2.0, log=True)
    moe_lb = trial.suggest_float("MOE_LB_REG", 1e-5, 1e-2, log=True)
    moe_sm = trial.suggest_float("MOE_SMOOTH_REG", 1e-5, 1e-2, log=True)
    moe_delta = trial.suggest_categorical("MOE_DELTA_REG", [0.0, 1e-4, 1e-3])
    moe_use_ctx = trial.suggest_categorical("MOE_USE_CTX", [0, 1])

    pretrn_epoch = trial.suggest_categorical("PRETRN_EPOCH", [50, 100])
    epoch = trial.suggest_categorical("EPOCH", [50, 100])
    pre_learn = trial.suggest_float("PRE_LEARN", 1e-4, 5e-3, log=True)
    hidden = trial.suggest_categorical("HIDDEN", [256, 320, 384])

    argv = [
        mapping["IS_PRETRN"], 0.7, 0, seed, 1.0, dataset, -1, 1, 0.0, 1, 1, 2,
        64, 0.01, pretrn_epoch, epoch, 0, pre_learn, 1, hidden,
        "topo_moe", gate_hidden, topo_lap_k, mapping["topk"],
        moe_tau, moe_lb, moe_sm, moe_delta, moe_use_ctx,
    ]

    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "MOE_EXPERTS": mapping["experts"],
        "MOE_TOP_K": str(mapping["topk"]),
        "MOE_RUN_TAG": f"hpo_{dataset}_{cfg}_s{seed}_t{trial.number}",
    }
    return argv, env


def parse_tst_u_mae(log_text: str) -> float | None:
    import re
    m = re.search(
        r"GraphWaveNet,\s*tst_u,\s*Masked MAE:\s*([0-9.]+)", log_text
    )
    return float(m.group(1)) if m else None


def run_success(log_text: str) -> bool:
    return "SCRIPT DURATION" in log_text
```

---

## 11. 汇总报告命令

```bash
cd /mnt/data728/linhui/9991/5_11/GWN
/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python \
  aggregate_topomoe_est_2ds_5seed.py \
  --logroot logs_topomoe/est_2ds_seed5_imgbase \
  --out PEMSBAY_PEMSD7M_topo_Estimation.md
```

---

## 12. 相关文件索引

| 文件 | 作用 |
|---|---|
| `pred_maskpredition_GWN_scpt_geo_topomoe.py` | Estimation + TopoMoE 入口 |
| `pred_maskpredition_GWN_scpt_geo.py` | 基础 Estimation 逻辑 |
| `topo_moe_utils.py` | MoE 融合、拓扑 embedding |
| `run_topomoe_est_2ds_resume_all8.sh` | 8 卡并行 + resume 批量跑 |
| `run_topomoe_est_2ds_5seed_8gpu.sh` | 按 seed 批量跑 |
| `aggregate_topomoe_est_2ds_5seed.py` | 日志汇总为 Markdown 报告 |
| `PEMSBAY_PEMSD7M_topo_Estimation.md` | 当前实验结果报告 |
| `README_RUN_TOPO_MOE.md` | MoE 使用说明（含 METRLA 示例） |
| `hpo_estimation_runner.py` | **6 卡 HPO 调度器**（topo_only / scpt_topo） |
| `hpo_estimation_live_report.py` | 从 `hpo_state.json` 生成实时榜单 |
| `run_hpo_estimation_6gpu.sh` | 后台启动 HPO |
| `HPO_ESTIMATION_LIVE.md` | **实时排名**（每完成一个子任务自动刷新） |

---

## 13. 6 卡超参搜索（已实现）

### 13.1 约束（按你的要求）

```yaml
configs: [topo_only, scpt_topo]          # 仅这两种
seeds: [100, 42]                         # 2 个种子
datasets: [PEMSBAY, PEMSD7M, METRLA]    # METRLA 放最后
gpus: [0, 1, 3, 4, 6, 7]                # 6 卡并行，留 2 卡
primary_metric: tst_u Masked MAE         # 越小越好
live_report: HPO_ESTIMATION_LIVE.md      # 每完成一个子任务即更新
```

每个 **trial** = 一组超参；包含 **12 个子任务** = 2 配置 × 3 数据集 × 2 种子。

### 13.2 启动

```bash
cd /mnt/data728/linhui/9991/5_11/GWN

# 默认 24 组超参 trial（可改环境变量）
N_TRIALS=24 bash ./run_hpo_estimation_6gpu.sh

# 或前台跑（调试用）
/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python hpo_estimation_runner.py --n-trials 24
```

### 13.3 随时看排名

```bash
less +F /mnt/data728/linhui/9991/5_11/GWN/HPO_ESTIMATION_LIVE.md
```

榜单含：总进度、Trial 总榜、分 TOPO / SCPT+TOPO 榜、各数据集明细、最近完成记录。

### 13.4 仅刷新榜单（不训练）

```bash
python hpo_estimation_live_report.py
# 或
python hpo_estimation_runner.py --report-only
```

### 13.5 状态与日志路径

```text
logs_hpo_est/hpo_state.json
logs_hpo_est/hpo_runner.nohup.log
logs_hpo_est/trial_<id>/<config>/<dataset>/s<seed>.log
```

---

*文档生成自当前仓库实际脚本；路径与默认值以 `run_topomoe_est_2ds_resume_all8.sh` 及 `pred_maskpredition_GWN_scpt_geo_topomoe.py` 为准。*
