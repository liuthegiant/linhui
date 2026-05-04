# PRETRN_MODE / 无预训练 实验记录（PEMSD7M，`pred_GWN_16_adpAdj.py`）

**本版设定**：随机种子 **41–50**（共 10 组）；**8 张物理 GPU（0–7）** 并行跑；其余超参与脚本参数与 METRLA 记录一致。数据与邻接来自 **`9991/PEMSD7M`**（`V_228.csv`、`W_228.csv`、站点表等）。汇总为 **均值 ± 样本标准差（n=10）**，与 Python `statistics.mean` / `statistics.stdev`（分母 n−1）一致。

---

## 1. 结果与日志保存位置


| 类型               | 路径（相对仓库根目录 `forecasting-on-new-roads/`）                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| **本说明与汇总表**      | `GWN_PEMSD7M_predict12.md`（当前文件）                                               |
| **各次运行的控制台日志**   | `logs/pretrn_seed_sweep_pemsd7m_41_50/geo_s{seed}.log`、`temporal_s*.log`、`dual_s*.log`、`nopre_s*.log`      |
| **扫种总控 / nohup** | 例如 `logs/pretrn_seed_sweep_pemsd7m_41_50/nohup_runner.log`（`scripts/run_pemsd7m_pretrn_gpu8_seeds41_50.sh`） |
| **自动汇总脚本输出**     | 执行 `python3 scripts/fill_run_pretrn_pemsd7m_predgwn16_md.py` 后刷新第 4 节各表与表 11                                            |
| **每次实验的模型与预测**   | `P.PATH = '../save/' + P.KEYWORD` → `../save/pred_PEMSD7M_GraphWaveNet_<yyMMddHHmm>_<pid>/`（`KEYWORD` 由 `pred_GWN_16_adpAdj.py` 内 `pred_` + 数据集 + 模型名 + 时间 + pid 拼接） |
| **指标原文**         | 各日志中 `all pred steps` 行；`../save/pred_PEMSD7M_GraphWaveNet_*/GraphWaveNet_prediction_scores.txt`                                                |
| **耗时 JSON / 文本**   | 各次运行目录下 `paper_timing.json`、`paper_timing.txt`；控制台含 `=== TIME SUMMARY ===` 块（见第 5 节）                         |


工作目录：在 `forecasting-on-new-roads/` 下执行 `python pred_GWN_16_adpAdj.py ...`。数据集根路径为上一级目录中的 `PEMSD7M/`（与 `graph.py` 中 `../PEMSD7M/...` 一致）。

### 1.1 命令行参数位次（`pred_GWN_16_adpAdj.py`，与第 2 节 PEMSD7M 命令一致）

下列下标为 `sys.argv[i]`（`i=0` 为脚本路径）。**第 2 节四条命令**均显式给到 **`i=1`…`27`**；未出现在命令中的 **`28`…`34`** 在代码中取默认值（见源码 `get_argv()`）。

| i | 变量（代码中） | 本 MD §2（geo / temporal / dual / 无预训练）取值 | 说明 |
| --- | --- | --- | --- |
| 1 | `IS_PRETRN` | `1` / `1` / `1` / `0` | 是否做预训练段 |
| 2 | `R_TRN` | `0.7` | 训练集比例 |
| 3 | `IS_EPOCH_1` | `0` | 单 epoch 调试开关 |
| 4 | `seed` | `41`…`50` | 随机种子 |
| 5 | `TEMPERATURE` | `1.0` | 对比/温度 |
| 6 | `DATANAME` | `PEMSD7M` | 数据集名 |
| 7 | `seed_SS` | `-1` | 空间划分种子 |
| 8 | `IS_DESEASONED` | `1` | 是否去季节分支 |
| 9 | `weight_decay` | `0.0` | 权重衰减 |
| 10 | `adp_adj` | `1` | 自适应邻接 |
| 11 | `is_SGA` | `1` | SGA |
| 12 | `FEATURES` | `2` | 特征维 |
| 13 | `SUBGRAPH_SIZE` | `64` | 子图规模 |
| 14 | `QUOTIENT_GRAPH_RADIUS` | `0.01` | 商图半径 |
| 15 | `PRETRN_EPOCH` | `100` | 预训练 epoch |
| 16 | `EPOCH` | `100` | 主训练 epoch |
| 17 | `NETWORK_CALLS` | `0` | 网络调用调试 |
| 18 | `PRE_LEARN` | `0.001` | 预训练学习率 |
| 19 | `GRAPH_NORM` | `1` | 图归一化 |
| 20 | `HIDDEN` | `320` | 隐藏维 |
| 21 | `IS_DUAL_PRETRN` | `0` | 双编码器预训练开关 |
| 22 | `FUSE_ALPHA` | `0.5` | 融合 α |
| 23 | `SKIP_PRETRAIN` | `0` | 跳过预训练 |
| 24 | `TEMP_ENCODER_NAME` | `encoder` | 时序编码器 ckpt 名 |
| 25 | `GEO_ENCODER_NAME` | `encoder` | 几何编码器 ckpt 名（`dual` 且与 temporal 同名时会自动改为 `encoder_geo`） |
| 26 | `PRETRAIN_CKPT_DIR` | `""` | 空则使用本次 `P.PATH` |
| 27 | `PRETRN_MODE` | `geo` / `temporal` / `dual` / `dual` | 无预训练时末尾仍为 `dual`，仅 `IS_PRETRN=0` 使预训练段不执行 |
| 28 | `FUSION_MODE` | （默认 `temporal_delta`） | 融合方式 |
| 29 | `GATE_SCALE` | （默认 `1.0`） | 门控 scale |
| 30 | `GATE_BIAS` | （默认 `0.0`） | 门控 bias |
| 31 | `GATE_HIDDEN` | （默认 `64`） | 门控隐藏维 |
| 32 | `FUSION_NORM` | （默认 `1`） | 融合归一化 |
| 33 | `GATE_REG` | （默认 `0.0`） | 门控正则 |
| 34 | `DELTA_REG` | （默认 `0.0`） | delta 正则 |

### 1.2 程序文件（本实验直接相关的源码）

| 文件 | 作用 |
| --- | --- |
| `pred_GWN_16_adpAdj.py` | **训练/评测入口**；解析 `sys.argv`、数据加载、`PEMSD7M` 分支、`KEYWORD`/`PATH`、保存权重与 `paper_timing` |
| `GWN_SCPT_14_adpAdj_future12step.py` | **GraphWaveNet 骨干与训练逻辑**（`from … import *` 引入模型与辅助函数）；**12 步预测**（`out_dim=12`） |
| `graph.py` | 商图/子图、`load_dataset`、`generate_graphs` 等 |
| `unseen_nodes.py` | 空间划分（unseen 节点） |
| `Metrics.py` | 指标 |
| `../PEMSD7M/V_228.csv` | 流量矩阵（节点 228） |
| `../PEMSD7M/W_228.csv` | 邻接（与 `load_adj` / `P.ADJTYPE` 一致） |

### 1.3 单次运行目录内：模型权重与日志（`../save/pred_PEMSD7M_GraphWaveNet_* /`）

以下为**常见**产物；是否出现依赖 `PRETRN_MODE` / `IS_PRETRN` / `FUSION_MODE` 等。预训练权重与 `*_log.txt` 的**具体前缀**以 `get_argv()` 得到的 `TEMP_ENCODER_NAME`、`GEO_ENCODER_NAME` 及预训练分支为准（例如仅 `temporal` 时多为 `encoder.pt`；`dual` 时常同时有 `encoder.pt` 与 `encoder_geo.pt`）。路径均相对 `forecasting-on-new-roads/`。

| 类型 | 典型文件名 | 说明 |
| --- | --- | --- |
| **预训练（temporal）** | `encoder.pt`、`encoder_log.txt` | 时序编码器权重与预训练 epoch 日志 |
| **预训练（geo，或 dual 的几何支路）** | `encoder_geo.pt`、`encoder_geo_log.txt` | `dual` 时几何支路常为 `encoder_geo`（由 `GEO_ENCODER_NAME` 决定） |
| **主模型 checkpoint** | `GraphWaveNet_u.pt`、`GraphWaveNet_a.pt` | 分别在 unseen / all-node 验证最优上保存的预测网络权重 |
| **融合门控（dual + temporal_delta 等）** | `GraphWaveNet_fusion_u.pt`、`GraphWaveNet_fusion_a.pt`（或历史命名 `*_gate_*.pt`） | 双支路融合模块 |
| **主训练日志** | `GraphWaveNet_log.txt` | 主训练各 epoch 损失等 |
| **指标** | `GraphWaveNet_prediction_scores.txt` | 训练 MAE 与 test 指标摘要 |
| **耗时汇总** | `paper_timing.json`、`paper_timing.txt` | 与第 5 节 `TIME SUMMARY` 一致的分层计时 |
| **预测结果** | `GraphWaveNet_test_u_GraphWaveNet_*.npy`、`GraphWaveNet_test_a_GraphWaveNet_*.npy` | `prediction` / `groundtruth` |
| **控制台备份** | `logs/pretrn_seed_sweep_pemsd7m_41_50/geo_s<seed>.log` 等 | 与 §1 表「各次运行的控制台日志」对应，内含 `SCRIPT DURATION`、`all pred steps` |

**对应关系**：第 3 节每一进程的标准输出若 `tee` 到 `geo_s41.log` 等，则该日志与当次进程打印的 `KEYWORD` 一致；在日志中搜索 `pred_PEMSD7M_GraphWaveNet_` 即可得到本次 `../save/<KEYWORD>/` 目录名。

---

## 2. 环境与启动命令（四种配置，参数不变）

环境（每次会话前）：

```bash
cd /path/to/forecasting-on-new-roads
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate fonr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs/pretrn_seed_sweep_pemsd7m_41_50
```

将 `**<SEED>**` 替换为 `41` … `50`。  
前三组：`IS_PRETRN=1`，仅 `PRETRN_MODE` 不同；第四组：`IS_PRETRN=0`，为与预训练组对齐随机种子，使用**完整 argv**（末尾 `PRETRN_MODE` 占位为 `dual`，预训练阶段不执行）。

### 2.1 `PRETRN_MODE=geo`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo
```

### 2.2 `PRETRN_MODE=temporal`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal
```

### 2.3 `PRETRN_MODE=dual`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual
```

### 2.4 `IS_PRETRN=False`（无预训练）

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 0 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual
```

**超参说明**：除 `IS_PRETRN` / `PRETRN_MODE` / `seed` / `CUDA_VISIBLE_DEVICES` 外，与 METRLA 记录及脚本默认一致（如 `TEMPERATURE=1.0`、`seed_SS=-1`、`weight_decay=0`、`FEATURES=2`、`PRE_LEARN=0.001` 等）。

---

## 3. 八卡并行调度（GPU 0–7 同时跑）

共 **10 个种子 × 4 种配置 = 40 个作业**。每轮同时起 **8 个进程**，占满 **GPU 0–7**：同一轮内「两个连续种子」各跑齐 geo / temporal / dual / 无预训练。


| 轮次  | 种子 A（GPU 0–3）                     | 种子 B（GPU 4–7）                     |
| --- | --------------------------------- | --------------------------------- |
| 1   | 41：geo→0，temporal→1，dual→2，无预训练→3 | 42：geo→4，temporal→5，dual→6，无预训练→7 |
| 2   | 43                                | 44                                |
| 3   | 45                                | 46                                |
| 4   | 47                                | 48                                |
| 5   | 49                                | 50                                |


**示例（第一轮：种子 41 与 42）**：在仓库根目录执行；日志路径可按需修改。

```bash
LOG=logs/pretrn_seed_sweep_pemsd7m_41_50

# 种子 41：IS_PRETRN=1 三组 + IS_PRETRN=0 一组（参数与 2.1–2.4 一致，仅换 SEED 与 GPU）
CUDA_VISIBLE_DEVICES=0 python pred_GWN_16_adpAdj.py 1 0.7 0 41 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo       2>&1 | tee "$LOG/geo_s41.log" &
CUDA_VISIBLE_DEVICES=1 python pred_GWN_16_adpAdj.py 1 0.7 0 41 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal  2>&1 | tee "$LOG/temporal_s41.log" &
CUDA_VISIBLE_DEVICES=2 python pred_GWN_16_adpAdj.py 1 0.7 0 41 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/dual_s41.log" &
CUDA_VISIBLE_DEVICES=3 python pred_GWN_16_adpAdj.py 0 0.7 0 41 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/nopre_s41.log" &

CUDA_VISIBLE_DEVICES=4 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo       2>&1 | tee "$LOG/geo_s42.log" &
CUDA_VISIBLE_DEVICES=5 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal  2>&1 | tee "$LOG/temporal_s42.log" &
CUDA_VISIBLE_DEVICES=6 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/dual_s42.log" &
CUDA_VISIBLE_DEVICES=7 python pred_GWN_16_adpAdj.py 0 0.7 0 42 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/nopre_s42.log" &

wait
```

其余四轮将上述块中的 `41`/`42` 与日志名改为 `43`/`44`、`45`/`46`、`47`/`48`、`49`/`50`，每轮结束执行 `wait` 后再起下一轮。

**一键跑五轮（与上表等价）**：在仓库根目录执行 `bash scripts/run_pemsd7m_pretrn_gpu8_seeds41_50.sh`（可用 `nohup … >> logs/pretrn_seed_sweep_pemsd7m_41_50/nohup_runner.log 2>&1 &` 后台跑）。**全部日志齐全后**执行 `python3 scripts/fill_run_pretrn_pemsd7m_predgwn16_md.py` 自动重写下面第 4 节各表与表 11。

**说明**：`CUDA_VISIBLE_DEVICES` 仅重映射为进程内 `cuda:0`，**不改变**给定 `seed` 与 argv 下的可复现性；物理 GPU 编号仅表示机器上的卡位。

---

## 4. 指标表（`all pred steps`）

以下为 **十张** 按种子的四配置对比表（种子 **41–50**），以及 **第十一张** 汇总。**数据**由 `scripts/fill_run_pretrn_pemsd7m_predgwn16_md.py` 自 `logs/pretrn_seed_sweep_pemsd7m_41_50` 解析填入（UTC：`2026-04-08T10:15:16Z`）。**规则**：每个种子仅当 geo / temporal / dual / nopre 四份日志均可解析时，才写入该种子的整表；否则该表四行均为占位「—」，避免与目录内历史/残缺日志混写。

**进度**：十种子里已有 **10** 个四配置日志齐全（`41, 42, 43, 44, 45, 46, 47, 48, 49, 50`）。表 11 的 **n=10**（仅统计已齐种子）。

| 列 | 含义 |
| --- | --- |
| 脚本总时长 | 日志中 `SCRIPT DURATION` 行；未跑完为 — |
| test_u / test_a | 日志 `all pred steps` 行 |

### 表 1 — 随机种子 `seed=41`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:26:58.077962 | 42.6979711617 | 6.5343684593 | 3.2641660296 | 8.1858306828 | 42.9161902247 | 6.5510449720 | 3.5587220883 | 8.6578736761 |
| temporal | 1 | True | temporal | 0:20:58.982372 | 38.1063683099 | 6.1730355831 | 3.1676911485 | 7.6876596964 | 34.9461935639 | 5.9115305602 | 3.0469712150 | 7.9269972024 |
| dual | 2 | True | dual | 0:28:26.282283 | 37.9737375107 | 6.1622834656 | 3.1552881417 | 7.7334962945 | 36.5480309751 | 6.0454967517 | 3.1590908395 | 7.9716154696 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:20:44.668713 | 42.0670545723 | 6.4859120078 | 3.2799901918 | 8.7848904684 | 41.9433477477 | 6.4763684073 | 3.4547990056 | 9.4770674539 |

---

### 表 2 — 随机种子 `seed=42`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:28:42.927142 | 36.9816445877 | 6.0812535375 | 3.0822271492 | 7.4612205539 | 39.3038885600 | 6.2692813432 | 3.2848054051 | 9.0147193664 |
| temporal | 5 | True | temporal | 0:21:01.565145 | 32.4032621990 | 5.6923863361 | 2.8954104915 | 6.8073545476 | 38.3644068707 | 6.1939007799 | 3.4577846264 | 8.6191866150 |
| dual | 6 | True | dual | 0:27:09.437384 | 31.8127321027 | 5.6402776618 | 2.9366951040 | 6.9739544250 | 37.2246572763 | 6.1012012978 | 3.4414687645 | 8.7738727063 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:20:33.667686 | 37.3536365027 | 6.1117621438 | 3.1131662913 | 7.3266146424 | 42.4371589074 | 6.5143809305 | 3.5635718892 | 10.1338978914 |

---

### 表 3 — `seed=43`（第三轮：GPU 0–3 为 geo/temporal/dual/无预训练，GPU 4–7 为表 4）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:25:10.788809 | 48.3487414393 | 6.9533259264 | 3.5836146194 | 9.6494335885 | 42.6670914906 | 6.5320051661 | 3.6213023113 | 8.5068993261 |
| temporal | 1 | True | temporal | 0:21:14.243629 | 42.3415792134 | 6.5070407416 | 3.3297416419 | 8.1523211549 | 41.1254453987 | 6.4129123960 | 3.3123656603 | 9.0488300647 |
| dual | 2 | True | dual | 0:27:45.219260 | 43.9344737139 | 6.6283085108 | 3.4347030454 | 8.9096779695 | 43.0482439334 | 6.5611160585 | 3.7300037043 | 9.8316818455 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:20:53.494247 | 45.3917849560 | 6.7373425737 | 3.3747124877 | 8.7429130354 | 41.8582069280 | 6.4697918767 | 3.5563839900 | 10.0955975711 |

---

### 表 4 — `seed=44`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:25:34.941700 | 45.2805721116 | 6.7290840470 | 3.4438198110 | 8.9964957679 | 38.3381843563 | 6.1917836167 | 3.3586598878 | 8.9256179914 |
| temporal | 5 | True | temporal | 0:21:13.939787 | 41.1380990313 | 6.4138988947 | 3.3813689098 | 8.4929351244 | 37.1686862495 | 6.0966126865 | 3.1112367556 | 7.9201265793 |
| dual | 6 | True | dual | 0:26:47.572218 | 40.4719353615 | 6.3617556823 | 3.3700189636 | 8.2911444998 | 37.8147658177 | 6.1493711725 | 3.2647848914 | 8.4562972580 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:20:36.249324 | 44.5951454315 | 6.6779596758 | 3.4123049022 | 8.6198498023 | 43.0234506845 | 6.5592263785 | 3.4196279086 | 9.7524852787 |

---

### 表 5 — `seed=45`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:25:52.303825 | 47.5485984586 | 6.8955491774 | 3.6110607522 | 9.5532810962 | 39.6459994267 | 6.2965069226 | 3.2189987757 | 8.0831938185 |
| temporal | 1 | True | temporal | 0:20:42.813329 | 40.5947867923 | 6.3714038321 | 3.3960758183 | 9.4540175674 | 36.5212432211 | 6.0432808326 | 3.2342333595 | 8.3011287600 |
| dual | 2 | True | dual | 0:27:05.248518 | 44.3567316825 | 6.6600849606 | 3.6504916325 | 10.0207408141 | 31.0271215633 | 5.5701994186 | 2.8984287915 | 7.4853451159 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:20:16.633702 | 45.7218359393 | 6.7617923614 | 3.4865407965 | 9.0530095879 | 41.3900722610 | 6.4335116586 | 3.3779413392 | 9.2772772163 |

---

### 表 6 — `seed=46`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:25:43.421550 | 45.4850627050 | 6.7442614648 | 3.4173891585 | 8.6289391470 | 42.2806838175 | 6.5023598653 | 3.3172832322 | 9.1049112970 |
| temporal | 5 | True | temporal | 0:20:50.352544 | 39.0846307864 | 6.2517702122 | 3.2353548951 | 8.4166699485 | 34.2940996062 | 5.8561164270 | 3.1091723768 | 8.0063524031 |
| dual | 6 | True | dual | 0:27:09.994284 | 44.4250911427 | 6.6652150110 | 3.4183035374 | 8.8374662792 | 34.9513977136 | 5.9119707132 | 3.1069088881 | 7.7446175140 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:20:12.415629 | 44.9414243995 | 6.7038365433 | 3.3181484922 | 8.6894809222 | 39.7160810996 | 6.3020695886 | 3.3030091720 | 8.4691223141 |

---

### 表 7 — `seed=47`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:26:33.043594 | 32.4138877617 | 5.6933195731 | 2.8596950274 | 6.6907304000 | 38.4269411769 | 6.1989467797 | 3.2835032564 | 8.6807531994 |
| temporal | 1 | True | temporal | 0:20:30.354815 | 31.0271918138 | 5.5702057245 | 2.9331439612 | 6.8363721276 | 46.7928932930 | 6.8405331147 | 3.8177310012 | 9.0718615479 |
| dual | 2 | True | dual | 0:27:03.127027 | 30.9229882456 | 5.5608442026 | 2.9381745963 | 6.4723335113 | 33.6647294060 | 5.8021314537 | 2.9877111756 | 7.6534620595 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:21:12.447426 | 32.9125071827 | 5.7369423200 | 2.9034125309 | 7.1065965439 | 41.2414087345 | 6.4219474254 | 3.4608307269 | 9.6082696549 |

---

### 表 8 — `seed=48`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:26:41.744125 | 35.0699516457 | 5.9219888252 | 3.0067847437 | 7.5790317888 | 42.7981364304 | 6.5420284645 | 3.4146910214 | 8.0654807021 |
| temporal | 5 | True | temporal | 0:20:30.234470 | 33.1346638105 | 5.7562716936 | 2.9086687016 | 7.3996343746 | 33.3967595620 | 5.7789929540 | 3.0016535658 | 8.0072741486 |
| dual | 6 | True | dual | 0:27:14.672058 | 33.5895732152 | 5.7956512331 | 2.9080271313 | 7.4147573128 | 32.3636628675 | 5.6889070011 | 2.9765398615 | 7.6394093406 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:20:09.981998 | 36.2015465300 | 6.0167721022 | 2.9671931798 | 7.0789986724 | 45.4714193601 | 6.7432499108 | 3.5591064098 | 9.1619502553 |

---

### 表 9 — `seed=49`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:27:06.547678 | 44.0517845832 | 6.6371518427 | 3.3485123890 | 8.8808842974 | 40.9913483764 | 6.4024486235 | 3.3419592140 | 8.9577367226 |
| temporal | 1 | True | temporal | 0:21:11.575518 | 40.9417479575 | 6.3985739003 | 3.2841284218 | 8.5769396415 | 31.3517662578 | 5.5992647962 | 2.9044841754 | 7.2145100498 |
| dual | 2 | True | dual | 0:28:05.090018 | 42.3547498741 | 6.5080526945 | 3.3593682592 | 8.4006192987 | 35.4893082613 | 5.9572903456 | 3.2315174765 | 7.8919459756 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:21:54.595716 | 45.8629539768 | 6.7722192800 | 3.4204506365 | 8.6733883759 | 102.5243982739 | 10.1254332388 | 5.3652622826 | 16.6890713639 |

---

### 表 10 — `seed=50`

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:28:09.989441 | 42.0807244147 | 6.4869657325 | 3.1821474122 | 8.3178094440 | 41.4195116720 | 6.4357992256 | 3.4564356852 | 8.2451499163 |
| temporal | 5 | True | temporal | 0:21:06.871050 | 35.0128973292 | 5.9171697060 | 2.9927999633 | 7.5708566990 | 34.9356172458 | 5.9106359426 | 3.1052193149 | 7.8263588015 |
| dual | 6 | True | dual | 0:27:55.762445 | 39.0130760553 | 6.2460448330 | 3.1895922935 | 8.2200606481 | 35.4964803754 | 5.9578922763 | 3.1299007710 | 7.8577396912 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:20:41.294751 | 44.0238340261 | 6.6350458948 | 3.3155098796 | 9.1813620189 | 50.1496553812 | 7.0816421387 | 3.8484462773 | 8.7161984277 |

---

### 表 11 — 随机种子汇总：均值 ± 标准差（n=10）

对 **表 1–10** 中同一配置、同一指标在十个种子上计算 **算术平均** 与 **样本标准差**（`statistics.stdev`，分母 n−1）。

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 42.00 ± 5.41 | 6.47 ± 0.43 | 3.28 ± 0.25 | 8.39 ± 0.95 | 40.88 ± 1.82 | 6.39 ± 0.14 | 3.39 ± 0.13 | 8.62 ± 0.39 |
| temporal | 37.38 ± 4.14 | 6.11 ± 0.34 | 3.15 ± 0.20 | 7.94 ± 0.83 | 36.89 ± 4.42 | 6.06 ± 0.35 | 3.21 ± 0.27 | 8.19 ± 0.58 |
| dual | 38.89 ± 5.20 | 6.22 ± 0.42 | 3.24 ± 0.25 | 8.13 ± 1.03 | 35.76 ± 3.32 | 5.97 ± 0.27 | 3.19 ± 0.25 | 8.13 ± 0.72 |
| 无预训练 | 41.91 ± 4.68 | 6.46 ± 0.37 | 3.26 ± 0.20 | 8.33 ± 0.82 | 48.98 ± 19.04 | 6.91 ± 1.15 | 3.69 ± 0.61 | 10.14 ± 2.36 |

---


## 5. Time summary（训练 / 推理耗时）

脚本在每次运行结束、打印 `SCRIPT DURATION` 之前，会输出 **`=== TIME SUMMARY (training & inference) ===`**，并将同类信息写入当次实验目录下的 **`paper_timing.txt`** 与 **`paper_timing.json`**。分层含义如下（与 METRLA 实验相比仅增加 PEMSD7M 数据路径，计时逻辑相同）。

| 层级 | 训练（training） | 推理（inference） |
| --- | --- | --- |
| **Overall** | `script_wall_sec`（整脚本墙钟）；`pretrain_total_sec`、`main_train_sec`；无预训练时预训练段为 0 | `test_u_all_sec`、`test_a_all_sec`（含加载 checkpoint、前向与指标）；另有 `test_*_eval_forward_sec` / `test_*_predict_forward_sec` 为 test 集上前向总时长 |
| **Epoch** | 预训练 geo：`pretrain_geo_epoch_sec_list` 及控制台「Epoch — pretrain geometric」汇总；预训练 temporal：同理；主训练：`main_train_epoch_sec_list` 及「Epoch — main forecast training」 | 推理阶段无 epoch；test 仅按 DataLoader 遍历 |
| **Iteration** | 预训练 geo 内层步、temporal 的 mini-batch、主训练每个 `train_iter` batch：对应 `pretrain_geo_iter_*`、`pretrain_temporal_iter_*`、`main_train_iter_*`（mean / stdev / n / sum 等） | `test_u`、`test_a` 各含 eval 与 predict 两趟前向：`test_*_eval_batch_*`、`test_*_predict_batch_*`（每个 batch 一次的统计） |

**解析提示**：扫种汇总时可在各 `../save/pred_PEMSD7M_GraphWaveNet_* /paper_timing.json` 中抽取上述字段；控制台日志从 `=== TIME SUMMARY` 起可复制到实验笔记。

---

## 6. 备注

- **训练/评测入口脚本**为仓库根目录下 `pred_GWN_16_adpAdj.py`（**12 步预测**，依赖骨干 `GWN_SCPT_14_adpAdj_future12step.py`，默认 `out_dim=12`）。
- 指标均来自各次运行日志中的 `all pred steps` 行（或与历史实验相同的导出文件）。
- 表 1–10 中的「物理 GPU」列对应该次八卡调度方案中的卡位（奇数种子用 4–7，偶数轮次中较小种子用 0–3）；若你改用其它绑卡方式，只需在填表时改为实际使用的物理 GPU 编号。
- PEMSD7M 流量文件为 `../PEMSD7M/V_228.csv`，邻接为 `../PEMSD7M/W_228.csv`，与 `pred_GWN_16_adpAdj.py` 中 `P.DATANAME == 'PEMSD7M'` 分支一致。
