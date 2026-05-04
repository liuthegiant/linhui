# PRETRN_MODE / 无预训练 实验记录（PEMSBAY，`pred_GWN_16_adpAdj.py`，五组种子）

**本版设定**：随机种子 **100、42、999、555、250**（共 **5** 组）；**8 张物理 GPU（0–7）** 上前两轮各跑两个种子（占满 8 卡），第三轮仅跑种子 **250**（占用 **GPU 0–3**）。**命令行 argv** 与 `GWN_PEMSBAY_predict12.md` 中 `pred_GWN_16_adpAdj.py` 示例**完全一致**（仅将 `<SEED>` 换为上述五值之一，`DATANAME` 仍为 **PEMSBAY**）。汇总为 **均值 ± 样本标准差（n=5）** 时，与 Python `statistics.mean` / `statistics.stdev`（分母 **n−1=4**）一致。

---

## 1. 结果与日志保存位置

| 类型               | 路径（相对仓库根目录 `forecasting-on-new-roads/`） |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| **本说明与汇总表**      | `GWN_PEMSBAY_predict12_5seeds.md`（当前文件） |
| **各次运行的控制台日志**   | `logs/pretrn_seed_sweep_pemsbay_5seeds/geo_s{seed}.log`、`temporal_s*.log`、`dual_s*.log`、`nopre_s*.log` |
| **扫种总控 / nohup** | 例如 `logs/pretrn_seed_sweep_pemsbay_5seeds/nohup_runner.log`（`scripts/run_pemsbay_pretrn_gpu8_5seeds.sh`） |
| **自动汇总脚本**     | 可仿照 `scripts/fill_run_pretrn_pemsbay_predgwn16_md.py`，将日志根目录改为 `logs/pretrn_seed_sweep_pemsbay_5seeds` 后解析填入第 4 节 |
| **每次实验的模型与预测**   | `P.PATH = '../save/' + P.KEYWORD` → `../save/pred_PEMSBAY_GraphWaveNet_<时间戳>_<pid>/`（相对脚本工作目录） |
| **指标原文**         | 各日志中 `all pred steps` 行；`save/.../*_prediction_scores.txt` |

工作目录：在 **`forecasting-on-new-roads/`** 下执行 `python pred_GWN_16_adpAdj.py ...`。

---

## 2. 环境与启动命令（四种配置，与 `GWN_PEMSBAY_predict12.md` 一致）

环境（每次会话前）：

```bash
cd /path/to/forecasting-on-new-roads
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate fonr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs/pretrn_seed_sweep_pemsbay_5seeds
```

将 `**<SEED>**` 替换为 **`100`**、**`42`**、**`999`**、**`555`**、**`250`** 之一。  
前三组：`IS_PRETRN=1`，仅 `PRETRN_MODE` 不同；第四组：`IS_PRETRN=0`，为与预训练组对齐随机种子，使用**完整 argv**（末尾 `PRETRN_MODE` 占位为 `dual`，预训练阶段不执行）。

### 2.1 `PRETRN_MODE=geo`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo
```

### 2.2 `PRETRN_MODE=temporal`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal
```

### 2.3 `PRETRN_MODE=dual`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual
```

### 2.4 `IS_PRETRN=False`（无预训练）

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 0 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual
```

**超参说明**：除 `IS_PRETRN` / `PRETRN_MODE` / `seed` / `CUDA_VISIBLE_DEVICES` 外，与 `GWN_PEMSBAY_predict12.md` 及脚本默认一致（如 `TEMPERATURE=1.0`、`seed_SS=-1`、`weight_decay=0`、`FEATURES=2`、`PRE_LEARN=0.001` 等）。

---

## 3. 八卡并行调度（GPU 0–7）

共 **5 个种子 × 4 种配置 = 20 个作业**。

| 轮次 | 种子 A（GPU 0–3） | 种子 B（GPU 4–7） |
| --- | --- | --- |
| 1 | **100**：geo→0，temporal→1，dual→2，无预训练→3 | **42**：geo→4，temporal→5，dual→6，无预训练→7 |
| 2 | **999**：geo→0，temporal→1，dual→2，无预训练→3 | **555**：geo→4，temporal→5，dual→6，无预训练→7 |
| 3 | **250**：geo→0，temporal→1，dual→2，无预训练→3 | （无）GPU 4–7 空闲 |

**示例（第一轮：种子 100 与 42）**：在仓库根目录执行。

```bash
LOG=logs/pretrn_seed_sweep_pemsbay_5seeds

CUDA_VISIBLE_DEVICES=0 python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo       2>&1 | tee "$LOG/geo_s100.log" &
CUDA_VISIBLE_DEVICES=1 python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal  2>&1 | tee "$LOG/temporal_s100.log" &
CUDA_VISIBLE_DEVICES=2 python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/dual_s100.log" &
CUDA_VISIBLE_DEVICES=3 python pred_GWN_16_adpAdj.py 0 0.7 0 100 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/nopre_s100.log" &

CUDA_VISIBLE_DEVICES=4 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo       2>&1 | tee "$LOG/geo_s42.log" &
CUDA_VISIBLE_DEVICES=5 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal  2>&1 | tee "$LOG/temporal_s42.log" &
CUDA_VISIBLE_DEVICES=6 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/dual_s42.log" &
CUDA_VISIBLE_DEVICES=7 python pred_GWN_16_adpAdj.py 0 0.7 0 42 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/nopre_s42.log" &

wait
```

第二轮将 `100`/`42` 与日志名改为 **`999`** / **`555`**；第三轮仅起 **250** 的四个进程（`CUDA_VISIBLE_DEVICES=0..3`，日志 `*_s250.log`），`wait` 后再结束。

**一键跑三轮**：在仓库根目录执行 `bash scripts/run_pemsbay_pretrn_gpu8_5seeds.sh`（可用 `nohup bash scripts/run_pemsbay_pretrn_gpu8_5seeds.sh >> logs/pretrn_seed_sweep_pemsbay_5seeds/nohup_runner.log 2>&1 &` 后台跑）。

**说明**：`CUDA_VISIBLE_DEVICES` 仅重映射为进程内 `cuda:0`，**不改变**给定 `seed` 与 argv 下的可复现性；物理 GPU 编号仅表示机器上的卡位。

---

## 4. 指标表（`all pred steps`）

共 **五张** 按种子的四配置对比表（种子 **100、42、999、555、250**），以及 **第六张** 汇总（**n=5**）。**数据**需在全部日志就绪后，从 `logs/pretrn_seed_sweep_pemsbay_5seeds` 解析填入（可改写 `scripts/fill_run_pretrn_pemsbay_predgwn16_md.py` 中的日志目录）。

**进度**：五种子里已有 **0** 个四配置日志齐全（待跑）。表 1–6 数值均为占位 **—**。

| 列 | 含义 |
| --- | --- |
| 脚本总时长 | 日志中 `SCRIPT DURATION` 行；未跑完为 — |
| test_u / test_a | 日志 `all pred steps` 行 |

### 表 1 — 随机种子 `seed=100`（第一轮，GPU 0–3）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | — | — | — | — | — | — | — | — | — |
| temporal | 1 | True | temporal | — | — | — | — | — | — | — | — | — |
| dual | 2 | True | dual | — | — | — | — | — | — | — | — | — |
| 无预训练 | 3 | False | （默认 dual，未预训练） | — | — | — | — | — | — | — | — | — |

---

### 表 2 — 随机种子 `seed=42`（第一轮，GPU 4–7）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | — | — | — | — | — | — | — | — | — |
| temporal | 5 | True | temporal | — | — | — | — | — | — | — | — | — |
| dual | 6 | True | dual | — | — | — | — | — | — | — | — | — |
| 无预训练 | 7 | False | （默认 dual，未预训练） | — | — | — | — | — | — | — | — | — |

---

### 表 3 — `seed=999`（第二轮，GPU 0–3）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | — | — | — | — | — | — | — | — | — |
| temporal | 1 | True | temporal | — | — | — | — | — | — | — | — | — |
| dual | 2 | True | dual | — | — | — | — | — | — | — | — | — |
| 无预训练 | 3 | False | （默认 dual，未预训练） | — | — | — | — | — | — | — | — | — |

---

### 表 4 — `seed=555`（第二轮，GPU 4–7）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | — | — | — | — | — | — | — | — | — |
| temporal | 5 | True | temporal | — | — | — | — | — | — | — | — | — |
| dual | 6 | True | dual | — | — | — | — | — | — | — | — | — |
| 无预训练 | 7 | False | （默认 dual，未预训练） | — | — | — | — | — | — | — | — | — |

---

### 表 5 — `seed=250`（第三轮，GPU 0–3）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | — | — | — | — | — | — | — | — | — |
| temporal | 1 | True | temporal | — | — | — | — | — | — | — | — | — |
| dual | 2 | True | dual | — | — | — | — | — | — | — | — | — |
| 无预训练 | 3 | False | （默认 dual，未预训练） | — | — | — | — | — | — | — | — | — |

---

### 表 6 — 随机种子汇总：均值 ± 标准差（n=5）

对 **表 1–5** 中同一配置、同一指标在五个种子上计算 **算术平均** 与 **样本标准差**（`statistics.stdev`，分母 **n−1=4**）。**日志未齐前本表为占位。**

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | — | — | — | — | — | — | — | — |
| temporal | — | — | — | — | — | — | — | — |
| dual | — | — | — | — | — | — | — | — |
| 无预训练 | — | — | — | — | — | — | — | — |

---

## 5. 备注

- **训练/评测入口脚本**为仓库根目录下 **`pred_GWN_16_adpAdj.py`**（**12 步预测**，依赖 **`GWN_SCPT_14_adpAdj_future12step.py`**）。与 `GWN_PEMSBAY_predict12.md` 中备注一致。
- 指标均来自各次运行日志中的 `all pred steps` 行（或 `*_prediction_scores.txt`）。
- 表 1–5 中的「物理 GPU」列与第 3 节三轮调度一致；若改用其它绑卡方式，填表时改为实际物理 GPU 编号即可。
- 原版十种子文档：`GWN_PEMSBAY_predict12.md`（种子 41–50，日志目录 `logs/pretrn_seed_sweep_pemsbay_41_50`）；补跑脚本见该文件第 3 节说明。
