# PRETRN_MODE / 无预训练 实验记录（METRLA，`pred_GWN_16_adpAdj.py`，五组种子）

**本版设定**：随机种子 **100、42、999、555、250**（共 **5** 组）；**8 张物理 GPU（0–7）** 上前两轮各跑两个种子（占满 8 卡），第三轮仅跑种子 **250**（占用 **GPU 0–3**）。**命令行 argv** 与 `GWN_METRLA_predict12.md` 中 `pred_GWN_16_adpAdj.py` 示例**完全一致**（仅替换 `<SEED>`）。汇总为 **均值 ± 样本标准差（n=5）** 时，与 Python `statistics.mean` / `statistics.stdev`（分母 **n−1=4**）一致。

---

## 1. 结果与日志保存位置

| 类型               | 路径（相对仓库根目录 `forecasting-on-new-roads/`） |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| **本说明与汇总表**      | `GWN_METRLA_predict12_5seeds.md`（当前文件） |
| **各次运行的控制台日志**   | `logs/pretrn_seed_sweep_metrla_5seeds/geo_s{seed}.log`、`temporal_s*.log`、`dual_s*.log`、`nopre_s*.log` |
| **扫种总控 / nohup** | 例如 `logs/pretrn_seed_sweep_metrla_5seeds/nohup_runner.log`（`scripts/run_metrla_pretrn_gpu8_5seeds.sh`） |
| **自动汇总脚本**     | 可仿照 `scripts/fill_run_pretrn_metrla_predgwn16_md.py`，将日志根目录改为 `logs/pretrn_seed_sweep_metrla_5seeds` 后解析填入第 4 节 |
| **每次实验的模型与预测**   | `P.PATH = '../save/' + P.KEYWORD` → `../save/pred_METRLA_GraphWaveNet_<时间戳>_<pid>/`（相对脚本工作目录） |
| **指标原文**         | 各日志中 `all pred steps` 行；`save/.../*_prediction_scores.txt` |

工作目录：在 **`forecasting-on-new-roads/`** 下执行 `python pred_GWN_16_adpAdj.py ...`。

---

## 2. 环境与启动命令（四种配置，与 `GWN_METRLA_predict12.md` 一致）

环境（每次会话前）：

```bash
cd /path/to/forecasting-on-new-roads
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate fonr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs/pretrn_seed_sweep_metrla_5seeds
```

将 `**<SEED>**` 替换为 **`100`**、**`42`**、**`999`**、**`555`**、**`250`** 之一。  
前三组：`IS_PRETRN=1`，仅 `PRETRN_MODE` 不同；第四组：`IS_PRETRN=0`，为与预训练组对齐随机种子，使用**完整 argv**（末尾 `PRETRN_MODE` 占位为 `dual`，预训练阶段不执行）。

### 2.1 `PRETRN_MODE=geo`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo
```

### 2.2 `PRETRN_MODE=temporal`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal
```

### 2.3 `PRETRN_MODE=dual`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual
```

### 2.4 `IS_PRETRN=False`（无预训练）

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python pred_GWN_16_adpAdj.py 0 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual
```

**超参说明**：除 `IS_PRETRN` / `PRETRN_MODE` / `seed` / `CUDA_VISIBLE_DEVICES` 外，与 `GWN_METRLA_predict12.md` 及脚本默认一致（如 `TEMPERATURE=1.0`、`seed_SS=-1`、`weight_decay=0`、`FEATURES=2`、`PRE_LEARN=0.001` 等）。

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
LOG=logs/pretrn_seed_sweep_metrla_5seeds

CUDA_VISIBLE_DEVICES=0 python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo       2>&1 | tee "$LOG/geo_s100.log" &
CUDA_VISIBLE_DEVICES=1 python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal  2>&1 | tee "$LOG/temporal_s100.log" &
CUDA_VISIBLE_DEVICES=2 python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/dual_s100.log" &
CUDA_VISIBLE_DEVICES=3 python pred_GWN_16_adpAdj.py 0 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/nopre_s100.log" &

CUDA_VISIBLE_DEVICES=4 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" geo       2>&1 | tee "$LOG/geo_s42.log" &
CUDA_VISIBLE_DEVICES=5 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" temporal  2>&1 | tee "$LOG/temporal_s42.log" &
CUDA_VISIBLE_DEVICES=6 python pred_GWN_16_adpAdj.py 1 0.7 0 42 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/dual_s42.log" &
CUDA_VISIBLE_DEVICES=7 python pred_GWN_16_adpAdj.py 0 0.7 0 42 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 0 0.5 0 encoder encoder "" dual      2>&1 | tee "$LOG/nopre_s42.log" &

wait
```

第二轮将 `100`/`42` 与日志名改为 **`999`** / **`555`**；第三轮仅起 **250** 的四个进程（`CUDA_VISIBLE_DEVICES=0..3`，日志 `*_s250.log`），`wait` 后再结束。

**一键跑三轮**：在仓库根目录执行 `bash scripts/run_metrla_pretrn_gpu8_5seeds.sh`（可用 `nohup bash scripts/run_metrla_pretrn_gpu8_5seeds.sh >> logs/pretrn_seed_sweep_metrla_5seeds/nohup_runner.log 2>&1 &` 后台跑）。

**说明**：`CUDA_VISIBLE_DEVICES` 仅重映射为进程内 `cuda:0`，**不改变**给定 `seed` 与 argv 下的可复现性；物理 GPU 编号仅表示机器上的卡位。

---

## 4. 指标表（`all pred steps`）

共 **五张** 按种子的四配置对比表（种子 **100、42、999、555、250**），以及 **第六张** 汇总（**n=5**）。**数据**由 `logs/pretrn_seed_sweep_metrla_5seeds` 下各 `geo_s*.log` / `temporal_s*.log` / `dual_s*.log` / `nopre_s*.log` 中的 `SCRIPT DURATION` 与 `all pred steps` 行解析填入（UTC：`2026-04-13T07:30:14Z` 对应 `ALL WAVES COMPLETE`）。

**进度**：五种子里已有 **5** 个四配置日志齐全（`100, 42, 999, 555, 250`）。表 6 的 **n=5**（仅统计已齐种子）。

| 列 | 含义 |
| --- | --- |
| 脚本总时长 | 日志中 `SCRIPT DURATION` 行；未跑完为 — |
| test_u / test_a | 日志 `all pred steps` 行（模型名 `GraphWaveNet`） |

### 表 1 — 随机种子 `seed=100`（第一轮，GPU 0–3）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:55:47.652550 | 110.9815931294 | 10.5347801652 | 4.4946749745 | 12.2298490071 | 106.2435042078 | 10.3074489670 | 4.5623028852 | 11.9357363558 |
| temporal | 1 | True | temporal | 0:50:08.604457 | 99.3712773390 | 9.9685142995 | 4.1229786305 | 11.7244715573 | 94.9596933509 | 9.7447264380 | 4.1554964389 | 10.7941137956 |
| dual | 2 | True | dual | 0:57:39.661557 | 102.2638973073 | 10.1125613624 | 4.2192928856 | 12.4447682817 | 94.8422747694 | 9.7386998501 | 4.1450786640 | 10.9388231886 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:49:39.108551 | 112.3867802767 | 10.6012631453 | 4.5070162822 | 12.2414758126 | 105.0475411564 | 10.2492702743 | 4.5589439470 | 11.7880144439 |

---

### 表 2 — 随机种子 `seed=42`（第一轮，GPU 4–7）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:54:56.726992 | 105.0585670739 | 10.2498081482 | 4.3817341639 | 11.0507981452 | 106.4962176011 | 10.3197004608 | 4.6481096033 | 11.5703407124 |
| temporal | 5 | True | temporal | 0:51:02.382947 | 99.7149721262 | 9.9857384367 | 4.1445828480 | 9.7275709182 | 96.3591164335 | 9.8162679483 | 4.1844272578 | 10.4569707831 |
| dual | 6 | True | dual | 0:57:59.869633 | 104.5520901983 | 10.2250716476 | 4.2358902275 | 9.8681067970 | 92.4628815576 | 9.6157621413 | 4.1508430848 | 10.5913144748 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:51:00.649347 | 106.7660559633 | 10.3327661332 | 4.3551695521 | 10.3156881608 | 106.4087921587 | 10.3154637394 | 4.7915370494 | 13.2565225466 |

---

### 表 3 — `seed=999`（第二轮，GPU 0–3）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:56:27.286037 | 114.4737782738 | 10.6992419486 | 4.7374837425 | 12.6834210726 | 101.8036570507 | 10.0897798316 | 4.5138061573 | 11.9866378016 |
| temporal | 1 | True | temporal | 0:49:59.497350 | 104.0526205875 | 10.2006186375 | 4.5216280735 | 12.2299472293 | 92.1491944366 | 9.5994371937 | 4.1719187037 | 10.3591459513 |
| dual | 2 | True | dual | 0:56:50.386586 | 104.4827275367 | 10.2216792914 | 4.5900290045 | 12.0758582663 | 94.3738661369 | 9.7146212555 | 4.2252733939 | 10.6206982962 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:49:26.163453 | 112.7798110904 | 10.6197839474 | 4.7040092454 | 12.7384263324 | 104.7310602482 | 10.2338194360 | 4.7103923961 | 13.2166817125 |

---

### 表 4 — `seed=555`（第二轮，GPU 4–7）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 4 | True | geo | 0:54:58.138245 | 98.9315402124 | 9.9464335423 | 4.3870085927 | 10.9473740605 | 98.2358498508 | 9.9113999945 | 4.5677143405 | 11.8010715781 |
| temporal | 5 | True | temporal | 0:50:31.490283 | 96.4695796692 | 9.8218928761 | 4.2473976663 | 10.0830833961 | 94.1286077233 | 9.7019898847 | 4.1459081005 | 10.7040535110 |
| dual | 6 | True | dual | 0:57:24.812831 | 96.2331628940 | 9.8098502993 | 4.2403211376 | 10.0061287363 | 93.9387989974 | 9.6922030002 | 4.1860895290 | 10.3092485144 |
| 无预训练 | 7 | False | （默认 dual，未预训练） | 0:52:04.529086 | 98.6593628106 | 9.9327419583 | 4.3643394460 | 10.5249489956 | 105.1946168932 | 10.2564427017 | 4.8393213659 | 13.2792159183 |

---

### 表 5 — `seed=250`（第三轮，GPU 0–3）

| 配置 | 物理 GPU | IS_PRETRN | PRETRN_MODE | 脚本总时长 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 0 | True | geo | 0:54:24.541500 | 112.8071949220 | 10.6210731530 | 4.5315516662 | 11.6186715711 | 108.1496242530 | 10.3995011540 | 4.9887118302 | 13.7569196272 |
| temporal | 1 | True | temporal | 0:49:43.542328 | 106.4846915851 | 10.3191419985 | 4.5455548648 | 11.3341637731 | 98.4247543632 | 9.9209250760 | 4.2919857891 | 11.3703327259 |
| dual | 2 | True | dual | 0:56:57.995160 | 104.7733454256 | 10.2358851804 | 4.3528511939 | 10.9070381243 | 91.7455479980 | 9.5783896349 | 4.1662767758 | 10.9316044718 |
| 无预训练 | 3 | False | （默认 dual，未预训练） | 0:48:37.809775 | 115.2495592510 | 10.7354347490 | 4.5876230304 | 11.6518744068 | 115.6655621921 | 10.7547925220 | 4.9534566330 | 14.9020205256 |

---

### 表 6 — 随机种子汇总：均值 ± 标准差（n=5）

对 **表 1–5** 中同一配置、同一指标在五个种子上计算 **算术平均** 与 **样本标准差**（`statistics.stdev`，分母 **n−1=4**）。

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geo | 108.45 ± 6.40 | 10.41 ± 0.31 | 4.51 ± 0.14 | 11.71 ± 0.75 | 104.19 ± 4.07 | 10.21 ± 0.20 | 4.66 ± 0.19 | 12.21 ± 0.88 |
| temporal | 101.22 ± 4.00 | 10.06 ± 0.20 | 4.32 ± 0.20 | 11.02 ± 1.07 | 95.20 ± 2.36 | 9.76 ± 0.12 | 4.19 ± 0.06 | 10.74 ± 0.40 |
| dual | 102.46 ± 3.63 | 10.12 ± 0.18 | 4.33 ± 0.16 | 11.06 ± 1.17 | 93.47 ± 1.31 | 9.67 ± 0.07 | 4.17 ± 0.03 | 10.68 ± 0.26 |
| 无预训练 | 109.17 ± 6.64 | 10.44 ± 0.32 | 4.50 ± 0.15 | 11.49 ± 1.06 | 107.41 ± 4.66 | 10.36 ± 0.22 | 4.77 ± 0.15 | 13.29 ± 1.10 |

---

## 5. 备注

- **训练/评测入口脚本**为仓库根目录下 **`pred_GWN_16_adpAdj.py`**（**12 步预测**，依赖 **`GWN_SCPT_14_adpAdj_future12step.py`**）。与 `GWN_METRLA_predict12.md` 中备注一致。
- 指标均来自各次运行日志中的 `all pred steps` 行（或 `*_prediction_scores.txt`）。
- 表 1–5 中的「物理 GPU」列与第 3 节三轮调度一致；若改用其它绑卡方式，填表时改为实际物理 GPU 编号即可。
- 原版十种子文档：`GWN_METRLA_predict12.md`（种子 41–50，日志目录 `logs/pretrn_seed_sweep_metrla_41_50`）。
