# 随机种子 41–50 统计表（PEMSD7M / GraphWaveNet）

- 生成时间: 2026-04-08T15:34:12
- 数据来源: `REPORT_batch_pemsd7m_seeds41_45_260408102740.md`（种子 41–45）+ `REPORT_batch_pemsd7m_seeds46_50_260408125511.md`（种子 46–50）汇总表中的 `est_PEMSD7M_GraphWaveNet_*` 目录。
- 指标文件: `../save/<KEYWORD>/GraphWaveNet_prediction_scores.txt`。
- **tst_u**：unseen 测试；**tst_a**：全图空间划分测试。
- 数值均为 **均值 ± 样本标准差**；四位小数。误差类指标列内对**最小均值**加粗。
- **时间**：预训练/主训练/脚本时间为 job 日志中的 wall clock；**iteration 级**为 `GraphWaveNet_log.txt` 中 `iter_epoch` 行的 `mean_batch_sec` 在各 epoch 上再平均；**epoch 级**为各 epoch 的 `time` 字段平均；**overall** 为 `SCRIPT DURATION`。推理时间为测试阶段 DataLoader 上纯推理循环 wall 时间（不含嵌入构造等前置）。

---

## 复现信息：参数、程序与产物路径

本报告对应 **状态估计 / 缺失填补** 流水线（`pred_maskpredition_GWN_*.py`），**不是** `pred_GWN_16_adpAdj.py` 的 12 步预测扫种。工作目录均为 `forecasting-on-new-roads/`（以下路径相对该目录）。

### 编排方式与批次报告

| 项 | 说明 |
| --- | --- |
| 环境变量 | `FONR_DATANAME=PEMSD7M`（传给 `run_batch_seeds41_45.py` / `run_batch_seeds46_50.py` 内 `DATANAME`，并写入各 job 的 `sys.argv[6]`） |
| 编排脚本 | `run_batch_seeds41_45.py`（种子 41–45）、`run_batch_seeds46_50.py`（种子 46–50）；或一键 `run_pemsd7m_batch_seeds41_50_and_report.sh` |
| 本报告数据来源 | `REPORT_batch_pemsd7m_seeds41_45_260408102740.md`、`REPORT_batch_pemsd7m_seeds46_50_260408125511.md` 汇总表中的 `KEYWORD` |
| 单次 job 控制台日志 | 各批次目录 `batch_run_pemsd7m_seeds41_45_<STAMP>/job_<jid>.log`、`batch_run_pemsd7m_seeds46_50_<STAMP>/job_<jid>.log`（`latest_batch_run_dir_pemsd7m_seeds41_45.txt` / `latest_batch_run_dir_pemsd7m_seeds46_50.txt` 可定位最新目录） |
| 汇总脚本 | `build_report_statistics_pemsd7m_seeds41_50.py`（由上述两份 batch 报告 + `../save/` 指标与时间解析生成本 MD） |

### `sys.argv` 位次（`pred_maskpredition_GWN_*.py` 共用 `get_argv()`）

批量命令**显式给出下标 1–12**；**13–20** 未传时使用源码默认值（与本批一致：`SUBGRAPH_SIZE=64`、`QUOTIENT_GRAPH_RADIUS=0.01`、`PRETRN_EPOCH=100`、`EPOCH=100`、`NETWORK_CALLS=0`、`PRE_LEARN` 默认、`GRAPH_NORM` 默认、`HIDDEN=320`）。

| i | 变量 | 本报告六配置下的取值 |
| --- | --- | --- |
| 1 | `IS_PRETRN` | geo / scpt_geo / scpt：`1`；`pred_maskpredition_GWN.py`：`0` |
| 2 | `R_TRN` | `0.7` |
| 3 | `IS_EPOCH_1` | `0` |
| 4 | `seed` | `41`…`50` |
| 5 | `TEMPERATURE` | `1.0` |
| 6 | `DATANAME` | `PEMSD7M` |
| 7 | `seed_SS` | `-1` |
| 8 | `IS_DESEASONED` | `1` |
| 9 | `weight_decay` | `0.0001` |
| 10 | `adp_adj` | `1` |
| 11 | `is_SGA` | `1` |
| 12 | `FEATURES` | geo / scpt_geo：`2` 或 `4`；scpt / GWN baseline：`4` |

### 六配置与等价启动命令（`<SEED>` 替换 41–50）

在仓库根目录、`FONR_DATANAME=PEMSD7M` 已生效的前提下，与批量脚本一致的一条命令形如：

```bash
# 1–2：pred_maskpredition_GWN_geo.py，FEATURES=2 或 4
python pred_maskpredition_GWN_geo.py 1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0001 1 1 <2|4>

# 3–4：pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 或 4
python pred_maskpredition_GWN_scpt_geo.py 1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0001 1 1 <2|4>

# 5：pred_maskpredition_GWN_scpt.py（显式 FEATURES=4，与 GEO 后缀一致）
python pred_maskpredition_GWN_scpt.py 1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0001 1 1 4

# 6：pred_maskpredition_GWN.py，无预训练，FEATURES=4
python pred_maskpredition_GWN.py 0 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0001 1 1 4
```

### 程序文件（直接依赖）

| 文件 | 作用 |
| --- | --- |
| `pred_maskpredition_GWN_geo.py` | 几何预训练 + 估计 |
| `pred_maskpredition_GWN_scpt_geo.py` | 对比学习 + 几何 + temporal_delta 融合 |
| `pred_maskpredition_GWN_scpt.py` | 对比预训练 + 估计 |
| `pred_maskpredition_GWN.py` | 基线（可关预训练） |
| `GWN_SCPT_14_adpAdj_mask_infill.py` | 模型与训练相关定义（各脚本 `from … import *`） |
| `graph.py` | 图、数据、`load_dataset` |
| `unseen_nodes.py` | 空间划分 |
| `Metrics.py` | 指标 |
| `../PEMSD7M/V_228.csv`、`../PEMSD7M/W_228.csv` | 流量与邻接（`P.N_NODE=228`） |

### 单次运行目录：`../save/est_PEMSD7M_GraphWaveNet_<yyMMddHHmm>_<pid>/`

`KEYWORD` 由脚本在运行时生成（`est_` + 数据集 + `GraphWaveNet` + 时间 + pid）。**指标**路径固定为：

`../save/<KEYWORD>/GraphWaveNet_prediction_scores.txt`

| 类型 | 常见文件名 | 说明 |
| --- | --- | --- |
| 预训练 | `encoder.pt`、`encoder_log.txt` | 常见：时序/对比支路 |
| scpt_geo（双支路+融合） | `encoderg.pt`、`encoderg_log.txt`、`GraphWaveNet_fusion_u.pt` 等 | 几何支路与 fusion 权重；具体文件名以当次目录为准 |
| 其它有预训练脚本 | 仅 `encoder.pt` 等 | 分支较简时可能无 `encoderg` / `fusion` |
| 主模型 | `GraphWaveNet_best.pt` | 验证最优权重 |
| 主训练日志 | `GraphWaveNet_log.txt` | epoch 与 `iter_epoch` 行（batch 均值耗时） |
| 推理计时 | `GraphWaveNet_timing_infer.txt` | 各次测试写入的 `tst_u` / `tst_a` 推理 wall（秒） |
| 预测输出 | `GraphWaveNet_tst_u_GraphWaveNet_*.npy`、`GraphWaveNet_tst_a_GraphWaveNet_*.npy` | `prediction` / `groundtruth` / `missmask` |
| 可视化 | `heatmap_*_tst_u.png`、`heatmap_*_tst_a.png` | 可选热图 |

在 **batch** 的 `job_*.log` 中搜索 `est_PEMSD7M_GraphWaveNet_` 即可对齐当次 `../save/<KEYWORD>/` 目录名。

### 如何在大量 `../save/` 目录里找到对应的一次运行？

1. **优先看批次报告的「汇总表」**  
   `REPORT_batch_pemsd7m_seeds41_45_*.md`、`REPORT_batch_pemsd7m_seeds46_50_*.md` 的 **「汇总表」** 中 **`输出 KEYWORD`** 列即为 `../save/<KEYWORD>/`。全文搜索 `est_PEMSD7M_GraphWaveNet` 可列出本批 60 个目录名。

2. **用 job 日志反查**  
   在 `batch_run_pemsd7m_seeds41_45_<STAMP>/`、`batch_run_pemsd7m_seeds46_50_<STAMP>/` 下执行：  
   `rg 'est_PEMSD7M_GraphWaveNet_[0-9]+_[0-9]+' job_*.log`

3. **按时间缩小范围（辅助）**  
   `ls -lt ../save/est_PEMSD7M_GraphWaveNet_* | head -30`  
   仅作粗筛，**以 batch 报告或日志中的完整 KEYWORD 为准**。

4. **不要用「只记 pid」**  
   务必使用 **完整 KEYWORD**（含 `yyMMddHHmm` 与 pid）对齐目录。

---

## 表一：六实验配置 × 十随机种子（41–50）

每组 **n=10**（10 个种子），标准差分母为 **9**（`statistics.stdev`）。

| 实验配置 | n | train MAE (est) | tst_u Masked MAE | tst_u RMSE | tst_u MAPE | tst_a Masked MAE | tst_a RMSE | tst_a MAPE |
|---------|---|-----------------|------------------|------------|------------|------------------|------------|------------|
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 10 | 0.1316 ± 0.0162 | 1.7755 ± 0.1359 | 2.8407 ± 0.2050 | 2.5221 ± 0.3902 | 1.8366 ± 0.3110 | 2.6727 ± 0.3426 | 3.2029 ± 1.0522 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 10 | 0.1265 ± 0.0073 | 1.7685 ± 0.1700 | 2.8102 ± 0.1717 | 2.6296 ± 0.4468 | 1.7648 ± 0.2767 | 2.6638 ± 0.3662 | 2.5893 ± 0.3412 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 10 | 0.0813 ± 0.0027 | 1.3344 ± 0.0863 | 2.0908 ± 0.1343 | 2.6374 ± 0.4880 | 1.1834 ± 0.0715 | 1.8662 ± 0.0920 | 2.4650 ± 0.1789 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 10 | **0.0803 ± 0.0026** | **1.3143 ± 0.0845** | **2.0768 ± 0.1229** | 2.6479 ± 0.3670 | **1.1647 ± 0.0872** | **1.8389 ± 0.1227** | **2.3811 ± 0.1547** |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 10 | 0.0836 ± 0.0028 | 1.3867 ± 0.1075 | 2.1602 ± 0.1705 | 2.8480 ± 0.4187 | 1.2782 ± 0.0853 | 1.9287 ± 0.1167 | 2.6048 ± 0.2380 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 10 | 0.2479 ± 0.0928 | 1.7854 ± 0.1748 | 2.7494 ± 0.2214 | **2.3244 ± 0.6201** | 3.6097 ± 2.3524 | 4.5833 ± 2.3570 | 7.2914 ± 6.1591 |

## 表二：十随机种子 × 六实验配置（跨配置汇总）

对每个种子 **s**，在六种配置上的指标再算 **均值 ± 标准差**；每组 **n=6**，标准差分母为 **5**。

| 随机种子 | n | train MAE (est) | tst_u Masked MAE | tst_u RMSE | tst_u MAPE | tst_a Masked MAE | tst_a RMSE | tst_a MAPE |
|----------|---|-----------------|------------------|------------|------------|------------------|------------|------------|
| 41 | 6 | 0.1314 ± 0.0835 | 1.5767 ± 0.3090 | 2.4436 ± 0.3867 | 2.5140 ± 0.3747 | 1.5899 ± 0.5450 | 2.4132 ± 0.7528 | 2.5254 ± 0.2450 |
| 42 | 6 | 0.1174 ± 0.0459 | **1.4333 ± 0.1221** | 2.3563 ± 0.2858 | 2.6945 ± 0.4810 | 1.4397 ± 0.2832 | 2.1690 ± 0.3553 | 2.4040 ± 0.1449 |
| 43 | 6 | **0.1143 ± 0.0479** | 1.6243 ± 0.2609 | 2.5071 ± 0.3480 | 2.6092 ± 0.5173 | 2.1707 ± 1.8014 | 2.9557 ± 1.9384 | 4.1308 ± 4.1308 |
| 44 | 6 | 0.1198 ± 0.0490 | 1.6425 ± 0.2332 | 2.6057 ± 0.4690 | 2.5769 ± 0.3916 | **1.4234 ± 0.2395** | 2.2421 ± 0.3713 | **2.3693 ± 0.2867** |
| 45 | 6 | 0.1215 ± 0.0522 | 1.7208 ± 0.3444 | 2.5705 ± 0.3957 | 2.5435 ± 0.3987 | 2.2133 ± 2.1192 | 2.9773 ± 2.1914 | 4.4027 ± 4.8188 |
| 46 | 6 | 0.1289 ± 0.0541 | 1.6149 ± 0.3063 | 2.5496 ± 0.4908 | 2.4003 ± 0.3493 | 1.9026 ± 1.4503 | 2.6895 ± 1.6437 | 3.8176 ± 3.3441 |
| 47 | 6 | 0.1169 ± 0.0516 | 1.4370 ± 0.3008 | **2.1105 ± 0.3619** | **2.0841 ± 0.2773** | 1.7091 ± 0.6266 | 2.5616 ± 0.8259 | 3.4817 ± 1.1299 |
| 48 | 6 | 0.1158 ± 0.0521 | 1.4581 ± 0.1582 | 2.4255 ± 0.3423 | 2.5998 ± 0.4377 | 1.6435 ± 0.5063 | 2.4724 ± 0.5780 | 2.9840 ± 1.3279 |
| 49 | 6 | 0.1685 ± 0.1632 | 1.5454 ± 0.2000 | 2.4776 ± 0.4276 | 3.3236 ± 0.3641 | 2.4775 ± 2.5424 | 3.2797 ± 2.5925 | 5.2997 ± 6.5005 |
| 50 | 6 | 0.1175 ± 0.0578 | 1.5548 ± 0.2287 | 2.5002 ± 0.4012 | 2.6700 ± 0.3285 | 1.4925 ± 0.3526 | **2.1621 ± 0.3940** | 2.8089 ± 0.3518 |

## 表三：Time summary（六配置 × 十种子，单位：秒）

对每种时间指标在 10 个种子上计算 **均值 ± 样本标准差**。无预训练的配置 `pretrain_sec` 多为 0 或缺失（记为 —）。`IS_PRETRN=False` 时 `infer_*` 仍包含测试循环计时。

| 实验配置 | n | 预训练 wall 时间 (s，日志 PRETIME 求和) | 主训练 wall 时间 (s，MODEL TRAINING DURATION) | 整脚本 wall 时间 (s，SCRIPT DURATION) | 推理 tst_u wall 时间 (s) | 推理 tst_a wall 时间 (s) | 单 epoch 耗时均值 (s，GraphWaveNet_log) | 单 batch 前向+反传均值 (s，iter_epoch 行) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 10 | 565.2668 ± 61.8358 | 1676.5765 ± 164.3330 | 2343.5697 ± 223.8289 | 1.0577 ± 0.2271 | 2.2927 ± 0.2705 | 15.9250 ± 1.6325 | 0.0918 ± 0.0100 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 10 | 566.5637 ± 63.0073 | 1674.7115 ± 157.7477 | 2346.2884 ± 221.7930 | 1.1332 ± 0.2002 | 2.2082 ± 0.2480 | 15.8970 ± 1.5568 | 0.0914 ± 0.0094 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 10 | 589.2028 ± 80.5591 | 1733.4019 ± 163.1730 | 2420.8284 ± 235.4840 | 1.1342 ± 0.1472 | 2.2388 ± 0.3151 | 16.5050 ± 1.6106 | 0.0960 ± 0.0101 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 10 | 673.3117 ± 70.7577 | 1889.6001 ± 84.0728 | 2669.1367 ± 142.9486 | 1.3306 ± 0.4287 | 2.4266 ± 0.2644 | 18.0190 ± 0.7903 | 0.1042 ± 0.0041 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 10 | 51.3620 ± 7.7774 | 1811.6288 ± 59.5049 | 1894.6830 ± 60.1663 | 1.1390 ± 0.1541 | 2.3670 ± 0.1368 | 17.5980 ± 0.5793 | 0.1014 ± 0.0025 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 10 | — | 1776.4531 ± 53.0468 | 1806.3501 ± 55.6394 | 1.1303 ± 0.2508 | 2.3635 ± 0.1858 | 17.2580 ± 0.5334 | 0.0999 ± 0.0020 |

## 附：各配置 × 各种子原始值（tst_u）

| 实验配置 | seed | Masked MAE | RMSE | MAPE |
|---------|------|------------|------|------|
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 41 | 1.713084 | 2.714249 | 2.221829 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 42 | 1.505630 | 2.619435 | 2.713802 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 43 | 1.834668 | 2.930642 | 1.849045 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 44 | 1.791437 | 2.926492 | 2.445128 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 45 | 2.030921 | 2.970198 | 2.449637 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 46 | 1.887970 | 3.127222 | 2.683524 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 47 | 1.720525 | 2.415466 | 2.058464 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 48 | 1.748113 | 2.961367 | 2.812916 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 49 | 1.718519 | 2.841737 | 3.062459 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 50 | 1.803723 | 2.899803 | 2.924040 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 41 | 1.985234 | 2.911332 | 2.137998 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 42 | 1.574377 | 2.714076 | 3.233230 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 43 | 1.794376 | 2.757773 | 2.147871 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 44 | 1.868796 | 3.030948 | 2.743445 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 45 | 2.032479 | 2.848909 | 3.156399 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 46 | 1.811966 | 2.967145 | 2.517155 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 47 | 1.686587 | 2.522333 | 2.448716 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 48 | 1.481527 | 2.554164 | 2.090892 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 49 | 1.717383 | 2.945994 | 3.206276 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 50 | 1.732268 | 2.849501 | 2.614396 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 41 | 1.422618 | 2.230097 | 2.409411 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 42 | 1.281095 | 2.046175 | 2.541490 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 43 | 1.291050 | 2.048456 | 2.663364 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 44 | 1.443967 | 2.217457 | 2.197459 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 45 | 1.346964 | 2.125542 | 2.404657 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 46 | 1.304492 | 2.068442 | 2.642034 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 47 | 1.152303 | 1.756938 | 2.323812 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 48 | 1.341160 | 2.131528 | 2.807741 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 49 | 1.425067 | 2.181751 | 3.930562 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 50 | 1.335522 | 2.101463 | 2.453809 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 41 | 1.182752 | 1.942472 | 2.354191 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 42 | 1.368498 | 2.129187 | 2.668079 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 43 | 1.383816 | 2.171876 | 2.808967 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 44 | 1.345263 | 2.097632 | 2.576046 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 45 | 1.398650 | 2.183364 | 2.475013 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 46 | 1.275282 | 2.085755 | 2.260560 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 47 | 1.156488 | 1.790806 | 2.119323 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 48 | 1.343810 | 2.134929 | 2.887640 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 49 | 1.312340 | 2.062009 | 3.312019 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 50 | 1.375654 | 2.170377 | 3.017355 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 41 | 1.340561 | 2.150192 | 2.936051 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 42 | 1.330078 | 2.141460 | 3.125785 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 43 | 1.521590 | 2.437674 | 3.227396 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 44 | 1.524687 | 2.236137 | 3.258526 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 45 | 1.480958 | 2.342568 | 2.798453 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 46 | 1.452222 | 2.172078 | 2.544137 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 47 | 1.179714 | 1.802451 | 1.819553 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 48 | 1.344869 | 2.152890 | 3.003518 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 49 | 1.360260 | 2.033276 | 2.907779 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 50 | 1.331659 | 2.132985 | 2.858800 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 41 | 1.816093 | 2.713341 | 3.024517 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 42 | 1.540297 | 2.487705 | 1.884369 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 43 | 1.920428 | 2.696461 | 2.958830 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 44 | 1.880813 | 3.125774 | 2.240546 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 45 | 2.034844 | 2.952394 | 1.976737 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 46 | 1.957558 | 2.877228 | 1.754251 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 47 | 1.726655 | 2.375045 | 1.734474 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 48 | 1.489073 | 2.618178 | 1.996058 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 49 | 1.738686 | 2.800661 | 3.522477 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 50 | 1.749851 | 2.847269 | 2.151432 |

## 附：各配置 × 各种子原始值（tst_a）

| 实验配置 | seed | Masked MAE | RMSE | MAPE |
|---------|------|------------|------|------|
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 41 | 1.704395 | 2.476636 | 2.696212 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 42 | 1.734781 | 2.497757 | 2.578355 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 43 | 1.709912 | 2.561703 | 2.196041 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 44 | 1.643154 | 2.545030 | 2.584407 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 45 | 1.741382 | 2.589110 | 2.988217 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 46 | 1.643907 | 2.518974 | 2.844761 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 47 | 2.077040 | 2.984916 | 4.496520 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 48 | 2.639213 | 3.549554 | 5.609732 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 49 | 1.634420 | 2.461666 | 2.692086 |
| pred_maskpredition_GWN_geo.py，FEATURES=2 | 50 | 1.837416 | 2.542044 | 3.342822 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 41 | 2.535231 | 3.652895 | 2.747522 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 42 | 1.679569 | 2.459446 | 2.509638 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 43 | 1.694824 | 2.464232 | 2.656804 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 44 | 1.627542 | 2.550939 | 2.182952 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 45 | 1.691052 | 2.506803 | 2.287208 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 46 | 1.630128 | 2.523178 | 2.532909 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 47 | 1.792215 | 2.759337 | 3.335166 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 48 | 1.590511 | 2.469374 | 2.314703 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 49 | 1.670376 | 2.767654 | 2.428792 |
| pred_maskpredition_GWN_geo.py，FEATURES=4 | 50 | 1.736392 | 2.484331 | 2.897108 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 41 | 1.128113 | 1.807065 | 2.729409 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 42 | 1.139477 | 1.831386 | 2.176000 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 43 | 1.227302 | 1.886729 | 2.459811 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 44 | 1.228374 | 1.925192 | 2.586381 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 45 | 1.068314 | 1.740138 | 2.298320 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 46 | 1.115749 | 1.742291 | 2.247256 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 47 | 1.214916 | 1.935528 | 2.560503 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 48 | 1.285164 | 1.969229 | 2.436309 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 49 | 1.268670 | 2.003630 | 2.645652 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 50 | 1.158125 | 1.821031 | 2.509901 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 41 | 1.154888 | 1.774641 | 2.265151 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 42 | 1.119029 | 1.787904 | 2.378950 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 43 | 1.278818 | 1.952879 | 2.309438 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 44 | 1.139445 | 1.839836 | 2.172368 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 45 | 1.125080 | 1.833420 | 2.384431 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 46 | 1.039768 | 1.628083 | 2.285124 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 47 | 1.141591 | 1.830415 | 2.530029 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 48 | 1.332630 | 2.094155 | 2.302834 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 49 | 1.215346 | 1.876350 | 2.707918 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 50 | 1.100432 | 1.771357 | 2.474972 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 41 | 1.223124 | 1.876222 | 2.524028 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 42 | 1.309876 | 1.923407 | 2.462727 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 43 | 1.291969 | 1.998328 | 2.607781 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 44 | 1.255227 | 1.953474 | 2.693109 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 45 | 1.157241 | 1.808409 | 2.235486 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 46 | 1.171686 | 1.779325 | 2.366536 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 47 | 1.283572 | 1.930929 | 2.744543 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 48 | 1.395160 | 2.157667 | 3.092523 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 49 | 1.422972 | 2.041232 | 2.757046 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4 | 50 | 1.271175 | 1.817523 | 2.564173 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 41 | 1.793873 | 2.891996 | 2.190180 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 42 | 1.655510 | 2.514212 | 2.318352 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 43 | 5.821480 | 6.870378 | 12.555216 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 44 | 1.646650 | 2.637954 | 1.996316 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 45 | 6.496903 | 7.385670 | 14.222601 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 46 | 4.814300 | 5.945331 | 10.629205 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 47 | 2.745301 | 3.928422 | 5.223622 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 48 | 1.618516 | 2.594664 | 2.147925 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 49 | 7.653418 | 8.527852 | 18.566745 |
| pred_maskpredition_GWN.py，IS_PRETRN=False | 50 | 1.851425 | 2.536558 | 3.064276 |
