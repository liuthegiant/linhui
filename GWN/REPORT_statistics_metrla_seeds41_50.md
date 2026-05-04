# 随机种子 41–50 统计表（METRLA / GraphWaveNet）

- 生成时间: 2026-04-06T02:30:43
- 数据来源: `REPORT_batch_seeds41_45_260405094819.md`（种子 41–45）+ `REPORT_batch_seeds46_50_260405143537.md`（种子 46–50）汇总表中的 `est_METRLA_GraphWaveNet_*` 目录。
- 指标文件: `../save/<KEYWORD>/GraphWaveNet_prediction_scores.txt`。
- **tst_u**：unseen 测试；**tst_a**：全图空间划分测试。
- 数值均为 **均值 ± 样本标准差**；四位小数。误差类指标列内对**最小均值**加粗。

---

## 复现信息：参数、程序与产物路径

本报告对应 **状态估计 / 缺失填补** 流水线（`pred_maskpredition_GWN_*.py`），**不是** `pred_GWN_16_adpAdj.py` 的 12 步预测扫种。工作目录均为 `forecasting-on-new-roads/`（以下路径相对该目录）。

### 编排方式与批次报告

| 项 | 说明 |
| --- | --- |
| 环境变量 | `FONR_DATANAME=METRLA`（传给 `run_batch_seeds41_45.py` / `run_batch_seeds46_50.py` 内 `DATANAME`，并写入各 job 的 `sys.argv[6]`） |
| 编排脚本 | `run_batch_seeds41_45.py`（种子 41–45）、`run_batch_seeds46_50.py`（种子 46–50）；历史上亦可用 `run_batch_seeds41_45.sh`（日志目录名可能**无** `metrla` 前缀，见下行） |
| 本报告数据来源 | `REPORT_batch_seeds41_45_260405094819.md`、`REPORT_batch_seeds46_50_260405143537.md` 汇总表中的 `KEYWORD` |
| 单次 job 控制台日志 | **当前 Python 编排**：`batch_run_metrla_seeds41_45_<STAMP>/job_<jid>.log`、`batch_run_metrla_seeds46_50_<STAMP>/job_<jid>.log`；**旧版 shell** 可能为 `batch_run_seeds41_45_<STAMP>/`（无数据集前缀） |
| 汇总脚本 | `export_statistics_seeds41_50_md.py`（METRLA 需在仓库内存在更新的 `REPORT_batch_pemsbay_*.md` 时**显式**指定输入，避免默认抓到 PEMSBAY），示例：`python3 export_statistics_seeds41_50_md.py --report-41-45 REPORT_batch_seeds41_45_<STAMP>.md --report-46-50 REPORT_batch_seeds46_50_<STAMP>.md -o REPORT_statistics_metrla_seeds41_50.md` |

### `sys.argv` 位次（`pred_maskpredition_GWN_*.py` 共用 `get_argv()`）

批量命令**显式给出下标 1–12**；**13–20** 未传时使用源码默认值（与本批一致：`SUBGRAPH_SIZE=64`、`QUOTIENT_GRAPH_RADIUS=0.01`、`PRETRN_EPOCH=100`、`EPOCH=100`、`NETWORK_CALLS=0`、`PRE_LEARN` 默认、`GRAPH_NORM` 默认、`HIDDEN=320`）。

| i | 变量 | 本报告六配置下的取值 |
| --- | --- | --- |
| 1 | `IS_PRETRN` | geo / scpt_geo / scpt：`1`；`pred_maskpredition_GWN.py`：`0` |
| 2 | `R_TRN` | `0.7` |
| 3 | `IS_EPOCH_1` | `0` |
| 4 | `seed` | `41`…`50` |
| 5 | `TEMPERATURE` | `1.0` |
| 6 | `DATANAME` | `METRLA` |
| 7 | `seed_SS` | `-1` |
| 8 | `IS_DESEASONED` | `1` |
| 9 | `weight_decay` | `0.0001` |
| 10 | `adp_adj` | `1` |
| 11 | `is_SGA` | `1` |
| 12 | `FEATURES` | geo / scpt_geo：`2` 或 `4`；scpt / GWN baseline：`4` |

### 六配置与等价启动命令（`<SEED>` 替换 41–50）

```bash
# 1–2：pred_maskpredition_GWN_geo.py，FEATURES=2 或 4
python pred_maskpredition_GWN_geo.py 1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0001 1 1 <2|4>

# 3–4：pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 或 4
python pred_maskpredition_GWN_scpt_geo.py 1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0001 1 1 <2|4>

# 5：pred_maskpredition_GWN_scpt.py（显式 FEATURES=4）
python pred_maskpredition_GWN_scpt.py 1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0001 1 1 4

# 6：pred_maskpredition_GWN.py，无预训练，FEATURES=4
python pred_maskpredition_GWN.py 0 0.7 0 <SEED> 1.0 METRLA -1 1 0.0001 1 1 4
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
| `../METRLA/metr-la.h5`、`../METRLA/adj_mx.pkl` | 流量与邻接（`P.N_NODE=207`，与 `pred_maskpredition_*.py` 中 `METRLA` 分支一致） |

### 单次运行目录：`../save/est_METRLA_GraphWaveNet_<yyMMddHHmm>_<pid>/`

`KEYWORD` 由脚本在运行时生成（`est_` + 数据集 + `GraphWaveNet` + 时间 + pid）。**指标**路径：

`../save/<KEYWORD>/GraphWaveNet_prediction_scores.txt`

| 类型 | 常见文件名 | 说明 |
| --- | --- | --- |
| 预训练 | `encoder.pt`、`encoder_log.txt` | 常见：时序/对比支路 |
| scpt_geo（双支路+融合） | `encoderg.pt`、`encoderg_log.txt`、`GraphWaveNet_fusion_u.pt` 等 | 几何支路与 fusion；以当次目录为准 |
| 其它有预训练脚本 | 仅 `encoder.pt` 等 | 分支较简时可能无 `encoderg` / `fusion` |
| 主模型 | `GraphWaveNet_best.pt` | 验证最优权重 |
| 主训练日志 | `GraphWaveNet_log.txt` | 各 epoch 训练记录 |
| 推理计时（若脚本版本已写入） | `GraphWaveNet_timing_infer.txt` | `tst_u` / `tst_a` 推理 wall（秒） |
| 预测输出 | `GraphWaveNet_tst_u_GraphWaveNet_*.npy`、`GraphWaveNet_tst_a_GraphWaveNet_*.npy` | `prediction` / `groundtruth` / `missmask` |
| 可视化 | `heatmap_*_tst_u.png`、`heatmap_*_tst_a.png` | 可选热图 |

在 **batch** 的 `job_*.log` 中搜索 `est_METRLA_GraphWaveNet_` 即可对齐当次 `../save/<KEYWORD>/` 目录名。

### 如何在大量 `../save/` 目录里找到对应的一次运行？

1. **优先看批次报告的「汇总表」**  
   本统计所依据的 `REPORT_batch_seeds41_45_*.md`、`REPORT_batch_seeds46_50_*.md` 里 **「汇总表」** 每行有一列 **`输出 KEYWORD`**（反引号内的字符串），即为 `../save/` 下的子目录名。在编辑器中全文搜索 `est_METRLA_GraphWaveNet` 可一次列出本批 60 个目录名。

2. **用 job 日志反查**  
   进入当次 batch 的日志目录（如 `batch_run_metrla_seeds41_45_<STAMP>/` 或历史 `batch_run_seeds41_45_<STAMP>/`），对全体日志做一次检索，例如在工作区根目录执行：  
   `rg 'est_METRLA_GraphWaveNet_[0-9]+_[0-9]+' /path/to/batch_run_.../job_*.log`  
   命中行附近或训练开头打印的 `KEYWORD` 与 `save` 子目录一致。

3. **按时间缩小范围（辅助）**  
   `ls -lt ../save/est_METRLA_GraphWaveNet_* | head -30`  
   只作粗筛；**最终以 batch 报告或日志中的完整 KEYWORD 为准**（时间戳+pid 组合仍可能混淆多次重跑）。

4. **不要用「只记 pid」**  
   目录名形如 `..._2604050948_1234567`，末尾为进程号；不同日期可能重复感强，务必用 **完整 KEYWORD** 对齐。

---

## 表一：六实验配置 × 十随机种子（41–50）

每组 **n=10**（10 个种子），标准差分母为 **9**（`statistics.stdev`）。


| 实验配置                                          | n   | train MAE (est)     | tst_u Masked MAE    | tst_u RMSE          | tst_u MAPE          | tst_a Masked MAE    | tst_a RMSE          | tst_a MAPE          |
| --------------------------------------------- | --- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 10  | 0.0837 ± 0.0076     | 2.4570 ± 0.3740     | 3.9001 ± 0.3973     | 1.5821 ± 0.3389     | 1.8594 ± 0.2345     | 3.1229 ± 0.2124     | 1.4374 ± 0.1298     |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 10  | 0.0839 ± 0.0077     | 2.2971 ± 0.2802     | 3.8252 ± 0.3947     | 1.5762 ± 0.3184     | 1.9921 ± 0.2638     | 3.2570 ± 0.2538     | 1.5478 ± 0.1504     |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 10  | 0.0727 ± 0.0065     | **1.9824 ± 0.2125** | **3.5603 ± 0.3174** | 1.4400 ± 0.3218     | 1.4827 ± 0.0909     | **2.7436 ± 0.0709** | 1.2879 ± 0.1216     |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 10  | **0.0722 ± 0.0063** | 1.9947 ± 0.2465     | 3.6011 ± 0.3663     | **1.3760 ± 0.2731** | **1.4595 ± 0.1283** | 2.7534 ± 0.1048     | **1.2548 ± 0.1061** |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 10  | 0.0727 ± 0.0067     | 2.0567 ± 0.2783     | 3.6032 ± 0.3646     | 1.4322 ± 0.3026     | 1.5943 ± 0.1845     | 2.8747 ± 0.1474     | 1.4382 ± 0.1802     |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 10  | 0.0859 ± 0.0082     | 2.5284 ± 0.3702     | 3.8369 ± 0.3223     | 1.4305 ± 0.2305     | 2.3140 ± 0.4743     | 3.5075 ± 0.3882     | 1.7526 ± 0.2953     |


## 表二：十随机种子 × 六实验配置（跨配置汇总）

对每个种子 **s**，在六种配置上的指标再算 **均值 ± 标准差**；每组 **n=6**，标准差分母为 **5**。


| 随机种子 | n   | train MAE (est)     | tst_u Masked MAE    | tst_u RMSE          | tst_u MAPE          | tst_a Masked MAE    | tst_a RMSE          | tst_a MAPE          |
| ---- | --- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| 41   | 6   | 0.0697 ± 0.0057     | 1.9889 ± 0.2954     | 3.5813 ± 0.1723     | 1.1315 ± 0.1128     | 1.8192 ± 0.5063     | 3.1339 ± 0.4591     | **1.3365 ± 0.1476** |
| 42   | 6   | 0.0854 ± 0.0076     | **1.8395 ± 0.1896** | **3.2959 ± 0.2470** | 1.2623 ± 0.1865     | 2.0034 ± 0.6817     | 3.2384 ± 0.5643     | 1.4513 ± 0.3210     |
| 43   | 6   | 0.0770 ± 0.0068     | 2.4549 ± 0.4381     | 4.1378 ± 0.1763     | 1.5490 ± 0.1535     | **1.5993 ± 0.2420** | **2.8249 ± 0.2094** | 1.3714 ± 0.1293     |
| 44   | 6   | 0.0822 ± 0.0061     | 2.0426 ± 0.1308     | 3.4788 ± 0.1051     | **1.0645 ± 0.0675** | 1.8011 ± 0.3677     | 3.0184 ± 0.3687     | 1.4868 ± 0.2882     |
| 45   | 6   | 0.0734 ± 0.0055     | 2.5578 ± 0.2508     | 4.1469 ± 0.1616     | 1.5359 ± 0.2546     | 1.8493 ± 0.6836     | 3.0968 ± 0.5685     | 1.4499 ± 0.4215     |
| 46   | 6   | 0.0904 ± 0.0088     | 2.2048 ± 0.2832     | 3.6033 ± 0.1462     | 1.6573 ± 0.0773     | 1.7636 ± 0.2285     | 3.1419 ± 0.2347     | 1.4195 ± 0.2195     |
| 47   | 6   | 0.0820 ± 0.0067     | 2.0632 ± 0.3133     | 3.4029 ± 0.2533     | 1.8350 ± 0.1821     | 1.7602 ± 0.3455     | 3.0365 ± 0.2787     | 1.5253 ± 0.1740     |
| 48   | 6   | 0.0823 ± 0.0068     | 2.3113 ± 0.2978     | 3.6763 ± 0.2208     | 1.6426 ± 0.1092     | 1.7059 ± 0.1026     | 2.9482 ± 0.1050     | 1.3777 ± 0.1221     |
| 49   | 6   | 0.0737 ± 0.0058     | 2.1837 ± 0.2387     | 3.6502 ± 0.0981     | 1.3755 ± 0.1309     | 1.8083 ± 0.2585     | 3.0065 ± 0.2809     | 1.5989 ± 0.2272     |
| 50   | 6   | **0.0690 ± 0.0069** | 2.5471 ± 0.4293     | 4.2379 ± 0.3577     | 1.6747 ± 0.3698     | 1.7263 ± 0.3303     | 2.9861 ± 0.2756     | 1.5137 ± 0.2178     |


## 附：各配置 × 各种子原始值（tst_u）


| 实验配置                                          | seed | Masked MAE | RMSE     | MAPE     |
| --------------------------------------------- | ---- | ---------- | -------- | -------- |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 41   | 2.091976   | 3.583424 | 1.259685 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 42   | 2.052478   | 3.638218 | 1.465464 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 43   | 2.435960   | 4.106839 | 1.426322 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 44   | 2.222894   | 3.614987 | 1.088462 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 45   | 2.703569   | 4.376601 | 1.588791 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 46   | 2.484637   | 3.745627 | 1.661497 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 47   | 2.173145   | 3.484615 | 2.022896 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 48   | 2.723154   | 4.011652 | 1.660923 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 49   | 2.394506   | 3.726026 | 1.420765 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 50   | 3.287411   | 4.713397 | 2.226365 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 41   | 2.203396   | 3.685168 | 1.280239 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 42   | 1.923384   | 3.336124 | 1.224599 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 43   | 2.341915   | 4.242385 | 1.418531 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 44   | 2.024384   | 3.425718 | 1.141319 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 45   | 2.787547   | 4.070535 | 1.903771 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 46   | 2.126414   | 3.794383 | 1.676204 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 47   | 2.250313   | 3.642794 | 1.891577 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 48   | 2.446168   | 3.751248 | 1.716098 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 49   | 2.159405   | 3.653890 | 1.449452 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 50   | 2.707704   | 4.649996 | 2.060625 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 41   | 1.767106   | 3.342843 | 1.128023 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 42   | 1.622307   | 3.009669 | 0.985546 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 43   | 2.227610   | 3.940093 | 1.532238 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 44   | 2.089666   | 3.567008 | 1.107059 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 45   | 2.345301   | 4.067897 | 1.735356 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 46   | 1.894161   | 3.449594 | 1.757291 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 47   | 1.926665   | 3.321007 | 1.934155 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 48   | 2.033312   | 3.529930 | 1.547124 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 49   | 1.879702   | 3.526470 | 1.188807 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 50   | 2.038360   | 3.848220 | 1.484747 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 41   | 1.793827   | 3.697711 | 1.032427 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 42   | 1.671058   | 3.089206 | 1.221559 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 43   | 2.316051   | 4.086696 | 1.613517 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 44   | 1.818029   | 3.317366 | 0.969616 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 45   | 2.371760   | 4.095402 | 1.300587 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 46   | 1.994529   | 3.545272 | 1.642808 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 47   | 1.821059   | 3.248358 | 1.822510 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 48   | 1.901788   | 3.361852 | 1.523378 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 49   | 1.971587   | 3.535433 | 1.279969 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 50   | 2.287279   | 4.033252 | 1.353438 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 41   | 1.659176   | 3.406319 | 1.050761 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 42   | 1.729251   | 3.184424 | 1.194647 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 43   | 2.090987   | 4.017261 | 1.478358 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 44   | 2.050845   | 3.472708 | 0.994029 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 45   | 2.285871   | 3.958297 | 1.251747 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 46   | 2.114523   | 3.453687 | 1.684920 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 47   | 1.678079   | 3.026924 | 1.846354 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 48   | 2.362220   | 3.656945 | 1.811338 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 49   | 2.193630   | 3.704820 | 1.559142 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 50   | 2.402098   | 4.150820 | 1.450820 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 41   | 2.417681   | 3.772418 | 1.037893 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 42   | 2.038665   | 3.517680 | 1.482116 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 43   | 3.316930   | 4.433332 | 1.825174 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 44   | 2.049556   | 3.475213 | 1.086712 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 45   | 2.852495   | 4.312809 | 1.434981 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 46   | 2.614305   | 3.631062 | 1.521203 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 47   | 2.530131   | 3.693841 | 1.492676 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 48   | 2.401342   | 3.746406 | 1.596741 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 49   | 2.503600   | 3.754759 | 1.354726 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 50   | 2.559726   | 4.031762 | 1.472317 |


## 附：各配置 × 各种子原始值（tst_a）


| 实验配置                                          | seed | Masked MAE | RMSE     | MAPE     |
| --------------------------------------------- | ---- | ---------- | -------- | -------- |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 41   | 2.056792   | 3.247845 | 1.552252 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 42   | 2.425370   | 3.629961 | 1.560428 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 43   | 1.865123   | 3.011513 | 1.383257 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 44   | 1.730639   | 2.930276 | 1.493533 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 45   | 1.823683   | 3.133530 | 1.228381 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 46   | 1.719537   | 3.184105 | 1.308322 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 47   | 1.859762   | 3.132043 | 1.621339 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 48   | 1.657191   | 3.065394 | 1.365432 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 49   | 1.835063   | 3.032757 | 1.524496 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 50   | 1.621170   | 2.861512 | 1.336587 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 41   | 1.774221   | 3.202264 | 1.427443 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 42   | 2.519182   | 3.713116 | 1.685945 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 43   | 1.700287   | 2.961240 | 1.413337 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 44   | 2.176043   | 3.429507 | 1.726296 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 45   | 1.854151   | 3.124620 | 1.419584 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 46   | 2.076037   | 3.534521 | 1.618624 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 47   | 2.017781   | 3.288936 | 1.621730 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 48   | 1.785041   | 2.962627 | 1.290313 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 49   | 1.772777   | 3.008641 | 1.563844 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 50   | 2.245690   | 3.344708 | 1.710982 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 41   | 1.368184   | 2.671254 | 1.165230 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 42   | 1.384827   | 2.676412 | 1.167703 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 43   | 1.478189   | 2.712246 | 1.402812 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 44   | 1.610378   | 2.782305 | 1.356179 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 45   | 1.476243   | 2.681906 | 1.194768 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 46   | 1.523682   | 2.871389 | 1.171330 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 47   | 1.583600   | 2.819564 | 1.463367 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 48   | 1.532697   | 2.788210 | 1.249895 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 49   | 1.525126   | 2.758455 | 1.459771 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 50   | 1.344101   | 2.673888 | 1.247648 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 41   | 1.438685   | 2.754510 | 1.192812 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 42   | 1.384846   | 2.704503 | 1.124172 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 43   | 1.348303   | 2.551885 | 1.249609 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 44   | 1.443145   | 2.746416 | 1.163668 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 45   | 1.307995   | 2.691962 | 1.148225 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 46   | 1.494702   | 2.935649 | 1.273062 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 47   | 1.326435   | 2.742949 | 1.228577 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 48   | 1.685737   | 2.889439 | 1.367650 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 49   | 1.637874   | 2.752622 | 1.351225 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 50   | 1.527254   | 2.763647 | 1.449011 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 41   | 1.559516   | 2.983963 | 1.386207 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 42   | 1.421720   | 2.811857 | 1.249025 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 43   | 1.342733   | 2.660310 | 1.209081 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 44   | 1.513491   | 2.689164 | 1.262932 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 45   | 1.460643   | 2.770018 | 1.432678 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 46   | 1.919935   | 3.124469 | 1.744233 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 47   | 1.519842   | 2.827241 | 1.491835 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 48   | 1.776862   | 2.927095 | 1.391504 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 49   | 1.798421   | 2.964262 | 1.693055 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 50   | 1.629655   | 2.988880 | 1.521019 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 41   | 2.717745   | 3.943687 | 1.295332 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 42   | 2.884166   | 3.894638 | 1.920299 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 43   | 1.861089   | 3.052034 | 1.570166 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 44   | 2.333183   | 3.533013 | 1.918415 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 45   | 3.173168   | 4.178992 | 2.275753 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 46   | 1.847844   | 3.201409 | 1.401227 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 47   | 2.253758   | 3.408530 | 1.724880 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 48   | 1.798138   | 3.056321 | 1.601583 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 49   | 2.280400   | 3.522356 | 2.001155 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 50   | 1.990179   | 3.283982 | 1.816985 |


