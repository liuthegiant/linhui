# 随机种子 41–50 统计表（PEMSBAY / GraphWaveNet）

- 生成时间: 2026-04-07T00:38:09
- 数据来源: `REPORT_batch_pemsbay_seeds41_45_260406044425.md`（种子 41–45）+ `REPORT_batch_pemsbay_seeds46_50_260406133356.md`（种子 46–50）汇总表中的 `est_<数据集>_GraphWaveNet_*` 目录。
- 指标文件: `../save/<KEYWORD>/GraphWaveNet_prediction_scores.txt`。
- **tst_u**：unseen 测试；**tst_a**：全图空间划分测试。
- 数值均为 **均值 ± 样本标准差**；四位小数。误差类指标列内对**最小均值**加粗。

---

## 复现信息：参数、程序与产物路径

本报告对应 **状态估计 / 缺失填补** 流水线（`pred_maskpredition_GWN_*.py`），**不是** `pred_GWN_16_adpAdj.py` 的 12 步预测扫种。工作目录均为 `forecasting-on-new-roads/`（以下路径相对该目录）。

### 编排方式与批次报告

| 项 | 说明 |
| --- | --- |
| 环境变量 | `run_batch_seeds41_45.py` / `run_batch_seeds46_50.py` 默认 `DATANAME=PEMSBAY`（即不设 `FONR_DATANAME` 即为 PEMSBAY）；若与其它数据集混跑，可显式 `FONR_DATANAME=PEMSBAY` |
| 编排脚本 | `run_batch_seeds41_45.py`（种子 41–45）、`run_batch_seeds46_50.py`（种子 46–50）；亦可使用仓库内历史 PEMSBAY 批量 shell / nohup 流程（与上述 Python 编排等价） |
| 本报告数据来源 | `REPORT_batch_pemsbay_seeds41_45_260406044425.md`、`REPORT_batch_pemsbay_seeds46_50_260406133356.md` 汇总表中的 `KEYWORD` |
| 单次 job 控制台日志 | `batch_run_pemsbay_seeds41_45_<STAMP>/job_<jid>.log`、`batch_run_pemsbay_seeds46_50_<STAMP>/job_<jid>.log`（可用 `latest_batch_run_dir_pemsbay_seeds41_45.txt`、`latest_batch_run_dir_pemsbay_seeds46_50.txt` 定位最新目录） |
| 汇总脚本 | `export_statistics_seeds41_50_md.py`（`default_report_pair()` 优先匹配 `REPORT_batch_pemsbay_seeds*.md` 时可直接 `-o` 指定本文件）；显式指定示例：`python3 export_statistics_seeds41_50_md.py --report-41-45 REPORT_batch_pemsbay_seeds41_45_<STAMP>.md --report-46-50 REPORT_batch_pemsbay_seeds46_50_<STAMP>.md -o REPORT_statistics_pemsbay_seeds41_50.md` |

### `sys.argv` 位次（`pred_maskpredition_GWN_*.py` 共用 `get_argv()`）

批量命令**显式给出下标 1–12**；**13–20** 未传时使用源码默认值（与本批一致：`SUBGRAPH_SIZE=64`、`QUOTIENT_GRAPH_RADIUS=0.01`、`PRETRN_EPOCH=100`、`EPOCH=100`、`NETWORK_CALLS=0`、`PRE_LEARN` 默认、`GRAPH_NORM` 默认、`HIDDEN=320`）。

| i | 变量 | 本报告六配置下的取值 |
| --- | --- | --- |
| 1 | `IS_PRETRN` | geo / scpt_geo / scpt：`1`；`pred_maskpredition_GWN.py`：`0` |
| 2 | `R_TRN` | `0.7` |
| 3 | `IS_EPOCH_1` | `0` |
| 4 | `seed` | `41`…`50` |
| 5 | `TEMPERATURE` | `1.0` |
| 6 | `DATANAME` | `PEMSBAY` |
| 7 | `seed_SS` | `-1` |
| 8 | `IS_DESEASONED` | `1` |
| 9 | `weight_decay` | `0.0001` |
| 10 | `adp_adj` | `1` |
| 11 | `is_SGA` | `1` |
| 12 | `FEATURES` | geo / scpt_geo：`2` 或 `4`；scpt / GWN baseline：`4` |

### 六配置与等价启动命令（`<SEED>` 替换 41–50）

```bash
# 1–2：pred_maskpredition_GWN_geo.py，FEATURES=2 或 4
python pred_maskpredition_GWN_geo.py 1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0001 1 1 <2|4>

# 3–4：pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 或 4
python pred_maskpredition_GWN_scpt_geo.py 1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0001 1 1 <2|4>

# 5：pred_maskpredition_GWN_scpt.py（显式 FEATURES=4）
python pred_maskpredition_GWN_scpt.py 1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0001 1 1 4

# 6：pred_maskpredition_GWN.py，无预训练，FEATURES=4
python pred_maskpredition_GWN.py 0 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0001 1 1 4
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
| `../PEMSBAY/pems-bay.h5`、`../PEMSBAY/adj_mx_bay.pkl` | 流量与邻接（`P.N_NODE=325`，与 `pred_maskpredition_*.py` 中 `PEMSBAY` 分支一致） |

### 单次运行目录：`../save/est_PEMSBAY_GraphWaveNet_<yyMMddHHmm>_<pid>/`

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

在 **batch** 的 `job_*.log` 中搜索 `est_PEMSBAY_GraphWaveNet_` 即可对齐当次 `../save/<KEYWORD>/` 目录名。

### 如何在大量 `../save/` 目录里找到对应的一次运行？

1. **优先看批次报告的「汇总表」**  
   `REPORT_batch_pemsbay_seeds41_45_*.md`、`REPORT_batch_pemsbay_seeds46_50_*.md` 的 **「汇总表」** 中 **`输出 KEYWORD`** 列即为 `../save/<KEYWORD>/`。全文搜索 `est_PEMSBAY_GraphWaveNet` 可列出本批 60 个目录名。

2. **用 job 日志反查**  
   在 `batch_run_pemsbay_seeds41_45_<STAMP>/`、`batch_run_pemsbay_seeds46_50_<STAMP>/` 下执行：  
   `rg 'est_PEMSBAY_GraphWaveNet_[0-9]+_[0-9]+' job_*.log`

3. **按时间缩小范围（辅助）**  
   `ls -lt ../save/est_PEMSBAY_GraphWaveNet_* | head -30`  
   仅作粗筛，**以 batch 报告或日志中的完整 KEYWORD 为准**。

4. **不要用「只记 pid」**  
   务必使用 **完整 KEYWORD**（含 `yyMMddHHmm` 与 pid）对齐目录。

---

## 表一：六实验配置 × 十随机种子（41–50）

每组 **n=10**（10 个种子），标准差分母为 **9**（`statistics.stdev`）。


| 实验配置                                          | n   | train MAE (est)     | tst_u Masked MAE    | tst_u RMSE          | tst_u MAPE          | tst_a Masked MAE    | tst_a RMSE          | tst_a MAPE          |
| --------------------------------------------- | --- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 10  | 0.0903 ± 0.0044     | 1.0991 ± 0.1252     | 1.8107 ± 0.2234     | 3.1578 ± 0.4344     | 0.8410 ± 0.0354     | 1.3677 ± 0.0488     | 2.8175 ± 0.3604     |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 10  | 0.0897 ± 0.0041     | 1.0952 ± 0.1445     | 1.8119 ± 0.1609     | 3.2131 ± 0.7590     | 0.8313 ± 0.0473     | 1.3518 ± 0.0504     | 2.7826 ± 0.2858     |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 10  | **0.0750 ± 0.0033** | 0.7934 ± 0.0438     | 1.3185 ± 0.0778     | 2.3586 ± 0.2959     | 0.6083 ± 0.0305     | 1.0119 ± 0.0330     | 2.1299 ± 0.1177     |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 10  | **0.0750 ± 0.0032** | **0.7775 ± 0.0635** | **1.3111 ± 0.1169** | **2.2878 ± 0.2902** | **0.5962 ± 0.0299** | **0.9990 ± 0.0351** | **2.1005 ± 0.1323** |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 10  | 0.0757 ± 0.0033     | 0.7975 ± 0.0623     | 1.3177 ± 0.1018     | 2.2990 ± 0.3080     | 0.6483 ± 0.0441     | 1.0505 ± 0.0400     | 2.2532 ± 0.1709     |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 10  | 0.1009 ± 0.0053     | 1.1309 ± 0.1805     | 1.8015 ± 0.2544     | 3.4707 ± 1.0159     | 0.9548 ± 0.0848     | 1.5515 ± 0.1560     | 3.4407 ± 0.5406     |


## 表二：十随机种子 × 六实验配置（跨配置汇总）

对每个种子 **s**，在六种配置上的指标再算 **均值 ± 标准差**；每组 **n=6**，标准差分母为 **5**。


| 随机种子 | n   | train MAE (est)     | tst_u Masked MAE    | tst_u RMSE          | tst_u MAPE          | tst_a Masked MAE    | tst_a RMSE          | tst_a MAPE          |
| ---- | --- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| 41   | 6   | 0.0802 ± 0.0100     | 0.9557 ± 0.1806     | 1.5194 ± 0.2801     | 2.6369 ± 0.8310     | **0.7124 ± 0.1299** | 1.1973 ± 0.2066     | 2.4745 ± 0.5537     |
| 42   | 6   | 0.0859 ± 0.0103     | 1.1621 ± 0.3122     | 1.8817 ± 0.4173     | 3.4445 ± 1.2980     | 0.7279 ± 0.1239     | 1.2031 ± 0.2141     | 2.4101 ± 0.3087     |
| 43   | 6   | 0.0886 ± 0.0133     | 0.8724 ± 0.1802     | 1.5194 ± 0.2500     | 2.5530 ± 0.6023     | 0.7633 ± 0.1471     | 1.2354 ± 0.2481     | **2.3794 ± 0.2281** |
| 44   | 6   | 0.0885 ± 0.0100     | 0.9131 ± 0.1410     | 1.5251 ± 0.2269     | 2.4817 ± 0.3069     | 0.7672 ± 0.1578     | 1.2319 ± 0.2263     | 2.8361 ± 0.6455     |
| 45   | 6   | 0.0849 ± 0.0111     | **0.8549 ± 0.1511** | **1.3896 ± 0.2254** | 2.3550 ± 0.4588     | 0.7912 ± 0.1994     | 1.2623 ± 0.2718     | 2.7827 ± 0.7607     |
| 46   | 6   | 0.0818 ± 0.0111     | 0.9033 ± 0.1580     | 1.5395 ± 0.3111     | **2.3348 ± 0.3327** | 0.7517 ± 0.1618     | 1.2370 ± 0.2521     | 2.6655 ± 0.6431     |
| 47   | 6   | 0.0871 ± 0.0122     | 0.9016 ± 0.1577     | 1.4950 ± 0.2373     | 2.8633 ± 0.6656     | 0.7355 ± 0.1327     | 1.2139 ± 0.2101     | 2.7394 ± 0.5796     |
| 48   | 6   | 0.0832 ± 0.0098     | 1.0063 ± 0.1914     | 1.6220 ± 0.2710     | 3.0792 ± 0.6030     | 0.7166 ± 0.1750     | **1.1694 ± 0.2398** | 2.3985 ± 0.6302     |
| 49   | 6   | 0.0863 ± 0.0113     | 0.9065 ± 0.1601     | 1.4176 ± 0.1996     | 2.9955 ± 0.5841     | 0.7498 ± 0.1043     | 1.2046 ± 0.1593     | 2.6137 ± 0.3938     |
| 50   | 6   | **0.0780 ± 0.0104** | 1.0133 ± 0.2045     | 1.7099 ± 0.3528     | 3.2345 ± 0.8565     | 0.7508 ± 0.1939     | 1.2660 ± 0.3522     | 2.5741 ± 0.8459     |


## 附：各配置 × 各种子原始值（tst_u）


| 实验配置                                          | seed | Masked MAE | RMSE     | MAPE     |
| --------------------------------------------- | ---- | ---------- | -------- | -------- |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 41   | 1.094299   | 1.709238 | 2.825926 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 42   | 1.341496   | 2.175695 | 3.339586 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 43   | 1.021009   | 1.697573 | 3.207585 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 44   | 1.018784   | 1.711342 | 2.972530 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 45   | 1.018064   | 1.604904 | 2.850234 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 46   | 1.130357   | 1.810755 | 2.884053 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 47   | 1.006787   | 1.765940 | 2.939825 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 48   | 1.099877   | 1.804609 | 2.806257 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 49   | 0.969161   | 1.583600 | 3.604766 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 50   | 1.290804   | 2.243099 | 4.146797 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 41   | 1.040298   | 1.839335 | 2.490202 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 42   | 1.407121   | 2.135527 | 4.442947 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 43   | 0.959489   | 1.797452 | 2.391356 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 44   | 1.003423   | 1.752984 | 2.375048 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 45   | 1.026981   | 1.654570 | 2.990341 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 46   | 0.989567   | 1.774995 | 2.539406 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 47   | 1.139749   | 1.740734 | 3.968158 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 48   | 1.270111   | 1.980454 | 3.897110 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 49   | 0.988343   | 1.566673 | 3.369796 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 50   | 1.126689   | 1.875921 | 3.666975 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 41   | 0.782800   | 1.255313 | 2.289793 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 42   | 0.853309   | 1.460661 | 2.203403 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 43   | 0.710697   | 1.277364 | 2.073210 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 44   | 0.811776   | 1.344293 | 2.468301 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 45   | 0.756440   | 1.192170 | 1.942727 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 46   | 0.775943   | 1.266005 | 2.016129 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 47   | 0.759750   | 1.307377 | 2.729879 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 48   | 0.820084   | 1.354383 | 2.504306 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 49   | 0.828853   | 1.320833 | 2.753279 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 50   | 0.834028   | 1.406937 | 2.604846 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 41   | 0.809200   | 1.285213 | 1.991999 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 42   | 0.889550   | 1.554153 | 2.713271 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 43   | 0.705123   | 1.303470 | 1.955551 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 44   | 0.735417   | 1.265688 | 2.052231 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 45   | 0.696574   | 1.163360 | 2.079379 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 46   | 0.774010   | 1.269377 | 2.167220 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 47   | 0.773470   | 1.290088 | 2.483480 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 48   | 0.842142   | 1.411432 | 2.634060 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 49   | 0.725036   | 1.171020 | 2.198048 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 50   | 0.824837   | 1.397474 | 2.602712 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 41   | 0.802842   | 1.259692 | 2.014345 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 42   | 0.914219   | 1.526326 | 2.441273 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 43   | 0.726431   | 1.298608 | 2.296906 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 44   | 0.822883   | 1.350293 | 2.637983 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 45   | 0.715500   | 1.210215 | 1.960625 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 46   | 0.749886   | 1.236434 | 2.051963 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 47   | 0.771730   | 1.255583 | 1.980089 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 48   | 0.855962   | 1.381641 | 2.851145 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 49   | 0.773840   | 1.232904 | 2.524413 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 50   | 0.841509   | 1.425627 | 2.231109 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 41   | 1.204731   | 1.767908 | 4.208916 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 42   | 1.567063   | 2.437692 | 5.526383 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 43   | 1.111623   | 1.741789 | 3.393527 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 44   | 1.086540   | 1.725847 | 2.384049 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 45   | 0.915880   | 1.512490 | 2.306830 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 46   | 0.999853   | 1.879154 | 2.350101 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 47   | 0.958166   | 1.610006 | 3.078244 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 48   | 1.149647   | 1.799499 | 3.782176 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 49   | 1.153470   | 1.630301 | 3.522798 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 50   | 1.162127   | 1.910205 | 4.154316 |


## 附：各配置 × 各种子原始值（tst_a）


| 实验配置                                          | seed | Masked MAE | RMSE     | MAPE     |
| --------------------------------------------- | ---- | ---------- | -------- | -------- |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 41   | 0.852376   | 1.409282 | 2.853554 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 42   | 0.819034   | 1.359917 | 2.372258 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 43   | 0.853554   | 1.376365 | 2.612669 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 44   | 0.849771   | 1.366589 | 3.211821 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 45   | 0.910192   | 1.464313 | 3.377485 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 46   | 0.856671   | 1.363255 | 2.934343 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 47   | 0.857029   | 1.396754 | 3.153609 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 48   | 0.805428   | 1.294227 | 2.543481 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 49   | 0.825006   | 1.322485 | 2.805063 |
| pred_maskpredition_GWN_geo.py，FEATURES=2      | 50   | 0.780630   | 1.323617 | 2.310685 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 41   | 0.756234   | 1.329967 | 2.356597 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 42   | 0.864982   | 1.438049 | 2.832131 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 43   | 0.805553   | 1.315941 | 2.656023 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 44   | 0.870444   | 1.377554 | 3.328472 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 45   | 0.901543   | 1.419366 | 2.927288 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 46   | 0.810559   | 1.281615 | 2.769888 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 47   | 0.763222   | 1.302796 | 2.937856 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 48   | 0.849866   | 1.348997 | 2.436459 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 49   | 0.830858   | 1.328497 | 2.980621 |
| pred_maskpredition_GWN_geo.py，FEATURES=4      | 50   | 0.860196   | 1.375277 | 2.600523 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 41   | 0.607320   | 1.039736 | 2.141354 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 42   | 0.653762   | 1.044647 | 2.195875 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 43   | 0.632266   | 1.022264 | 2.199011 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 44   | 0.628765   | 1.033665 | 2.226020 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 45   | 0.615205   | 1.011245 | 2.097475 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 46   | 0.611582   | 1.042505 | 2.274902 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 47   | 0.606559   | 0.994422 | 2.171722 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 48   | 0.557842   | 0.955141 | 1.949031 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 49   | 0.612941   | 1.017398 | 2.136495 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=2 | 50   | 0.556352   | 0.958125 | 1.907049 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 41   | 0.593382   | 0.990668 | 2.012741 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 42   | 0.566711   | 0.955696 | 2.069054 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 43   | 0.589317   | 0.992237 | 2.104865 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 44   | 0.579278   | 0.980204 | 2.057767 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 45   | 0.593986   | 1.007223 | 2.002938 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 46   | 0.595731   | 1.014313 | 2.192161 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 47   | 0.621697   | 1.037209 | 2.245918 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 48   | 0.558705   | 0.946425 | 1.898822 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 49   | 0.665456   | 1.063870 | 2.357111 |
| pred_maskpredition_GWN_scpt_geo.py，FEATURES=4 | 50   | 0.597277   | 1.002499 | 2.063439 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 41   | 0.596788   | 1.002962 | 2.067361 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 42   | 0.636134   | 1.032669 | 2.250562 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 43   | 0.715762   | 1.079529 | 2.252325 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 44   | 0.691336   | 1.093548 | 2.554540 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 45   | 0.648665   | 1.048390 | 2.377912 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 46   | 0.639237   | 1.058235 | 2.048409 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 47   | 0.643529   | 1.061288 | 2.330208 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 48   | 0.572962   | 0.975710 | 2.001084 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 49   | 0.697844   | 1.108055 | 2.317359 |
| pred_maskpredition_GWN_scpt.py，默认 FEATURES=4  | 50   | 0.640763   | 1.044620 | 2.332518 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 41   | 0.868050   | 1.411062 | 3.415334 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 42   | 0.826547   | 1.387394 | 2.740734 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 43   | 0.983632   | 1.626010 | 2.451805 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 44   | 0.983683   | 1.539856 | 3.637830 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 45   | 1.077909   | 1.622985 | 3.912821 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 46   | 0.996349   | 1.661822 | 3.773260 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 47   | 0.920995   | 1.490941 | 3.596872 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 48   | 0.954696   | 1.495905 | 3.561919 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 49   | 0.866523   | 1.387506 | 3.085496 |
| pred_maskpredition_GWN.py，IS_PRETRN=False     | 50   | 1.069645   | 1.891921 | 4.230549 |


