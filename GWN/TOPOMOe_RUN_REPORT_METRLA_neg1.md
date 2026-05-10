# TopoMoE（METRLA）**五随机种子**实验汇总（METRLA `-1` / 100+100 epoch BASE）

- **BASE（估计与预测共用 argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`
- **日志根目录**：`logs_topomoe/seed5_metrla_neg1`
- **种子**：`100, 42, 999, 555, 250`（**n=5**）
- **一键扫种**：`./run_topomoe_5seed_sweep.sh`（每轮：`A` 7 并行 → `B` 7 并行，占用 GPU **0–6**）
- **生成/更新本报告**：`python3 aggregate_topomoe_5seed.py --out TOPOMOe_RUN_REPORT_METRLA_neg1.md`（默认日志目录为 `logs_topomoe/seed5_metrla_neg1`）

---

## A2. Estimation：**n=5 汇总（均值 ± 样本标准差）**

*种子集合*：`100, 42, 999, 555, 250`，与仓库内 METRLA 五种子文档一致。*标准差*：`statistics.stdev`（分母 **n−1=4**）。

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 0 | — | — | — | — | — | — |
| GEO only | 0 | — | — | — | — | — | — |
| TOPO only | 0 | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | 0 | — | — | — | — | — | — |
| SCPT + TOPO | 0 | — | — | — | — | — | — |
| GEO + TOPO | 0 | — | — | — | — | — | — |
| SCPT + GEO + TOPO | 0 | — | — | — | — | — | — |

## B2. Forecasting：**n=5 汇总（均值 ± 样本标准差）**

| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 0 | — | — | — | — | — | — | — | — |
| GEO only | 0 | — | — | — | — | — | — | — | — |
| TOPO only | 0 | — | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | 0 | — | — | — | — | — | — | — | — |
| SCPT + TOPO | 0 | — | — | — | — | — | — | — | — |
| GEO + TOPO | 0 | — | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | 0 | — | — | — | — | — | — | — | — |

---

## A1. Estimation：按种子分项（完整指标）

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_only_s100.log` | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/100/est/A_geo_only_s100.log` | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/100/est/A_topo_only_s100.log` | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_geo_s100.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_topo_s100.log` | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/est/A_geo_topo_s100.log` | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_geo_topo_s100.log` | — | — | — | — | — | — | — |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_only_s42.log` | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/42/est/A_geo_only_s42.log` | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/42/est/A_topo_only_s42.log` | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_geo_s42.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_topo_s42.log` | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/est/A_geo_topo_s42.log` | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_geo_topo_s42.log` | — | — | — | — | — | — | — |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_only_s999.log` | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/999/est/A_geo_only_s999.log` | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/999/est/A_topo_only_s999.log` | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_geo_s999.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_topo_s999.log` | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/est/A_geo_topo_s999.log` | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_geo_topo_s999.log` | — | — | — | — | — | — | — |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_only_s555.log` | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/555/est/A_geo_only_s555.log` | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/555/est/A_topo_only_s555.log` | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_geo_s555.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_topo_s555.log` | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/est/A_geo_topo_s555.log` | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_geo_topo_s555.log` | — | — | — | — | — | — | — |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_only_s250.log` | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/250/est/A_geo_only_s250.log` | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/250/est/A_topo_only_s250.log` | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_geo_s250.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_topo_s250.log` | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/est/A_geo_topo_s250.log` | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_geo_topo_s250.log` | — | — | — | — | — | — | — |

## B1. Forecasting：按种子分项（`all pred steps`）

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_only_s100.log` | — | — | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/100/pred/B_geo_only_s100.log` | — | — | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/100/pred/B_topo_only_s100.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_geo_s100.log` | — | — | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_topo_s100.log` | — | — | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/pred/B_geo_topo_s100.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_geo_topo_s100.log` | — | — | — | — | — | — | — | — | — |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_only_s42.log` | — | — | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/42/pred/B_geo_only_s42.log` | — | — | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/42/pred/B_topo_only_s42.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_geo_s42.log` | — | — | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_topo_s42.log` | — | — | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/pred/B_geo_topo_s42.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_geo_topo_s42.log` | — | — | — | — | — | — | — | — | — |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_only_s999.log` | — | — | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/999/pred/B_geo_only_s999.log` | — | — | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/999/pred/B_topo_only_s999.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_geo_s999.log` | — | — | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_topo_s999.log` | — | — | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/pred/B_geo_topo_s999.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_geo_topo_s999.log` | — | — | — | — | — | — | — | — | — |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_only_s555.log` | — | — | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/555/pred/B_geo_only_s555.log` | — | — | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/555/pred/B_topo_only_s555.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_geo_s555.log` | — | — | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_topo_s555.log` | — | — | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/pred/B_geo_topo_s555.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_geo_topo_s555.log` | — | — | — | — | — | — | — | — | — |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_only_s250.log` | — | — | — | — | — | — | — | — | — |
| GEO only | `logs_topomoe/seed5_metrla_neg1/250/pred/B_geo_only_s250.log` | — | — | — | — | — | — | — | — | — |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/250/pred/B_topo_only_s250.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_geo_s250.log` | — | — | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_topo_s250.log` | — | — | — | — | — | — | — | — | — |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/pred/B_geo_topo_s250.log` | — | — | — | — | — | — | — | — | — |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_geo_topo_s250.log` | — | — | — | — | — | — | — | — | — |

