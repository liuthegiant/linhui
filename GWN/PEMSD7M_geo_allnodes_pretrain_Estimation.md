# PEMSD7M Geo 预训练（全节点池）Estimation（5 seeds，无 SCPT）

- **任务**：Estimation 掩码预测
- **数据集**：PEMSD7M
- **配置**：`GEO only` / `GEO+TOPO`（与主实验相同 MoE 设置）
- **与主实验差异**：`GEO_PRETRAIN_TRAIN_ONLY=0`，几何预训练从**全部节点**抽样，而非仅 `spatialSplit_unseen.i_trn`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`
- **日志根目录**：`logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5`
- **种子**：`100, 42, 999, 555, 250`（**n=5**）
- **运行脚本**：`./run_pemsd7m_notopo_allnodes_6gpu.sh`（Batch B）
- **主指标**：`tst_u` MAE（unseen）
- **报告更新时间**：2026-05-17 05:56:10

---

## 进度（共 10 项：2 配置 × 5 种子）

| 配置 | 种子 | 状态 | tst_u MAE | tst_a MAE | 耗时 |
| --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | 100 | 完成 | 2.244641 | 2.082822 | 0:28:42.028040 |
| GEO（全节点 geo 预训练） | 42 | 完成 | 2.403778 | 2.785916 | 0:28:35.129787 |
| GEO（全节点 geo 预训练） | 999 | 完成 | 1.843607 | 1.695579 | 0:28:28.725594 |
| GEO（全节点 geo 预训练） | 555 | 完成 | 1.626416 | 1.854521 | 0:27:33.788526 |
| GEO（全节点 geo 预训练） | 250 | 完成 | 2.078090 | 1.770981 | 0:27:37.756826 |
| GEO+TOPO（全节点 geo 预训练） | 100 | 完成 | 1.775785 | 1.557084 | 0:28:46.031297 |
| GEO+TOPO（全节点 geo 预训练） | 42 | 完成 | 1.912354 | 1.423705 | 0:28:33.420744 |
| GEO+TOPO（全节点 geo 预训练） | 999 | 完成 | 1.825624 | 1.530172 | 0:28:57.197176 |
| GEO+TOPO（全节点 geo 预训练） | 555 | 完成 | 1.719411 | 1.411338 | 0:28:06.172024 |
| GEO+TOPO（全节点 geo 预训练） | 250 | 完成 | 1.836908 | 1.562677 | 0:27:57.140510 |

**进度**：完成 **10/10**，进行中 **0**，未开始 **0**

## PEMSD7M 汇总（tst_u MAE，均值 ± 样本标准差）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |
| --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | 5 | 2.0393 ± 0.3104 | 3.0575 ± 0.3426 | 2.8510 ± 0.6695 | 2.0380 ± 0.4426 |
| GEO+TOPO（全节点 geo 预训练） | 5 | 1.8140 ± 0.0720 | 2.8028 ± 0.0961 | 2.5265 ± 0.4760 | 1.4970 ± 0.0737 |

## 与主报告（train 节点 geo 预训练）对照

| 配置 | 本表（全节点 geo 预训练）tst_u MAE | 主报告 GEO 预训练池=tst_u MAE |
| --- | --- | --- |
| GEO | 2.0393 ± 0.3104 | 2.0198 ± 0.2263 |
| GEO+TOPO | 1.8140 ± 0.0720 | 1.7850 ± 0.1215 |

---

## 按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |
| --- | --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/100/est/A_geo_only_s100.log` | 0:28:42.028040 | 2.244641 | 2.945120 | 3.913817 | 2.082822 |
| GEO+TOPO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/100/est/A_geo_topo_s100.log` | 0:28:46.031297 | 1.775785 | 2.698839 | 3.232412 | 1.557084 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |
| --- | --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/42/est/A_geo_only_s42.log` | 0:28:35.129787 | 2.403778 | 3.381281 | 2.813206 | 2.785916 |
| GEO+TOPO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/42/est/A_geo_topo_s42.log` | 0:28:33.420744 | 1.912354 | 2.931751 | 2.360590 | 1.423705 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |
| --- | --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/999/est/A_geo_only_s999.log` | 0:28:28.725594 | 1.843607 | 2.663595 | 2.595334 | 1.695579 |
| GEO+TOPO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/999/est/A_geo_topo_s999.log` | 0:28:57.197176 | 1.825624 | 2.798603 | 2.025736 | 1.530172 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |
| --- | --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/555/est/A_geo_only_s555.log` | 0:27:33.788526 | 1.626416 | 2.848526 | 2.853992 | 1.854521 |
| GEO+TOPO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/555/est/A_geo_topo_s555.log` | 0:28:06.172024 | 1.719411 | 2.724270 | 2.760881 | 1.411338 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |
| --- | --- | --- | --- | --- | --- | --- |
| GEO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/250/est/A_geo_only_s250.log` | 0:27:37.756826 | 2.078090 | 3.448988 | 2.078695 | 1.770981 |
| GEO+TOPO（全节点 geo 预训练） | `logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5/PEMSD7M/250/est/A_geo_topo_s250.log` | 0:27:57.140510 | 1.836908 | 2.860623 | 2.253019 | 1.562677 |

