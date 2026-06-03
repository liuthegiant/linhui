# PEMSBAY + PEMSD7M + METRLA Geo/Topo Estimation（5 seeds，无 SCPT）

- **任务**：Estimation 掩码预测
- **配置**：`无预训练 / GEO only / TOPO only / GEO+TOPO`（不含 SCPT）
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 <DATASET> -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`
- **日志根目录**：`logs_topomoe/est_geotopo_3ds_seed5`
- **种子**：`100, 42, 999, 555, 250`（**n=5**）
- **运行脚本**：`./run_topomoe_est_geotopo_3ds_6gpu.sh`
- **主指标**：`tst_u` MAE（unseen）；`tst_a` 仅作参考
- **报告更新时间**：2026-05-17 09:20:49

---

## 补充批次进度（RUN_SCOPE=supplement，共 25 项）

| 数据集 | 配置 | 种子 | 状态 | tst_u MAE | tst_a MAE | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| PEMSBAY | GEO | 100 | 完成 | 1.025147 | 0.777177 | 2:06:32.983110 |
| PEMSBAY | GEO | 42 | 完成 | 1.213080 | 0.825169 | 2:05:57.935140 |
| PEMSBAY | GEO | 999 | 完成 | 1.127935 | 0.833106 | 3:15:50.763770 |
| PEMSBAY | GEO | 555 | 完成 | 1.136884 | 0.725858 | 2:04:33.920477 |
| PEMSBAY | GEO | 250 | 完成 | 1.000720 | 0.706964 | 2:44:47.894706 |
| PEMSBAY | GEO+TOPO | 100 | 完成 | 0.974593 | 0.621823 | 2:07:07.250741 |
| PEMSBAY | GEO+TOPO | 42 | 完成 | 1.070837 | 0.695710 | 2:08:58.047389 |
| PEMSBAY | GEO+TOPO | 999 | 完成 | 1.006945 | 0.647720 | 2:09:00.399926 |
| PEMSBAY | GEO+TOPO | 555 | 完成 | 1.051367 | 0.645362 | 2:04:28.726245 |
| PEMSBAY | GEO+TOPO | 250 | 完成 | 0.984915 | 0.678833 | 2:24:27.863898 |
| PEMSD7M | GEO | 100 | 完成 | 1.995271 | 2.367550 | 0:29:14.246937 |
| PEMSD7M | GEO | 42 | 完成 | 2.187472 | 2.426868 | 0:31:00.201149 |
| PEMSD7M | GEO | 999 | 完成 | 2.163097 | 1.709516 | 0:29:18.550376 |
| PEMSD7M | GEO | 555 | 完成 | 1.637156 | 2.027420 | 0:29:17.922645 |
| PEMSD7M | GEO | 250 | 完成 | 2.116018 | 2.429504 | 0:29:25.384119 |
| PEMSD7M | GEO+TOPO | 100 | 完成 | 1.728855 | 1.491260 | 0:29:39.079679 |
| PEMSD7M | GEO+TOPO | 42 | 完成 | 1.911388 | 1.576600 | 0:45:57.587407 |
| PEMSD7M | GEO+TOPO | 999 | 完成 | 1.739349 | 1.484112 | 0:44:27.767476 |
| PEMSD7M | GEO+TOPO | 555 | 完成 | 1.635724 | 1.445743 | 0:55:33.540136 |
| PEMSD7M | GEO+TOPO | 250 | 完成 | 1.909832 | 1.457125 | 0:29:21.542735 |
| METRLA | 无预训练 | 100 | 完成 | 2.785440 | 2.291689 | 0:52:33.078253 |
| METRLA | 无预训练 | 42 | 完成 | 2.204587 | 2.953044 | 1:09:13.343692 |
| METRLA | 无预训练 | 999 | 完成 | 2.636102 | 2.744405 | 1:19:08.524885 |
| METRLA | 无预训练 | 555 | 完成 | 2.485818 | 2.311193 | 1:20:49.498228 |
| METRLA | 无预训练 | 250 | 完成 | 2.469482 | 2.695698 | 0:51:02.021416 |

**进度**：完成 **25/25**（含备份日志），进行中 **0**，未开始 **0**

## 补充批次已完成项汇总（tst_u MAE，均值 ± 标准差）

| 数据集 | 配置 | n | tst_u MAE | tst_u RMSE | tst_a MAE |
| --- | --- | --- | --- | --- | --- |
| PEMSBAY | GEO | 5 | 1.1008 ± 0.0872 | 1.8411 ± 0.1207 | 0.7737 ± 0.0569 |
| PEMSBAY | GEO+TOPO | 5 | 1.0177 ± 0.0419 | 1.7394 ± 0.0916 | 0.6579 ± 0.0293 |
| PEMSD7M | GEO | 5 | 2.0198 ± 0.2263 | 3.0335 ± 0.2996 | 2.1922 ± 0.3171 |
| PEMSD7M | GEO+TOPO | 5 | 1.7850 ± 0.1215 | 2.7689 ± 0.2000 | 1.4910 ± 0.0514 |
| METRLA | 无预训练 | 5 | 2.5163 ± 0.2163 | 3.9239 ± 0.2719 | 2.5992 ± 0.2886 |

---

## PEMSBAY：n=5 汇总（均值 ± 样本标准差）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 3 | 1.4845 ± 0.6994 | 2.1053 ± 0.6651 | 5.4544 ± 3.8966 | 0.8854 ± 0.0306 | 1.4484 ± 0.0649 | 3.1066 ± 0.5145 |
| GEO | 5 | 1.1008 ± 0.0872 | 1.8411 ± 0.1207 | 2.9645 ± 0.6664 | 0.7737 ± 0.0569 | 1.2762 ± 0.0764 | 2.4069 ± 0.2180 |
| TOPO | 2 | 1.1059 ± 0.0425 | 1.7519 ± 0.1297 | 3.4346 ± 1.0749 | 0.7950 ± 0.0821 | 1.2817 ± 0.1028 | 2.6059 ± 0.2014 |
| GEO+TOPO | 5 | 1.0177 ± 0.0419 | 1.7394 ± 0.0916 | 3.1512 ± 0.4708 | 0.6579 ± 0.0293 | 1.1195 ± 0.0468 | 2.1552 ± 0.1028 |

## PEMSBAY：按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/100/est/A_no_pretrain_s100.log` | — | — | — | — | — | — | — |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/100/est/A_geo_only_s100.log` | 2:06:32.983110 | 1.025147 | 1.689027 | 2.448396 | 0.777177 | 1.271439 | 2.724560 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/100/est/A_topo_only_s100.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/100/est/A_geo_topo_s100.log` | 2:07:07.250741 | 0.974593 | 1.645874 | 2.537428 | 0.621823 | 1.059750 | 2.189833 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/42/est/A_no_pretrain_s42.log` | — | — | — | — | — | — | — |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/42/est/A_geo_only_s42.log` | 2:05:57.935140 | 1.213080 | 1.926391 | 3.956065 | 0.825169 | 1.346350 | 2.151967 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/42/est/A_topo_only_s42.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/42/est/A_geo_topo_s42.log` | 2:08:58.047389 | 1.070837 | 1.879201 | 3.009193 | 0.695710 | 1.179263 | 2.108172 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/999/est/A_no_pretrain_s999.log` | — | — | — | — | — | — | — |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/999/est/A_geo_only_s999.log` | 3:15:50.763770 | 1.127935 | 1.995973 | 2.457536 | 0.833106 | 1.360758 | 2.425208 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/999/est/A_topo_only_s999.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/999/est/A_geo_topo_s999.log` | 2:09:00.399926 | 1.006945 | 1.747929 | 3.059783 | 0.647720 | 1.109734 | 2.094023 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/555/est/A_no_pretrain_s555.log` | — | — | — | — | — | — | — |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/555/est/A_geo_only_s555.log` | 2:04:33.920477 | 1.136884 | 1.796578 | 2.613134 | 0.725858 | 1.201467 | 2.467616 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/555/est/A_topo_only_s555.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/555/est/A_geo_topo_s555.log` | 2:04:28.726245 | 1.051367 | 1.755126 | 3.822119 | 0.645362 | 1.097179 | 2.064727 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/250/est/A_no_pretrain_s250.log` | — | — | — | — | — | — | — |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/250/est/A_geo_only_s250.log` | 2:44:47.894706 | 1.000720 | 1.797428 | 3.347332 | 0.706964 | 1.200911 | 2.265152 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/250/est/A_topo_only_s250.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSBAY/250/est/A_geo_topo_s250.log` | 2:24:27.863898 | 0.984915 | 1.669006 | 3.327570 | 0.678833 | 1.151426 | 2.319202 |

---

## PEMSD7M：n=5 汇总（均值 ± 样本标准差）


| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 5 | 1.8967 ± 0.2364 | 2.9501 ± 0.2027 | 2.2771 ± 0.2571 | 2.5655 ± 1.5243 | 3.4192 ± 1.4767 | 4.8387 ± 5.0970 |
| GEO | 5 | 1.9867 ± 0.2448 | 3.0329 ± 0.3103 | 2.7112 ± 0.4180 | 2.0390 ± 0.3653 | 2.8982 ± 0.3991 | 3.3330 ± 0.9715 |
| TOPO | 5 | 1.7301 ± 0.1479 | 2.7227 ± 0.1662 | 2.2118 ± 0.1028 | 1.5897 ± 0.0469 | 2.3915 ± 0.0600 | 3.1414 ± 0.1961 |
| GEO+TOPO | 5 | 1.7797 ± 0.1346 | 2.7806 ± 0.1883 | 2.3749 ± 0.3406 | 1.5158 ± 0.0623 | 2.3054 ± 0.0957 | 2.9301 ± 0.3908 |


- **种子**：`432, 250, 999, 42, 233`
## PEMSD7M：按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/100/est/A_no_pretrain_s100.log` | 0:20:30.272963 | 1.696565 | 2.654391 | 2.396259 | 1.616529 | 2.543739 | 2.392255 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/100/est/A_geo_only_s100.log` | 0:29:14.246937 | 1.995271 | 2.871973 | 2.734088 | 2.367550 | 3.185395 | 5.568120 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/100/est/A_topo_only_s100.log` | 0:28:22.286656 | 1.726586 | 2.656468 | 2.479632 | 1.620260 | 2.445664 | 3.567958 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/100/est/A_geo_topo_s100.log` | 0:29:39.079679 | 1.728855 | 2.598628 | 2.637412 | 1.491260 | 2.279000 | 2.829202 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/42/est/A_no_pretrain_s42.log` | 0:20:29.535754 | 1.927381 | 2.963386 | 2.530321 | 5.230775 | 6.029004 | 13.895087 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/42/est/A_geo_only_s42.log` | 0:31:00.201149 | 2.187472 | 3.177984 | 2.492194 | 2.426868 | 3.087553 | 2.751267 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/42/est/A_topo_only_s42.log` | 0:27:44.176326 | 1.778683 | 2.734244 | 2.231784 | 1.545387 | 2.309281 | 3.105759 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/42/est/A_geo_topo_s42.log` | 0:45:57.587407 | 1.911388 | 2.999535 | 2.576656 | 1.576600 | 2.239623 | 2.881016 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/999/est/A_no_pretrain_s999.log` | 0:20:37.166641 | 1.792772 | 2.956689 | 2.256297 | 1.679329 | 2.738594 | 2.237219 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/999/est/A_geo_only_s999.log` | 0:29:18.550376 | 2.163097 | 2.881000 | 3.164890 | 1.709516 | 2.862978 | 2.106488 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/999/est/A_topo_only_s999.log` | 0:28:14.795128 | 1.722472 | 2.744324 | 2.302864 | 1.630811 | 2.468159 | 3.177356 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/999/est/A_geo_topo_s999.log` | 0:44:27.767476 | 1.739349 | 2.728889 | 2.140556 | 1.484112 | 2.288110 | 2.734704 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/555/est/A_no_pretrain_s555.log` | 0:20:34.694752 | 1.598405 | 2.688021 | 2.115265 | 1.709816 | 2.563138 | 2.000888 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/555/est/A_geo_only_s555.log` | 0:29:17.922645 | 1.637156 | 2.747699 | 2.964882 | 2.027420 | 2.648879 | 4.526682 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/555/est/A_topo_only_s555.log` | 0:29:16.392177 | 1.516418 | 2.323921 | 2.133832 | 1.612436 | 2.396981 | 3.340420 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/555/est/A_geo_topo_s555.log` | 0:55:33.540136 | 1.635724 | 2.563904 | 3.163544 | 1.445743 | 2.235587 | 3.022420 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/250/est/A_no_pretrain_s250.log` | 0:20:50.441858 | 2.008259 | 3.250887 | 1.897598 | 1.641133 | 2.660945 | 1.827535 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/250/est/A_geo_only_s250.log` | 0:29:25.384119 | 2.116018 | 3.488730 | 2.139898 | 2.429504 | 3.475048 | 4.679962 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/250/est/A_topo_only_s250.log` | 0:28:25.720489 | 1.907298 | 2.977532 | 2.037121 | 1.534743 | 2.375583 | 2.940313 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/250/est/A_geo_topo_s250.log` | 0:29:21.542735 | 1.909832 | 2.953613 | 2.716633 | 1.457125 | 2.274814 | 2.591879 |

### 种子 `88`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/88/est/A_no_pretrain_s88.log` | 0:21:03.597388 | 1.753652 | 2.722938 | 1.788206 | 1.730453 | 2.514659 | 2.445992 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/88/est/A_geo_only_s88.log` | 0:29:12.497377 | 2.038464 | 2.924945 | 2.137492 | 3.247328 | 3.709446 | 5.262566 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/88/est/A_topo_only_s88.log` | 0:29:41.300895 | 1.796411 | 2.711002 | 1.870550 | 1.583908 | 2.386058 | 3.162729 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/88/est/A_geo_topo_s88.log` | 0:30:12.273353 | 1.792438 | 2.869123 | 2.107100 | 1.836407 | 2.711643 | 4.537059 |

### 种子 `66`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/66/est/A_no_pretrain_s66.log` | 0:21:04.760345 | 1.730701 | 2.715819 | 2.377335 | 1.681115 | 2.753453 | 1.927491 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/66/est/A_geo_only_s66.log` | 0:29:07.769270 | 1.912559 | 2.872017 | 3.505471 | 2.411489 | 3.078036 | 3.260113 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/66/est/A_topo_only_s66.log` | 0:29:14.634690 | 1.916023 | 2.811673 | 3.110901 | 1.616372 | 2.411083 | 3.340756 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/66/est/A_geo_topo_s66.log` | 0:30:15.332998 | 2.060020 | 3.020946 | 3.911398 | 1.641742 | 2.476982 | 3.800740 |

### 种子 `233`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/233/est/A_no_pretrain_s233.log` | 0:20:53.890635 | 1.562477 | 2.684301 | 2.500775 | 1.833875 | 2.527686 | 2.855369 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/233/est/A_geo_only_s233.log` | 0:29:17.833103 | 1.625610 | 2.680943 | 3.055566 | 1.917954 | 2.546613 | 3.485993 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/233/est/A_topo_only_s233.log` | 0:29:18.643371 | 1.498721 | 2.537212 | 2.266128 | 1.633649 | 2.427735 | 3.454690 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/233/est/A_geo_topo_s233.log` | 0:29:28.588068 | 1.591425 | 2.569498 | 2.539568 | 1.589468 | 2.473147 | 3.599622 |

### 种子 `38`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/38/est/A_no_pretrain_s38.log` | 0:20:48.485762 | 2.044316 | 3.069440 | 1.536654 | 3.332944 | 4.537230 | 6.123188 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/38/est/A_geo_only_s38.log` | 0:28:54.252303 | 1.917191 | 2.810101 | 3.172591 | 1.639271 | 2.694026 | 1.978643 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/38/est/A_topo_only_s38.log` | 0:29:32.682703 | 1.706939 | 2.591175 | 2.525985 | 1.456269 | 2.259918 | 2.497307 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/38/est/A_geo_topo_s38.log` | 0:30:08.492992 | 2.138739 | 2.987837 | 2.213464 | 1.521087 | 2.284374 | 2.507927 |

### 种子 `432`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/432/est/A_no_pretrain_s432.log` | 0:21:01.730773 | 2.192654 | 2.895383 | 2.200547 | 2.442407 | 3.139767 | 3.378340 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/432/est/A_geo_only_s432.log` | 0:29:32.963688 | 1.841317 | 2.935711 | 2.703618 | 1.711035 | 2.518983 | 3.641167 |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/432/est/A_topo_only_s432.log` | 0:28:38.539857 | 1.743527 | 2.620234 | 2.220978 | 1.604005 | 2.376938 | 3.028675 |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/PEMSD7M/432/est/A_geo_topo_s432.log` | 0:29:47.497832 | 1.746461 | 2.651363 | 1.900948 | 1.471757 | 2.251061 | 2.843309 |

---

## METRLA：n=5 汇总（均值 ± 样本标准差）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 5 | 2.5163 ± 0.2163 | 3.9239 ± 0.2719 | 1.3937 ± 0.2223 | 2.5992 ± 0.2886 | 3.6784 ± 0.2010 | 1.7338 ± 0.3071 |
| GEO | 5 | 2.4881 ± 0.3073 | 3.9886 ± 0.2847 | 1.5333 ± 0.2021 | 1.6344 ± 0.1281 | 2.9517 ± 0.1449 | 1.3418 ± 0.0588 |
| TOPO | 5 | 2.4182 ± 0.1588 | 3.9169 ± 0.2032 | 1.6064 ± 0.2276 | 1.6791 ± 0.2449 | 2.9355 ± 0.2305 | 1.3745 ± 0.2299 |
| GEO+TOPO | 5 | 2.3099 ± 0.1940 | 3.8819 ± 0.1079 | 1.5620 ± 0.3345 | 1.5202 ± 0.1834 | 2.7739 ± 0.1464 | 1.2254 ± 0.1202 |

## METRLA：按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/100/est/A_no_pretrain_s100.log` | 0:52:33.078253 | 2.785440 | 3.785666 | 1.309389 | 2.291689 | 3.462756 | 1.913071 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/100/est/A_geo_only_s100.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/100/est/A_topo_only_s100.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/100/est/A_geo_topo_s100.log` | — | — | — | — | — | — | — |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/42/est/A_no_pretrain_s42.log` | 1:09:13.343692 | 2.204587 | 3.728118 | 1.208343 | 2.953044 | 3.943464 | 2.034893 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/42/est/A_geo_only_s42.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/42/est/A_topo_only_s42.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/42/est/A_geo_topo_s42.log` | — | — | — | — | — | — | — |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/999/est/A_no_pretrain_s999.log` | 1:19:08.524885 | 2.636102 | 4.385916 | 1.636543 | 2.744405 | 3.720730 | 1.745048 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/999/est/A_geo_only_s999.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/999/est/A_topo_only_s999.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/999/est/A_geo_topo_s999.log` | — | — | — | — | — | — | — |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/555/est/A_no_pretrain_s555.log` | 1:20:49.498228 | 2.485818 | 3.951103 | 1.627173 | 2.311193 | 3.492990 | 1.229573 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/555/est/A_geo_only_s555.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/555/est/A_topo_only_s555.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/555/est/A_geo_topo_s555.log` | — | — | — | — | — | — | — |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/250/est/A_no_pretrain_s250.log` | 0:51:02.021416 | 2.469482 | 3.768451 | 1.187089 | 2.695698 | 3.771826 | 1.746175 |
| GEO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/250/est/A_geo_only_s250.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/250/est/A_topo_only_s250.log` | — | — | — | — | — | — | — |
| GEO+TOPO | `logs_topomoe/est_geotopo_3ds_seed5/METRLA/250/est/A_geo_topo_s250.log` | — | — | — | — | — | — | — |

---

