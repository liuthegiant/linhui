# PEMSD7M Virtual-Node SplitMask Estimation（10 seeds，无 SCPT）

- **任务**：Estimation 掩码预测
- **训练掩码**：只使用原始随机点掩码，不固定整节点
- **测试汇报**：`tst_u` 排除 fixed virtual nodes；`tst_v` 只在 fixed virtual nodes 上算；`tst_a` 为 all-node 随机点掩码
- **配置**：`无预训练 / TOPO only / GEO only / GEO+TOPO`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`
- **日志根目录**：`logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10`
- **种子**：`100, 42, 999, 555, 250, 88, 66, 233, 38, 432`（n=10）
- **运行脚本**：`./run_pems2ds_virtualnode_splitmask_7gpu.sh`
- **报告更新时间**：2026-05-19 01:39:56

## 进度

| 配置 | 种子 | 状态 | tst_u MAE | tst_v MAE | tst_a MAE | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 100 | 完成 | 2.414301 | 2.914006 | 3.074920 | 0:20:27.721335 |
| 无预训练 | 42 | 完成 | 2.164783 | 2.088832 | 1.810430 | 0:20:30.298517 |
| 无预训练 | 999 | 完成 | 2.289066 | 2.588079 | 3.256384 | 0:20:43.024166 |
| 无预训练 | 555 | 完成 | 1.743870 | 2.720177 | 5.070911 | 0:20:31.593188 |
| 无预训练 | 250 | 完成 | 2.382046 | 3.179300 | 8.697079 | 0:20:34.546890 |
| 无预训练 | 88 | 完成 | 1.869459 | 2.591092 | 1.866160 | 0:20:19.882842 |
| 无预训练 | 66 | 完成 | 1.966522 | 2.190860 | 5.469573 | 0:20:30.328784 |
| 无预训练 | 233 | 完成 | 1.940678 | 2.472897 | 2.860519 | 0:20:31.125629 |
| 无预训练 | 38 | 完成 | 1.897276 | 3.042605 | 6.256475 | 0:20:44.470957 |
| 无预训练 | 432 | 完成 | 1.915723 | 1.979054 | 2.034662 | 0:20:43.528211 |
| TOPO | 100 | 完成 | 1.952934 | 2.138684 | 1.809385 | 0:28:42.058318 |
| TOPO | 42 | 完成 | 2.055393 | 2.356928 | 1.717722 | 0:28:53.968379 |
| TOPO | 999 | 完成 | 1.830063 | 3.100958 | 1.800246 | 0:28:42.385357 |
| TOPO | 555 | 完成 | 1.904672 | 2.282480 | 1.686457 | 0:28:52.138236 |
| TOPO | 250 | 完成 | 1.988107 | 3.205289 | 1.806174 | 0:28:51.464317 |
| TOPO | 88 | 完成 | 1.891853 | 2.893144 | 1.773036 | 0:29:00.108094 |
| TOPO | 66 | 完成 | 2.186276 | 2.442848 | 1.769206 | 0:29:08.483636 |
| TOPO | 233 | 完成 | 1.653530 | 2.925681 | 1.764822 | 0:28:44.244630 |
| TOPO | 38 | 完成 | 1.912252 | 3.098244 | 1.657213 | 0:28:52.859985 |
| TOPO | 432 | 完成 | 2.004917 | 2.110063 | 1.745333 | 0:29:11.876102 |
| GEO | 100 | 完成 | 2.298432 | 2.946230 | 2.222407 | 0:28:21.483161 |
| GEO | 42 | 完成 | 2.238721 | 1.938767 | 1.815441 | 0:28:45.360699 |
| GEO | 999 | 完成 | 1.873495 | 2.643595 | 1.809855 | 0:28:56.908141 |
| GEO | 555 | 完成 | 1.874717 | 2.260643 | 2.581519 | 0:28:37.986725 |
| GEO | 250 | 完成 | 2.320572 | 3.454715 | 3.065380 | 0:28:55.235389 |
| GEO | 88 | 完成 | 2.107305 | 2.868627 | 2.004519 | 0:29:01.744166 |
| GEO | 66 | 完成 | 2.723991 | 3.465728 | 2.462837 | 0:29:30.609298 |
| GEO | 233 | 完成 | 2.032553 | 3.009493 | 3.555797 | 0:28:59.760115 |
| GEO | 38 | 完成 | 2.073709 | 3.131489 | 2.347717 | 0:28:48.544636 |
| GEO | 432 | 完成 | 2.068564 | 1.959050 | 3.937230 | 0:29:11.360769 |
| GEO+TOPO | 100 | 完成 | 1.955608 | 2.173433 | 1.696074 | 0:28:34.673682 |
| GEO+TOPO | 42 | 完成 | 1.988122 | 1.972260 | 1.823928 | 0:28:52.459804 |
| GEO+TOPO | 999 | 完成 | 1.951329 | 2.516517 | 1.607609 | 0:29:21.502480 |
| GEO+TOPO | 555 | 完成 | 1.626975 | 2.226061 | 1.842555 | 0:28:35.763059 |
| GEO+TOPO | 250 | 完成 | 2.010184 | 3.310999 | 1.933539 | 0:29:31.111389 |
| GEO+TOPO | 88 | 完成 | 1.909059 | 2.758514 | 1.890559 | 0:29:56.490101 |
| GEO+TOPO | 66 | 完成 | 2.273869 | 2.378663 | 1.481984 | 0:29:09.003770 |
| GEO+TOPO | 233 | 完成 | 1.839889 | 2.657471 | 2.061282 | 0:29:05.638625 |
| GEO+TOPO | 38 | 完成 | 2.045228 | 3.137146 | 1.850537 | 0:29:31.310586 |
| GEO+TOPO | 432 | 完成 | 2.070737 | 1.980398 | 1.737697 | 0:28:55.029404 |

**进度**：完成 **40/40**，进行中 **0**，未开始 **0**

## `tst_u` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 2.0584 ± 0.2355 | 3.1185 ± 0.2507 | 1.7104 ± 0.4193 |
| TOPO | 10 | 1.9380 ± 0.1411 | 2.9213 ± 0.1477 | 2.4697 ± 0.7434 |
| GEO | 10 | 2.1612 ± 0.2511 | 3.2363 ± 0.2391 | 2.3392 ± 0.7363 |
| GEO+TOPO | 10 | 1.9671 ± 0.1663 | 3.0343 ± 0.1427 | 2.3890 ± 0.6698 |

## `tst_v` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 2.5767 ± 0.4041 | 3.8700 ± 0.5068 | 2.4815 ± 2.1746 |
| TOPO | 10 | 2.6554 ± 0.4300 | 3.7501 ± 0.5192 | 2.1122 ± 0.8371 |
| GEO | 10 | 2.7678 ± 0.5583 | 3.8094 ± 0.5158 | 2.7722 ± 1.6057 |
| GEO+TOPO | 10 | 2.5111 ± 0.4594 | 3.7486 ± 0.5132 | 1.9011 ± 0.7201 |

## `tst_a` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 4.0397 ± 2.2697 | 5.0919 ± 2.5120 | 7.7743 ± 5.8890 |
| TOPO | 10 | 1.7530 ± 0.0516 | 2.6062 ± 0.0599 | 3.5927 ± 0.3307 |
| GEO | 10 | 2.5803 ± 0.7253 | 3.5734 ± 0.8140 | 4.4317 ± 2.1465 |
| GEO+TOPO | 10 | 1.7926 ± 0.1672 | 2.6523 ± 0.1766 | 3.8469 ± 0.7573 |

## 按种子分项

### 种子 `100`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/100/est/A_no_pretrain_s100.log` | 2.414301 / 3.322499 / 1.618372 | 2.914006 / 4.246775 / 8.280399 | 3.074920 / 3.719619 / 4.977590 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/100/est/A_topo_only_s100.log` | 1.952934 / 2.915145 / 2.427523 | 2.138684 / 3.135656 / 2.075438 | 1.809385 / 2.609736 / 3.892597 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/100/est/A_geo_only_s100.log` | 2.298432 / 2.977656 / 3.628824 | 2.946230 / 3.406867 / 4.895245 | 2.222407 / 3.166165 / 2.908852 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/100/est/A_geo_topo_s100.log` | 1.955608 / 3.013868 / 1.816904 | 2.173433 / 3.202582 / 1.552687 | 1.696074 / 2.552147 / 3.620340 |

### 种子 `42`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/42/est/A_no_pretrain_s42.log` | 2.164783 / 3.229359 / 1.288220 | 2.088832 / 3.127497 / 1.285897 | 1.810430 / 2.656430 / 2.979688 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/42/est/A_topo_only_s42.log` | 2.055393 / 2.936600 / 2.488655 | 2.356928 / 3.236981 / 2.219899 | 1.717722 / 2.536097 / 3.482270 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/42/est/A_geo_only_s42.log` | 2.238721 / 3.584996 / 2.062098 | 1.938767 / 3.171183 / 1.770589 | 1.815441 / 2.748212 / 3.129170 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/42/est/A_geo_topo_s42.log` | 1.988122 / 3.085565 / 2.005449 | 1.972260 / 3.209380 / 2.912096 | 1.823928 / 2.606475 / 4.430456 |

### 种子 `999`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/999/est/A_no_pretrain_s999.log` | 2.289066 / 3.275573 / 1.282877 | 2.588079 / 3.710747 / 1.214904 | 3.256384 / 4.040416 / 4.393843 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/999/est/A_topo_only_s999.log` | 1.830063 / 2.887087 / 2.005640 | 3.100958 / 3.847271 / 3.982533 | 1.800246 / 2.663807 / 3.865727 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/999/est/A_geo_only_s999.log` | 1.873495 / 2.960149 / 1.833908 | 2.643595 / 3.742862 / 1.607734 | 1.809855 / 2.752557 / 2.868081 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/999/est/A_geo_topo_s999.log` | 1.951329 / 3.094902 / 2.893393 | 2.516517 / 3.765378 / 1.915918 | 1.607609 / 2.550630 / 2.680381 |

### 种子 `555`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/555/est/A_no_pretrain_s555.log` | 1.743870 / 2.868289 / 2.200375 | 2.720177 / 3.547274 / 1.508896 | 5.070911 / 7.662800 / 10.892762 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/555/est/A_topo_only_s555.log` | 1.904672 / 3.020646 / 3.632324 | 2.282480 / 3.651036 / 1.595530 | 1.686457 / 2.652805 / 3.772054 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/555/est/A_geo_only_s555.log` | 1.874717 / 3.106313 / 3.192597 | 2.260643 / 3.750149 / 2.325438 | 2.581519 / 3.300932 / 4.497469 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/555/est/A_geo_topo_s555.log` | 1.626975 / 2.766808 / 2.487106 | 2.226061 / 3.841960 / 3.345714 | 1.842555 / 2.677275 / 4.427988 |

### 种子 `250`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/250/est/A_no_pretrain_s250.log` | 2.382046 / 3.567623 / 1.261311 | 3.179300 / 4.537428 / 1.110086 | 8.697079 / 9.777714 / 19.726708 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/250/est/A_topo_only_s250.log` | 1.988107 / 3.164561 / 1.903710 | 3.205289 / 4.629664 / 1.150039 | 1.806174 / 2.707277 / 3.939208 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/250/est/A_geo_only_s250.log` | 2.320572 / 3.525984 / 1.906529 | 3.454715 / 4.450058 / 1.317320 | 3.065380 / 3.815611 / 6.952555 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/250/est/A_geo_topo_s250.log` | 2.010184 / 3.098371 / 1.572862 | 3.310999 / 4.413103 / 1.073770 | 1.933539 / 2.846764 / 4.942658 |

### 种子 `88`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/88/est/A_no_pretrain_s88.log` | 1.869459 / 2.942967 / 1.416880 | 2.591092 / 3.850885 / 1.399467 | 1.866160 / 2.689264 / 2.591621 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/88/est/A_topo_only_s88.log` | 1.891853 / 2.729800 / 1.698603 | 2.893144 / 3.743193 / 1.447669 | 1.773036 / 2.604659 / 3.449854 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/88/est/A_geo_only_s88.log` | 2.107305 / 3.102833 / 1.557386 | 2.868627 / 3.773251 / 2.038213 | 2.004519 / 3.116186 / 2.328522 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/88/est/A_geo_topo_s88.log` | 1.909059 / 3.071810 / 1.924236 | 2.758514 / 3.844593 / 1.184656 | 1.890559 / 2.752145 / 3.817420 |

### 种子 `66`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/66/est/A_no_pretrain_s66.log` | 1.966522 / 3.018021 / 2.083137 | 2.190860 / 3.750717 / 2.517224 | 5.469573 / 6.438442 / 11.451690 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/66/est/A_topo_only_s66.log` | 2.186276 / 3.032176 / 2.431336 | 2.442848 / 3.730488 / 2.722940 | 1.769206 / 2.584188 / 3.311867 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/66/est/A_geo_only_s66.log` | 2.723991 / 3.534777 / 3.297499 | 3.465728 / 4.032583 / 6.284294 | 2.462837 / 3.587754 / 4.199516 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/66/est/A_geo_topo_s66.log` | 2.273869 / 3.187001 / 3.589501 | 2.378663 / 3.702070 / 1.631721 | 1.481984 / 2.333123 / 2.656463 |

### 种子 `233`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/233/est/A_no_pretrain_s233.log` | 1.940678 / 2.768911 / 2.328053 | 2.472897 / 4.229573 / 2.494083 | 2.860519 / 3.823378 / 2.244052 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/233/est/A_topo_only_s233.log` | 1.653530 / 2.831749 / 2.051306 | 2.925681 / 4.106807 / 1.815563 | 1.764822 / 2.626144 / 3.951909 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/233/est/A_geo_only_s233.log` | 2.032553 / 3.075786 / 1.911474 | 3.009493 / 3.925990 / 2.338528 | 3.555797 / 4.839897 / 6.732624 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/233/est/A_geo_topo_s233.log` | 1.839889 / 2.788118 / 2.376013 | 2.657471 / 4.036384 / 1.515626 | 2.061282 / 2.966455 / 4.465894 |

### 种子 `38`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/38/est/A_no_pretrain_s38.log` | 1.897276 / 2.924414 / 1.544891 | 3.042605 / 4.519636 / 1.527346 | 6.256475 / 7.303234 / 13.826421 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/38/est/A_topo_only_s38.log` | 1.912252 / 2.677192 / 3.947692 | 3.098244 / 4.366450 / 1.467142 | 1.657213 / 2.509895 / 3.120831 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/38/est/A_geo_only_s38.log` | 2.073709 / 3.157769 / 1.884604 | 3.131489 / 4.732376 / 1.909385 | 2.347717 / 3.280361 / 2.418901 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/38/est/A_geo_topo_s38.log` | 2.045228 / 3.145921 / 3.274185 | 3.137146 / 4.521202 / 1.855834 | 1.850537 / 2.678824 / 3.843226 |

### 种子 `432`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/432/est/A_no_pretrain_s432.log` | 1.915723 / 3.267081 / 2.080201 | 1.979054 / 3.179230 / 3.476855 | 2.034662 / 2.807398 / 4.658561 |
| TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/432/est/A_topo_only_s432.log` | 2.004917 / 3.017793 / 2.110235 | 2.110063 / 3.053518 / 2.644969 | 1.745333 / 2.567370 / 3.140299 |
| GEO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/432/est/A_geo_only_s432.log` | 2.068564 / 3.337057 / 2.117130 | 1.959050 / 3.109017 / 3.235000 | 3.937230 / 5.125964 / 8.281769 |
| GEO+TOPO | `logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10/PEMSD7M/432/est/A_geo_topo_s432.log` | 2.070737 / 3.090545 / 1.950657 | 1.980398 / 2.948919 / 2.023113 | 1.737697 / 2.559614 / 3.584418 |

