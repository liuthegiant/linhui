# METRLA Virtual-Node SplitMask Estimation（10 seeds，无 SCPT）

- **任务**：Estimation 掩码预测
- **训练掩码**：只使用原始随机点掩码，不固定整节点
- **测试汇报**：`tst_u` 排除 fixed virtual nodes；`tst_v` 只在 fixed virtual nodes 上算；`tst_a` 为 all-node 随机点掩码
- **配置**：`无预训练 / TOPO only / GEO only / GEO+TOPO`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`
- **日志根目录**：`logs_topomoe/est_metrla_virtualnode_splitmask_seed10`
- **种子**：`100, 42, 999, 555, 250, 88, 66, 233, 38, 432`（n=10）
- **运行脚本**：`./run_metrla_virtualnode_splitmask_7gpu.sh`
- **报告更新时间**：2026-05-17 21:50:05

## 进度

| 配置 | 种子 | 状态 | tst_u MAE | tst_v MAE | tst_a MAE | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 100 | 完成 | 4.103152 | 5.951870 | 2.173661 | 0:52:10.687409 |
| 无预训练 | 42 | 完成 | 2.766722 | 4.776599 | 3.081300 | 0:52:31.461760 |
| 无预训练 | 999 | 完成 | 3.072155 | 3.483698 | 2.388206 | 0:52:16.007903 |
| 无预训练 | 555 | 完成 | 2.795447 | 5.075512 | 2.344488 | 0:52:54.489469 |
| 无预训练 | 250 | 完成 | 2.859340 | 4.604501 | 2.516746 | 0:51:55.951169 |
| 无预训练 | 88 | 完成 | 2.890114 | 4.769041 | 2.943437 | 0:51:49.153080 |
| 无预训练 | 66 | 完成 | 3.005379 | 4.446416 | 2.711567 | 0:52:00.468516 |
| 无预训练 | 233 | 完成 | 2.820651 | 3.920725 | 1.985639 | 0:51:07.420515 |
| 无预训练 | 38 | 完成 | 3.240429 | 4.873539 | 2.654799 | 0:51:03.417025 |
| 无预训练 | 432 | 完成 | 3.325987 | 4.603965 | 2.375166 | 0:51:27.582967 |
| TOPO | 100 | 完成 | 2.699305 | 5.064707 | 1.866389 | 1:01:34.116999 |
| TOPO | 42 | 完成 | 2.415879 | 4.567456 | 1.654031 | 1:01:43.211370 |
| TOPO | 999 | 完成 | 2.897249 | 4.278140 | 1.703775 | 1:01:29.667713 |
| TOPO | 555 | 完成 | 2.506763 | 5.017044 | 1.980448 | 1:02:30.341399 |
| TOPO | 250 | 完成 | 2.775084 | 5.227496 | 2.008651 | 1:02:48.856119 |
| TOPO | 88 | 完成 | 3.207340 | 5.494446 | 1.969946 | 1:00:52.442233 |
| TOPO | 66 | 完成 | 3.387046 | 3.839196 | 1.973362 | 1:02:38.824892 |
| TOPO | 233 | 完成 | 2.717136 | 4.039581 | 1.638092 | 1:01:01.404459 |
| TOPO | 38 | 完成 | 2.533571 | 3.929612 | 1.662787 | 1:01:24.030957 |
| TOPO | 432 | 完成 | 2.524922 | 4.560082 | 1.827117 | 1:00:36.031406 |
| GEO | 100 | 完成 | 3.007334 | 6.156266 | 1.838096 | 1:01:07.672136 |
| GEO | 42 | 完成 | 2.368390 | 4.643855 | 1.915470 | 1:01:05.719408 |
| GEO | 999 | 完成 | 3.188326 | 4.590887 | 1.887009 | 1:02:47.159474 |
| GEO | 555 | 完成 | 2.720336 | 5.452513 | 2.155333 | 1:02:32.272985 |
| GEO | 250 | 完成 | 2.899655 | 4.251844 | 1.791905 | 1:02:39.686481 |
| GEO | 88 | 完成 | 3.026243 | 5.060161 | 2.140767 | 1:00:47.858218 |
| GEO | 66 | 完成 | 2.932708 | 5.077244 | 1.882878 | 1:02:23.946278 |
| GEO | 233 | 完成 | 2.970202 | 4.729161 | 1.863563 | 1:01:57.525296 |
| GEO | 38 | 完成 | 3.132498 | 3.964544 | 1.902294 | 1:02:24.719882 |
| GEO | 432 | 完成 | 2.867370 | 4.309831 | 1.797460 | 1:00:54.295330 |
| GEO+TOPO | 100 | 完成 | 2.485648 | 5.392785 | 2.009483 | 1:02:28.639230 |
| GEO+TOPO | 42 | 完成 | 2.305914 | 4.387551 | 1.595629 | 1:04:18.497884 |
| GEO+TOPO | 999 | 完成 | 2.824425 | 4.552106 | 1.516181 | 1:01:27.902923 |
| GEO+TOPO | 555 | 完成 | 2.549269 | 5.388599 | 1.719741 | 1:02:00.559762 |
| GEO+TOPO | 250 | 完成 | 2.418535 | 3.624444 | 1.750090 | 1:02:44.701298 |
| GEO+TOPO | 88 | 完成 | 2.813664 | 4.439335 | 1.811637 | 1:01:35.016410 |
| GEO+TOPO | 66 | 完成 | 2.689490 | 4.365421 | 1.748060 | 1:01:35.605717 |
| GEO+TOPO | 233 | 完成 | 2.700743 | 4.421612 | 1.627546 | 1:02:16.224942 |
| GEO+TOPO | 38 | 完成 | 2.543251 | 4.407416 | 1.709736 | 1:01:05.831386 |
| GEO+TOPO | 432 | 完成 | 2.714395 | 4.262846 | 1.633055 | 1:00:46.466203 |

**进度**：完成 **40/40**，进行中 **0**，未开始 **0**

## `tst_u` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 3.0879 ± 0.4040 | 4.3605 ± 0.3376 | 1.5454 ± 0.3651 |
| TOPO | 10 | 2.7664 ± 0.3174 | 4.2227 ± 0.3120 | 1.8103 ± 0.3627 |
| GEO | 10 | 2.9113 ± 0.2322 | 4.3156 ± 0.2611 | 1.6385 ± 0.2587 |
| GEO+TOPO | 10 | 2.6045 ± 0.1717 | 4.1840 ± 0.2404 | 1.6598 ± 0.4119 |

## `tst_v` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 4.6506 ± 0.6578 | 6.3464 ± 0.3866 | 1.5735 ± 0.7559 |
| TOPO | 10 | 4.6018 ± 0.5804 | 6.4067 ± 0.3321 | 1.5141 ± 0.4762 |
| GEO | 10 | 4.8236 ± 0.6426 | 6.3821 ± 0.3415 | 1.6489 ± 0.6512 |
| GEO+TOPO | 10 | 4.5242 ± 0.5223 | 6.3421 ± 0.3151 | 1.3976 ± 0.4994 |

## `tst_a` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 2.5175 ± 0.3373 | 3.7553 ± 0.2848 | 1.7679 ± 0.2390 |
| TOPO | 10 | 1.8285 ± 0.1519 | 3.1161 ± 0.1538 | 1.4863 ± 0.1408 |
| GEO | 10 | 1.9175 ± 0.1283 | 3.2026 ± 0.0713 | 1.5707 ± 0.1308 |
| GEO+TOPO | 10 | 1.7121 ± 0.1360 | 3.0321 ± 0.1348 | 1.4309 ± 0.1494 |

## 按种子分项

### 种子 `100`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/100/est/A_no_pretrain_s100.log` | 4.103152 / 4.773695 / 2.085971 | 5.951870 / 6.798828 / 1.978930 | 2.173661 / 3.428423 / 1.570349 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/100/est/A_topo_only_s100.log` | 2.699305 / 3.869084 / 1.420728 | 5.064707 / 6.514545 / 1.376161 | 1.866389 / 3.223792 / 1.478408 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/100/est/A_geo_only_s100.log` | 3.007334 / 4.134588 / 1.254722 | 6.156266 / 7.063335 / 2.323111 | 1.838096 / 3.256279 / 1.694622 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/100/est/A_geo_topo_s100.log` | 2.485648 / 3.947661 / 1.277479 | 5.392785 / 6.675242 / 1.480571 | 2.009483 / 3.333271 / 1.719092 |

### 种子 `42`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/42/est/A_no_pretrain_s42.log` | 2.766722 / 3.870117 / 1.302651 | 4.776599 / 6.260833 / 1.626865 | 3.081300 / 4.099248 / 2.061451 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/42/est/A_topo_only_s42.log` | 2.415879 / 4.067616 / 1.511677 | 4.567456 / 6.636923 / 1.929490 | 1.654031 / 2.905641 / 1.231163 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/42/est/A_geo_only_s42.log` | 2.368390 / 3.954223 / 1.626511 | 4.643855 / 6.390233 / 1.196153 | 1.915470 / 3.177828 / 1.365809 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/42/est/A_geo_topo_s42.log` | 2.305914 / 3.741403 / 1.118938 | 4.387551 / 6.532525 / 1.136853 | 1.595629 / 2.876070 / 1.201192 |

### 种子 `999`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/999/est/A_no_pretrain_s999.log` | 3.072155 / 4.966575 / 1.150675 | 3.483698 / 6.276576 / 1.917606 | 2.388206 / 3.591447 / 1.839308 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/999/est/A_topo_only_s999.log` | 2.897249 / 4.526394 / 1.305314 | 4.278140 / 6.059885 / 1.143628 | 1.703775 / 3.048287 / 1.566748 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/999/est/A_geo_only_s999.log` | 3.188326 / 4.802573 / 1.534672 | 4.590887 / 6.263654 / 2.810576 | 1.887009 / 3.213707 / 1.709478 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/999/est/A_geo_topo_s999.log` | 2.824425 / 4.574837 / 1.279873 | 4.552106 / 6.123034 / 1.308547 | 1.516181 / 2.844401 / 1.293220 |

### 种子 `555`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/555/est/A_no_pretrain_s555.log` | 2.795447 / 4.197360 / 1.583266 | 5.075512 / 7.253925 / 1.234091 | 2.344488 / 3.535137 / 1.396954 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/555/est/A_topo_only_s555.log` | 2.506763 / 3.894182 / 1.991578 | 5.017044 / 7.142586 / 1.384759 | 1.980448 / 3.202032 / 1.635749 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/555/est/A_geo_only_s555.log` | 2.720336 / 4.031621 / 1.551650 | 5.452513 / 6.903528 / 1.648493 | 2.155333 / 3.331610 / 1.682858 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/555/est/A_geo_topo_s555.log` | 2.549269 / 3.990266 / 2.069402 | 5.388599 / 7.019394 / 1.642441 | 1.719741 / 2.968987 / 1.444094 |

### 种子 `250`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/250/est/A_no_pretrain_s250.log` | 2.859340 / 4.035947 / 1.232033 | 4.604501 / 5.995586 / 1.079354 | 2.516746 / 3.836491 / 1.625206 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/250/est/A_topo_only_s250.log` | 2.775084 / 4.138708 / 1.765953 | 5.227496 / 6.151375 / 2.012314 | 2.008651 / 3.278532 / 1.641593 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/250/est/A_geo_only_s250.log` | 2.899655 / 4.354636 / 1.428364 | 4.251844 / 6.042122 / 1.083578 | 1.791905 / 3.185511 / 1.448163 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/250/est/A_geo_topo_s250.log` | 2.418535 / 4.089921 / 1.581209 | 3.624444 / 5.993584 / 1.096695 | 1.750090 / 3.044835 / 1.418018 |

### 种子 `88`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/88/est/A_no_pretrain_s88.log` | 2.890114 / 4.099097 / 1.111091 | 4.769041 / 6.163168 / 1.042693 | 2.943437 / 4.127310 / 1.838976 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/88/est/A_topo_only_s88.log` | 3.207340 / 4.463072 / 1.765711 | 5.494446 / 6.511086 / 1.759213 | 1.969946 / 3.282070 / 1.592831 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/88/est/A_geo_only_s88.log` | 3.026243 / 4.414807 / 1.609825 | 5.060161 / 6.255059 / 1.308495 | 2.140767 / 3.271993 / 1.706168 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/88/est/A_geo_topo_s88.log` | 2.813664 / 4.319179 / 1.228743 | 4.439335 / 6.267978 / 1.257767 | 1.811637 / 3.067394 / 1.526083 |

### 种子 `66`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/66/est/A_no_pretrain_s66.log` | 3.005379 / 4.380933 / 1.656274 | 4.446416 / 6.234806 / 0.884553 | 2.711567 / 3.782051 / 2.136573 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/66/est/A_topo_only_s66.log` | 3.387046 / 4.819523 / 2.439244 | 3.839196 / 6.495195 / 0.790623 | 1.973362 / 3.228252 / 1.560315 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/66/est/A_geo_only_s66.log` | 2.932708 / 4.325675 / 1.746441 | 5.077244 / 6.214918 / 1.142185 | 1.882878 / 3.159653 / 1.421625 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/66/est/A_geo_topo_s66.log` | 2.689490 / 4.267455 / 2.156867 | 4.365421 / 6.254894 / 0.867242 | 1.748060 / 3.074893 / 1.382248 |

### 种子 `233`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/233/est/A_no_pretrain_s233.log` | 2.820651 / 4.365835 / 1.906198 | 3.920725 / 6.292165 / 1.061044 | 1.985639 / 3.285960 / 1.598491 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/233/est/A_topo_only_s233.log` | 2.717136 / 4.268956 / 2.289331 | 4.039581 / 6.247019 / 1.039073 | 1.638092 / 2.892334 / 1.380885 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/233/est/A_geo_only_s233.log` | 2.970202 / 4.381335 / 2.177733 | 4.729161 / 6.329069 / 1.320431 | 1.863563 / 3.130018 / 1.495416 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/233/est/A_geo_topo_s233.log` | 2.700743 / 4.285599 / 2.168054 | 4.421612 / 6.210195 / 1.331095 | 1.627546 / 3.070183 / 1.546366 |

### 种子 `38`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/38/est/A_no_pretrain_s38.log` | 3.240429 / 4.575069 / 2.033837 | 4.873539 / 6.042757 / 3.435793 | 2.654799 / 3.930890 / 1.969404 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/38/est/A_topo_only_s38.log` | 2.533571 / 4.274942 / 1.920746 | 3.929612 / 6.276086 / 2.304951 | 1.662787 / 2.950708 / 1.307226 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/38/est/A_geo_only_s38.log` | 3.132498 / 4.611672 / 1.918882 | 3.964544 / 6.345938 / 2.515087 | 1.902294 / 3.211351 / 1.553995 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/38/est/A_geo_topo_s38.log` | 2.543251 / 4.334024 / 1.855738 | 4.407416 / 6.294444 / 2.683909 | 1.709736 / 2.973916 / 1.482780 |

### 种子 `432`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/432/est/A_no_pretrain_s432.log` | 3.325987 / 4.340143 / 1.392274 | 4.603965 / 6.145415 / 1.474322 | 2.375166 / 3.935915 / 1.642696 |
| TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/432/est/A_topo_only_s432.log` | 2.524922 / 3.904978 / 1.693071 | 4.560082 / 6.032766 / 1.400891 | 1.827117 / 3.149778 / 1.467818 |
| GEO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/432/est/A_geo_only_s432.log` | 2.867370 / 4.144868 / 1.535740 | 4.309831 / 6.013421 / 1.141308 | 1.797460 / 3.087854 / 1.628710 |
| GEO+TOPO | `logs_topomoe/est_metrla_virtualnode_splitmask_seed10/METRLA/432/est/A_geo_topo_s432.log` | 2.714395 / 4.289811 / 1.861411 | 4.262846 / 6.049611 / 1.171373 | 1.633055 / 3.066946 / 1.295765 |

