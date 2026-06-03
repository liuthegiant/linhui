# PEMSBAY + PEMSD7M Topo Estimation（5 seeds）

- **任务**：Estimation，仅 `无预训练 / SCPT / TOPO / SCPT+TOPO`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 <DATASET> -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`
- **无预训练设置**：将 `argv[1]`（`IS_PRETRN`）设为 `0`；其余配置与 Stage-1 保持一致
- **日志根目录**：`logs_topomoe/est_2ds_seed5_imgbase`
- **种子**：`100, 42, 999, 555, 250`（**n=5**）
- **运行脚本**：`./run_topomoe_est_2ds_5seed_8gpu.sh`

---

## PEMSBAY：n=5 汇总（均值 ± 样本标准差）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 3 | 1.4845 ± 0.6994 | 2.1053 ± 0.6651 | 5.4544 ± 3.8966 | 0.8854 ± 0.0306 | 1.4484 ± 0.0649 | 3.1066 ± 0.5145 |
| SCPT | 3 | 0.8827 ± 0.0481 | 1.4388 ± 0.0540 | 2.5519 ± 0.3069 | 0.6327 ± 0.0395 | 1.0420 ± 0.0398 | 2.2777 ± 0.1326 |
| TOPO | 2 | 1.1059 ± 0.0425 | 1.7519 ± 0.1297 | 3.4346 ± 1.0749 | 0.7950 ± 0.0821 | 1.2817 ± 0.1028 | 2.6059 ± 0.2014 |
| SCPT + TOPO | 3 | 0.8673 ± 0.0127 | 1.4238 ± 0.0409 | 2.6671 ± 0.2959 | 0.6176 ± 0.0344 | 1.0181 ± 0.0529 | 2.1754 ± 0.2099 |

## PEMSBAY：按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/100/est/A_no_pretrain_s100.log` | 2:20:52.954753 | 0.955209 | 1.596309 | 2.689637 | 0.908997 | 1.485961 | 3.444703 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/100/est/A_scpt_only_s100.log` | 3:47:11.004606 | 0.850226 | 1.377603 | 2.422572 | 0.677928 | 1.086578 | 2.423464 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/100/est/A_topo_only_s100.log` | 3:48:35.732577 | 1.075893 | 1.660195 | 2.674501 | 0.736948 | 1.208979 | 2.748329 |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/100/est/A_scpt_topo_s100.log` | 2:53:08.022540 | 0.875407 | 1.395956 | 2.332668 | 0.637944 | 1.032394 | 2.327756 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/42/est/A_no_pretrain_s42.log` | 2:14:16.455721 | 2.277333 | 2.857925 | 9.911042 | 0.896266 | 1.485830 | 2.514460 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/42/est/A_scpt_only_s42.log` | 3:44:19.198706 | 0.938028 | 1.458973 | 2.902295 | 0.604808 | 1.009851 | 2.164094 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/42/est/A_topo_only_s42.log` | 2:32:20.529078 | 1.135967 | 1.843679 | 4.194656 | 0.853105 | 1.354366 | 2.463439 |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/42/est/A_scpt_topo_s42.log` | 2:46:35.329000 | 0.852671 | 1.404756 | 2.894731 | 0.577974 | 0.959539 | 1.935968 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/999/est/A_no_pretrain_s999.log` | 3:22:46.135218 | 1.220820 | 1.861750 | 3.762640 | 0.850817 | 1.373427 | 3.360501 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/999/est/A_scpt_only_s999.log` | 3:56:53.286219 | 0.859923 | 1.479755 | 2.330894 | 0.615466 | 1.029653 | 2.245533 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/999/est/A_topo_only_s999.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/999/est/A_scpt_topo_s999.log` | 2:34:23.929863 | 0.873756 | 1.470761 | 2.773916 | 0.637001 | 1.062328 | 2.262578 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/555/est/A_no_pretrain_s555.log` | — | — | — | — | — | — | — |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/555/est/A_scpt_only_s555.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/555/est/A_topo_only_s555.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/555/est/A_scpt_topo_s555.log` | — | — | — | — | — | — | — |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/250/est/A_no_pretrain_s250.log` | — | — | — | — | — | — | — |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/250/est/A_scpt_only_s250.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/250/est/A_topo_only_s250.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSBAY/250/est/A_scpt_topo_s250.log` | — | — | — | — | — | — | — |

---

## PEMSD7M：n=5 汇总（均值 ± 样本标准差）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 4 | 1.7965 ± 0.2151 | 2.7967 ± 0.1481 | 2.3115 ± 0.1555 | 2.0938 ± 0.8551 | 3.0898 ± 0.9465 | 3.5246 ± 2.6503 |
| SCPT | 4 | 1.4498 ± 0.1247 | 2.3019 ± 0.1179 | 2.8685 ± 0.6779 | 1.3041 ± 0.0630 | 1.9895 ± 0.0945 | 2.6105 ± 0.5316 |
| TOPO | 3 | 1.7414 ± 0.0301 | 2.7104 ± 0.0462 | 2.3368 ± 0.1231 | 1.5979 ± 0.0480 | 2.4070 ± 0.0881 | 3.2805 ± 0.2463 |
| SCPT + TOPO | 3 | 1.5263 ± 0.1349 | 2.3544 ± 0.0393 | 3.0367 ± 0.9574 | 1.2668 ± 0.0342 | 1.9558 ± 0.0831 | 2.3872 ± 0.0788 |

## PEMSD7M：按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/100/est/A_no_pretrain_s100.log` | 0:43:28.488601 | 1.695402 | 2.656192 | 2.399441 | 1.616957 | 2.547309 | 2.393533 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/100/est/A_scpt_only_s100.log` | 0:53:36.115390 | 1.574168 | 2.350399 | 3.719639 | 1.327963 | 2.059153 | 2.781784 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/100/est/A_topo_only_s100.log` | 0:52:01.549491 | 1.725774 | 2.657396 | 2.474555 | 1.619548 | 2.444809 | 3.560640 |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/100/est/A_scpt_topo_s100.log` | 0:52:16.516854 | 1.673682 | 2.375345 | 4.040745 | 1.290502 | 2.051695 | 2.315265 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/42/est/A_no_pretrain_s42.log` | 0:30:25.516607 | 2.096634 | 2.882013 | 2.464914 | 3.375233 | 4.503699 | 7.491639 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/42/est/A_scpt_only_s42.log` | 0:51:49.638152 | 1.382453 | 2.255370 | 2.107228 | 1.215029 | 1.890596 | 2.063442 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/42/est/A_topo_only_s42.log` | 0:49:58.034871 | 1.776040 | 2.731365 | 2.237517 | 1.542915 | 2.306338 | 3.097971 |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/42/est/A_scpt_topo_s42.log` | 0:46:39.751137 | 1.408976 | 2.309011 | 2.133890 | 1.227684 | 1.905834 | 2.374796 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/999/est/A_no_pretrain_s999.log` | 0:21:01.478093 | 1.793674 | 2.960840 | 2.268513 | 1.679625 | 2.738929 | 2.238746 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/999/est/A_scpt_only_s999.log` | 0:35:30.696940 | 1.533192 | 2.437166 | 2.635822 | 1.362055 | 2.080692 | 3.272025 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/999/est/A_topo_only_s999.log` | 0:35:05.247327 | 1.722258 | 2.742430 | 2.298457 | 1.631303 | 2.469863 | 3.182951 |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/999/est/A_scpt_topo_s999.log` | 0:56:32.744588 | 1.496205 | 2.378773 | 2.935549 | 1.282349 | 1.909800 | 2.471479 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/555/est/A_no_pretrain_s555.log` | 0:25:12.310411 | 1.600361 | 2.687792 | 2.112973 | 1.703381 | 2.569161 | 1.974576 |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/555/est/A_scpt_only_s555.log` | 0:31:41.351856 | 1.309363 | 2.164565 | 3.011472 | 1.311265 | 1.927434 | 2.324737 |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/555/est/A_topo_only_s555.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/555/est/A_scpt_topo_s555.log` | — | — | — | — | — | — | — |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/250/est/A_no_pretrain_s250.log` | — | — | — | — | — | — | — |
| SCPT | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/250/est/A_scpt_only_s250.log` | — | — | — | — | — | — | — |
| TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/250/est/A_topo_only_s250.log` | — | — | — | — | — | — | — |
| SCPT + TOPO | `logs_topomoe/est_2ds_seed5_imgbase/PEMSD7M/250/est/A_scpt_topo_s250.log` | — | — | — | — | — | — | — |

---

