# PEMSBAY Virtual-Node SplitMask Estimation（10 seeds，无 SCPT）

- **任务**：Estimation 掩码预测
- **训练掩码**：只使用原始随机点掩码，不固定整节点
- **测试汇报**：`tst_u` 排除 fixed virtual nodes；`tst_v` 只在 fixed virtual nodes 上算；`tst_a` 为 all-node 随机点掩码
- **配置**：`无预训练 / TOPO only / GEO only / GEO+TOPO`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 PEMSBAY -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`
- **日志根目录**：`logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10`
- **种子**：`100, 42, 999, 555, 250, 88, 66, 233, 38, 432`（n=10）
- **运行脚本**：`./run_pems2ds_virtualnode_splitmask_7gpu.sh`
- **报告更新时间**：2026-05-19 01:39:55

## 进度

| 配置 | 种子 | 状态 | tst_u MAE | tst_v MAE | tst_a MAE | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 无预训练 | 100 | 完成 | 1.021851 | 1.633156 | 0.962703 | 1:52:37.803975 |
| 无预训练 | 42 | 完成 | 2.294333 | 2.743114 | 0.931165 | 1:52:35.220905 |
| 无预训练 | 999 | 完成 | 1.126789 | 1.721077 | 0.897300 | 1:50:52.045655 |
| 无预训练 | 555 | 完成 | 1.264504 | 1.762152 | 1.053921 | 1:50:47.401118 |
| 无预训练 | 250 | 完成 | 1.190561 | 1.283855 | 0.991939 | 1:49:33.919277 |
| 无预训练 | 88 | 完成 | 1.035294 | 1.452348 | 0.913731 | 1:49:42.970036 |
| 无预训练 | 66 | 完成 | 1.040302 | 1.220058 | 0.922944 | 1:49:24.382800 |
| 无预训练 | 233 | 完成 | 1.171708 | 1.490934 | 0.921683 | 1:49:45.837665 |
| 无预训练 | 38 | 完成 | 1.443225 | 2.143613 | 0.892407 | 1:49:21.165653 |
| 无预训练 | 432 | 完成 | 1.205930 | 1.167780 | 1.110715 | 1:49:31.129532 |
| TOPO | 100 | 完成 | 1.042612 | 1.782067 | 0.827848 | 2:16:25.269651 |
| TOPO | 42 | 完成 | 1.282859 | 1.295844 | 0.777775 | 2:15:54.844445 |
| TOPO | 999 | 完成 | 1.050566 | 1.661268 | 0.764641 | 2:11:12.088651 |
| TOPO | 555 | 完成 | 1.173174 | 0.924760 | 0.882438 | 2:10:05.920062 |
| TOPO | 250 | 完成 | 1.109937 | 1.642830 | 0.996212 | 2:03:06.803572 |
| TOPO | 88 | 完成 | 1.127530 | 1.509111 | 0.735976 | 2:03:41.498852 |
| TOPO | 66 | 完成 | 1.173000 | 1.239582 | 0.886588 | 2:02:17.995350 |
| TOPO | 233 | 完成 | 1.086972 | 1.603898 | 0.842451 | 2:03:59.650642 |
| TOPO | 38 | 完成 | 1.087374 | 1.393442 | 0.800752 | 2:02:30.570283 |
| TOPO | 432 | 完成 | 1.102700 | 1.333580 | 0.773741 | 2:03:15.352644 |
| GEO | 100 | 完成 | 1.060648 | 1.868707 | 0.752829 | 2:15:58.608169 |
| GEO | 42 | 完成 | 1.366241 | 2.000715 | 0.864337 | 2:16:25.540839 |
| GEO | 999 | 完成 | 1.305553 | 1.924332 | 0.902463 | 2:10:22.163817 |
| GEO | 555 | 完成 | 1.244850 | 1.613649 | 0.857361 | 2:03:35.934238 |
| GEO | 250 | 完成 | 1.227292 | 1.003238 | 0.777809 | 2:03:36.276966 |
| GEO | 88 | 完成 | 1.507636 | 1.589880 | 0.898370 | 2:03:52.520709 |
| GEO | 66 | 完成 | 1.142692 | 1.259824 | 0.769009 | 2:03:53.559897 |
| GEO | 233 | 完成 | 1.172692 | 1.797764 | 0.845510 | 2:03:30.055561 |
| GEO | 38 | 完成 | 1.226139 | 1.525977 | 0.778180 | 2:02:33.256035 |
| GEO | 432 | 完成 | 1.354629 | 1.567380 | 0.897943 | 2:03:01.152063 |
| GEO+TOPO | 100 | 完成 | 0.965992 | 1.844008 | 0.687635 | 2:16:15.457231 |
| GEO+TOPO | 42 | 完成 | 1.273941 | 1.617062 | 0.768348 | 2:04:33.456874 |
| GEO+TOPO | 999 | 完成 | 1.071736 | 1.637236 | 0.716291 | 2:12:03.937258 |
| GEO+TOPO | 555 | 完成 | 1.193511 | 1.063540 | 0.717139 | 2:04:11.530611 |
| GEO+TOPO | 250 | 完成 | 1.106154 | 1.461097 | 0.786522 | 2:04:20.805866 |
| GEO+TOPO | 88 | 完成 | 1.038908 | 1.451822 | 0.692791 | 2:04:17.762829 |
| GEO+TOPO | 66 | 完成 | 1.083930 | 1.158295 | 0.720021 | 2:03:41.246955 |
| GEO+TOPO | 233 | 完成 | 1.025655 | 1.387365 | 0.748531 | 2:04:16.201321 |
| GEO+TOPO | 38 | 完成 | 1.140666 | 1.455090 | 0.705587 | 2:03:34.166161 |
| GEO+TOPO | 432 | 完成 | 1.163457 | 1.220917 | 0.733761 | 2:03:13.420516 |

**进度**：完成 **40/40**，进行中 **0**，未开始 **0**

## `tst_u` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 1.2794 ± 0.3784 | 2.0328 ± 0.4487 | 3.9678 ± 2.3419 |
| TOPO | 10 | 1.1237 ± 0.0711 | 1.8971 ± 0.1163 | 3.2146 ± 0.6041 |
| GEO | 10 | 1.2608 ± 0.1279 | 2.0314 ± 0.0961 | 3.3939 ± 0.8793 |
| GEO+TOPO | 10 | 1.1064 ± 0.0898 | 1.8730 ± 0.1381 | 3.0629 ± 0.5428 |

## `tst_v` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 1.6618 ± 0.4797 | 2.4707 ± 0.4654 | 3.4656 ± 2.5400 |
| TOPO | 10 | 1.4386 ± 0.2536 | 2.2925 ± 0.3879 | 2.7093 ± 0.6148 |
| GEO | 10 | 1.6151 ± 0.3079 | 2.4355 ± 0.3732 | 3.6830 ± 1.3822 |
| GEO+TOPO | 10 | 1.4296 ± 0.2369 | 2.2484 ± 0.3589 | 2.5902 ± 0.6476 |

## `tst_a` 汇总（均值 ± 样本标准差）

| 配置 | n | MAE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | 10 | 0.9599 ± 0.0722 | 1.5477 ± 0.0984 | 3.1724 ± 0.3055 |
| TOPO | 10 | 0.8288 ± 0.0772 | 1.3623 ± 0.0751 | 2.7934 ± 0.3228 |
| GEO | 10 | 0.8344 ± 0.0593 | 1.4034 ± 0.0808 | 2.6738 ± 0.2595 |
| GEO+TOPO | 10 | 0.7277 ± 0.0320 | 1.2387 ± 0.0486 | 2.3436 ± 0.0911 |

## 按种子分项

### 种子 `100`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/100/est/A_no_pretrain_s100.log` | 1.021851 / 1.764225 / 2.172862 | 1.633156 / 2.693372 / 1.301340 | 0.962703 / 1.541317 / 3.708978 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/100/est/A_topo_only_s100.log` | 1.042612 / 1.744782 / 2.355318 | 1.782067 / 2.736454 / 2.317741 | 0.827848 / 1.377411 / 3.107176 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/100/est/A_geo_only_s100.log` | 1.060648 / 1.851520 / 2.206812 | 1.868707 / 2.849286 / 2.835295 | 0.752829 / 1.272882 / 2.462221 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/100/est/A_geo_topo_s100.log` | 0.965992 / 1.733325 / 2.297251 | 1.844008 / 2.723587 / 2.228362 | 0.687635 / 1.166898 / 2.227595 |

### 种子 `42`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/42/est/A_no_pretrain_s42.log` | 2.294333 / 3.292892 / 10.296499 | 2.743114 / 3.609977 / 9.212765 | 0.931165 / 1.519111 / 3.095654 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/42/est/A_topo_only_s42.log` | 1.282859 / 2.160235 / 3.920437 | 1.295844 / 2.268801 / 2.769555 | 0.777775 / 1.319219 / 2.468239 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/42/est/A_geo_only_s42.log` | 1.366241 / 2.111850 / 4.769567 | 2.000715 / 2.508937 / 6.060883 | 0.864337 / 1.462131 / 2.721938 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/42/est/A_geo_topo_s42.log` | 1.273941 / 2.200620 / 3.715260 | 1.617062 / 2.417342 / 4.109402 | 0.768348 / 1.289096 / 2.402833 |

### 种子 `999`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/999/est/A_no_pretrain_s999.log` | 1.126789 / 1.802678 / 2.674603 | 1.721077 / 2.678085 / 3.553633 | 0.897300 / 1.454412 / 3.336396 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/999/est/A_topo_only_s999.log` | 1.050566 / 1.874125 / 2.798343 | 1.661268 / 2.750658 / 2.611811 | 0.764641 / 1.316397 / 2.467770 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/999/est/A_geo_only_s999.log` | 1.305553 / 2.081938 / 3.101048 | 1.924332 / 3.043218 / 4.162902 | 0.902463 / 1.473811 / 3.093682 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/999/est/A_geo_topo_s999.log` | 1.071736 / 1.879308 / 3.202557 | 1.637236 / 2.785959 / 2.882458 | 0.716291 / 1.228075 / 2.164673 |

### 种子 `555`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/555/est/A_no_pretrain_s555.log` | 1.264504 / 1.966481 / 2.955805 | 1.762152 / 2.204200 / 4.602321 | 1.053921 / 1.648090 / 2.858503 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/555/est/A_topo_only_s555.log` | 1.173174 / 2.011459 / 4.074404 | 0.924760 / 1.595288 / 2.140716 | 0.882438 / 1.410418 / 3.124691 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/555/est/A_geo_only_s555.log` | 1.244850 / 2.042980 / 2.692912 | 1.613649 / 2.033627 / 3.094035 | 0.857361 / 1.387605 / 2.734410 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/555/est/A_geo_topo_s555.log` | 1.193511 / 1.949208 / 3.530653 | 1.063540 / 1.669235 / 2.176771 | 0.717139 / 1.211406 / 2.390738 |

### 种子 `250`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/250/est/A_no_pretrain_s250.log` | 1.190561 / 1.872640 / 4.053676 | 1.283855 / 1.888633 / 1.595772 | 0.991939 / 1.703467 / 3.018024 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/250/est/A_topo_only_s250.log` | 1.109937 / 1.885689 / 3.360388 | 1.642830 / 2.117080 / 3.690892 | 0.996212 / 1.501073 / 3.337476 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/250/est/A_geo_only_s250.log` | 1.227292 / 2.086663 / 4.646119 | 1.003238 / 1.925584 / 1.996705 | 0.777809 / 1.314760 / 2.450443 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/250/est/A_geo_topo_s250.log` | 1.106154 / 1.791563 / 3.193874 | 1.461097 / 1.948294 / 2.507653 | 0.786522 / 1.345995 / 2.455125 |

### 种子 `88`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/88/est/A_no_pretrain_s88.log` | 1.035294 / 1.872269 / 2.744021 | 1.452348 / 2.449785 / 1.794732 | 0.913731 / 1.450763 / 3.194070 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/88/est/A_topo_only_s88.log` | 1.127530 / 1.896108 / 2.657932 | 1.509111 / 2.503753 / 2.122974 | 0.735976 / 1.272427 / 2.421054 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/88/est/A_geo_only_s88.log` | 1.507636 / 2.144849 / 4.079240 | 1.589880 / 2.537045 / 4.144665 | 0.898370 / 1.508492 / 2.993176 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/88/est/A_geo_topo_s88.log` | 1.038908 / 1.951183 / 2.178705 | 1.451822 / 2.431346 / 2.146155 | 0.692791 / 1.208964 / 2.386902 |

### 种子 `66`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/66/est/A_no_pretrain_s66.log` | 1.040302 / 1.872619 / 2.717300 | 1.220058 / 2.218638 / 1.308175 | 0.922944 / 1.489952 / 3.162947 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/66/est/A_topo_only_s66.log` | 1.173000 / 1.907064 / 3.276036 | 1.239582 / 2.310364 / 2.181123 | 0.886588 / 1.462351 / 2.934275 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/66/est/A_geo_only_s66.log` | 1.142692 / 1.898903 / 2.389708 | 1.259824 / 2.247705 / 1.724040 | 0.769009 / 1.319844 / 2.557188 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/66/est/A_geo_topo_s66.log` | 1.083930 / 1.760663 / 2.592079 | 1.158295 / 2.234633 / 1.781222 | 0.720021 / 1.242243 / 2.331148 |

### 种子 `233`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/233/est/A_no_pretrain_s233.log` | 1.171708 / 1.945053 / 4.034846 | 1.490934 / 2.348095 / 2.014161 | 0.921683 / 1.516281 / 2.788241 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/233/est/A_topo_only_s233.log` | 1.086972 / 1.844520 / 3.803069 | 1.603898 / 2.666129 / 3.838627 | 0.842451 / 1.345930 / 2.830608 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/233/est/A_geo_only_s233.log` | 1.172692 / 2.032402 / 3.259193 | 1.797764 / 2.716799 / 5.502331 | 0.845510 / 1.466170 / 2.925145 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/233/est/A_geo_topo_s233.log` | 1.025655 / 1.759771 / 3.619101 | 1.387365 / 2.311048 / 2.850811 | 0.748531 / 1.223416 / 2.364348 |

### 种子 `38`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/38/est/A_no_pretrain_s38.log` | 1.443225 / 2.000332 / 4.408490 | 2.143613 / 2.401637 / 5.912741 | 0.892407 / 1.458978 / 2.944979 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/38/est/A_topo_only_s38.log` | 1.087374 / 1.792414 / 2.554037 | 1.393442 / 1.814768 / 2.867900 | 0.800752 / 1.331263 / 2.657150 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/38/est/A_geo_only_s38.log` | 1.226139 / 1.967853 / 3.310905 | 1.525977 / 2.034660 / 3.517746 | 0.778180 / 1.375364 / 2.412931 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/38/est/A_geo_topo_s38.log` | 1.140666 / 1.850280 / 2.981645 | 1.455090 / 1.910902 / 2.885571 | 0.705587 / 1.235952 / 2.291913 |

### 种子 `432`

| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |
| --- | --- | --- | --- | --- |
| 无预训练 | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/432/est/A_no_pretrain_s432.log` | 1.205930 / 1.938933 / 3.619778 | 1.167780 / 2.215037 / 3.360002 | 1.110715 / 1.695111 / 3.616345 |
| TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/432/est/A_topo_only_s432.log` | 1.102700 / 1.854384 / 3.346518 | 1.333580 / 2.161516 / 2.551928 | 0.773741 / 1.286983 / 2.585555 |
| GEO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/432/est/A_geo_only_s432.log` | 1.354629 / 2.095425 / 3.483793 | 1.567380 / 2.457751 / 3.791805 | 0.897943 / 1.452516 / 2.387147 |
| GEO+TOPO | `logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10/PEMSBAY/432/est/A_geo_topo_s432.log` | 1.163457 / 1.853774 / 3.317749 | 1.220917 / 2.051272 / 2.333886 | 0.733761 / 1.235072 / 2.421001 |

