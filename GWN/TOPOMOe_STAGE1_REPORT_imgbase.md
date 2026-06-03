# TopoMoE（METRLA）Stage-1：五随机种子汇总（截图 BASE 参数）

- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`
- **日志根目录**：`logs_topomoe/stage1_seed5_imgbase_rerun`
- **种子**：`100, 42, 999, 555, 250`（**n=5**）
- **一键运行**：`./run_topomoe_stage1_5seed.sh`

---

## A2. Estimation：n=5 汇总（均值 ± 样本标准差）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 5 | 2.0648 ± 0.1197 | 3.6342 ± 0.1665 | 1.2965 ± 0.1646 | 1.4267 ± 0.1116 | 2.6943 ± 0.0515 | 1.1803 ± 0.1077 |
| GEO only | 5 | 2.4881 ± 0.3073 | 3.9886 ± 0.2847 | 1.5333 ± 0.2021 | 1.6344 ± 0.1281 | 2.9517 ± 0.1449 | 1.3418 ± 0.0588 |
| TOPO only | 5 | 2.4182 ± 0.1588 | 3.9169 ± 0.2032 | 1.6064 ± 0.2276 | 1.6791 ± 0.2449 | 2.9355 ± 0.2305 | 1.3745 ± 0.2299 |
| SCPT + GEO (`sparse_moe`) | 5 | 2.0188 ± 0.1588 | 3.6431 ± 0.0497 | 1.3077 ± 0.1355 | 1.3522 ± 0.1069 | 2.6275 ± 0.1118 | 1.1268 ± 0.1085 |
| SCPT + TOPO | 5 | 1.9280 ± 0.2105 | 3.5026 ± 0.1934 | 1.3435 ± 0.2389 | 1.4132 ± 0.1844 | 2.6703 ± 0.1214 | 1.1624 ± 0.1189 |
| GEO + TOPO | 5 | 2.3099 ± 0.1940 | 3.8819 ± 0.1079 | 1.5620 ± 0.3345 | 1.5202 ± 0.1834 | 2.7739 ± 0.1464 | 1.2254 ± 0.1202 |
| SCPT + GEO + TOPO | 5 | 2.0316 ± 0.1312 | 3.6727 ± 0.1081 | 1.3907 ± 0.2290 | 1.4349 ± 0.1403 | 2.6963 ± 0.0795 | 1.2257 ± 0.1184 |

## B2. Forecasting：n=5 汇总（均值 ± 样本标准差，all pred steps）

| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 5 | 110.1902 ± 5.1968 | 10.4948 ± 0.2475 | 4.6695 ± 0.2586 | 12.1135 ± 1.0523 | 95.9417 ± 1.4848 | 9.7947 ± 0.0761 | 4.2145 ± 0.0277 | 10.9859 ± 0.5333 |
| GEO only | 5 | 116.4667 ± 5.2352 | 10.7897 ± 0.2452 | 4.8769 ± 0.2249 | 12.8682 ± 0.9671 | 104.8434 ± 3.3950 | 10.2382 ± 0.1661 | 4.6732 ± 0.1381 | 11.6451 ± 0.3655 |
| TOPO only | 5 | 112.8754 ± 6.4119 | 10.6210 ± 0.2971 | 4.8217 ± 0.2360 | 12.7125 ± 1.1503 | 100.1516 ± 2.7806 | 10.0068 ± 0.1386 | 4.5513 ± 0.1222 | 11.5146 ± 0.3910 |
| SCPT + GEO (`sparse_moe`) | 5 | 108.0392 ± 7.3399 | 10.3895 ± 0.3494 | 4.6386 ± 0.2747 | 12.2292 ± 1.1113 | 94.9427 ± 0.9467 | 9.7438 ± 0.0485 | 4.1893 ± 0.0406 | 10.6996 ± 0.4840 |
| SCPT + TOPO | 5 | 108.4343 ± 7.5950 | 10.4080 ± 0.3687 | 4.6193 ± 0.2861 | 12.0851 ± 0.7689 | 93.1027 ± 2.3775 | 9.6484 ± 0.1223 | 4.1461 ± 0.0541 | 10.4809 ± 0.2251 |
| GEO + TOPO | 5 | 115.1388 ± 11.2727 | 10.7205 ± 0.5125 | 5.0060 ± 0.3647 | 13.4878 ± 1.7553 | 101.6818 ± 2.5869 | 10.0831 ± 0.1287 | 4.5904 ± 0.1398 | 11.8804 ± 0.4454 |
| SCPT + GEO + TOPO | 5 | 108.7526 ± 7.3697 | 10.4237 ± 0.3518 | 4.6814 ± 0.3339 | 12.6653 ± 1.2633 | 94.0779 ± 1.1321 | 9.6992 ± 0.0584 | 4.2210 ± 0.0608 | 10.6889 ± 0.2380 |

---

## A1. Estimation：按种子分项

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_scpt_only_s100.log` | 1:02:29.687293 | 1.879339 | 3.361980 | 1.098760 | 1.364759 | 2.713008 | 1.046793 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_geo_only_s100.log` | 1:01:59.900459 | 2.694285 | 4.009594 | 1.420558 | 1.561472 | 2.903703 | 1.373511 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_topo_only_s100.log` | 1:01:55.560421 | 2.422768 | 3.686857 | 1.343921 | 1.771198 | 3.060179 | 1.472046 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_scpt_geo_s100.log` | 1:02:49.772365 | 1.890881 | 3.583471 | 1.249310 | 1.371726 | 2.737126 | 1.263126 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_scpt_topo_s100.log` | 1:02:04.568949 | 1.735263 | 3.313950 | 1.140350 | 1.261743 | 2.650313 | 1.049611 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_geo_topo_s100.log` | 1:01:05.687095 | 2.110772 | 3.784442 | 1.287786 | 1.248296 | 2.563071 | 1.034843 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/100/est/A_scpt_geo_topo_s100.log` | 1:03:50.528998 | 1.851057 | 3.497648 | 1.077393 | 1.321815 | 2.719727 | 1.075378 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_scpt_only_s42.log` | 1:02:23.998829 | 2.014059 | 3.666759 | 1.233903 | 1.452373 | 2.683043 | 1.145164 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_geo_only_s42.log` | 1:01:58.344952 | 2.100890 | 3.648714 | 1.352473 | 1.657558 | 2.982353 | 1.296879 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_topo_only_s42.log` | 1:02:41.680411 | 2.215855 | 3.994672 | 1.644926 | 1.312776 | 2.605036 | 1.038933 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_scpt_geo_s42.log` | 1:03:37.157146 | 1.885175 | 3.713525 | 1.324159 | 1.333640 | 2.639726 | 1.052450 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_scpt_topo_s42.log` | 1:02:18.524202 | 1.751931 | 3.386665 | 1.156619 | 1.287787 | 2.561022 | 1.037685 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_geo_topo_s42.log` | 1:03:20.644672 | 2.094544 | 3.776382 | 1.260283 | 1.698622 | 2.908841 | 1.206518 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/42/est/A_scpt_geo_topo_s42.log` | 1:03:49.335207 | 2.019706 | 3.668801 | 1.365686 | 1.418919 | 2.683843 | 1.239274 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_scpt_only_s999.log` | 1:01:23.686573 | 2.175890 | 3.766190 | 1.239816 | 1.371359 | 2.690758 | 1.144440 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_geo_only_s999.log` | 1:01:10.660967 | 2.643829 | 4.240641 | 1.448731 | 1.453217 | 2.744311 | 1.261508 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_topo_only_s999.log` | 1:02:43.372110 | 2.354727 | 3.985075 | 1.447392 | 1.554045 | 2.782693 | 1.250076 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_scpt_geo_s999.log` | 1:01:31.142547 | 2.256392 | 3.621657 | 1.271890 | 1.525159 | 2.718254 | 1.225935 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_scpt_topo_s999.log` | 1:03:00.305161 | 1.998194 | 3.448481 | 1.226383 | 1.576139 | 2.820016 | 1.312918 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_geo_topo_s999.log` | 1:01:14.542853 | 2.378612 | 4.030790 | 1.422008 | 1.421530 | 2.680570 | 1.229714 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/999/est/A_scpt_geo_topo_s999.log` | 1:03:17.416843 | 1.985996 | 3.691160 | 1.309314 | 1.300087 | 2.596451 | 1.165073 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_scpt_only_s555.log` | 1:02:15.311854 | 2.117504 | 3.767195 | 1.531960 | 1.334533 | 2.621141 | 1.232208 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_geo_only_s555.log` | 1:01:28.412527 | 2.784252 | 4.288820 | 1.862577 | 1.761143 | 2.984965 | 1.391710 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_topo_only_s555.log` | 1:02:20.628099 | 2.653222 | 4.180161 | 1.936155 | 1.847684 | 3.119790 | 1.619789 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_scpt_geo_s555.log` | 1:03:11.202370 | 2.101107 | 3.669155 | 1.527386 | 1.268209 | 2.462116 | 1.038748 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_scpt_topo_s555.log` | 1:02:16.257408 | 1.904542 | 3.552583 | 1.527412 | 1.290927 | 2.550645 | 1.175519 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_geo_topo_s555.log` | 1:01:23.339169 | 2.487250 | 3.943175 | 1.972730 | 1.598102 | 2.853182 | 1.320558 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/555/est/A_scpt_geo_topo_s555.log` | 1:02:17.933901 | 2.204064 | 3.790929 | 1.690998 | 1.487098 | 2.667557 | 1.252962 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_scpt_only_s250.log` | 0:59:46.334004 | 2.137291 | 3.609007 | 1.378073 | 1.610499 | 2.763450 | 1.333136 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_geo_only_s250.log` | 1:01:16.112603 | 2.217093 | 3.755078 | 1.582077 | 1.738522 | 3.143042 | 1.385195 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_topo_only_s250.log` | 1:00:33.762088 | 2.444202 | 3.737550 | 1.659758 | 1.909800 | 3.109881 | 1.491451 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_scpt_geo_s250.log` | 1:01:00.487599 | 1.960607 | 3.627815 | 1.165661 | 1.262509 | 2.580349 | 1.053565 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_scpt_topo_s250.log` | 1:00:55.220703 | 2.249845 | 3.811082 | 1.666765 | 1.649561 | 2.769593 | 1.236095 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_geo_topo_s250.log` | 1:00:24.967030 | 2.478134 | 3.874478 | 1.867073 | 1.634278 | 2.863600 | 1.335203 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/250/est/A_scpt_geo_topo_s250.log` | 1:02:36.199728 | 2.097395 | 3.714893 | 1.509971 | 1.646585 | 2.813745 | 1.395762 |

## B1. Forecasting：按种子分项（all pred steps）

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_scpt_only_s100.log` | 0:59:27.137834 | 111.691902 | 10.568439 | 4.754460 | 12.070482 | 97.277435 | 9.862932 | 4.255769 | 10.432120 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_geo_only_s100.log` | 1:00:03.241541 | 117.727486 | 10.850230 | 5.089447 | 13.796781 | 108.903107 | 10.435665 | 4.650156 | 11.495946 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_topo_only_s100.log` | 1:00:03.864683 | 112.330376 | 10.598602 | 4.910201 | 12.907752 | 97.711494 | 9.884912 | 4.403728 | 11.489345 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_scpt_geo_s100.log` | 1:00:39.780920 | 108.642815 | 10.423186 | 4.776371 | 12.882316 | 94.407898 | 9.716372 | 4.182369 | 10.201007 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_scpt_topo_s100.log` | 1:00:42.981883 | 110.188210 | 10.497057 | 4.710130 | 12.361713 | 97.211990 | 9.859614 | 4.180154 | 10.452132 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_geo_topo_s100.log` | 0:59:52.167626 | 113.441261 | 10.650881 | 4.960896 | 12.851457 | 100.700981 | 10.034987 | 4.501920 | 11.871014 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/100/pred/B_scpt_geo_topo_s100.log` | 1:01:11.380993 | 106.938408 | 10.341103 | 4.758323 | 13.128504 | 94.449768 | 9.718527 | 4.174449 | 10.453788 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_scpt_only_s42.log` | 0:59:47.307556 | 107.502098 | 10.368321 | 4.599881 | 11.049572 | 96.348839 | 9.815744 | 4.198024 | 11.515485 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_geo_only_s42.log` | 1:00:30.545773 | 107.715576 | 10.378612 | 4.512597 | 11.307631 | 102.324615 | 10.115563 | 4.565209 | 11.088594 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_topo_only_s42.log` | 1:01:00.450511 | 109.961998 | 10.486277 | 4.603709 | 11.388886 | 97.848198 | 9.891825 | 4.434344 | 11.245914 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_scpt_geo_s42.log` | 1:00:16.112840 | 100.507652 | 10.025351 | 4.395660 | 10.913197 | 95.019852 | 9.747812 | 4.145293 | 10.621709 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_scpt_topo_s42.log` | 1:00:48.340204 | 110.038712 | 10.489934 | 4.603157 | 11.464193 | 92.444305 | 9.614797 | 4.094397 | 10.382975 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_geo_topo_s42.log` | 0:59:40.981674 | 106.917404 | 10.340087 | 4.623087 | 11.690900 | 102.472496 | 10.122869 | 4.638355 | 11.391661 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/42/pred/B_scpt_geo_topo_s42.log` | 1:00:39.192218 | 101.906769 | 10.094889 | 4.485556 | 11.238436 | 92.420776 | 9.613572 | 4.139119 | 10.480133 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_scpt_only_s999.log` | 0:59:46.823045 | 103.370705 | 10.167138 | 4.274595 | 11.132999 | 93.392181 | 9.663963 | 4.230306 | 10.436760 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_geo_only_s999.log` | 1:00:29.501161 | 119.493828 | 10.931323 | 4.826722 | 12.830298 | 106.390793 | 10.314591 | 4.911903 | 12.018317 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_topo_only_s999.log` | 1:00:13.390683 | 107.989906 | 10.391819 | 4.538498 | 11.872784 | 103.670021 | 10.181848 | 4.621885 | 11.294791 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_scpt_geo_s999.log` | 0:59:55.598710 | 103.403397 | 10.168746 | 4.292654 | 11.124945 | 94.509720 | 9.721611 | 4.231885 | 10.436296 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_scpt_topo_s999.log` | 1:00:35.049180 | 96.495255 | 9.823200 | 4.136682 | 11.088326 | 92.839600 | 9.635331 | 4.223435 | 10.334212 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_geo_topo_s999.log` | 1:00:35.387446 | 107.574631 | 10.371819 | 4.756147 | 12.777340 | 104.545288 | 10.224739 | 4.795798 | 11.928753 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/999/pred/B_scpt_geo_topo_s999.log` | 1:02:10.160860 | 102.209587 | 10.109876 | 4.234329 | 11.584435 | 93.580612 | 9.673707 | 4.271857 | 10.634215 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_scpt_only_s555.log` | 0:59:13.943899 | 117.354980 | 10.833051 | 4.742320 | 13.436273 | 96.492073 | 9.823038 | 4.190660 | 11.067304 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_geo_only_s555.log` | 0:59:37.531901 | 121.181526 | 11.008248 | 4.949153 | 12.873460 | 100.460495 | 10.022998 | 4.644823 | 11.748360 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_topo_only_s555.log` | 0:58:51.822973 | 124.011009 | 11.136023 | 5.060197 | 13.040689 | 98.956299 | 9.947678 | 4.631969 | 11.348290 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_scpt_geo_s555.log` | 1:00:21.545106 | 119.739670 | 10.942562 | 4.829507 | 13.094120 | 96.551872 | 9.826081 | 4.230725 | 10.758607 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_scpt_topo_s555.log` | 0:59:16.678666 | 117.506584 | 10.840045 | 4.786404 | 12.824853 | 91.244064 | 9.552176 | 4.123928 | 10.359409 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_geo_topo_s555.log` | 0:59:48.669330 | 134.552155 | 11.599662 | 5.558548 | 16.333270 | 97.755783 | 9.887153 | 4.429501 | 11.630135 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/555/pred/B_scpt_geo_topo_s555.log` | 1:00:14.241999 | 118.441895 | 10.883101 | 4.820635 | 13.026257 | 95.419807 | 9.768307 | 4.272337 | 10.953900 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_scpt_only_s250.log` | 0:58:58.693060 | 111.031479 | 10.537148 | 4.976282 | 12.878075 | 96.197983 | 9.808057 | 4.197986 | 11.477992 |
| GEO only | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_geo_only_s250.log` | 0:59:29.871712 | 116.215034 | 10.780308 | 5.006356 | 13.533026 | 106.137787 | 10.302320 | 4.593727 | 11.874071 |
| TOPO only | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_topo_only_s250.log` | 0:59:08.311450 | 110.083893 | 10.492087 | 4.995720 | 14.352512 | 102.571960 | 10.127782 | 4.664387 | 12.194730 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_scpt_geo_s250.log` | 0:59:51.652848 | 107.902603 | 10.387618 | 4.898939 | 13.131255 | 94.224304 | 9.706921 | 4.156235 | 11.480229 |
| SCPT + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_scpt_topo_s250.log` | 0:59:53.077809 | 107.942902 | 10.389557 | 4.860229 | 12.686570 | 91.773636 | 9.579856 | 4.108423 | 10.875908 |
| GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_geo_topo_s250.log` | 1:00:14.054907 | 113.208542 | 10.639951 | 5.131199 | 13.785960 | 102.934273 | 10.145653 | 4.586566 | 12.580232 |
| SCPT + GEO + TOPO | `logs_topomoe/stage1_seed5_imgbase_rerun/250/pred/B_scpt_geo_topo_s250.log` | 0:59:56.981153 | 114.266151 | 10.689535 | 5.108367 | 14.348833 | 94.518532 | 9.722064 | 4.247455 | 10.922375 |

