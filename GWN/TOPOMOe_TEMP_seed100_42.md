# TopoMoE（METRLA）临时小表（已完成种子）

- **日志根目录**：`logs_topomoe/seed5_metrla_neg1`
- **种子**：`100, 42`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`

## A. Estimation（按配置汇总：tst_u / tst_a）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 2 | 1.9385 ± 0.0092 | 3.5274 ± 0.1378 | 1.1792 ± 0.1136 | 1.4108 ± 0.0959 | 2.6704 ± 0.0968 | 1.2069 ± 0.1095 |
| GEO only | 2 | 2.2983 ± 0.1589 | 3.8578 ± 0.1327 | 1.3459 ± 0.1222 | 1.9960 ± 0.0487 | 3.1746 ± 0.1080 | 1.3892 ± 0.0994 |
| TOPO only | 2 | 2.1397 ± 0.1555 | 3.8443 ± 0.1182 | 1.4194 ± 0.0484 | 1.3688 ± 0.0342 | 2.6520 ± 0.0386 | 1.0592 ± 0.0752 |
| SCPT + GEO (`sparse_moe`) | 2 | 1.9847 ± 0.0363 | 3.5440 ± 0.0718 | 1.2765 ± 0.0524 | 1.4303 ± 0.1056 | 2.7514 ± 0.0870 | 1.1577 ± 0.0617 |
| SCPT + TOPO | 2 | 1.9297 ± 0.0772 | 3.6636 ± 0.2741 | 1.2246 ± 0.0687 | 1.2786 ± 0.0022 | 2.6012 ± 0.0607 | 1.1117 ± 0.0568 |
| GEO + TOPO | 2 | 2.1497 ± 0.2016 | 3.7514 ± 0.0073 | 1.3701 ± 0.1807 | 1.5688 ± 0.3117 | 2.8195 ± 0.2393 | 1.2114 ± 0.1777 |
| SCPT + GEO + TOPO | 2 | 2.0038 ± 0.0110 | 3.4879 ± 0.2169 | 1.2823 ± 0.2228 | 1.4055 ± 0.0061 | 2.6796 ± 0.0255 | 1.1611 ± 0.0340 |

## B. Forecasting（按配置汇总：test_u / test_a，all pred steps）

| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 2 | 106.7508 ± 4.6613 | 10.3308 ± 0.2256 | 4.5653 ± 0.2332 | 11.8627 ± 0.4975 | 94.4150 ± 1.4482 | 9.7166 ± 0.0745 | 4.1525 ± 0.0047 | 10.4183 ± 0.1751 |
| GEO only | 2 | 109.6924 ± 3.2025 | 10.4729 ± 0.1529 | 4.7090 ± 0.3395 | 12.2695 ± 1.1608 | 105.5103 ± 4.9118 | 10.2704 ± 0.2391 | 4.7648 ± 0.0713 | 12.7333 ± 0.6249 |
| TOPO only | 2 | 110.5733 ± 3.0322 | 10.5149 ± 0.1442 | 4.7441 ± 0.2276 | 12.1315 ± 1.0118 | 97.8183 ± 0.0101 | 9.8903 ± 0.0005 | 4.4049 ± 0.0382 | 11.0713 ± 0.0981 |
| SCPT + GEO (`sparse_moe`) | 2 | 110.3836 ± 3.6194 | 10.5057 ± 0.1723 | 4.6045 ± 0.2506 | 12.3844 ± 0.1770 | 94.2816 ± 2.1269 | 9.7096 ± 0.1095 | 4.1539 ± 0.0823 | 10.5076 ± 0.2108 |
| SCPT + TOPO | 2 | 108.0347 ± 4.1079 | 10.3930 ± 0.1976 | 4.6808 ± 0.2400 | 12.5005 ± 1.3608 | 98.6842 ± 2.8043 | 9.9335 ± 0.1412 | 4.3064 ± 0.0302 | 10.9466 ± 0.2006 |
| GEO + TOPO | 2 | 110.3821 ± 2.8359 | 10.5059 ± 0.1350 | 4.7565 ± 0.2124 | 12.2196 ± 1.3345 | 98.0539 ± 1.2726 | 9.9021 ± 0.0643 | 4.4399 ± 0.0509 | 11.5450 ± 0.4556 |
| SCPT + GEO + TOPO | 2 | 105.7920 ± 2.3318 | 10.2852 ± 0.1134 | 4.5617 ± 0.2090 | 11.7149 ± 0.7385 | 96.9318 ± 1.6156 | 9.8452 ± 0.0820 | 4.2267 ± 0.0357 | 10.5828 ± 0.1636 |

## 明细日志（逐种子）

### 种子 `100`

**A. Estimation**

| 配置 | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 1.932006 | 3.430026 | 1.098865 | 1.478631 | 2.738800 | 1.284318 | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_only_s100.log` |
| GEO only | 2.410634 | 3.951651 | 1.432279 | 1.961524 | 3.250957 | 1.459421 | `logs_topomoe/seed5_metrla_neg1/100/est/A_geo_only_s100.log` |
| TOPO only | 2.249653 | 3.927899 | 1.385228 | 1.392982 | 2.679261 | 1.112437 | `logs_topomoe/seed5_metrla_neg1/100/est/A_topo_only_s100.log` |
| SCPT + GEO (`sparse_moe`) | 2.010303 | 3.493291 | 1.313485 | 1.504984 | 2.812949 | 1.201303 | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_geo_s100.log` |
| SCPT + TOPO | 1.875103 | 3.469825 | 1.176009 | 1.280187 | 2.644163 | 1.071539 | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_topo_s100.log` |
| GEO + TOPO | 2.292253 | 3.756593 | 1.497934 | 1.789222 | 2.988776 | 1.337013 | `logs_topomoe/seed5_metrla_neg1/100/est/A_geo_topo_s100.log` |
| SCPT + GEO + TOPO | 1.995996 | 3.334572 | 1.124768 | 1.409843 | 2.697641 | 1.137099 | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_geo_topo_s100.log` |

**B. Forecasting (all pred steps)**

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 110.046883 | 10.490323 | 4.730153 | 12.214536 | 95.439026 | 9.769290 | 4.149149 | 10.294565 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_only_s100.log` |
| GEO only | 111.956909 | 10.580969 | 4.949038 | 13.090260 | 108.983398 | 10.439511 | 4.815280 | 13.175189 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_geo_only_s100.log` |
| TOPO only | 112.717392 | 10.616845 | 4.905080 | 12.846957 | 97.825462 | 9.890676 | 4.377889 | 11.001922 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_topo_only_s100.log` |
| SCPT + GEO (`sparse_moe`) | 112.942841 | 10.627457 | 4.781718 | 12.509504 | 92.777695 | 9.632118 | 4.095767 | 10.358617 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_geo_s100.log` |
| SCPT + TOPO | 110.939392 | 10.532777 | 4.850528 | 13.462748 | 100.667206 | 10.033305 | 4.327734 | 10.804801 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_topo_s100.log` |
| GEO + TOPO | 112.387390 | 10.601292 | 4.906709 | 13.163310 | 97.154015 | 9.856673 | 4.403944 | 11.222790 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_geo_topo_s100.log` |
| SCPT + GEO + TOPO | 104.143188 | 10.205057 | 4.709433 | 12.237172 | 95.789421 | 9.787207 | 4.201476 | 10.698550 | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_geo_topo_s100.log` |

### 种子 `42`

**A. Estimation**

| 配置 | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 1.945083 | 3.624848 | 1.259540 | 1.343043 | 2.601944 | 1.129417 | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_only_s42.log` |
| GEO only | 2.185883 | 3.763955 | 1.259464 | 2.030384 | 3.098224 | 1.318916 | `logs_topomoe/seed5_metrla_neg1/42/est/A_geo_only_s42.log` |
| TOPO only | 2.029777 | 3.760801 | 1.453646 | 1.344585 | 2.624640 | 1.006021 | `logs_topomoe/seed5_metrla_neg1/42/est/A_topo_only_s42.log` |
| SCPT + GEO (`sparse_moe`) | 1.958999 | 3.594786 | 1.239438 | 1.355613 | 2.689924 | 1.114026 | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_geo_s42.log` |
| SCPT + TOPO | 1.984222 | 3.857473 | 1.273171 | 1.277103 | 2.558292 | 1.151876 | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_topo_s42.log` |
| GEO + TOPO | 2.007185 | 3.746205 | 1.242357 | 1.348459 | 2.650316 | 1.085764 | `logs_topomoe/seed5_metrla_neg1/42/est/A_geo_topo_s42.log` |
| SCPT + GEO + TOPO | 2.011569 | 3.641273 | 1.439888 | 1.401197 | 2.661613 | 1.185146 | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_geo_topo_s42.log` |

**B. Forecasting (all pred steps)**

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 103.454742 | 10.171270 | 4.400359 | 11.510909 | 93.390938 | 9.663898 | 4.155843 | 10.542129 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_only_s42.log` |
| GEO only | 107.427826 | 10.364739 | 4.468875 | 11.448698 | 102.037109 | 10.101342 | 4.714378 | 12.291467 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_geo_only_s42.log` |
| TOPO only | 108.429176 | 10.412933 | 4.583190 | 11.416111 | 97.811142 | 9.889952 | 4.431937 | 11.140684 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_topo_only_s42.log` |
| SCPT + GEO (`sparse_moe`) | 107.824295 | 10.383848 | 4.427284 | 12.259234 | 95.785561 | 9.787010 | 4.212114 | 10.656668 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_geo_s42.log` |
| SCPT + TOPO | 105.129982 | 10.253291 | 4.511162 | 11.538234 | 96.701279 | 9.833681 | 4.285035 | 11.088460 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_topo_s42.log` |
| GEO + TOPO | 108.376770 | 10.410417 | 4.606364 | 11.275972 | 98.953812 | 9.947553 | 4.475900 | 11.867139 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_geo_topo_s42.log` |
| SCPT + GEO + TOPO | 107.440811 | 10.365366 | 4.413929 | 11.192719 | 98.074158 | 9.903240 | 4.251971 | 10.467114 | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_geo_topo_s42.log` |

