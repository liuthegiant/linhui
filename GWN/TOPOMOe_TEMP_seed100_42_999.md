# TopoMoE（METRLA）临时小表（已完成种子）

- **日志根目录**：`logs_topomoe/seed5_metrla_neg1`
- **种子**：`100, 42, 999`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`

## A. Estimation（按配置汇总：tst_u / tst_a）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 3 | 2.0712 ± 0.2298 | 3.6304 ± 0.2032 | 1.1813 ± 0.0804 | 1.4046 ± 0.0687 | 2.6902 ± 0.0766 | 1.1680 ± 0.1027 |
| GEO only | 3 | 2.3353 ± 0.1294 | 3.8941 ± 0.1129 | 1.3660 ± 0.0932 | 1.9332 ± 0.1140 | 3.1456 ± 0.0914 | 1.3974 ± 0.0717 |
| TOPO only | 3 | 2.2485 ± 0.2181 | 3.9247 ± 0.1623 | 1.4394 ± 0.0487 | 1.3478 ± 0.0436 | 2.6298 ± 0.0471 | 1.0574 ± 0.0533 |
| SCPT + GEO (`sparse_moe`) | 3 | 2.1255 ± 0.2453 | 3.6121 ± 0.1283 | 1.3249 ± 0.0918 | 1.4767 ± 0.1097 | 2.7668 ± 0.0670 | 1.2204 ± 0.1170 |
| SCPT + TOPO | 3 | 1.9228 ± 0.0558 | 3.5971 ± 0.2255 | 1.2056 ± 0.0587 | 1.2811 ± 0.0045 | 2.6069 ± 0.0440 | 1.1215 ± 0.0436 |
| GEO + TOPO | 3 | 2.1994 ± 0.1665 | 3.8076 ± 0.0974 | 1.4202 ± 0.1544 | 1.4989 ± 0.2515 | 2.7823 ± 0.1811 | 1.2032 ± 0.1264 |
| SCPT + GEO + TOPO | 3 | 1.9866 ± 0.0307 | 3.5340 ± 0.1729 | 1.2524 ± 0.1658 | 1.3802 ± 0.0441 | 2.6656 ± 0.0302 | 1.1513 ± 0.0294 |

## B. Forecasting（按配置汇总：test_u / test_a，all pred steps）

| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 3 | 106.2174 ± 3.4231 | 10.3053 ± 0.1655 | 4.4712 ± 0.2318 | 11.8959 ± 0.3565 | 94.0016 ± 1.2495 | 9.6953 ± 0.0643 | 4.1611 ± 0.0152 | 10.5950 ± 0.3300 |
| GEO only | 3 | 109.5729 ± 2.2740 | 10.4673 ± 0.1085 | 4.6078 ± 0.2972 | 12.0427 ± 0.9099 | 105.9617 ± 3.5601 | 10.2928 ± 0.1735 | 4.7205 ± 0.0919 | 12.5846 ± 0.5114 |
| TOPO only | 3 | 110.2884 ± 2.2001 | 10.5015 ± 0.1046 | 4.6855 ± 0.1903 | 11.9975 ± 0.7522 | 99.0644 ± 2.1583 | 9.9527 ± 0.1081 | 4.4807 ± 0.1340 | 11.1560 ± 0.1623 |
| SCPT + GEO (`sparse_moe`) | 3 | 107.2403 ± 6.0159 | 10.3530 ± 0.2912 | 4.4722 ± 0.2897 | 11.8502 ± 0.9337 | 94.1976 ± 1.5110 | 9.7053 ± 0.0778 | 4.1707 ± 0.0650 | 10.5139 ± 0.1494 |
| SCPT + TOPO | 3 | 104.8651 ± 6.2109 | 10.2374 ± 0.3037 | 4.5219 ± 0.3234 | 12.0615 ± 1.2264 | 96.6785 ± 4.0002 | 9.8311 ± 0.2035 | 4.2577 ± 0.0870 | 10.8757 ± 0.1876 |
| GEO + TOPO | 3 | 108.7746 ± 3.4312 | 10.4286 ± 0.1643 | 4.7169 ± 0.1651 | 12.2647 ± 0.9469 | 99.2418 ± 2.2457 | 9.9616 ± 0.1126 | 4.4784 ± 0.0758 | 11.5393 ± 0.3223 |
| SCPT + GEO + TOPO | 3 | 106.6028 ± 2.1659 | 10.3245 ± 0.1052 | 4.5290 ± 0.1582 | 12.1192 ± 0.8735 | 95.7976 ± 2.2724 | 9.7872 ± 0.1161 | 4.2118 ± 0.0361 | 10.5819 ± 0.1157 |

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

### 种子 `999`

**A. Estimation**

| 配置 | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 2.336459 | 3.836392 | 1.185399 | 1.391984 | 2.729973 | 1.090137 | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_only_s999.log` |
| GEO only | 2.409250 | 3.966601 | 1.406190 | 1.807645 | 3.087724 | 1.413806 | `logs_topomoe/seed5_metrla_neg1/999/est/A_geo_only_s999.log` |
| TOPO only | 2.466033 | 4.085425 | 1.479439 | 1.305973 | 2.585526 | 1.053666 | `logs_topomoe/seed5_metrla_neg1/999/est/A_topo_only_s999.log` |
| SCPT + GEO (`sparse_moe`) | 2.407182 | 3.748202 | 1.421868 | 1.569557 | 2.797522 | 1.345754 | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_geo_s999.log` |
| SCPT + TOPO | 1.909206 | 3.464046 | 1.167495 | 1.286022 | 2.618272 | 1.140954 | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_topo_s999.log` |
| GEO + TOPO | 2.298728 | 3.919897 | 1.520217 | 1.359007 | 2.707875 | 1.186789 | `logs_topomoe/seed5_metrla_neg1/999/est/A_geo_topo_s999.log` |
| SCPT + GEO + TOPO | 1.952305 | 3.626183 | 1.192687 | 1.329430 | 2.637686 | 1.131659 | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_geo_topo_s999.log` |

**B. Forecasting (all pred steps)**

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 105.150681 | 10.254300 | 4.283087 | 11.962118 | 93.174843 | 9.652712 | 4.178180 | 10.948211 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_only_s999.log` |
| GEO only | 109.333832 | 10.456282 | 4.405535 | 11.589254 | 106.864540 | 10.337531 | 4.631808 | 12.287233 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_geo_only_s999.log` |
| TOPO only | 109.718666 | 10.474668 | 4.568296 | 11.729393 | 101.556618 | 10.077530 | 4.632159 | 11.325423 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_topo_only_s999.log` |
| SCPT + GEO (`sparse_moe`) | 100.953621 | 10.047568 | 4.207577 | 10.781772 | 94.029587 | 9.696885 | 4.204167 | 10.526545 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_geo_s999.log` |
| SCPT + TOPO | 98.526024 | 9.926027 | 4.204013 | 11.183663 | 92.666893 | 9.626365 | 4.160355 | 10.733921 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_topo_s999.log` |
| GEO + TOPO | 105.559723 | 10.274226 | 4.637646 | 12.354912 | 101.617691 | 10.080560 | 4.555400 | 11.528078 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_geo_topo_s999.log` |
| SCPT + GEO + TOPO | 108.224548 | 10.403103 | 4.463523 | 12.927733 | 93.529327 | 9.671056 | 4.181955 | 10.579924 | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_geo_topo_s999.log` |

