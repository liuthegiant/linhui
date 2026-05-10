# TopoMoE（METRLA）临时小表（已完成种子）

- **日志根目录**：`logs_topomoe/seed5_metrla_neg1`
- **种子**：`100, 42, 999, 555`
- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`

## A. Estimation（按配置汇总：tst_u / tst_a）

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 4 | 2.0862 ± 0.1900 | 3.6302 ± 0.1659 | 1.2616 ± 0.1736 | 1.4055 ± 0.0561 | 2.6889 ± 0.0626 | 1.1967 ± 0.1017 |
| GEO only | 4 | 2.3834 ± 0.1430 | 3.9237 ± 0.1096 | 1.5416 ± 0.3593 | 1.9634 ± 0.1110 | 3.1921 ± 0.1191 | 1.4195 ± 0.0734 |
| TOPO only | 4 | 2.3386 ± 0.2533 | 4.0397 ± 0.2654 | 1.6189 ± 0.3611 | 1.3358 ± 0.0430 | 2.6040 ± 0.0644 | 1.0624 ± 0.0447 |
| SCPT + GEO (`sparse_moe`) | 4 | 2.1612 ± 0.2126 | 3.6003 ± 0.1074 | 1.4335 ± 0.2296 | 1.5182 ± 0.1221 | 2.7769 ± 0.0583 | 1.2789 ± 0.1511 |
| SCPT + TOPO | 4 | 1.9909 ± 0.1435 | 3.5798 ± 0.1874 | 1.3158 ± 0.2257 | 1.3722 ± 0.1822 | 2.6462 ± 0.0865 | 1.1926 ± 0.1467 |
| GEO + TOPO | 4 | 2.2288 ± 0.1481 | 3.8438 ± 0.1076 | 1.5338 ± 0.2599 | 1.5513 ± 0.2306 | 2.8128 ± 0.1599 | 1.2620 ± 0.1565 |
| SCPT + GEO + TOPO | 4 | 2.0454 ± 0.1202 | 3.5406 ± 0.1418 | 1.3413 ± 0.2234 | 1.4479 ± 0.1402 | 2.6956 ± 0.0648 | 1.2066 ± 0.1133 |

## B. Forecasting（按配置汇总：test_u / test_a，all pred steps）

| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 4 | 107.6735 ± 4.0364 | 10.3752 ± 0.1945 | 4.5247 ± 0.2174 | 12.0620 ± 0.4418 | 95.0323 ± 2.3001 | 9.7479 ± 0.1176 | 4.1726 ± 0.0262 | 10.6533 ± 0.2937 |
| GEO only | 4 | 113.2536 ± 7.5920 | 10.6377 ± 0.3521 | 4.6927 ± 0.2961 | 12.2349 ± 0.8364 | 106.4869 ± 3.0908 | 10.3184 ± 0.1506 | 4.7220 ± 0.0751 | 12.7154 ± 0.4928 |
| TOPO only | 4 | 113.6596 ± 6.9776 | 10.6574 ± 0.3234 | 4.7734 ± 0.2346 | 12.2421 ± 0.7852 | 99.3098 ± 1.8293 | 9.9651 ± 0.0917 | 4.4944 ± 0.1128 | 11.2482 ± 0.2271 |
| SCPT + GEO (`sparse_moe`) | 4 | 108.9697 ± 6.0076 | 10.4358 ± 0.2898 | 4.5271 ± 0.2608 | 11.8607 ± 0.7626 | 95.1283 ± 2.2331 | 9.7529 ± 0.1143 | 4.1965 ± 0.0740 | 10.7882 ± 0.5620 |
| SCPT + TOPO | 4 | 108.1699 ± 8.3309 | 10.3947 ± 0.4006 | 4.6041 ± 0.3111 | 12.4283 ± 1.2412 | 97.0304 ± 3.3411 | 9.8493 ± 0.1701 | 4.2456 ± 0.0750 | 10.7193 ± 0.3483 |
| GEO + TOPO | 4 | 113.9860 ± 10.7926 | 10.6678 ± 0.4967 | 4.9231 ± 0.4338 | 13.1638 ± 1.9574 | 99.8070 ± 2.1541 | 9.9899 ± 0.1080 | 4.4804 ± 0.0620 | 11.5773 ± 0.2739 |
| SCPT + GEO + TOPO | 4 | 108.8355 ± 4.8027 | 10.4305 ± 0.2288 | 4.5840 ± 0.1697 | 12.1835 ± 0.7247 | 94.9167 ± 2.5586 | 9.7419 ± 0.1311 | 4.1933 ± 0.0474 | 10.5858 ± 0.0948 |

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

### 种子 `555`

**A. Estimation**

| 配置 | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 2.131318 | 3.629573 | 1.502648 | 1.408212 | 2.684754 | 1.283031 | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_only_s555.log` |
| GEO only | 2.527966 | 4.012728 | 2.068367 | 2.054124 | 3.331401 | 1.485862 | `logs_topomoe/seed5_metrla_neg1/555/est/A_geo_only_s555.log` |
| TOPO only | 2.608834 | 4.384667 | 2.157161 | 1.299661 | 2.526376 | 1.077626 | `logs_topomoe/seed5_metrla_neg1/555/est/A_topo_only_s555.log` |
| SCPT + GEO (`sparse_moe`) | 2.268402 | 3.564876 | 1.759060 | 1.642562 | 2.807273 | 1.454484 | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_geo_s555.log` |
| SCPT + TOPO | 2.195009 | 3.527775 | 1.646722 | 1.645525 | 2.764154 | 1.406026 | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_topo_s555.log` |
| GEO + TOPO | 2.317128 | 3.952527 | 1.874628 | 1.708635 | 2.904227 | 1.438378 | `logs_topomoe/seed5_metrla_neg1/555/est/A_geo_topo_s555.log` |
| SCPT + GEO + TOPO | 2.221814 | 3.560400 | 1.607850 | 1.651227 | 2.785523 | 1.372643 | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_geo_topo_s555.log` |

**B. Forecasting (all pred steps)**

| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 112.041740 | 10.584977 | 4.685093 | 12.560609 | 98.124397 | 9.905776 | 4.207133 | 10.828444 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_only_s555.log` |
| GEO only | 124.295883 | 11.148807 | 4.947243 | 12.811382 | 108.062523 | 10.395312 | 4.726436 | 13.107902 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_geo_only_s555.log` |
| TOPO only | 123.773239 | 11.125342 | 5.037110 | 12.975946 | 100.045837 | 10.002292 | 4.535789 | 11.524751 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_topo_only_s555.log` |
| SCPT + GEO (`sparse_moe`) | 114.158012 | 10.684475 | 4.691998 | 11.892176 | 97.920357 | 9.895472 | 4.273907 | 11.611149 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_geo_s555.log` |
| SCPT + TOPO | 118.084373 | 10.866663 | 4.850852 | 13.528492 | 98.086128 | 9.903844 | 4.209264 | 10.250029 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_topo_s555.log` |
| GEO + TOPO | 129.620010 | 11.385078 | 5.541512 | 15.861163 | 101.502670 | 10.074853 | 4.486201 | 11.691014 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_geo_topo_s555.log` |
| SCPT + GEO + TOPO | 115.533340 | 10.748644 | 4.749018 | 12.376519 | 92.274010 | 9.605936 | 4.137603 | 10.597768 | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_geo_topo_s555.log` |

