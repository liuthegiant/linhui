# TopoMoE（METRLA）**五随机种子**实验汇总（METRLA `-1` / 100+100 epoch BASE）

- **BASE（估计与预测共用 argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`
- **日志根目录**：`logs_topomoe/seed5_metrla_neg1`
- **种子**：`100, 42, 999, 555, 250`（**n=5**）
- **一键扫种**：`./run_topomoe_5seed_sweep.sh`（每轮：`A` 7 并行 → `B` 7 并行，占用 GPU **0–6**）
- **生成/更新本报告**：`python3 aggregate_topomoe_5seed.py --out TOPOMOe_RUN_REPORT_METRLA_neg1.md`（默认日志目录为 `logs_topomoe/seed5_metrla_neg1`）

---

## A2. Estimation：**n=5 汇总（均值 ± 样本标准差）**

*种子集合*：`100, 42, 999, 555, 250`，与仓库内 METRLA 五种子文档一致。*标准差*：`statistics.stdev`（分母 **n−1=4**）。

| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 5 | 2.1159 ± 0.1775 | 3.6542 ± 0.1534 | 1.3062 ± 0.1804 | 1.4456 ± 0.1020 | 2.7026 ± 0.0623 | 1.2156 ± 0.0977 |
| GEO only | 5 | 2.3571 ± 0.1372 | 3.8743 ± 0.1458 | 1.4987 ± 0.3256 | 1.9437 ± 0.1058 | 3.2038 ± 0.1064 | 1.4362 ± 0.0737 |
| TOPO only | 5 | 2.3580 ± 0.2237 | 3.9790 ± 0.2670 | 1.6239 ± 0.3129 | 1.4499 ± 0.2577 | 2.7046 ± 0.2319 | 1.1483 ± 0.1959 |
| SCPT + GEO (`sparse_moe`) | 5 | 2.1631 ± 0.1842 | 3.6323 ± 0.1174 | 1.4467 ± 0.2010 | 1.5301 ± 0.1090 | 2.7756 ± 0.0506 | 1.2767 ± 0.1310 |
| SCPT + TOPO | 5 | 2.0362 ± 0.1604 | 3.6262 ± 0.1927 | 1.3912 ± 0.2581 | 1.4273 ± 0.2002 | 2.6678 ± 0.0891 | 1.1992 ± 0.1279 |
| GEO + TOPO | 5 | 2.2153 ± 0.1318 | 3.8196 ± 0.1077 | 1.5213 ± 0.2268 | 1.5780 ± 0.2084 | 2.8322 ± 0.1452 | 1.2629 ± 0.1355 |
| SCPT + GEO + TOPO | 5 | 2.0652 ± 0.1132 | 3.6002 ± 0.1812 | 1.3606 ± 0.1982 | 1.4409 ± 0.1225 | 2.6976 ± 0.0563 | 1.2021 ± 0.0986 |

## B2. Forecasting：**n=5 汇总（均值 ± 样本标准差）**

| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | 5 | 108.2141 ± 3.6987 | 10.4014 ± 0.1783 | 4.6067 ± 0.2629 | 12.3676 ± 0.7831 | 94.7035 ± 2.1233 | 9.7311 ± 0.1086 | 4.1647 ± 0.0286 | 10.6821 ± 0.2623 |
| GEO only | 5 | 113.8768 ± 6.7209 | 10.6677 ± 0.3122 | 4.7515 ± 0.2882 | 12.4844 ± 0.9143 | 106.3080 ± 2.7064 | 10.3099 ± 0.1318 | 4.6940 ± 0.0902 | 12.5420 ± 0.5766 |
| TOPO only | 5 | 113.3733 ± 6.0766 | 10.6447 ± 0.2815 | 4.8168 ± 0.2251 | 12.5913 ± 1.0355 | 99.5915 ± 1.7049 | 9.9793 ± 0.0855 | 4.5181 ± 0.1111 | 11.4259 ± 0.4434 |
| SCPT + GEO (`sparse_moe`) | 5 | 109.3112 ± 5.2585 | 10.4527 ± 0.2538 | 4.6249 ± 0.3143 | 12.1193 ± 0.8779 | 94.8185 ± 2.0542 | 9.7370 ± 0.1052 | 4.2019 ± 0.0653 | 10.8442 ± 0.5025 |
| SCPT + TOPO | 5 | 107.9547 ± 7.2308 | 10.3855 ± 0.3475 | 4.6538 ± 0.2914 | 12.5405 ± 1.1038 | 96.0436 ± 3.6388 | 9.7988 ± 0.1856 | 4.2256 ± 0.0789 | 10.7876 ± 0.3381 |
| GEO + TOPO | 5 | 113.8136 ± 9.3546 | 10.6614 ± 0.4304 | 4.9470 ± 0.3795 | 13.2345 ± 1.7025 | 100.3301 ± 2.2018 | 10.0160 ± 0.1102 | 4.4954 ± 0.0633 | 11.6970 ± 0.3577 |
| SCPT + GEO + TOPO | 5 | 109.7004 ± 4.5869 | 10.4720 ± 0.2187 | 4.6722 ± 0.2460 | 12.5167 ± 0.9741 | 95.0369 ± 2.2321 | 9.7482 ± 0.1144 | 4.2031 ± 0.0466 | 10.7627 ± 0.4039 |

---

## A1. Estimation：按种子分项（完整指标）

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_only_s100.log` | 0:58:53.943901 | 1.932006 | 3.430026 | 1.098865 | 1.478631 | 2.738800 | 1.284318 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/100/est/A_geo_only_s100.log` | 0:59:13.147611 | 2.410634 | 3.951651 | 1.432279 | 1.961524 | 3.250957 | 1.459421 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/100/est/A_topo_only_s100.log` | 0:59:24.672737 | 2.249653 | 3.927899 | 1.385228 | 1.392982 | 2.679261 | 1.112437 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_geo_s100.log` | 0:59:59.065468 | 2.010303 | 3.493291 | 1.313485 | 1.504984 | 2.812949 | 1.201303 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_topo_s100.log` | 0:59:32.853371 | 1.875103 | 3.469825 | 1.176009 | 1.280187 | 2.644163 | 1.071539 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/est/A_geo_topo_s100.log` | 1:00:03.720107 | 2.292253 | 3.756593 | 1.497934 | 1.789222 | 2.988776 | 1.337013 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/est/A_scpt_geo_topo_s100.log` | 0:59:43.638878 | 1.995996 | 3.334572 | 1.124768 | 1.409843 | 2.697641 | 1.137099 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_only_s42.log` | 0:58:55.034589 | 1.945083 | 3.624848 | 1.259540 | 1.343043 | 2.601944 | 1.129417 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/42/est/A_geo_only_s42.log` | 0:59:10.765271 | 2.185883 | 3.763955 | 1.259464 | 2.030384 | 3.098224 | 1.318916 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/42/est/A_topo_only_s42.log` | 0:58:20.301058 | 2.029777 | 3.760801 | 1.453646 | 1.344585 | 2.624640 | 1.006021 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_geo_s42.log` | 0:59:54.813916 | 1.958999 | 3.594786 | 1.239438 | 1.355613 | 2.689924 | 1.114026 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_topo_s42.log` | 0:58:35.144628 | 1.984222 | 3.857473 | 1.273171 | 1.277103 | 2.558292 | 1.151876 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/est/A_geo_topo_s42.log` | 0:59:23.286294 | 2.007185 | 3.746205 | 1.242357 | 1.348459 | 2.650316 | 1.085764 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/est/A_scpt_geo_topo_s42.log` | 1:00:31.332103 | 2.011569 | 3.641273 | 1.439888 | 1.401197 | 2.661613 | 1.185146 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_only_s999.log` | 0:59:12.349952 | 2.336459 | 3.836392 | 1.185399 | 1.391984 | 2.729973 | 1.090137 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/999/est/A_geo_only_s999.log` | 0:59:47.335912 | 2.409250 | 3.966601 | 1.406190 | 1.807645 | 3.087724 | 1.413806 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/999/est/A_topo_only_s999.log` | 0:59:31.543564 | 2.466033 | 4.085425 | 1.479439 | 1.305973 | 2.585526 | 1.053666 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_geo_s999.log` | 1:00:28.384090 | 2.407182 | 3.748202 | 1.421868 | 1.569557 | 2.797522 | 1.345754 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_topo_s999.log` | 1:00:52.266638 | 1.909206 | 3.464046 | 1.167495 | 1.286022 | 2.618272 | 1.140954 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/est/A_geo_topo_s999.log` | 1:01:36.731714 | 2.298728 | 3.919897 | 1.520217 | 1.359007 | 2.707875 | 1.186789 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/est/A_scpt_geo_topo_s999.log` | 1:00:19.784454 | 1.952305 | 3.626183 | 1.192687 | 1.329430 | 2.637686 | 1.131659 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_only_s555.log` | 0:59:24.197828 | 2.131318 | 3.629573 | 1.502648 | 1.408212 | 2.684754 | 1.283031 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/555/est/A_geo_only_s555.log` | 0:59:28.108390 | 2.527966 | 4.012728 | 2.068367 | 2.054124 | 3.331401 | 1.485862 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/555/est/A_topo_only_s555.log` | 0:59:58.362042 | 2.608834 | 4.384667 | 2.157161 | 1.299661 | 2.526376 | 1.077626 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_geo_s555.log` | 1:00:05.605586 | 2.268402 | 3.564876 | 1.759060 | 1.642562 | 2.807273 | 1.454484 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_topo_s555.log` | 1:00:06.621694 | 2.195009 | 3.527775 | 1.646722 | 1.645525 | 2.764154 | 1.406026 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/est/A_geo_topo_s555.log` | 0:59:23.657198 | 2.317128 | 3.952527 | 1.874628 | 1.708635 | 2.904227 | 1.438378 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/est/A_scpt_geo_topo_s555.log` | 1:01:36.948452 | 2.221814 | 3.560400 | 1.607850 | 1.651227 | 2.785523 | 1.372643 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_only_s250.log` | 0:59:52.835305 | 2.234773 | 3.749930 | 1.484670 | 1.606129 | 2.757468 | 1.291208 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/250/est/A_geo_only_s250.log` | 1:00:26.982107 | 2.251523 | 3.676451 | 1.327386 | 1.864827 | 3.250628 | 1.503095 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/250/est/A_topo_only_s250.log` | 1:00:13.570980 | 2.435915 | 3.736004 | 1.644197 | 1.906064 | 3.107218 | 1.491939 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_geo_s250.log` | 1:00:43.448952 | 2.170412 | 3.760584 | 1.499492 | 1.577895 | 2.770221 | 1.267966 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_topo_s250.log` | 1:00:40.449728 | 2.217701 | 3.812040 | 1.692603 | 1.647652 | 2.754241 | 1.225849 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/est/A_geo_topo_s250.log` | 1:01:15.151318 | 2.161276 | 3.723002 | 1.471259 | 1.684655 | 2.910040 | 1.266465 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/est/A_scpt_geo_topo_s250.log` | 1:01:23.391493 | 2.144496 | 3.838741 | 1.437618 | 1.412599 | 2.705687 | 1.184183 |

## B1. Forecasting：按种子分项（`all pred steps`）

### 种子 `100`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_only_s100.log` | 0:58:26.438343 | 110.046883 | 10.490323 | 4.730153 | 12.214536 | 95.439026 | 9.769290 | 4.149149 | 10.294565 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/100/pred/B_geo_only_s100.log` | 0:57:09.945416 | 111.956909 | 10.580969 | 4.949038 | 13.090260 | 108.983398 | 10.439511 | 4.815280 | 13.175189 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/100/pred/B_topo_only_s100.log` | 0:57:19.195849 | 112.717392 | 10.616845 | 4.905080 | 12.846957 | 97.825462 | 9.890676 | 4.377889 | 11.001922 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_geo_s100.log` | 0:58:32.813113 | 112.942841 | 10.627457 | 4.781718 | 12.509504 | 92.777695 | 9.632118 | 4.095767 | 10.358617 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_topo_s100.log` | 0:58:29.559927 | 110.939392 | 10.532777 | 4.850528 | 13.462748 | 100.667206 | 10.033305 | 4.327734 | 10.804801 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/pred/B_geo_topo_s100.log` | 0:58:21.336758 | 112.387390 | 10.601292 | 4.906709 | 13.163310 | 97.154015 | 9.856673 | 4.403944 | 11.222790 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/100/pred/B_scpt_geo_topo_s100.log` | 0:59:30.316869 | 104.143188 | 10.205057 | 4.709433 | 12.237172 | 95.789421 | 9.787207 | 4.201476 | 10.698550 |

### 种子 `42`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_only_s42.log` | 0:57:55.458359 | 103.454742 | 10.171270 | 4.400359 | 11.510909 | 93.390938 | 9.663898 | 4.155843 | 10.542129 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/42/pred/B_geo_only_s42.log` | 0:58:24.535157 | 107.427826 | 10.364739 | 4.468875 | 11.448698 | 102.037109 | 10.101342 | 4.714378 | 12.291467 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/42/pred/B_topo_only_s42.log` | 0:57:23.240848 | 108.429176 | 10.412933 | 4.583190 | 11.416111 | 97.811142 | 9.889952 | 4.431937 | 11.140684 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_geo_s42.log` | 0:57:58.080253 | 107.824295 | 10.383848 | 4.427284 | 12.259234 | 95.785561 | 9.787010 | 4.212114 | 10.656668 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_topo_s42.log` | 0:58:33.961928 | 105.129982 | 10.253291 | 4.511162 | 11.538234 | 96.701279 | 9.833681 | 4.285035 | 11.088460 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/pred/B_geo_topo_s42.log` | 0:58:41.292199 | 108.376770 | 10.410417 | 4.606364 | 11.275972 | 98.953812 | 9.947553 | 4.475900 | 11.867139 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/42/pred/B_scpt_geo_topo_s42.log` | 0:58:46.898159 | 107.440811 | 10.365366 | 4.413929 | 11.192719 | 98.074158 | 9.903240 | 4.251971 | 10.467114 |

### 种子 `999`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_only_s999.log` | 0:58:59.038310 | 105.150681 | 10.254300 | 4.283087 | 11.962118 | 93.174843 | 9.652712 | 4.178180 | 10.948211 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/999/pred/B_geo_only_s999.log` | 0:57:37.729780 | 109.333832 | 10.456282 | 4.405535 | 11.589254 | 106.864540 | 10.337531 | 4.631808 | 12.287233 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/999/pred/B_topo_only_s999.log` | 0:58:36.520037 | 109.718666 | 10.474668 | 4.568296 | 11.729393 | 101.556618 | 10.077530 | 4.632159 | 11.325423 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_geo_s999.log` | 0:58:26.684856 | 100.953621 | 10.047568 | 4.207577 | 10.781772 | 94.029587 | 9.696885 | 4.204167 | 10.526545 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_topo_s999.log` | 0:58:20.150711 | 98.526024 | 9.926027 | 4.204013 | 11.183663 | 92.666893 | 9.626365 | 4.160355 | 10.733921 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/pred/B_geo_topo_s999.log` | 0:58:41.702089 | 105.559723 | 10.274226 | 4.637646 | 12.354912 | 101.617691 | 10.080560 | 4.555400 | 11.528078 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/999/pred/B_scpt_geo_topo_s999.log` | 0:59:43.197189 | 108.224548 | 10.403103 | 4.463523 | 12.927733 | 93.529327 | 9.671056 | 4.181955 | 10.579924 |

### 种子 `555`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_only_s555.log` | 0:57:40.658896 | 112.041740 | 10.584977 | 4.685093 | 12.560609 | 98.124397 | 9.905776 | 4.207133 | 10.828444 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/555/pred/B_geo_only_s555.log` | 0:58:26.970984 | 124.295883 | 11.148807 | 4.947243 | 12.811382 | 108.062523 | 10.395312 | 4.726436 | 13.107902 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/555/pred/B_topo_only_s555.log` | 0:58:21.811020 | 123.773239 | 11.125342 | 5.037110 | 12.975946 | 100.045837 | 10.002292 | 4.535789 | 11.524751 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_geo_s555.log` | 0:58:37.991028 | 114.158012 | 10.684475 | 4.691998 | 11.892176 | 97.920357 | 9.895472 | 4.273907 | 11.611149 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_topo_s555.log` | 0:58:28.288009 | 118.084373 | 10.866663 | 4.850852 | 13.528492 | 98.086128 | 9.903844 | 4.209264 | 10.250029 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/pred/B_geo_topo_s555.log` | 0:58:42.089354 | 129.620010 | 11.385078 | 5.541512 | 15.861163 | 101.502670 | 10.074853 | 4.486201 | 11.691014 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/555/pred/B_scpt_geo_topo_s555.log` | 1:00:11.212055 | 115.533340 | 10.748644 | 4.749018 | 12.376519 | 92.274010 | 9.605936 | 4.137603 | 10.597768 |

### 种子 `250`

| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCPT only | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_only_s250.log` | 0:58:41.817220 | 110.376602 | 10.506027 | 4.934849 | 13.589768 | 93.388161 | 9.663755 | 4.133439 | 10.796963 |
| GEO only | `logs_topomoe/seed5_metrla_neg1/250/pred/B_geo_only_s250.log` | 0:58:22.624375 | 116.369316 | 10.787461 | 4.986564 | 13.482203 | 105.592613 | 10.275826 | 4.582204 | 11.848375 |
| TOPO only | `logs_topomoe/seed5_metrla_neg1/250/pred/B_topo_only_s250.log` | 0:58:32.376150 | 112.228088 | 10.593776 | 4.990128 | 13.988283 | 100.718567 | 10.035864 | 4.612566 | 12.136891 |
| SCPT + GEO (`sparse_moe`) | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_geo_s250.log` | 0:59:26.049193 | 110.677185 | 10.520323 | 5.015715 | 13.153951 | 93.579353 | 9.673642 | 4.223728 | 11.067897 |
| SCPT + TOPO | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_topo_s250.log` | 0:59:11.039999 | 107.093658 | 10.348607 | 4.852684 | 12.989311 | 92.096672 | 9.596701 | 4.145496 | 11.060908 |
| GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/pred/B_geo_topo_s250.log` | 0:58:49.630279 | 113.124260 | 10.635989 | 5.042579 | 13.516973 | 102.422249 | 10.120388 | 4.555456 | 12.176092 |
| SCPT + GEO + TOPO | `logs_topomoe/seed5_metrla_neg1/250/pred/B_scpt_geo_topo_s250.log` | 0:59:47.832254 | 113.160248 | 10.637681 | 5.025016 | 13.849209 | 95.517639 | 9.773313 | 4.242681 | 11.470216 |

