# SplitMask Estimation（三数据集，10 seeds）— `tst_v_full` 全图预测汇总

- `tst_u`、`tst_a`：来自原 splitmask 训练日志
- `tst_v_full`：eval-only 重跑（V 节点全 mask，但全图 forward，仅在 V 上计分）
- 生成时间：2026-05-20 08:21:24

## 报告文件

- `METRLA_virtualnode_splitmask_TSTVFULL_Estimation.md`
- `PEMSBAY_virtualnode_splitmask_TSTVFULL_Estimation.md`
- `PEMSD7M_virtualnode_splitmask_TSTVFULL_Estimation.md`

## 三数据集总览（MAE，均值 ± 样本标准差）

| 数据集 | split | 无预训练 | TOPO | GEO | GEO+TOPO |
| --- | --- | --- | --- | --- | --- |
| METRLA | `tst_u` | 3.0879 ± 0.4040 | 2.7664 ± 0.3174 | 2.9113 ± 0.2322 | 2.6045 ± 0.1717 |
| METRLA | `tst_v_full` | 2.4446 ± 0.6723 | 2.1051 ± 0.4880 | 2.2079 ± 0.3674 | 2.0220 ± 0.2965 |
| METRLA | `tst_a` | 2.5175 ± 0.3373 | 1.8285 ± 0.1519 | 1.9175 ± 0.1283 | 1.7121 ± 0.1360 |
| PEMSBAY | `tst_u` | 1.2794 ± 0.3784 | 1.1237 ± 0.0711 | 1.2608 ± 0.1279 | 1.1064 ± 0.0898 |
| PEMSBAY | `tst_v_full` | 1.0649 ± 0.1880 | 1.1096 ± 0.2306 | 1.0770 ± 0.3096 | 1.0209 ± 0.1879 |
| PEMSBAY | `tst_a` | 0.9599 ± 0.0722 | 0.8288 ± 0.0772 | 0.8344 ± 0.0593 | 0.7277 ± 0.0320 |
| PEMSD7M | `tst_u` | 2.0584 ± 0.2355 | 1.9380 ± 0.1411 | 2.1612 ± 0.2511 | 1.9671 ± 0.1663 |
| PEMSD7M | `tst_v_full` | 4.2742 ± 2.4491 | 2.1191 ± 0.2178 | 2.7770 ± 0.8765 | 2.1569 ± 0.3630 |
| PEMSD7M | `tst_a` | 4.0397 ± 2.2697 | 1.7530 ± 0.0516 | 2.5803 ± 0.7253 | 1.7926 ± 0.1672 |

