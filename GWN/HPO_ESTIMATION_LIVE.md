# Estimation HPO 实时榜单（topo_only / scpt_topo）

- **更新时间（UTC）**：2026-05-16 09:46:04 UTC
- **说明**：每完成一个子任务自动刷新；主榜按 trial 的 tst_u MAE 均值排序
- **配置**：`topo_only, scpt_topo`
- **数据集顺序**：`PEMSBAY → PEMSD7M → METRLA`（METRLA 最后）
- **种子（n=2）**：`100, 42`
- **GPU**：`0, 1, 3, 4, 6, 7`（6 卡并行，留 2 卡）
- **主优化指标**：`tst_u` Masked MAE（越小越好）
- **状态文件**：`logs_hpo_est/hpo_state.json`

---

## 总进度

| 项目 | 值 |
| --- | --- |
| 试验数（trials） | 24 |
| 子任务完成 | 13 / 288 |
| 运行中 | 6 |
| 失败 | 8 |

## Trial 总榜（按 tst_u MAE 均值，含 topo_only + scpt_topo）

| 排名 | trial_id | 进度 | mean tst_u MAE | GATE_HIDDEN | TOPO_LAP_K | MOE_TAU | MOE_LB | MOE_SM | MOE_DELTA | MOE_CTX | PRE_LEARN |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 8/12 | 1.3192 | 64 | 32 | 1.500 | 0.00100 | 0.00500 | 0.0001 | 1 | 0.00050 |
| 2 | 1 | 5/12 | 1.4617 | 32 | 16 | 1.500 | 0.00500 | 0.00500 | 0.0 | 1 | 0.00100 |

## 分配置榜单 — TOPO (`topo_only`)

| 排名 | trial_id | 进度 | mean tst_u MAE | 参数摘要 |
| ---: | ---: | --- | --- | --- |
| 1 | 1 | 5/12 | 1.4263 | gh=32 lap=16 tau=1.5 lb=0.005 |
| 2 | 0 | 8/12 | 1.4660 | gh=64 lap=32 tau=1.5 lb=0.001 |

## 分配置榜单 — SCPT+TOPO (`scpt_topo`)

| 排名 | trial_id | 进度 | mean tst_u MAE | 参数摘要 |
| ---: | ---: | --- | --- | --- |
| 1 | 0 | 8/12 | 1.1724 | gh=64 lap=32 tau=1.5 lb=0.001 |
| 2 | 1 | 5/12 | 1.6036 | gh=32 lap=16 tau=1.5 lb=0.005 |

## 数据集 `PEMSBAY` — 已完成子任务（按 tst_u MAE 升序）

| trial_id | 配置 | seed | tst_u MAE | tst_u RMSE | tst_u MAPE | 时长 |
| ---: | --- | ---: | --- | --- | --- | --- |
| 0 | SCPT+TOPO | 100 | 0.8639 | 1.3602 | 2.3201 | 3:25:39.583658 |
| 0 | SCPT+TOPO | 42 | 0.8695 | 1.4188 | 3.0213 | 2:33:01.455472 |
| 0 | TOPO | 100 | 1.0430 | 1.6256 | 2.8316 | 2:30:51.974956 |
| 1 | TOPO | 100 | 1.0815 | 1.6841 | 3.0531 | 2:16:35.143264 |
| 0 | TOPO | 42 | 1.1053 | 1.7106 | 4.1526 | 2:36:29.924946 |
| 1 | TOPO | 42 | 1.1244 | 1.8374 | 4.1086 | 2:16:54.995455 |

## 数据集 `PEMSD7M` — 已完成子任务（按 tst_u MAE 升序）

| trial_id | 配置 | seed | tst_u MAE | tst_u RMSE | tst_u MAPE | 时长 |
| ---: | --- | ---: | --- | --- | --- | --- |
| 0 | SCPT+TOPO | 42 | 1.4529 | 2.3868 | 2.0968 | 0:30:23.177902 |
| 0 | SCPT+TOPO | 100 | 1.5035 | 2.3168 | 3.4186 | 0:30:26.537656 |
| 1 | SCPT+TOPO | 100 | 1.6036 | 2.3320 | 3.8691 | 0:29:04.597125 |
| 1 | TOPO | 100 | 1.7303 | 2.6854 | 2.4104 | 0:29:04.596305 |
| 1 | TOPO | 42 | 1.7688 | 2.7657 | 2.2143 | 0:29:42.096809 |
| 0 | TOPO | 42 | 1.7825 | 2.7230 | 2.3820 | 0:37:44.925862 |
| 0 | TOPO | 100 | 1.9334 | 2.7331 | 3.0175 | 0:36:21.497653 |

## 数据集 `METRLA` — 已完成子任务（按 tst_u MAE 升序）

| trial_id | 配置 | seed | tst_u MAE | tst_u RMSE | tst_u MAPE | 时长 |
| ---: | --- | ---: | --- | --- | --- | --- |
| — | — | — | — | — | — | — |

## 最近完成的子任务

| 完成时间（UTC） | trial_id | 配置 | 数据集 | seed | tst_u MAE | GPU | 日志 |
| --- | ---: | --- | --- | ---: | --- | ---: | --- |
| 2026-05-16 09:46:04 | 1 | SCPT+TOPO | PEMSD7M | 100 | 1.6036 | 7 | `logs_hpo_est/trial_0001/scpt_topo/PEMSD7M/s100.log` |
| 2026-05-16 09:39:43 | 0 | SCPT+TOPO | PEMSBAY | 100 | 0.8639 | 6 | `logs_hpo_est/trial_0000/scpt_topo/PEMSBAY/s100.log` |
| 2026-05-16 09:39:14 | 1 | TOPO | PEMSBAY | 42 | 1.1244 | 4 | `logs_hpo_est/trial_0001/topo_only/PEMSBAY/s42.log` |
| 2026-05-16 09:37:55 | 1 | TOPO | PEMSBAY | 100 | 1.0815 | 3 | `logs_hpo_est/trial_0001/topo_only/PEMSBAY/s100.log` |
| 2026-05-16 09:16:54 | 1 | TOPO | PEMSD7M | 42 | 1.7688 | 7 | `logs_hpo_est/trial_0001/topo_only/PEMSD7M/s42.log` |
| 2026-05-16 09:14:03 | 1 | TOPO | PEMSD7M | 100 | 1.7303 | 0 | `logs_hpo_est/trial_0001/topo_only/PEMSD7M/s100.log` |
| 2026-05-16 08:50:28 | 0 | TOPO | PEMSBAY | 42 | 1.1053 | 1 | `logs_hpo_est/trial_0000/topo_only/PEMSBAY/s42.log` |
| 2026-05-16 08:47:05 | 0 | SCPT+TOPO | PEMSBAY | 42 | 0.8695 | 7 | `logs_hpo_est/trial_0000/scpt_topo/PEMSBAY/s42.log` |
| 2026-05-16 08:44:50 | 0 | TOPO | PEMSBAY | 100 | 1.0430 | 0 | `logs_hpo_est/trial_0000/topo_only/PEMSBAY/s100.log` |
| 2026-05-16 07:22:12 | 0 | SCPT+TOPO | PEMSD7M | 42 | 1.4529 | 4 | `logs_hpo_est/trial_0000/scpt_topo/PEMSD7M/s42.log` |
| 2026-05-16 07:21:00 | 0 | SCPT+TOPO | PEMSD7M | 100 | 1.5035 | 3 | `logs_hpo_est/trial_0000/scpt_topo/PEMSD7M/s100.log` |
| 2026-05-16 06:51:42 | 0 | TOPO | PEMSD7M | 42 | 1.7825 | 4 | `logs_hpo_est/trial_0000/topo_only/PEMSD7M/s42.log` |
| 2026-05-16 06:50:20 | 0 | TOPO | PEMSD7M | 100 | 1.9334 | 3 | `logs_hpo_est/trial_0000/topo_only/PEMSD7M/s100.log` |

## 运行中 / 失败

| 状态 | trial_id | 配置 | 数据集 | seed | GPU | 日志 | 备注 |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| failed | 0 | TOPO | METRLA | 100 | 6 | `logs_hpo_est/trial_0000/topo_only/METRLA/s100.log` | exit code 1 |
| failed | 0 | TOPO | METRLA | 42 | 7 | `logs_hpo_est/trial_0000/topo_only/METRLA/s42.log` | exit code 1 |
| failed | 0 | SCPT+TOPO | METRLA | 100 | 3 | `logs_hpo_est/trial_0000/scpt_topo/METRLA/s100.log` | exit code 1 |
| failed | 0 | SCPT+TOPO | METRLA | 42 | 3 | `logs_hpo_est/trial_0000/scpt_topo/METRLA/s42.log` | exit code 1 |
| failed | 1 | TOPO | METRLA | 100 | 1 | `logs_hpo_est/trial_0001/topo_only/METRLA/s100.log` | exit code 1 |
| failed | 1 | TOPO | METRLA | 42 | 1 | `logs_hpo_est/trial_0001/topo_only/METRLA/s42.log` | exit code 1 |
| running | 1 | SCPT+TOPO | PEMSBAY | 100 | 1 | `logs_hpo_est/trial_0001/scpt_topo/PEMSBAY/s100.log` | — |
| running | 1 | SCPT+TOPO | PEMSBAY | 42 | 0 | `logs_hpo_est/trial_0001/scpt_topo/PEMSBAY/s42.log` | — |
| running | 1 | SCPT+TOPO | PEMSD7M | 42 | 3 | `logs_hpo_est/trial_0001/scpt_topo/PEMSD7M/s42.log` | — |
| failed | 1 | SCPT+TOPO | METRLA | 100 | 4 | `logs_hpo_est/trial_0001/scpt_topo/METRLA/s100.log` | exit code 1 |
| failed | 1 | SCPT+TOPO | METRLA | 42 | 4 | `logs_hpo_est/trial_0001/scpt_topo/METRLA/s42.log` | exit code 1 |
| running | 2 | TOPO | PEMSBAY | 100 | 4 | `logs_hpo_est/trial_0002/topo_only/PEMSBAY/s100.log` | — |
| running | 2 | TOPO | PEMSBAY | 42 | 6 | `logs_hpo_est/trial_0002/topo_only/PEMSBAY/s42.log` | — |
| running | 2 | TOPO | PEMSD7M | 100 | 7 | `logs_hpo_est/trial_0002/topo_only/PEMSD7M/s100.log` | — |
| queued | 2 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0002/topo_only/PEMSD7M/s42.log` | — |
| queued | 2 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0002/topo_only/METRLA/s100.log` | — |
| queued | 2 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0002/topo_only/METRLA/s42.log` | — |
| queued | 2 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0002/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 2 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0002/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 2 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0002/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 2 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0002/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 2 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0002/scpt_topo/METRLA/s100.log` | — |
| queued | 2 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0002/scpt_topo/METRLA/s42.log` | — |
| queued | 3 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0003/topo_only/PEMSBAY/s100.log` | — |
| queued | 3 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0003/topo_only/PEMSBAY/s42.log` | — |
| queued | 3 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0003/topo_only/PEMSD7M/s100.log` | — |
| queued | 3 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0003/topo_only/PEMSD7M/s42.log` | — |
| queued | 3 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0003/topo_only/METRLA/s100.log` | — |
| queued | 3 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0003/topo_only/METRLA/s42.log` | — |
| queued | 3 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0003/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 3 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0003/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 3 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0003/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 3 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0003/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 3 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0003/scpt_topo/METRLA/s100.log` | — |
| queued | 3 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0003/scpt_topo/METRLA/s42.log` | — |
| queued | 4 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0004/topo_only/PEMSBAY/s100.log` | — |
| queued | 4 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0004/topo_only/PEMSBAY/s42.log` | — |
| queued | 4 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0004/topo_only/PEMSD7M/s100.log` | — |
| queued | 4 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0004/topo_only/PEMSD7M/s42.log` | — |
| queued | 4 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0004/topo_only/METRLA/s100.log` | — |
| queued | 4 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0004/topo_only/METRLA/s42.log` | — |
| queued | 4 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0004/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 4 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0004/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 4 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0004/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 4 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0004/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 4 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0004/scpt_topo/METRLA/s100.log` | — |
| queued | 4 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0004/scpt_topo/METRLA/s42.log` | — |
| queued | 5 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0005/topo_only/PEMSBAY/s100.log` | — |
| queued | 5 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0005/topo_only/PEMSBAY/s42.log` | — |
| queued | 5 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0005/topo_only/PEMSD7M/s100.log` | — |
| queued | 5 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0005/topo_only/PEMSD7M/s42.log` | — |
| queued | 5 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0005/topo_only/METRLA/s100.log` | — |
| queued | 5 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0005/topo_only/METRLA/s42.log` | — |
| queued | 5 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0005/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 5 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0005/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 5 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0005/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 5 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0005/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 5 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0005/scpt_topo/METRLA/s100.log` | — |
| queued | 5 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0005/scpt_topo/METRLA/s42.log` | — |
| queued | 6 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0006/topo_only/PEMSBAY/s100.log` | — |
| queued | 6 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0006/topo_only/PEMSBAY/s42.log` | — |
| queued | 6 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0006/topo_only/PEMSD7M/s100.log` | — |
| queued | 6 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0006/topo_only/PEMSD7M/s42.log` | — |
| queued | 6 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0006/topo_only/METRLA/s100.log` | — |
| queued | 6 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0006/topo_only/METRLA/s42.log` | — |
| queued | 6 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0006/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 6 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0006/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 6 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0006/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 6 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0006/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 6 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0006/scpt_topo/METRLA/s100.log` | — |
| queued | 6 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0006/scpt_topo/METRLA/s42.log` | — |
| queued | 7 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0007/topo_only/PEMSBAY/s100.log` | — |
| queued | 7 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0007/topo_only/PEMSBAY/s42.log` | — |
| queued | 7 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0007/topo_only/PEMSD7M/s100.log` | — |
| queued | 7 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0007/topo_only/PEMSD7M/s42.log` | — |
| queued | 7 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0007/topo_only/METRLA/s100.log` | — |
| queued | 7 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0007/topo_only/METRLA/s42.log` | — |
| queued | 7 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0007/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 7 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0007/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 7 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0007/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 7 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0007/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 7 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0007/scpt_topo/METRLA/s100.log` | — |
| queued | 7 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0007/scpt_topo/METRLA/s42.log` | — |
| queued | 8 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0008/topo_only/PEMSBAY/s100.log` | — |
| queued | 8 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0008/topo_only/PEMSBAY/s42.log` | — |
| queued | 8 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0008/topo_only/PEMSD7M/s100.log` | — |
| queued | 8 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0008/topo_only/PEMSD7M/s42.log` | — |
| queued | 8 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0008/topo_only/METRLA/s100.log` | — |
| queued | 8 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0008/topo_only/METRLA/s42.log` | — |
| queued | 8 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0008/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 8 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0008/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 8 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0008/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 8 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0008/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 8 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0008/scpt_topo/METRLA/s100.log` | — |
| queued | 8 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0008/scpt_topo/METRLA/s42.log` | — |
| queued | 9 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0009/topo_only/PEMSBAY/s100.log` | — |
| queued | 9 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0009/topo_only/PEMSBAY/s42.log` | — |
| queued | 9 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0009/topo_only/PEMSD7M/s100.log` | — |
| queued | 9 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0009/topo_only/PEMSD7M/s42.log` | — |
| queued | 9 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0009/topo_only/METRLA/s100.log` | — |
| queued | 9 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0009/topo_only/METRLA/s42.log` | — |
| queued | 9 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0009/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 9 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0009/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 9 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0009/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 9 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0009/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 9 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0009/scpt_topo/METRLA/s100.log` | — |
| queued | 9 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0009/scpt_topo/METRLA/s42.log` | — |
| queued | 10 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0010/topo_only/PEMSBAY/s100.log` | — |
| queued | 10 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0010/topo_only/PEMSBAY/s42.log` | — |
| queued | 10 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0010/topo_only/PEMSD7M/s100.log` | — |
| queued | 10 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0010/topo_only/PEMSD7M/s42.log` | — |
| queued | 10 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0010/topo_only/METRLA/s100.log` | — |
| queued | 10 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0010/topo_only/METRLA/s42.log` | — |
| queued | 10 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0010/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 10 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0010/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 10 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0010/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 10 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0010/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 10 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0010/scpt_topo/METRLA/s100.log` | — |
| queued | 10 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0010/scpt_topo/METRLA/s42.log` | — |
| queued | 11 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0011/topo_only/PEMSBAY/s100.log` | — |
| queued | 11 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0011/topo_only/PEMSBAY/s42.log` | — |
| queued | 11 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0011/topo_only/PEMSD7M/s100.log` | — |
| queued | 11 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0011/topo_only/PEMSD7M/s42.log` | — |
| queued | 11 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0011/topo_only/METRLA/s100.log` | — |
| queued | 11 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0011/topo_only/METRLA/s42.log` | — |
| queued | 11 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0011/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 11 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0011/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 11 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0011/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 11 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0011/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 11 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0011/scpt_topo/METRLA/s100.log` | — |
| queued | 11 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0011/scpt_topo/METRLA/s42.log` | — |
| queued | 12 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0012/topo_only/PEMSBAY/s100.log` | — |
| queued | 12 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0012/topo_only/PEMSBAY/s42.log` | — |
| queued | 12 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0012/topo_only/PEMSD7M/s100.log` | — |
| queued | 12 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0012/topo_only/PEMSD7M/s42.log` | — |
| queued | 12 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0012/topo_only/METRLA/s100.log` | — |
| queued | 12 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0012/topo_only/METRLA/s42.log` | — |
| queued | 12 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0012/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 12 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0012/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 12 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0012/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 12 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0012/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 12 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0012/scpt_topo/METRLA/s100.log` | — |
| queued | 12 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0012/scpt_topo/METRLA/s42.log` | — |
| queued | 13 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0013/topo_only/PEMSBAY/s100.log` | — |
| queued | 13 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0013/topo_only/PEMSBAY/s42.log` | — |
| queued | 13 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0013/topo_only/PEMSD7M/s100.log` | — |
| queued | 13 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0013/topo_only/PEMSD7M/s42.log` | — |
| queued | 13 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0013/topo_only/METRLA/s100.log` | — |
| queued | 13 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0013/topo_only/METRLA/s42.log` | — |
| queued | 13 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0013/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 13 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0013/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 13 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0013/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 13 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0013/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 13 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0013/scpt_topo/METRLA/s100.log` | — |
| queued | 13 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0013/scpt_topo/METRLA/s42.log` | — |
| queued | 14 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0014/topo_only/PEMSBAY/s100.log` | — |
| queued | 14 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0014/topo_only/PEMSBAY/s42.log` | — |
| queued | 14 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0014/topo_only/PEMSD7M/s100.log` | — |
| queued | 14 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0014/topo_only/PEMSD7M/s42.log` | — |
| queued | 14 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0014/topo_only/METRLA/s100.log` | — |
| queued | 14 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0014/topo_only/METRLA/s42.log` | — |
| queued | 14 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0014/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 14 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0014/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 14 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0014/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 14 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0014/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 14 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0014/scpt_topo/METRLA/s100.log` | — |
| queued | 14 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0014/scpt_topo/METRLA/s42.log` | — |
| queued | 15 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0015/topo_only/PEMSBAY/s100.log` | — |
| queued | 15 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0015/topo_only/PEMSBAY/s42.log` | — |
| queued | 15 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0015/topo_only/PEMSD7M/s100.log` | — |
| queued | 15 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0015/topo_only/PEMSD7M/s42.log` | — |
| queued | 15 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0015/topo_only/METRLA/s100.log` | — |
| queued | 15 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0015/topo_only/METRLA/s42.log` | — |
| queued | 15 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0015/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 15 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0015/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 15 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0015/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 15 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0015/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 15 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0015/scpt_topo/METRLA/s100.log` | — |
| queued | 15 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0015/scpt_topo/METRLA/s42.log` | — |
| queued | 16 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0016/topo_only/PEMSBAY/s100.log` | — |
| queued | 16 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0016/topo_only/PEMSBAY/s42.log` | — |
| queued | 16 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0016/topo_only/PEMSD7M/s100.log` | — |
| queued | 16 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0016/topo_only/PEMSD7M/s42.log` | — |
| queued | 16 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0016/topo_only/METRLA/s100.log` | — |
| queued | 16 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0016/topo_only/METRLA/s42.log` | — |
| queued | 16 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0016/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 16 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0016/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 16 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0016/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 16 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0016/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 16 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0016/scpt_topo/METRLA/s100.log` | — |
| queued | 16 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0016/scpt_topo/METRLA/s42.log` | — |
| queued | 17 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0017/topo_only/PEMSBAY/s100.log` | — |
| queued | 17 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0017/topo_only/PEMSBAY/s42.log` | — |
| queued | 17 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0017/topo_only/PEMSD7M/s100.log` | — |
| queued | 17 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0017/topo_only/PEMSD7M/s42.log` | — |
| queued | 17 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0017/topo_only/METRLA/s100.log` | — |
| queued | 17 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0017/topo_only/METRLA/s42.log` | — |
| queued | 17 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0017/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 17 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0017/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 17 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0017/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 17 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0017/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 17 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0017/scpt_topo/METRLA/s100.log` | — |
| queued | 17 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0017/scpt_topo/METRLA/s42.log` | — |
| queued | 18 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0018/topo_only/PEMSBAY/s100.log` | — |
| queued | 18 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0018/topo_only/PEMSBAY/s42.log` | — |
| queued | 18 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0018/topo_only/PEMSD7M/s100.log` | — |
| queued | 18 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0018/topo_only/PEMSD7M/s42.log` | — |
| queued | 18 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0018/topo_only/METRLA/s100.log` | — |
| queued | 18 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0018/topo_only/METRLA/s42.log` | — |
| queued | 18 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0018/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 18 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0018/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 18 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0018/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 18 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0018/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 18 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0018/scpt_topo/METRLA/s100.log` | — |
| queued | 18 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0018/scpt_topo/METRLA/s42.log` | — |
| queued | 19 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0019/topo_only/PEMSBAY/s100.log` | — |
| queued | 19 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0019/topo_only/PEMSBAY/s42.log` | — |
| queued | 19 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0019/topo_only/PEMSD7M/s100.log` | — |
| queued | 19 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0019/topo_only/PEMSD7M/s42.log` | — |
| queued | 19 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0019/topo_only/METRLA/s100.log` | — |
| queued | 19 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0019/topo_only/METRLA/s42.log` | — |
| queued | 19 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0019/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 19 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0019/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 19 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0019/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 19 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0019/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 19 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0019/scpt_topo/METRLA/s100.log` | — |
| queued | 19 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0019/scpt_topo/METRLA/s42.log` | — |
| queued | 20 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0020/topo_only/PEMSBAY/s100.log` | — |
| queued | 20 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0020/topo_only/PEMSBAY/s42.log` | — |
| queued | 20 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0020/topo_only/PEMSD7M/s100.log` | — |
| queued | 20 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0020/topo_only/PEMSD7M/s42.log` | — |
| queued | 20 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0020/topo_only/METRLA/s100.log` | — |
| queued | 20 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0020/topo_only/METRLA/s42.log` | — |
| queued | 20 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0020/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 20 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0020/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 20 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0020/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 20 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0020/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 20 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0020/scpt_topo/METRLA/s100.log` | — |
| queued | 20 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0020/scpt_topo/METRLA/s42.log` | — |
| queued | 21 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0021/topo_only/PEMSBAY/s100.log` | — |
| queued | 21 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0021/topo_only/PEMSBAY/s42.log` | — |
| queued | 21 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0021/topo_only/PEMSD7M/s100.log` | — |
| queued | 21 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0021/topo_only/PEMSD7M/s42.log` | — |
| queued | 21 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0021/topo_only/METRLA/s100.log` | — |
| queued | 21 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0021/topo_only/METRLA/s42.log` | — |
| queued | 21 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0021/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 21 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0021/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 21 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0021/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 21 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0021/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 21 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0021/scpt_topo/METRLA/s100.log` | — |
| queued | 21 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0021/scpt_topo/METRLA/s42.log` | — |
| queued | 22 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0022/topo_only/PEMSBAY/s100.log` | — |
| queued | 22 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0022/topo_only/PEMSBAY/s42.log` | — |
| queued | 22 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0022/topo_only/PEMSD7M/s100.log` | — |
| queued | 22 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0022/topo_only/PEMSD7M/s42.log` | — |
| queued | 22 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0022/topo_only/METRLA/s100.log` | — |
| queued | 22 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0022/topo_only/METRLA/s42.log` | — |
| queued | 22 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0022/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 22 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0022/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 22 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0022/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 22 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0022/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 22 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0022/scpt_topo/METRLA/s100.log` | — |
| queued | 22 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0022/scpt_topo/METRLA/s42.log` | — |
| queued | 23 | TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0023/topo_only/PEMSBAY/s100.log` | — |
| queued | 23 | TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0023/topo_only/PEMSBAY/s42.log` | — |
| queued | 23 | TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0023/topo_only/PEMSD7M/s100.log` | — |
| queued | 23 | TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0023/topo_only/PEMSD7M/s42.log` | — |
| queued | 23 | TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0023/topo_only/METRLA/s100.log` | — |
| queued | 23 | TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0023/topo_only/METRLA/s42.log` | — |
| queued | 23 | SCPT+TOPO | PEMSBAY | 100 | None | `logs_hpo_est/trial_0023/scpt_topo/PEMSBAY/s100.log` | — |
| queued | 23 | SCPT+TOPO | PEMSBAY | 42 | None | `logs_hpo_est/trial_0023/scpt_topo/PEMSBAY/s42.log` | — |
| queued | 23 | SCPT+TOPO | PEMSD7M | 100 | None | `logs_hpo_est/trial_0023/scpt_topo/PEMSD7M/s100.log` | — |
| queued | 23 | SCPT+TOPO | PEMSD7M | 42 | None | `logs_hpo_est/trial_0023/scpt_topo/PEMSD7M/s42.log` | — |
| queued | 23 | SCPT+TOPO | METRLA | 100 | None | `logs_hpo_est/trial_0023/scpt_topo/METRLA/s100.log` | — |
| queued | 23 | SCPT+TOPO | METRLA | 42 | None | `logs_hpo_est/trial_0023/scpt_topo/METRLA/s42.log` | — |

## 超参搜索空间（本 run）

```yaml
GATE_HIDDEN: [32, 64, 128]
TOPO_LAP_K: [8, 16, 32]
MOE_TAU: [0.5, 1.0, 1.5, 2.0]
MOE_LB_REG: [0.0001, 0.0005, 0.001, 0.005]
MOE_SMOOTH_REG: [0.0001, 0.0005, 0.001, 0.005]
MOE_DELTA_REG: [0.0, 0.0001, 0.001]
MOE_USE_CTX: [0, 1]
PRE_LEARN: [0.0005, 0.001, 0.002]
```

