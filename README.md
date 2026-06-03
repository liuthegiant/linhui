# linhui/9991/gith

这个目录是我在交通时空预测实验里整理的代码与结果汇总仓库（放到 GitHub：`liuthegiant/linhui`）。

## 目录结构（每个文件夹是干啥的）

- **`AGCN/`**：AGCRN/AGCN 相关实验代码与预测脚本  
  - **`pred_AGCRN_16_adpAdj.py`**：主预测/实验脚本（跑训练/评估/统计的入口之一）  
  - **`AGCRN.py` / `AGCRNCell.py` / `AGCN.py`**：模型实现  
  - **`graph.py` / `unseen_nodes.py`**：图构建、新节点/未见节点相关处理  
  - **`AGCRN_METRLA_predict12_5seeds.md`**：METR-LA 上预测 12-step 的实验记录

- **`GWN/`**：Graph WaveNet 相关实验代码与预测脚本  
  - **`pred_GWN_16_adpAdj.py`**：主预测/实验脚本（训练/评估/统计入口之一）  
  - **`pred_maskpredition_GWN*.py`**：mask/补全（infill）相关实验脚本（含不同特征组合）  
  - **`graph.py` / `graph_new_sensor.py` / `unseen_nodes.py`**：图与新增传感器/未见节点处理  
  - **`GWN_*predict12*.md`**：各数据集的预测实验记录  
  - **`REPORT_statistics_*`**：多 seed 统计汇总报告

- **`STGfomer/`**：STGFormer 相关（用于替换/对比下游模型）  
  - **`train.py`**：STGFormer 的训练脚本（来自其工程化实现，依赖其 `lib/`、`model/` 等目录；若缺少需要自行补齐对应工程）  
  - **`STGformer.py`**：STGFormer 模型定义文件（同上，依赖其工程目录结构）  
  - **`pred_STGFormer_16_adpAdj_12_step.py`**：将 STGFormer 适配成 GWN 同接口的“替换式”脚本  
    - 会把 `GWN/` 加到 `sys.path`，然后复用 `GWN/pred_GWN_16_adpAdj.py` 的数据/流程，仅替换 `getModel`
  - **`STGFormer_METRLA_predict12_5seeds_no_phy5.md`**：对应实验记录

- **`METRLA/`**：METR-LA 数据与图相关的辅助文件（含邻接矩阵/距离等）  
  - **注意**：`METRLA/metr-la.h5` 是数据文件，已用 **Git LFS** 管理（见 `.gitattributes`）。
  - **带 `new`/`with_newnodes` 的文件**：表示在原始 METR-LA 节点集合基础上**加入了虚拟节点（new/virtual nodes）**后重新生成的版本（例如 `distances_la_2012_with_newnodes.csv`、`adj_mat_with_newnodes.pkl`、`graph_sensor_ids_with_newnodes.txt`、`graph_sensor_locations_new.csv`）。

- **`PEMSBAY/`**：PEMS-BAY 相关目录  
  - **注意**：`PEMSBAY/*.h5` 默认不上传（见 `.gitignore`），避免触发 GitHub 大文件限制。

- **`PEMSD7M/`**：PEMSD7M 相关目录（如你后续加入的话）  
  - 如果包含 `.h5`，建议同样走 **Git LFS**。

- **`save/`**：临时/中间产物目录（当前为空或用于本地运行时落盘）。

## 大文件与 Git LFS / .gitignore 约定

- **LFS**：仓库已配置 `*.h5` 走 Git LFS（见 `.gitattributes`）。
- **忽略规则**（见 `.gitignore`）：默认忽略所有 `.h5`，但允许 `METRLA/*.h5`；`PEMSBAY/*.h5` 明确忽略。

## 快速定位“跑哪个脚本”

- **GWN 主流程**：`GWN/pred_GWN_16_adpAdj.py`
- **AGCRN 主流程**：`AGCN/pred_AGCRN_16_adpAdj.py`
- **STGFormer 替换式跑法**：`STGfomer/pred_STGFormer_16_adpAdj_12_step.py`（复用 GWN 脚本流程）

## 运行这些脚本（参数怎么传、跑出什么、结果怎么看）

### 0) 通用注意事项

- **运行目录必须对**：脚本里大量用到 `../METRLA/...`、`../PEMSBAY/...`、`../save/...` 这种相对路径，所以要在脚本所属目录运行。
  - 跑 `GWN/*.py`：先 `cd /mnt/data728/linhui/9991/gith/GWN`
  - 跑 `AGCN/*.py`：先 `cd /mnt/data728/linhui/9991/gith/AGCN`
  - 跑 `STGfomer/pred_STGFormer_*.py`：先 `cd /mnt/data728/linhui/9991/gith/STGfomer`
- **输出目录**：都会写到 `../save/<KEYWORD>/`，其中 `<KEYWORD>` 一般是：
  - 预测类：`pred_<DATANAME>_<MODELNAME>_<时间>_<pid>`
  - 状态估计/补全类：`est_<DATANAME>_<MODELNAME>_<时间>_<pid>`

### 1) 参数传递方式（不是 argparse，是 sys.argv 按位置）

这些脚本都用 `sys.argv[1]、sys.argv[2]...` 的方式取参数，所以**参数顺序固定**。你可以只传前几个，后面的不传会用默认值。

#### 1.1 预测脚本（`pred_GWN_16_adpAdj.py` / `pred_AGCRN_16_adpAdj.py`）的前 13 个常用参数

- **argv[1] `IS_PRETRN`**：`1/0` 是否跑预训练阶段
- **argv[2] `R_TRN`**：训练集比例（默认 0.7）
- **argv[3] `IS_EPOCH_1`**：`1/0`（脚本内部开关，通常保持 0）
- **argv[4] `seed`**：随机种子（默认 100）
- **argv[5] `TEMPERATURE`**：温度参数（默认 1.0）
- **argv[6] `DATANAME`**：数据集名（常用：`METRLA`、`PEMSBAY`、`PEMSD7M`）
- **argv[7] `seed_SS`**：空间划分 seed（默认 -1）
- **argv[8] `IS_DESEASONED`**：`1/0` 是否去季节（为 1 时会把输入 `CHANNEL` 变成 2：主信号 + deseason 分量）
- **argv[9] `weight_decay`**
- **argv[10] `adp_adj`**：`1/0` 是否启用自适应邻接
- **argv[11] `is_SGA`**：`1/0`（脚本内部开关）
- **argv[12] `FEATURES`**：特征数（不同脚本默认不同，GWN 常见 2 或 4）

后续还有 `SUBGRAPH_SIZE`、`QUOTIENT_GRAPH_RADIUS`、`EPOCH`、`PRETRN_EPOCH`、融合/门控相关参数等（不传就用脚本默认值）。

#### 1.2 状态估计/补全脚本（`pred_GWN_sensor2.py` / `pred_maskpredition_GWN*.py`）

参数顺序与上面基本一致（同样从 `argv[1]` 开始），区别在于这些脚本会额外做 **missing/mask**（例如 `missing_ratio=0.2`）并保存掩码与热力图。

### 2) 每个脚本怎么跑（最小可用命令）

#### 2.1 `GWN/pred_GWN_16_adpAdj.py`（主预测：训练 + test_u/test_a）

```bash
cd /mnt/data728/linhui/9991/gith/GWN
python pred_GWN_16_adpAdj.py 1 0.7 0 100 1.0 METRLA
```

#### 2.2 `AGCN/pred_AGCRN_16_adpAdj.py`（主预测：训练 + test_u/test_a）

```bash
cd /mnt/data728/linhui/9991/gith/AGCN
python pred_AGCRN_16_adpAdj.py 1 0.7 0 100 1.0 METRLA
```

#### 2.3 `GWN/pred_GWN_sensor2.py`（状态估计：含虚拟节点/缺失恢复，输出 est_*）

```bash
cd /mnt/data728/linhui/9991/gith/GWN
python pred_GWN_sensor2.py 1 0.7 0 100 1.0 METRLA
```

#### 2.4 `GWN/pred_maskpredition_GWN*.py`（mask/补全：输出 prediction/groundtruth/missmask + heatmap）

以 `PEMSBAY` 为例：

```bash
cd /mnt/data728/linhui/9991/gith/GWN
python pred_maskpredition_GWN.py 0 0.7 0 100 1.0 PEMSBAY
python pred_maskpredition_GWN_geo.py 0 0.7 0 100 1.0 PEMSBAY
python pred_maskpredition_GWN_scpt.py 0 0.7 0 100 1.0 PEMSBAY
python pred_maskpredition_GWN_scpt_geo.py 0 0.7 0 100 1.0 PEMSBAY
```

#### 2.5 `STGfomer/pred_STGFormer_16_adpAdj_12_step.py`（用 STGFormer 替换 GWN 的 getModel）

这个脚本会 `import GWN/pred_GWN_16_adpAdj.py` 并直接调用 `base.main()`，所以**参数与 `pred_GWN_16_adpAdj.py` 完全一致**：

```bash
cd /mnt/data728/linhui/9991/gith/STGfomer
python pred_STGFormer_16_adpAdj_12_step.py 1 0.7 0 100 1.0 METRLA
```

可用环境变量：
- `STGFORMER_EMBED_DIM`：拼接到输入的 embed 维度（默认 32）
- `STGFORMER_CLIP_NORM`：梯度裁剪阈值（默认 5.0）
- `STGFORMER_SMOKE=1`：只做一次前向自检（打印 `SMOKE y.shape`）

### 3) 跑出来的文件各是什么意思（你要看哪些结果）

在 `../save/<KEYWORD>/` 里常见的结果：

- **`*_log.txt`**：训练日志（loss、epoch 信息等）
- **`*_prediction_scores.txt`**：评估指标汇总（通常含 MAE/RMSE/MAPE；按 `mode`/horizon/节点集统计）
- **`*.pt`**：模型权重
  - 预测类常见：`<MODELNAME>_u.pt`（unseen 节点）与 `<MODELNAME>_a.pt`（all nodes）
  - 补全/估计类常见：`<name>_best.pt`（best checkpoint），以及可能的 `encoder.pt / encoderg.pt`
- **`*_prediction.npy` / `*_groundtruth.npy`**：预测值与真值数组（用于后处理画图/复算指标）
- **`*_missmask.npy`**：缺失/掩码位置（仅补全/估计类脚本会保存）
- **`heatmap_*.png`**：真值/预测/误差热力图（仅补全/估计类脚本会保存）
- **`paper_timing.json` / `paper_timing.txt`**：训练/推理耗时统计（预测类脚本常见）

