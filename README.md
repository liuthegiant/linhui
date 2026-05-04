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

