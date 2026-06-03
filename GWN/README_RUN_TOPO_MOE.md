# Topology + Selective Sparse MoE patch for `liuthegiant/linhui`

This patch adds selective Sparse-MoE fusion for the GWN estimation and forecasting scripts.

You can now choose the exact expert embeddings used in the router with:

```bash
MOE_EXPERTS=scpt,geo,topo
```

Supported expert names and aliases:

```text
temporal: temporal, temp, tmp, scpt
geometric: geometric, geo, geom
topology: topology, topo, lap, laplacian
```

The original scripts are not overwritten. These wrappers import the original scripts and monkey-patch only the training/testing functions needed by the new fusion.

## Files

Copy these files into the repo:

```bash
cp -r GWN/* /path/to/linhui/GWN/
cd /path/to/linhui/GWN
```

Files included:

```text
GWN/topo_moe_utils.py
GWN/pred_maskpredition_GWN_scpt_geo_topomoe.py
GWN/pred_GWN_16_adpAdj_topomoe.py
```

Run from `GWN/`, because the original repo uses relative paths such as `../METRLA/...` and `../save/...`.

---

## Key switch: `MOE_EXPERTS`

Examples:

```bash
MOE_EXPERTS=scpt,geo          # temporal + geometric
MOE_EXPERTS=scpt,topo         # temporal + topology
MOE_EXPERTS=geo,topo          # geometric + topology
MOE_EXPERTS=scpt,geo,topo     # all three
MOE_EXPERTS=topo              # topology only
```

`MOE_TOP_K` controls how many experts are active per node.

```text
MOE_TOP_K=1   node-wise sparse selection: one expert per node
MOE_TOP_K=2   default for three experts; two experts per node
MOE_TOP_K=3   dense soft mixture for three experts
```

For a two-expert run, both `MOE_TOP_K=1` and `MOE_TOP_K=2` are useful ablations. For a one-expert run, the code automatically clamps top-k to 1.

`MOE_FORCE_EXPERT` still works, but it is mostly for diagnostics. For example, these two are similar:

```bash
MOE_EXPERTS=topo
MOE_EXPERTS=scpt,geo,topo MOE_FORCE_EXPERT=topology
```

The first is cleaner for a topology-only run.

---

# A. Estimation first: unseen masked estimation

This is the main setting to run first.

Base estimation command, with the extra MoE arguments after the original 20 arguments:

```bash
python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

Use environment variables before this command to pick the embeddings.

## A1. SCPT + GEO Sparse MoE, no topology

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=scpt,geo python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  sparse_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

## A2. SCPT + TOPO Sparse MoE

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=scpt,topo python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

## A3. GEO + TOPO Sparse MoE

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=geo,topo python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

## A4. SCPT + GEO + TOPO full Sparse MoE

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=scpt,geo,topo python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

## A5. Single-expert sanity checks

```bash
# SCPT only through the same MoE path
MOE_EXPERTS=scpt MOE_TOP_K=1 python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1

# GEO only through the same MoE path
MOE_EXPERTS=geo MOE_TOP_K=1 python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1

# TOPO only through the same MoE path
MOE_EXPERTS=topo MOE_TOP_K=1 python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

## A6. Smoke run

```bash
MOE_EXPERTS=scpt,geo,topo python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  1 0.7 1 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 1 1 0 0.001 1 320 \
  topo_moe 64 16 2 1.0 0.001 0.001 0.0 1
```

---

# B. Forecasting second: 12-step GWN forecasting

The original forecasting script has more positional arguments. The wrapper follows the same ordering and adds MoE arguments after `DELTA_REG`.

Base forecasting command:

```bash
python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

Use environment variables before this command to pick the embeddings.

## B1. SCPT + GEO Sparse MoE, no topology

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=scpt,geo python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual sparse_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

## B2. SCPT + TOPO Sparse MoE

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=scpt,topo python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

## B3. GEO + TOPO Sparse MoE

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=geo,topo python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

## B4. SCPT + GEO + TOPO full Sparse MoE

```bash
cd /path/to/linhui/GWN
MOE_EXPERTS=scpt,geo,topo python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

## B5. Single-expert sanity checks

```bash
# SCPT only
MOE_EXPERTS=scpt MOE_TOP_K=1 python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1

# GEO only
MOE_EXPERTS=geo MOE_TOP_K=1 python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1

# TOPO only
MOE_EXPERTS=topo MOE_TOP_K=1 python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 0 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

## B6. Smoke run

```bash
MOE_EXPERTS=scpt,geo,topo python pred_GWN_16_adpAdj_topomoe.py \
  1 0.7 1 100 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 1 1 0 0.001 1 320 \
  1 0.5 0 encoder encoderg "" dual topo_moe 1.0 0.0 64 1 0.0 0.0 \
  16 2 1.0 0.001 0.001 0.0 1
```

---

## Extra arguments

### Estimation wrapper extra args

After original `argv[1]..argv[20]`:

```text
argv[21] FUSION_MODE       sparse_moe | topo_moe
argv[22] GATE_HIDDEN       default 64
argv[23] TOPO_LAP_K        default 16
argv[24] MOE_TOP_K         default 2
argv[25] MOE_TAU           default 1.0
argv[26] MOE_LB_REG        default 0.001
argv[27] MOE_SMOOTH_REG    default 0.001
argv[28] MOE_DELTA_REG     default 0.0
argv[29] MOE_USE_CTX       default 1
argv[30] MOE_EXPERTS       optional, e.g. scpt,geo,topo
```

### Forecast wrapper extra args

Original forecast script consumes up to `argv[34]`. New args start at `argv[35]`:

```text
argv[35] TOPO_LAP_K        default 16
argv[36] MOE_TOP_K         default 2
argv[37] MOE_TAU           default 1.0
argv[38] MOE_LB_REG        default 0.001
argv[39] MOE_SMOOTH_REG    default 0.001
argv[40] MOE_DELTA_REG     default 0.0
argv[41] MOE_USE_CTX       default 1
argv[42] MOE_EXPERTS       optional, e.g. scpt,geo,topo
```

All of these can also be overridden with environment variables of the same name. Recommended usage is environment variables, because it avoids changing the long positional command.

Useful environment variables:

```bash
MOE_EXPERTS=scpt,geo,topo   # choose which embeddings enter Sparse MoE
MOE_TOP_K=1                 # choose only one expert per node
MOE_TOP_K=2                 # choose two experts per node
MOE_FORCE_EXPERT=topology   # force one expert inside a multi-expert run
TOPO_FORCE_RECOMPUTE=1      # recompute cached topology embedding
MOE_INIT_TEMPORAL_BIAS=0    # no initial bias toward temporal expert
```

---

## Outputs to check

The wrappers save the same prediction/groundtruth/score files as the original scripts, plus router diagnostics:

```text
*_moe_alpha.npy
*_moe_alpha_summary.json
```

The expert order in these files is exactly the order in `MOE_EXPERTS` after alias normalization. For example:

```bash
MOE_EXPERTS=geo,topo
```

will save:

```json
{"expert_names": ["geometric", "topology"], ...}
```

When running multiple ablations, use a separate save folder/run tag when possible, or move the result folder after each run. Different `MOE_EXPERTS` settings have different fusion checkpoint shapes, so reusing the same path can overwrite `*_fusion_*.pt` files.

Recommended estimation ablation table:

```text
SCPT only
GEO only
TOPO only
SCPT + GEO Sparse MoE
SCPT + TOPO Sparse MoE
GEO + TOPO Sparse MoE
SCPT + GEO + TOPO Sparse MoE
```
