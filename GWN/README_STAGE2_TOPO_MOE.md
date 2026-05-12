# TopoMoE Stage-2 Operations

This patch keeps the model design from the selective TopoMoE patch and adds two conveniences:

1. `MOE_RUN_TAG` / `MOE_AUTO_TAG` output subfolders, so different expert subsets and tau/reg settings no longer overwrite each other.
2. Small helper scripts for alpha inspection, score collection, METR-LA estimation sweep, quick forecast check, and running the selected best setting on other datasets.

## Files to update

Copy these updated model wrappers into `GWN/`:

```bash
cp topomoe_patch/GWN/topo_moe_utils.py /path/to/linhui/GWN/
cp topomoe_patch/GWN/pred_maskpredition_GWN_scpt_geo_topomoe.py /path/to/linhui/GWN/
cp topomoe_patch/GWN/pred_GWN_16_adpAdj_topomoe.py /path/to/linhui/GWN/
```

Copy helper scripts into the repo root:

```bash
cp -r topomoe_patch/scripts /path/to/linhui/
chmod +x /path/to/linhui/scripts/topomoe_ops/*.sh
```

## New environment variables

- `MOE_RUN_TAG=...`: explicit result subfolder under the original `P.PATH`.
- `MOE_AUTO_TAG=0`: disable automatic result subfoldering.
- `MOE_EXPERTS=scpt,topo`: choose experts.
- `MOE_TOP_K`, `MOE_TAU`, `MOE_LB_REG`, `MOE_SMOOTH_REG`, `MOE_DELTA_REG`, `MOE_USE_CTX`: same as before.

By default, the wrappers automatically put outputs into a config-specific subfolder. The sweep scripts set explicit tags such as:

```text
est_sctopo_tau05_noreg_seed100
est_full_tau05_noreg_seed100
```

## Stage 0: sanity check

```bash
cd /path/to/linhui/GWN
python -m py_compile topo_moe_utils.py pred_maskpredition_GWN_scpt_geo_topomoe.py pred_GWN_16_adpAdj_topomoe.py
```

## Stage 1: inspect current alpha summaries

```bash
cd /path/to/linhui
python scripts/topomoe_ops/show_alpha.py --root logs_topomoe
python scripts/topomoe_ops/collect_scores.py --root logs_topomoe --csv topomoe_existing_scores.csv
```

## Stage 2 and 3: METR-LA estimation sweep

This runs SCPT+TOPO tau/reg sweep and full SCPT+GEO+TOPO no-reg sweep for seeds `100 42 999`.

```bash
cd /path/to/linhui
SEEDS="100 42 999" ./scripts/topomoe_ops/run_stage2_metrla_estimation_sweep.sh
```

Smoke test:

```bash
cd /path/to/linhui
SEEDS="100" PRE_EPOCH=1 MAIN_EPOCH=1 ./scripts/topomoe_ops/run_stage2_metrla_estimation_sweep.sh
```

Optional diagnostic top-k=1:

```bash
cd /path/to/linhui
RUN_TOPK1_DIAG=1 SEEDS="100 42 999" ./scripts/topomoe_ops/run_stage2_metrla_estimation_sweep.sh
```

## Stage 4: choose best METR-LA estimation setting

After the sweep:

```bash
cd /path/to/linhui
python scripts/topomoe_ops/collect_scores.py --root logs_topomoe --csv topomoe_stage2_metrla_estimation.csv
python scripts/topomoe_ops/show_alpha.py --root logs_topomoe
```

Main decision metric: estimation `test_u` MAE. If tied, use `test_u` RMSE, then stability/std.

## Stage 5: run selected best setting on PEMS-BAY / PEMSD7M

Example if best is SCPT+TOPO, tau=0.5, no regularization:

```bash
cd /path/to/linhui
DATASET=PEMSBAY BEST_TAG=sctopo_tau05_noreg BEST_EXPERTS=scpt,topo BEST_TOPK=2 BEST_TAU=0.5 BEST_LB=0.0 BEST_SMOOTH=0.0 ./scripts/topomoe_ops/run_best_estimation_dataset.sh

DATASET=PEMSD7M BEST_TAG=sctopo_tau05_noreg BEST_EXPERTS=scpt,topo BEST_TOPK=2 BEST_TAU=0.5 BEST_LB=0.0 BEST_SMOOTH=0.0 ./scripts/topomoe_ops/run_best_estimation_dataset.sh
```

If the best is full three-expert:

```bash
DATASET=PEMSBAY BEST_TAG=full_tau05_noreg BEST_EXPERTS=scpt,geo,topo BEST_TOPK=2 BEST_TAU=0.5 ./scripts/topomoe_ops/run_best_estimation_dataset.sh
DATASET=PEMSD7M BEST_TAG=full_tau05_noreg BEST_EXPERTS=scpt,geo,topo BEST_TOPK=2 BEST_TAU=0.5 ./scripts/topomoe_ops/run_best_estimation_dataset.sh
```

## Stage 6: quick forecast check

```bash
cd /path/to/linhui
SEEDS="100 42 999" ./scripts/topomoe_ops/run_stage2_metrla_forecast_quick.sh
```

## Manual single-run example

```bash
cd /path/to/linhui/GWN
SEED=100
BASE_ARGS=(1 0.7 0 "$SEED" 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320)
MOE_EXPERTS=scpt,topo MOE_RUN_TAG=est_sctopo_tau05_noreg_seed100 \
python pred_maskpredition_GWN_scpt_geo_topomoe.py \
  "${BASE_ARGS[@]}" topo_moe 64 16 2 0.5 0.0 0.0 0.0 1
```
