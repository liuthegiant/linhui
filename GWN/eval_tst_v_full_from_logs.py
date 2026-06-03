#!/usr/bin/env python3
"""Re-run tst_v_full (full-graph forward, V nodes fully masked) using existing checkpoints.

This script:
  - Scans the existing batch log files to find each run's saved output directory (P.PATH).
  - Re-invokes the splitmask entrypoint in eval-only mode:
      FORECASTING_EVAL_DIR=<run_dir> FORECASTING_EVAL_MODE=tst_v_full
    so it loads the existing weights and runs only the tst_v_full evaluation.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

SEEDS = (100, 42, 999, 555, 250, 88, 66, 233, 38, 432)
CONFIGS = ("no_pretrain", "topo_only", "geo_only", "geo_topo")
DATASETS = ("METRLA", "PEMSBAY", "PEMSD7M")

PYBIN_DEFAULT = "/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python"
ENTRYPOINT = "pred_maskpredition_GWN_scpt_geo_topomoe_virtualnode_splitmask.py"

LOGROOTS_DEFAULT = {
    "METRLA": "logs_topomoe/est_metrla_virtualnode_splitmask_seed10",
    "PEMSBAY": "logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10",
    "PEMSD7M": "logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10",
}

CFG_INFO = {
    "no_pretrain": ("geo", "1", "0"),
    "topo_only": ("topo", "1", "1"),
    "geo_only": ("geo", "1", "1"),
    "geo_topo": ("geo,topo", "2", "1"),
}

# Log lines are like:
#   [TopoMoE] output dir: /abs/path
#   [splitmask] saved ../save/.../mask_policy.json
_RE_OUTDIR = re.compile(r"\[TopoMoE\]\s+output dir:\s+(?P<path>.+)\s*$", re.MULTILINE)
_RE_SAVE_MASKPOLICY = re.compile(r"\[splitmask\]\s+saved\s+(?P<path>.+?/mask_policy\.json)\s*$", re.MULTILINE)
_RE_PATH_LINE = re.compile(r"^PATH\\s+(?P<path>\\S.+?)\\s*$", re.MULTILINE)


def base_argv(seed: int, ds: str, pretrain_flag: str) -> list[str]:
    # argv[1]..argv[20]
    base = "1 0.7 0 {seed} 1.0 {ds} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320".format(seed=seed, ds=ds)
    arr = base.split()
    arr[0] = pretrain_flag
    return arr


def extra_topo() -> list[str]:
    # argv[21]..argv[29] in our runs
    return "topo_moe 64 16 2 1.0 0.001 0.001 0.0 1".split()


def log_path(root: Path, logroot: Path, ds: str, seed: int, cfg: str) -> Path:
    return logroot / ds / str(seed) / "est" / f"A_{cfg}_s{seed}.log"


def extract_run_dir(text: str) -> str | None:
    m = _RE_OUTDIR.search(text)
    if m:
        return m.group("path").strip()
    m2 = _RE_SAVE_MASKPOLICY.search(text)
    if m2:
        # .../<P.PATH>/mask_policy.json
        return str(Path(m2.group("path")).resolve().parent)
    m3 = _RE_PATH_LINE.search(text)
    if m3:
        return m3.group("path").strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pybin", default=PYBIN_DEFAULT)
    ap.add_argument("--entrypoint", default=ENTRYPOINT)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--only-missing", action="store_true", help="Skip if *_tst_v_full_* files already exist.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    pybin = args.pybin
    entry = str((root / args.entrypoint).resolve())

    for ds in args.datasets:
        logroot = (root / LOGROOTS_DEFAULT[ds]).resolve()
        for seed in args.seeds:
            for cfg in args.configs:
                lp = log_path(root, logroot, ds, seed, cfg)
                if not lp.is_file():
                    print("[skip] missing log:", lp)
                    continue
                txt = lp.read_text(encoding="utf-8", errors="ignore")
                run_dir = extract_run_dir(txt)
                if not run_dir:
                    print("[skip] cannot find output dir in log:", lp)
                    continue
                # Logs often store PATH as ../save/..., resolve relative to repo root.
                run_dir_p = (root / run_dir).resolve() if not os.path.isabs(run_dir) else Path(run_dir)
                if not run_dir_p.exists():
                    print("[skip] output dir not found:", run_dir_p)
                    continue
                if args.only_missing:
                    any_done = any(run_dir_p.glob(f"*tst_v_full*_prediction.npy"))
                    if any_done:
                        print("[skip] already has tst_v_full outputs:", ds, seed, cfg, run_dir_p)
                        continue

                experts, topk, pretrain_flag = CFG_INFO[cfg]
                env = os.environ.copy()
                env["MOE_EXPERTS"] = experts
                env["MOE_TOP_K"] = topk
                env["FORECASTING_EVAL_DIR"] = str(run_dir_p)
                env["FORECASTING_EVAL_MODE"] = "tst_v_full"

                argv = base_argv(seed, ds, pretrain_flag) + extra_topo()
                cmd = [pybin, entry, *argv]
                print("[eval]", ds, "seed", seed, "cfg", cfg, "dir", run_dir_p)
                if args.dry_run:
                    continue
                subprocess.run(cmd, check=False, env=env)


if __name__ == "__main__":
    main()

