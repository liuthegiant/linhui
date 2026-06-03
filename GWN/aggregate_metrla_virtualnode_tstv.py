#!/usr/bin/env python3
"""Compute tst_v metrics on fixed test nodes from saved fixedmask predictions."""
from __future__ import annotations

import json
import math
import os
import re
import statistics
from pathlib import Path

import numpy as np

import unseen_nodes

SEEDS = (100, 42, 999, 555, 250)
CONFIGS = (
    ("no_pretrain", "无预训练"),
    ("topo_only", "TOPO"),
    ("geo_only", "GEO"),
    ("geo_topo", "GEO+TOPO"),
)
LOGROOT = Path("logs_topomoe/est_metrla_virtualnode_fixedmask_seed5")
SAVE_ROOT = Path("../save").resolve()
REPORT = Path("METRLA_virtualnode_fixedmask_Estimation.md")


def fmt(x: float) -> str:
    return f"{x:.6f}"


def fmt_pm(xs: list[float]) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"


def discover_runs() -> dict[tuple[str, int], Path]:
    txt = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in LOGROOT.glob("*.log")
    )
    pat = re.compile(
        r"MOE_RUN_TAG\s+est_vnode_fm_metrla_(?P<cfg>.+?)_s(?P<seed>\d+)\s+"
        r".*?PATH\s+(?P<path>\.\./save/[^\s]+)",
        re.S,
    )
    out: dict[tuple[str, int], Path] = {}
    valid = {c for c, _ in CONFIGS}
    for m in pat.finditer(txt):
        cfg = m.group("cfg")
        seed = int(m.group("seed"))
        if cfg not in valid or seed not in SEEDS:
            continue
        out[(cfg, seed)] = (Path.cwd() / m.group("path")).resolve()
    return out


def local_fixed_indices(save_dir: Path) -> list[int]:
    policy = json.loads((save_dir / "mask_policy.json").read_text(encoding="utf-8"))
    fixed = [int(x) for x in policy["fixed_test_nodes"]]
    split = unseen_nodes.SpatialSplit(
        int(policy["n_nodes"]),
        r_trn=0.7,
        r_val=0.15,
        r_tst=0.15,
        seed=int(policy["seed_SS"]),
    )
    mapping = {int(node): i for i, node in enumerate(split.i_tst)}
    return [mapping[n] for n in fixed if n in mapping]


def eval_one(save_dir: Path) -> tuple[float, float, float, int, list[int]]:
    y_true = np.load(save_dir / "GraphWaveNet_tst_u_GraphWaveNet_groundtruth.npy")
    y_pred = np.load(save_dir / "GraphWaveNet_tst_u_GraphWaveNet_prediction.npy")
    miss = np.load(save_dir / "GraphWaveNet_tst_u_GraphWaveNet_missmask.npy")
    idx = local_fixed_indices(save_dir)
    if not idx:
        raise RuntimeError(f"no fixed test nodes mapped for {save_dir}")
    y_true = y_true[:, idx, :]
    y_pred = y_pred[:, idx, :]
    miss = miss[:, idx, :]
    eps = 1e-6
    abs_err = np.abs(y_true - y_pred) * miss
    sq_err = ((y_true - y_pred) ** 2) * miss
    den = float(miss.sum()) + eps
    mae = float(abs_err.sum() / den)
    rmse = float(math.sqrt(float(sq_err.sum() / den)))
    mape = float((np.abs((y_true - y_pred) / (np.abs(y_true) + eps)) * miss).sum() / den)
    return mae, rmse, mape, int(miss.sum()), idx


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    runs = discover_runs()

    lines: list[str] = []
    lines.append("## `tst_v`：固定全掩码测试节点（virtual nodes only）")
    lines.append("")
    lines.append("`tst_v` 只在 `fixed_test_nodes` 上计算 MAE/RMSE/MAPE；这些节点在所有测试时间步都被 100% mask，用于模拟完全没有历史数据的虚拟节点。")
    lines.append("")
    lines.append("| 配置 | n | tst_v MAE | tst_v RMSE | tst_v MAPE |")
    lines.append("| --- | --- | --- | --- | --- |")

    detail: list[str] = []
    detail.append("## `tst_v` 按种子分项")
    detail.append("")
    detail.append("| 配置 | 种子 | fixed test nodes | tst_v MAE | tst_v RMSE | tst_v MAPE | mask 点数 |")
    detail.append("| --- | --- | --- | --- | --- | --- | --- |")

    for cfg, label in CONFIGS:
        maes: list[float] = []
        rmses: list[float] = []
        mapes: list[float] = []
        for seed in SEEDS:
            save_dir = runs[(cfg, seed)]
            mae, rmse, mape, n_mask, _ = eval_one(save_dir)
            policy = json.loads((save_dir / "mask_policy.json").read_text(encoding="utf-8"))
            nodes = ", ".join(map(str, policy["fixed_test_nodes"]))
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            detail.append(
                f"| {label} | {seed} | `{nodes}` | {fmt(mae)} | {fmt(rmse)} | {fmt(mape)} | {n_mask} |"
            )
        lines.append(f"| {label} | {len(maes)} | {fmt_pm(maes)} | {fmt_pm(rmses)} | {fmt_pm(mapes)} |")

    section = "\n".join(lines + [""] + detail) + "\n"
    report_text = REPORT.read_text(encoding="utf-8")
    marker = "## `tst_v`：固定全掩码测试节点"
    if marker in report_text:
        report_text = report_text[: report_text.index(marker)].rstrip() + "\n\n" + section
    else:
        report_text = report_text.rstrip() + "\n\n" + section
    REPORT.write_text(report_text, encoding="utf-8")
    print(section)


if __name__ == "__main__":
    main()
