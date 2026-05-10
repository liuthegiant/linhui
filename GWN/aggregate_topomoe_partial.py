#!/usr/bin/env python3
"""
Generate a small interim markdown report for a subset of seeds.

It parses the same log formats as aggregate_topomoe_5seed.py, but allows selecting seeds.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MetricsEst:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


@dataclass
class MetricsPred:
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None


_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)
_RE_PRED = re.compile(
    r"all pred steps,\s*GraphWaveNet,\s*(test_[ua]),\s*MSE,\s*RMSE,\s*MAE,\s*MAPE,\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
)


LABELS_A = [
    ("A_scpt_only", "SCPT only"),
    ("A_geo_only", "GEO only"),
    ("A_topo_only", "TOPO only"),
    ("A_scpt_geo", "SCPT + GEO (`sparse_moe`)"),
    ("A_scpt_topo", "SCPT + TOPO"),
    ("A_geo_topo", "GEO + TOPO"),
    ("A_scpt_geo_topo", "SCPT + GEO + TOPO"),
]

LABELS_B = [
    ("B_scpt_only", "SCPT only"),
    ("B_geo_only", "GEO only"),
    ("B_topo_only", "TOPO only"),
    ("B_scpt_geo", "SCPT + GEO (`sparse_moe`)"),
    ("B_scpt_topo", "SCPT + TOPO"),
    ("B_geo_topo", "GEO + TOPO"),
    ("B_scpt_geo_topo", "SCPT + GEO + TOPO"),
]


def _fmt(x: Optional[float], nd: int = 6) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _fmt_pm(xs: list[Optional[float]], nd: int = 4) -> tuple[str, int]:
    ys = [x for x in xs if x is not None]
    if not ys:
        return ("—", 0)
    if len(ys) == 1:
        return (f"{ys[0]:.{nd}f}", 1)
    return (f"{statistics.mean(ys):.{nd}f} ± {statistics.stdev(ys):.{nd}f}", len(ys))


def parse_est(p: Path) -> tuple[MetricsEst, MetricsEst]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    u = MetricsEst()
    a = MetricsEst()
    for m in _RE_EST.finditer(txt):
        split = m.group(1)
        row = MetricsEst(mae=float(m.group(2)), rmse=float(m.group(3)), mape=float(m.group(4)))
        if split == "tst_u":
            u = row
        else:
            a = row
    return (u, a)


def parse_pred(p: Path) -> tuple[MetricsPred, MetricsPred]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    u = MetricsPred()
    a = MetricsPred()
    for m in _RE_PRED.finditer(txt):
        split = m.group(1)
        row = MetricsPred(
            mse=float(m.group(2)),
            rmse=float(m.group(3)),
            mae=float(m.group(4)),
            mape=float(m.group(5)),
        )
        if split == "test_u":
            u = row
        else:
            a = row
    return (u, a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path("logs_topomoe/seed5_metrla_neg1"))
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("TOPOMOe_TEMP_seed100_42.md"))
    ap.add_argument(
        "--base_note",
        default="`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    seeds = list(dict.fromkeys(args.seeds))

    md: list[str] = []
    md.append("# TopoMoE（METRLA）临时小表（已完成种子）")
    md.append("")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, seeds))}`")
    md.append(f"- **BASE（argv[1]–[20]）**：{args.base_note}")
    md.append("")
    md.append("## A. Estimation（按配置汇总：tst_u / tst_a）")
    md.append("")
    md.append("| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in LABELS_A:
        u_mae: list[Optional[float]] = []
        u_rmse: list[Optional[float]] = []
        u_mape: list[Optional[float]] = []
        a_mae: list[Optional[float]] = []
        a_rmse: list[Optional[float]] = []
        a_mape: list[Optional[float]] = []
        for s in seeds:
            p = logroot / str(s) / "est" / f"{key}_s{s}.log"
            if not p.is_file():
                continue
            uu, aa = parse_est(p)
            u_mae.append(uu.mae)
            u_rmse.append(uu.rmse)
            u_mape.append(uu.mape)
            a_mae.append(aa.mae)
            a_rmse.append(aa.rmse)
            a_mape.append(aa.mape)
        pm1, n1 = _fmt_pm(u_mae)
        pm2, _ = _fmt_pm(u_rmse)
        pm3, _ = _fmt_pm(u_mape)
        pm4, _ = _fmt_pm(a_mae)
        pm5, _ = _fmt_pm(a_rmse)
        pm6, _ = _fmt_pm(a_mape)
        md.append(f"| {label} | {n1} | {pm1} | {pm2} | {pm3} | {pm4} | {pm5} | {pm6} |")

    md.append("")
    md.append("## B. Forecasting（按配置汇总：test_u / test_a，all pred steps）")
    md.append("")
    md.append("| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in LABELS_B:
        cols: dict[str, list[Optional[float]]] = {
            "u_mse": [],
            "u_rmse": [],
            "u_mae": [],
            "u_mape": [],
            "a_mse": [],
            "a_rmse": [],
            "a_mae": [],
            "a_mape": [],
        }
        for s in seeds:
            p = logroot / str(s) / "pred" / f"{key}_s{s}.log"
            if not p.is_file():
                continue
            uu, aa = parse_pred(p)
            cols["u_mse"].append(uu.mse)
            cols["u_rmse"].append(uu.rmse)
            cols["u_mae"].append(uu.mae)
            cols["u_mape"].append(uu.mape)
            cols["a_mse"].append(aa.mse)
            cols["a_rmse"].append(aa.rmse)
            cols["a_mae"].append(aa.mae)
            cols["a_mape"].append(aa.mape)
        pm_u_mse, n_ok = _fmt_pm(cols["u_mse"])
        pm_u_rmse, _ = _fmt_pm(cols["u_rmse"])
        pm_u_mae, _ = _fmt_pm(cols["u_mae"])
        pm_u_mape, _ = _fmt_pm(cols["u_mape"])
        pm_a_mse, _ = _fmt_pm(cols["a_mse"])
        pm_a_rmse, _ = _fmt_pm(cols["a_rmse"])
        pm_a_mae, _ = _fmt_pm(cols["a_mae"])
        pm_a_mape, _ = _fmt_pm(cols["a_mape"])
        md.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(n_ok),
                    pm_u_mse,
                    pm_u_rmse,
                    pm_u_mae,
                    pm_u_mape,
                    pm_a_mse,
                    pm_a_rmse,
                    pm_a_mae,
                    pm_a_mape,
                ]
            )
            + " |"
        )

    md.append("")
    md.append("## 明细日志（逐种子）")
    md.append("")
    for s in seeds:
        md.append(f"### 种子 `{s}`")
        md.append("")
        md.append("**A. Estimation**")
        md.append("")
        md.append("| 配置 | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE | 日志 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in LABELS_A:
            p = logroot / str(s) / "est" / f"{key}_s{s}.log"
            uu, aa = (MetricsEst(), MetricsEst())
            if p.is_file():
                uu, aa = parse_est(p)
            md.append(
                "| "
                + " | ".join(
                    [
                        label,
                        _fmt(uu.mae),
                        _fmt(uu.rmse),
                        _fmt(uu.mape),
                        _fmt(aa.mae),
                        _fmt(aa.rmse),
                        _fmt(aa.mape),
                        f"`{os.path.relpath(p, root).replace(os.sep, '/')}`" if p.is_file() else "—",
                    ]
                )
                + " |"
            )
        md.append("")
        md.append("**B. Forecasting (all pred steps)**")
        md.append("")
        md.append("| 配置 | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE | 日志 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in LABELS_B:
            p = logroot / str(s) / "pred" / f"{key}_s{s}.log"
            uu2, aa2 = (MetricsPred(), MetricsPred())
            if p.is_file():
                uu2, aa2 = parse_pred(p)
            md.append(
                "| "
                + " | ".join(
                    [
                        label,
                        _fmt(uu2.mse),
                        _fmt(uu2.rmse),
                        _fmt(uu2.mae),
                        _fmt(uu2.mape),
                        _fmt(aa2.mse),
                        _fmt(aa2.rmse),
                        _fmt(aa2.mae),
                        _fmt(aa2.mape),
                        f"`{os.path.relpath(p, root).replace(os.sep, '/')}`" if p.is_file() else "—",
                    ]
                )
                + " |"
            )
        md.append("")

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()

