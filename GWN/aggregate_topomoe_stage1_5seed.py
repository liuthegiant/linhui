#!/usr/bin/env python3
"""
Aggregate Stage-1 5-seed sweep logs under:
  logs_topomoe/stage1_seed5_imgbase/<seed>/{est,pred}/*.log

Outputs a markdown report with mean ± sample stdev across 5 seeds,
plus per-seed tables.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SEEDS = (100, 42, 999, 555, 250)
LOG_SUBDIR_DEFAULT = "logs_topomoe/stage1_seed5_imgbase"
OUT_DEFAULT = "TOPOMOe_STAGE1_REPORT_imgbase.md"
BASE_ARGV_NOTE = "`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`"


@dataclass
class MetricsEst:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    duration: Optional[str] = None


@dataclass
class MetricsPred:
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    duration: Optional[str] = None


_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
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


def fmt(x: Optional[float], nd: int = 6) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_pm(mean: Optional[float], stdev: Optional[float], nd: int = 4) -> str:
    if mean is None:
        return "—"
    if stdev is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {stdev:.{nd}f}"


def agg(vals: list[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return (None, None, 0)
    if len(xs) == 1:
        return (xs[0], None, 1)
    return (statistics.mean(xs), statistics.stdev(xs), len(xs))


def parse_est_log(p: Path) -> tuple[MetricsEst, MetricsEst]:
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
    dur_m = _RE_DURATION.search(txt)
    if dur_m:
        d = dur_m.group(1).strip()
        u.duration = d
        a.duration = d
    return (u, a)


def parse_pred_log(p: Path) -> tuple[MetricsPred, MetricsPred]:
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
    dur_m = _RE_DURATION.search(txt)
    if dur_m:
        d = dur_m.group(1).strip()
        u.duration = d
        a.duration = d
    return (u, a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path(LOG_SUBDIR_DEFAULT))
    ap.add_argument("--out", type=Path, default=Path(OUT_DEFAULT))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    def p_est(seed: int, key: str) -> Path:
        return logroot / str(seed) / "est" / f"{key}_s{seed}.log"

    def p_pred(seed: int, key: str) -> Path:
        return logroot / str(seed) / "pred" / f"{key}_s{seed}.log"

    md: list[str] = []
    md.append("# TopoMoE（METRLA）Stage-1：五随机种子汇总（截图 BASE 参数）")
    md.append("")
    md.append(f"- **BASE（argv[1]–[20]）**：{BASE_ARGV_NOTE}")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（**n=5**）")
    md.append(f"- **一键运行**：`./run_topomoe_stage1_5seed.sh`")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## A2. Estimation：n=5 汇总（均值 ± 样本标准差）")
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
        for s in SEEDS:
            p = p_est(s, key)
            if not p.is_file():
                continue
            uu, aa = parse_est_log(p)
            u_mae.append(uu.mae)
            u_rmse.append(uu.rmse)
            u_mape.append(uu.mape)
            a_mae.append(aa.mae)
            a_rmse.append(aa.rmse)
            a_mape.append(aa.mape)
        m1, sd1, n_ok = agg(u_mae)
        m2, sd2, _ = agg(u_rmse)
        m3, sd3, _ = agg(u_mape)
        m4, sd4, _ = agg(a_mae)
        m5, sd5, _ = agg(a_rmse)
        m6, sd6, _ = agg(a_mape)
        md.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(n_ok),
                    fmt_pm(m1, sd1),
                    fmt_pm(m2, sd2),
                    fmt_pm(m3, sd3),
                    fmt_pm(m4, sd4),
                    fmt_pm(m5, sd5),
                    fmt_pm(m6, sd6),
                ]
            )
            + " |"
        )
    md.append("")

    md.append("## B2. Forecasting：n=5 汇总（均值 ± 样本标准差，all pred steps）")
    md.append("")
    md.append("| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in LABELS_B:
        cols: dict[str, list[Optional[float]]] = {k: [] for k in ["u_mse","u_rmse","u_mae","u_mape","a_mse","a_rmse","a_mae","a_mape"]}
        for s in SEEDS:
            p = p_pred(s, key)
            if not p.is_file():
                continue
            uu, aa = parse_pred_log(p)
            cols["u_mse"].append(uu.mse)
            cols["u_rmse"].append(uu.rmse)
            cols["u_mae"].append(uu.mae)
            cols["u_mape"].append(uu.mape)
            cols["a_mse"].append(aa.mse)
            cols["a_rmse"].append(aa.rmse)
            cols["a_mae"].append(aa.mae)
            cols["a_mape"].append(aa.mape)
        def pm(k: str) -> tuple[str, int]:
            m, sd, n_ok = agg(cols[k])
            return (fmt_pm(m, sd), n_ok)
        u_mse, n_ok = pm("u_mse")
        u_rmse, _ = pm("u_rmse")
        u_mae, _ = pm("u_mae")
        u_mape, _ = pm("u_mape")
        a_mse, _ = pm("a_mse")
        a_rmse, _ = pm("a_rmse")
        a_mae, _ = pm("a_mae")
        a_mape, _ = pm("a_mape")
        md.append(
            "| "
            + " | ".join([label, str(n_ok), u_mse, u_rmse, u_mae, u_mape, a_mse, a_rmse, a_mae, a_mape])
            + " |"
        )
    md.append("")

    md.append("---")
    md.append("")
    md.append("## A1. Estimation：按种子分项")
    md.append("")
    for s in SEEDS:
        md.append(f"### 种子 `{s}`")
        md.append("")
        md.append("| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in LABELS_A:
            p = p_est(s, key)
            uu, aa = (MetricsEst(), MetricsEst())
            if p.is_file():
                uu, aa = parse_est_log(p)
            md.append(
                "| "
                + " | ".join(
                    [
                        label,
                        f"`{os.path.relpath(p, root).replace(os.sep, '/')}`",
                        uu.duration or aa.duration or "—",
                        fmt(uu.mae),
                        fmt(uu.rmse),
                        fmt(uu.mape),
                        fmt(aa.mae),
                        fmt(aa.rmse),
                        fmt(aa.mape),
                    ]
                )
                + " |"
            )
        md.append("")

    md.append("## B1. Forecasting：按种子分项（all pred steps）")
    md.append("")
    for s in SEEDS:
        md.append(f"### 种子 `{s}`")
        md.append("")
        md.append("| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in LABELS_B:
            p = p_pred(s, key)
            uu, aa = (MetricsPred(), MetricsPred())
            if p.is_file():
                uu, aa = parse_pred_log(p)
            md.append(
                "| "
                + " | ".join(
                    [
                        label,
                        f"`{os.path.relpath(p, root).replace(os.sep, '/')}`",
                        uu.duration or aa.duration or "—",
                        fmt(uu.mse),
                        fmt(uu.rmse),
                        fmt(uu.mae),
                        fmt(uu.mape),
                        fmt(aa.mse),
                        fmt(aa.rmse),
                        fmt(aa.mae),
                        fmt(aa.mape),
                    ]
                )
                + " |"
            )
        md.append("")

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()

