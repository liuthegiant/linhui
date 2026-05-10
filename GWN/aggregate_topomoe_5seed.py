#!/usr/bin/env python3
"""
Parse logs from logs_topomoe/seed5_metrla_neg1/<seed>/{est,pred}/*.log and emit markdown tables.
Summaries use statistics.mean / statistics.stdev with sample stdev (n-1 denominator), n=5.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SEEDS_EXPECTED = (100, 42, 999, 555, 250)
# Subdir under GWN/ where run_topomoe_5seed_sweep.sh writes logs (must match script LOGROOT)
SEED_LOG_SUBDIR = "logs_topomoe/seed5_metrla_neg1"
# Default output (new file; does not overwrite legacy TOPOMOe_RUN_REPORT.md)
REPORT_OUT_DEFAULT = "TOPOMOe_RUN_REPORT_METRLA_neg1.md"
BASE_ARGV_NOTE = (
    "`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.001 100 100 0 0.01 1 320`"
)


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
_RE_SEED_SUFFIX = re.compile(r"^(.*)_s(\d+)\.log$")


def fmt(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def fmt_pm(mean: Optional[float], stdev: Optional[float], nd: int = 4) -> str:
    if mean is None:
        return "—"
    if stdev is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {stdev:.{nd}f}"


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
    dur = dur_m.group(1).strip() if dur_m else None
    if dur:
        u.duration = dur
        a.duration = dur
    return (u, a)


def collect_est(logroot: Path) -> dict[int, dict[str, tuple[MetricsEst, MetricsEst]]]:
    by_seed: dict[int, dict[str, tuple[MetricsEst, MetricsEst]]] = {}
    for seed in SEEDS_EXPECTED:
        by_seed[seed] = {}
        edir = logroot / str(seed) / "est"
        if not edir.is_dir():
            continue
        for p in sorted(edir.glob("*.log")):
            m = _RE_SEED_SUFFIX.match(p.name)
            if not m:
                continue
            key, fs = m.group(1), int(m.group(2))
            if fs != seed:
                continue
            uu, ua = parse_est_log(p)
            by_seed[seed][key] = (uu, ua)
    return by_seed


def collect_pred(logroot: Path) -> dict[int, dict[str, tuple[MetricsPred, MetricsPred]]]:
    by_seed: dict[int, dict[str, tuple[MetricsPred, MetricsPred]]] = {}
    for seed in SEEDS_EXPECTED:
        by_seed[seed] = {}
        pdir = logroot / str(seed) / "pred"
        if not pdir.is_dir():
            continue
        for p in sorted(pdir.glob("*.log")):
            m = _RE_SEED_SUFFIX.match(p.name)
            if not m:
                continue
            key, fs = m.group(1), int(m.group(2))
            if fs != seed:
                continue
            uu, ua = parse_pred_log(p)
            by_seed[seed][key] = (uu, ua)
    return by_seed


def agg(vals: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    xs = [v for v in vals if v is not None]
    if len(xs) < 2:
        mn = xs[0] if xs else None
        return (mn, None)
    return (statistics.mean(xs), statistics.stdev(xs))


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


def md_est_per_seed(by_seed: dict[int, dict[str, tuple[MetricsEst, MetricsEst]]]) -> list[str]:
    lines: list[str] = []
    lines.append("## A1. Estimation：按种子分项（完整指标）")
    lines.append("")
    for seed in SEEDS_EXPECTED:
        lines.append(f"### 种子 `{seed}`")
        lines.append("")
        lines.append(
            "| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        dmap = by_seed.get(seed, {})
        for key, label in LABELS_A:
            pth = f"`{SEED_LOG_SUBDIR}/{seed}/est/{key}_s{seed}.log`"
            uu, ua = dmap.get(key, (MetricsEst(), MetricsEst()))
            dur = uu.duration or ua.duration or "—"
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        pth,
                        dur if dur else "—",
                        fmt(uu.mae),
                        fmt(uu.rmse),
                        fmt(uu.mape),
                        fmt(ua.mae),
                        fmt(ua.rmse),
                        fmt(ua.mape),
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def md_est_summary(by_seed: dict[int, dict[str, tuple[MetricsEst, MetricsEst]]]) -> list[str]:
    lines: list[str] = []
    lines.append("## A2. Estimation：**n=5 汇总（均值 ± 样本标准差）**")
    lines.append("")
    lines.append("*种子集合*：`100, 42, 999, 555, 250`，与仓库内 METRLA 五种子文档一致。*标准差*：`statistics.stdev`（分母 **n−1=4**）。")
    lines.append("")
    lines.append("| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in LABELS_A:
        u_mae = []
        u_rmse = []
        u_mape = []
        a_mae = []
        a_rmse = []
        a_mape = []
        n_ok = 0
        for seed in SEEDS_EXPECTED:
            dmap = by_seed.get(seed, {})
            pair = dmap.get(key)
            if not pair:
                continue
            uu, ua = pair
            if uu.mae is not None and ua.mae is not None:
                n_ok += 1
            u_mae.append(uu.mae)
            u_rmse.append(uu.rmse)
            u_mape.append(uu.mape)
            a_mae.append(ua.mae)
            a_rmse.append(ua.rmse)
            a_mape.append(ua.mape)
        m1, s1 = agg(u_mae)
        m2, s2 = agg(u_rmse)
        m3, s3 = agg(u_mape)
        m4, s4 = agg(a_mae)
        m5, s5 = agg(a_rmse)
        m6, s6 = agg(a_mape)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(n_ok),
                    fmt_pm(m1, s1),
                    fmt_pm(m2, s2),
                    fmt_pm(m3, s3),
                    fmt_pm(m4, s4),
                    fmt_pm(m5, s5),
                    fmt_pm(m6, s6),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def md_pred_per_seed(by_seed: dict[int, dict[str, tuple[MetricsPred, MetricsPred]]]) -> list[str]:
    lines: list[str] = []
    lines.append("## B1. Forecasting：按种子分项（`all pred steps`）")
    lines.append("")
    for seed in SEEDS_EXPECTED:
        lines.append(f"### 种子 `{seed}`")
        lines.append("")
        lines.append(
            "| 配置 | 日志路径 | SCRIPT DURATION | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        dmap = by_seed.get(seed, {})
        for key, label in LABELS_B:
            pth = f"`{SEED_LOG_SUBDIR}/{seed}/pred/{key}_s{seed}.log`"
            uu, ua = dmap.get(key, (MetricsPred(), MetricsPred()))
            dur = uu.duration or ua.duration or "—"
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        pth,
                        dur,
                        fmt(uu.mse),
                        fmt(uu.rmse),
                        fmt(uu.mae),
                        fmt(uu.mape),
                        fmt(ua.mse),
                        fmt(ua.rmse),
                        fmt(ua.mae),
                        fmt(ua.mape),
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def md_pred_summary(by_seed: dict[int, dict[str, tuple[MetricsPred, MetricsPred]]]) -> list[str]:
    lines: list[str] = []
    lines.append("## B2. Forecasting：**n=5 汇总（均值 ± 样本标准差）**")
    lines.append("")
    lines.append(
        "| 配置 | n | test_u MSE | test_u RMSE | test_u MAE | test_u MAPE | test_a MSE | test_a RMSE | test_a MAE | test_a MAPE |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in LABELS_B:
        cols: dict[str, list[Optional[float]]] = defaultdict(list)
        n_ok = 0
        for seed in SEEDS_EXPECTED:
            dmap = by_seed.get(seed, {})
            pair = dmap.get(key)
            if not pair:
                continue
            uu, ua = pair
            if uu.mse is not None and ua.mse is not None:
                n_ok += 1
            cols["u_mse"].append(uu.mse)
            cols["u_rmse"].append(uu.rmse)
            cols["u_mae"].append(uu.mae)
            cols["u_mape"].append(uu.mape)
            cols["a_mse"].append(ua.mse)
            cols["a_rmse"].append(ua.rmse)
            cols["a_mae"].append(ua.mae)
            cols["a_mape"].append(ua.mape)
        def pm(k: str) -> str:
            m, sd = agg(cols[k])
            return fmt_pm(m, sd)

        lines.append(
            "| "
            + " | ".join(
                [label, str(n_ok), pm("u_mse"), pm("u_rmse"), pm("u_mae"), pm("u_mape"), pm("a_mse"), pm("a_rmse"), pm("a_mae"), pm("a_mape")]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path(SEED_LOG_SUBDIR))
    ap.add_argument("--out", type=Path, default=Path(REPORT_OUT_DEFAULT))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    by_est = collect_est(logroot)
    by_pred = collect_pred(logroot)

    md: list[str] = []
    md.append("# TopoMoE（METRLA）**五随机种子**实验汇总（METRLA `-1` / 100+100 epoch BASE）")
    md.append("")
    md.append("- **BASE（估计与预测共用 argv[1]–[20]）**：" + BASE_ARGV_NOTE)
    md.append("- **日志根目录**：`" + os.path.relpath(logroot, root).replace(os.sep, "/") + "`")
    md.append("- **种子**：`" + ", ".join(map(str, SEEDS_EXPECTED)) + "`（**n=5**）")
    md.append("- **一键扫种**：`./run_topomoe_5seed_sweep.sh`（每轮：`A` 7 并行 → `B` 7 并行，占用 GPU **0–6**）")
    md.append(
        "- **生成/更新本报告**：`python3 aggregate_topomoe_5seed.py --out "
        + REPORT_OUT_DEFAULT
        + "`（默认日志目录为 `"
        + SEED_LOG_SUBDIR
        + "`）"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.extend(md_est_summary(by_est))
    md.extend(md_pred_summary(by_pred))
    md.append("---")
    md.append("")
    md.extend(md_est_per_seed(by_est))
    md.extend(md_pred_per_seed(by_pred))

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
