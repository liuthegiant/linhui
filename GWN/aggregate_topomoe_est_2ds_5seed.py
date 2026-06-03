#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SEEDS = (100, 42, 999, 555, 250)
DATASETS = ("PEMSBAY", "PEMSD7M")
CONFIGS = (
    ("A_no_pretrain", "无预训练"),
    ("A_scpt_only", "SCPT"),
    ("A_topo_only", "TOPO"),
    ("A_scpt_topo", "SCPT + TOPO"),
)
LOGROOT_DEFAULT = "logs_topomoe/est_2ds_seed5_imgbase"
OUT_DEFAULT = "PEMSBAY_PEMSD7M_topo_Estimation.md"
BASE_ARGV_NOTE = "`1 0.7 0 <SEED> 1.0 <DATASET> -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`"

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)


@dataclass
class MetricsEst:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    duration: Optional[str] = None


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
    dm = _RE_DURATION.search(txt)
    if dm:
        dur = dm.group(1).strip()
        u.duration = dur
        a.duration = dur
    return u, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path(LOGROOT_DEFAULT))
    ap.add_argument("--out", type=Path, default=Path(OUT_DEFAULT))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    def p_est(dataset: str, seed: int, key: str) -> Path:
        return logroot / dataset / str(seed) / "est" / f"{key}_s{seed}.log"

    md: list[str] = []
    md.append("# PEMSBAY + PEMSD7M Topo Estimation（5 seeds）")
    md.append("")
    md.append("- **任务**：Estimation，仅 `无预训练 / SCPT / TOPO / SCPT+TOPO`")
    md.append(f"- **BASE（argv[1]–[20]）**：{BASE_ARGV_NOTE}")
    md.append("- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`")
    md.append("- **无预训练设置**：将 `argv[1]`（`IS_PRETRN`）设为 `0`；其余配置与 Stage-1 保持一致")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（**n=5**）")
    md.append("- **运行脚本**：`./run_topomoe_est_2ds_5seed_8gpu.sh`")
    md.append("")
    md.append("---")
    md.append("")

    for dataset in DATASETS:
        md.append(f"## {dataset}：n=5 汇总（均值 ± 样本标准差）")
        md.append("")
        md.append("| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in CONFIGS:
            u_mae: list[Optional[float]] = []
            u_rmse: list[Optional[float]] = []
            u_mape: list[Optional[float]] = []
            a_mae: list[Optional[float]] = []
            a_rmse: list[Optional[float]] = []
            a_mape: list[Optional[float]] = []
            for s in SEEDS:
                p = p_est(dataset, s, key)
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

        md.append(f"## {dataset}：按种子分项")
        md.append("")
        for s in SEEDS:
            md.append(f"### 种子 `{s}`")
            md.append("")
            md.append("| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
            md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for key, label in CONFIGS:
                p = p_est(dataset, s, key)
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

        md.append("---")
        md.append("")

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
