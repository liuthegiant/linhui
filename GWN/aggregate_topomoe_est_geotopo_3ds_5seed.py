#!/usr/bin/env python3
"""Aggregate geo/topo estimation logs (no SCPT) for PEMSBAY, PEMSD7M, METRLA."""
from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

SEEDS = (100, 42, 999, 555, 250)
PEMSD7M_SEEDS = (100, 42, 999, 555, 250, 88, 66, 233, 38, 432)
DATASETS = ("PEMSBAY", "PEMSD7M", "METRLA")
CONFIGS = (
    ("A_no_pretrain", "无预训练"),
    ("A_geo_only", "GEO"),
    ("A_topo_only", "TOPO"),
    ("A_geo_topo", "GEO+TOPO"),
)
LOGROOT_DEFAULT = "logs_topomoe/est_geotopo_3ds_seed5"
OUT_DEFAULT = "PEMSBAY_PEMSD7M_METRLA_geo_topo_Estimation.md"
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


def parse_est_text(txt: str) -> tuple[MetricsEst, MetricsEst]:
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


def parse_est_log(p: Path) -> tuple[MetricsEst, MetricsEst]:
    if not p.is_file():
        return MetricsEst(), MetricsEst()
    return parse_est_text(p.read_text(encoding="utf-8", errors="ignore"))


def moe_tag(dataset: str, key: str, seed: int) -> str:
    ds = dataset.lower()
    if key == "A_no_pretrain":
        return f"est_geotopo_{ds}_no_pretrain_s{seed}"
    cfg = key.replace("A_", "").replace("_", "_")
    return f"est_geotopo_{ds}_{cfg}_s{seed}"


def bundle_logs(logroot: Path, dataset: str) -> list[Path]:
    if dataset == "PEMSD7M":
        names = ("pemsd7m_gpu34.log",)
    elif dataset == "METRLA":
        names = ("metrla_gpu6.log", "remaining_gpu134.log")
    else:
        return []
    return [logroot / n for n in names if (logroot / n).is_file()]


def _run_header_match(dataset: str, key: str, seed: int, blk: str) -> bool:
    cfg = key.replace("A_", "")
    head = blk[:300]
    if not head.lstrip().startswith("[run]"):
        return False
    if dataset == "METRLA":
        return (
            "METRLA" in head
            and f"seed={seed}" in head
            and ("no_pretrain" in head or cfg.replace("_", "") in head.replace(" ", ""))
        )
    return dataset in head and f"seed={seed}" in head and cfg in head


def parse_from_bundles(logroot: Path, dataset: str, seed: int, key: str) -> tuple[MetricsEst, MetricsEst]:
    tag = moe_tag(dataset, key, seed)
    best_u, best_a = MetricsEst(), MetricsEst()
    for bp in bundle_logs(logroot, dataset):
        txt = bp.read_text(encoding="utf-8", errors="ignore")
        parts = re.split(r"(?=\[run\])", txt)
        for blk in parts:
            if tag not in blk or not _run_header_match(dataset, key, seed, blk):
                continue
            if "SCRIPT DURATION" not in blk:
                continue
            u, a = parse_est_text(blk)
            if u.mae is not None:
                best_u, best_a = u, a
    return best_u, best_a


def resolve_job(
    logroot: Path, dataset: str, seed: int, key: str
) -> tuple[str, MetricsEst, MetricsEst]:
    p = logroot / dataset / str(seed) / "est" / f"{key}_s{seed}.log"
    if p.is_file():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "SCRIPT DURATION" in txt:
            u, a = parse_est_text(txt)
            return "完成", u, a
        fr = FROZEN_METRICS.get((dataset, key, seed))
        if fr is not None:
            return "完成(冻结)", fr[0], fr[1]
        bu, ba = parse_from_bundles(logroot, dataset, seed, key)
        if bu.mae is not None and bu.duration:
            return "完成(备份)", bu, ba
        u, a = parse_est_text(txt)
        return "进行中", u, a
    fr = FROZEN_METRICS.get((dataset, key, seed))
    if fr is not None:
        return "完成(冻结)", fr[0], fr[1]
    bu, ba = parse_from_bundles(logroot, dataset, seed, key)
    if bu.mae is not None and bu.duration:
        return "完成(备份)", bu, ba
    return "未开始", MetricsEst(), MetricsEst()


# Metrics recovered from per-seed logs / pemsd7m_gpu34 before scheduler overwrote files.
FROZEN_METRICS: dict[tuple[str, str, int], tuple[MetricsEst, MetricsEst]] = {
    ("PEMSD7M", "A_geo_only", 999): (
        MetricsEst(1.835812, 2.956423, 2.348643, "0:29:50.405957"),
        MetricsEst(1.697671, 2.719906, 2.314945, "0:29:50.405957"),
    ),
}

SUPPLEMENT_JOBS = (
    ("PEMSBAY", "A_geo_only", "GEO"),
    ("PEMSBAY", "A_geo_topo", "GEO+TOPO"),
    ("PEMSD7M", "A_geo_only", "GEO"),
    ("PEMSD7M", "A_geo_topo", "GEO+TOPO"),
    ("METRLA", "A_no_pretrain", "无预训练"),
)


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
    md.append("# PEMSBAY + PEMSD7M + METRLA Geo/Topo Estimation（5 seeds，无 SCPT）")
    md.append("")
    md.append("- **任务**：Estimation 掩码预测")
    md.append("- **配置**：`无预训练 / GEO only / TOPO only / GEO+TOPO`（不含 SCPT）")
    md.append(f"- **BASE（argv[1]–[20]）**：{BASE_ARGV_NOTE}")
    md.append("- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（**n=5**）")
    md.append("- **运行脚本**：`./run_topomoe_est_geotopo_3ds_6gpu.sh`")
    md.append("- **主指标**：`tst_u` MAE（unseen）；`tst_a` 仅作参考")
    md.append(f"- **报告更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 补充批次进度（RUN_SCOPE=supplement，共 25 项）")
    md.append("")
    n_done = n_run = n_pend = 0
    md.append("| 数据集 | 配置 | 种子 | 状态 | tst_u MAE | tst_a MAE | 耗时 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for dataset, key, label in SUPPLEMENT_JOBS:
        for s in SEEDS:
            st, uu, aa = resolve_job(logroot, dataset, s, key)
            if st.startswith("完成"):
                n_done += 1
                dur = uu.duration or "—"
                md.append(
                    f"| {dataset} | {label} | {s} | {st} | {fmt(uu.mae)} | {fmt(aa.mae)} | {dur} |"
                )
            elif st == "进行中":
                n_run += 1
                u_txt = fmt(uu.mae) if uu.mae is not None else "—"
                a_txt = fmt(aa.mae) if aa.mae is not None else "—"
                md.append(f"| {dataset} | {label} | {s} | 进行中 | {u_txt} | {a_txt} | — |")
            else:
                n_pend += 1
                md.append(f"| {dataset} | {label} | {s} | 未开始 | — | — | — |")
    md.append("")
    md.append(
        f"**进度**：完成 **{n_done}/25**（含备份日志），进行中 **{n_run}**，未开始 **{n_pend}**"
    )
    md.append("")
    md.append("## 补充批次已完成项汇总（tst_u MAE，均值 ± 标准差）")
    md.append("")
    md.append("| 数据集 | 配置 | n | tst_u MAE | tst_u RMSE | tst_a MAE |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for dataset, key, label in SUPPLEMENT_JOBS:
        u_mae: list[Optional[float]] = []
        u_rmse: list[Optional[float]] = []
        a_mae: list[Optional[float]] = []
        for s in SEEDS:
            st, uu, aa = resolve_job(logroot, dataset, s, key)
            if not st.startswith("完成") or uu.mae is None:  # 完成 / 完成(备份) / 完成(冻结)
                continue
            u_mae.append(uu.mae)
            u_rmse.append(uu.rmse)
            a_mae.append(aa.mae)
        m1, sd1, n_ok = agg(u_mae)
        m2, sd2, _ = agg(u_rmse)
        m4, sd4, _ = agg(a_mae)
        md.append(
            f"| {dataset} | {label} | {n_ok} | {fmt_pm(m1, sd1)} | {fmt_pm(m2, sd2)} | {fmt_pm(m4, sd4)} |"
        )
    md.append("")
    md.append("---")
    md.append("")

    for dataset in DATASETS:
        ds_seeds = PEMSD7M_SEEDS if dataset == "PEMSD7M" else SEEDS
        n_label = len(ds_seeds)
        md.append(f"## {dataset}：n={n_label} 汇总（均值 ± 样本标准差）")
        if dataset == "PEMSD7M":
            md.append("")
            md.append(
                f"- **种子**：`{', '.join(map(str, ds_seeds))}`（原 5 + 扩种 5：`88, 66, 233, 38, 432`）"
            )
        md.append("")
        md.append("| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in CONFIGS:
            u_mae, u_rmse, u_mape = [], [], []
            a_mae, a_rmse, a_mape = [], [], []
            for s in ds_seeds:
                st, uu, aa = resolve_job(logroot, dataset, s, key)
                if not st.startswith("完成") or uu.mae is None:
                    continue
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
        for s in ds_seeds:
            md.append(f"### 种子 `{s}`")
            md.append("")
            md.append("| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
            md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for key, label in CONFIGS:
                p = p_est(dataset, s, key)
                st, uu, aa = resolve_job(logroot, dataset, s, key)
                note = f"`{os.path.relpath(p, root).replace(os.sep, '/')}`"
                if st == "完成(备份)":
                    note += " (备份)"
                md.append(
                    "| "
                    + " | ".join(
                        [
                            label,
                            note,
                            uu.duration or aa.duration or ("—" if st == "未开始" else st),
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
