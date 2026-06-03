#!/usr/bin/env python3
"""Aggregate PEMSD7M geo_only / geo_topo with all-node geometric pretraining."""
from __future__ import annotations

import argparse
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

SEEDS = (100, 42, 999, 555, 250)
CONFIGS = (
    ("A_geo_only", "GEO（全节点 geo 预训练）"),
    ("A_geo_topo", "GEO+TOPO（全节点 geo 预训练）"),
)
LOGROOT_DEFAULT = "logs_topomoe/est_geotopo_pemsd7m_geo_allnodes_seed5"
OUT_DEFAULT = "PEMSD7M_geo_allnodes_pretrain_Estimation.md"
BASE_ARGV_NOTE = "`1 0.7 0 <SEED> 1.0 PEMSD7M -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`"

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)


class MetricsEst:
    def __init__(
        self,
        mae: Optional[float] = None,
        rmse: Optional[float] = None,
        mape: Optional[float] = None,
        duration: Optional[str] = None,
    ):
        self.mae, self.rmse, self.mape, self.duration = mae, rmse, mape, duration


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
    u, a = MetricsEst(), MetricsEst()
    for m in _RE_EST.finditer(txt):
        row = MetricsEst(float(m.group(2)), float(m.group(3)), float(m.group(4)))
        if m.group(1) == "tst_u":
            u = row
        else:
            a = row
    dm = _RE_DURATION.search(txt)
    if dm:
        dur = dm.group(1).strip()
        u.duration = dur
        a.duration = dur
    return u, a


def resolve_job(logroot: Path, seed: int, key: str) -> tuple[str, MetricsEst, MetricsEst]:
    p = logroot / "PEMSD7M" / str(seed) / "est" / f"{key}_s{seed}.log"
    if not p.is_file():
        return "未开始", MetricsEst(), MetricsEst()
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if "SCRIPT DURATION" in txt:
        u, a = parse_est_text(txt)
        return "完成", u, a
    u, a = parse_est_text(txt)
    return "进行中", u, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path(LOGROOT_DEFAULT))
    ap.add_argument("--out", type=Path, default=Path(OUT_DEFAULT))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    md: list[str] = []
    md.append("# PEMSD7M Geo 预训练（全节点池）Estimation（5 seeds，无 SCPT）")
    md.append("")
    md.append("- **任务**：Estimation 掩码预测")
    md.append("- **数据集**：PEMSD7M")
    md.append("- **配置**：`GEO only` / `GEO+TOPO`（与主实验相同 MoE 设置）")
    md.append("- **与主实验差异**：`GEO_PRETRAIN_TRAIN_ONLY=0`，几何预训练从**全部节点**抽样，而非仅 `spatialSplit_unseen.i_trn`")
    md.append(f"- **BASE（argv[1]–[20]）**：{BASE_ARGV_NOTE}")
    md.append("- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（**n=5**）")
    md.append("- **运行脚本**：`./run_pemsd7m_notopo_allnodes_6gpu.sh`（Batch B）")
    md.append("- **主指标**：`tst_u` MAE（unseen）")
    md.append(f"- **报告更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("---")
    md.append("")

    n_done = n_run = n_pend = 0
    md.append("## 进度（共 10 项：2 配置 × 5 种子）")
    md.append("")
    md.append("| 配置 | 种子 | 状态 | tst_u MAE | tst_a MAE | 耗时 |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in CONFIGS:
        for s in SEEDS:
            st, uu, aa = resolve_job(logroot, s, key)
            if st == "完成":
                n_done += 1
            elif st == "进行中":
                n_run += 1
            else:
                n_pend += 1
            md.append(
                f"| {label} | {s} | {st} | {fmt(uu.mae)} | {fmt(aa.mae)} | {uu.duration or '—'} |"
            )
    md.append("")
    md.append(f"**进度**：完成 **{n_done}/10**，进行中 **{n_run}**，未开始 **{n_pend}**")
    md.append("")
    md.append("## PEMSD7M 汇总（tst_u MAE，均值 ± 样本标准差）")
    md.append("")
    md.append("| 配置 | n | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in CONFIGS:
        u_mae, u_rmse, u_mape, a_mae = [], [], [], []
        for s in SEEDS:
            st, uu, aa = resolve_job(logroot, s, key)
            if st != "完成" or uu.mae is None:
                continue
            u_mae.append(uu.mae)
            u_rmse.append(uu.rmse)
            u_mape.append(uu.mape)
            a_mae.append(aa.mae)
        m1, sd1, n_ok = agg(u_mae)
        m2, sd2, _ = agg(u_rmse)
        m3, sd3, _ = agg(u_mape)
        m4, sd4, _ = agg(a_mae)
        md.append(
            f"| {label} | {n_ok} | {fmt_pm(m1, sd1)} | {fmt_pm(m2, sd2)} | {fmt_pm(m3, sd3)} | {fmt_pm(m4, sd4)} |"
        )
    md.append("")
    md.append("## 与主报告（train 节点 geo 预训练）对照")
    md.append("")
    md.append("| 配置 | 本表（全节点 geo 预训练）tst_u MAE | 主报告 GEO 预训练池=tst_u MAE |")
    md.append("| --- | --- | --- |")
    main_root = root / "logs_topomoe" / "est_geotopo_3ds_seed5"
    train_labels = {"A_geo_only": "GEO", "A_geo_topo": "GEO+TOPO"}
    for key, label in CONFIGS:
        u_mae = []
        for s in SEEDS:
            st, uu, _ = resolve_job(logroot, s, key)
            if st == "完成" and uu.mae is not None:
                u_mae.append(uu.mae)
        m_all, sd_all, _ = agg(u_mae)
        u_mae_tr = []
        for s in SEEDS:
            p = main_root / "PEMSD7M" / str(s) / "est" / f"{key}_s{s}.log"
            if p.is_file() and "SCRIPT DURATION" in p.read_text(encoding="utf-8", errors="ignore"):
                u, _ = parse_est_text(p.read_text(encoding="utf-8", errors="ignore"))
                if u.mae is not None:
                    u_mae_tr.append(u.mae)
        m_tr, sd_tr, _ = agg(u_mae_tr)
        md.append(
            f"| {train_labels[key]} | {fmt_pm(m_all, sd_all)} | {fmt_pm(m_tr, sd_tr)} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 按种子分项")
    md.append("")
    for s in SEEDS:
        md.append(f"### 种子 `{s}`")
        md.append("")
        md.append("| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for key, label in CONFIGS:
            p = logroot / "PEMSD7M" / str(s) / "est" / f"{key}_s{s}.log"
            st, uu, aa = resolve_job(logroot, s, key)
            rel = f"`{os.path.relpath(p, root).replace(os.sep, '/')}`"
            md.append(
                "| "
                + " | ".join(
                    [
                        label,
                        rel,
                        uu.duration or ("—" if st == "未开始" else st),
                        fmt(uu.mae),
                        fmt(uu.rmse),
                        fmt(uu.mape),
                        fmt(aa.mae),
                    ]
                )
                + " |"
            )
        md.append("")

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
