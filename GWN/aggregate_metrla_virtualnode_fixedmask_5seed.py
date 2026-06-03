#!/usr/bin/env python3
"""Aggregate METRLA virtual-node fixedmask estimation logs."""
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
CONFIGS = (
    ("A_no_pretrain", "无预训练"),
    ("A_topo_only", "TOPO"),
    ("A_geo_only", "GEO"),
    ("A_geo_topo", "GEO+TOPO"),
)
LOGROOT_DEFAULT = "logs_topomoe/est_metrla_virtualnode_fixedmask_seed5"
OUT_DEFAULT = "METRLA_virtualnode_fixedmask_Estimation.md"

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)


@dataclass
class Metrics:
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
        return None, None, 0
    if len(xs) == 1:
        return xs[0], None, 1
    return statistics.mean(xs), statistics.stdev(xs), len(xs)


def parse_text(txt: str) -> tuple[Metrics, Metrics]:
    u, a = Metrics(), Metrics()
    for m in _RE_EST.finditer(txt):
        row = Metrics(float(m.group(2)), float(m.group(3)), float(m.group(4)))
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


def resolve(logroot: Path, seed: int, key: str) -> tuple[str, Metrics, Metrics]:
    p = logroot / "METRLA" / str(seed) / "est" / f"{key}_s{seed}.log"
    if not p.is_file():
        return "未开始", Metrics(), Metrics()
    txt = p.read_text(encoding="utf-8", errors="ignore")
    u, a = parse_text(txt)
    if "SCRIPT DURATION" in txt:
        return "完成", u, a
    return "进行中", u, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path(LOGROOT_DEFAULT))
    ap.add_argument("--out", type=Path, default=Path(OUT_DEFAULT))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    md: list[str] = []
    md.append("# METRLA Virtual-Node FixedMask Estimation（5 seeds，无 SCPT）")
    md.append("")
    md.append("- **任务**：Estimation 掩码预测")
    md.append("- **数据集**：METRLA")
    md.append("- **掩码策略**：固定一部分整节点 100% 掩码，模拟无历史数据的虚拟节点；其余节点沿用原随机点掩码")
    md.append("- **配置**：`无预训练 / TOPO only / GEO only / GEO+TOPO`")
    md.append("- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`")
    md.append("- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（n=5）")
    md.append("- **运行脚本**：`./run_metrla_virtualnode_fixedmask_7gpu.sh`")
    md.append("- **主指标**：`tst_u` MAE（unseen）")
    md.append(f"- **报告更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    done = running = pending = 0
    md.append("## 进度")
    md.append("")
    md.append("| 配置 | 种子 | 状态 | tst_u MAE | tst_a MAE | 耗时 |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in CONFIGS:
        for seed in SEEDS:
            st, u, a = resolve(logroot, seed, key)
            done += int(st == "完成")
            running += int(st == "进行中")
            pending += int(st == "未开始")
            md.append(f"| {label} | {seed} | {st} | {fmt(u.mae)} | {fmt(a.mae)} | {u.duration or '—'} |")
    md.append("")
    md.append(f"**进度**：完成 **{done}/20**，进行中 **{running}**，未开始 **{pending}**")
    md.append("")

    md.append("## 汇总（均值 ± 样本标准差）")
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
        for seed in SEEDS:
            st, u, a = resolve(logroot, seed, key)
            if st != "完成" or u.mae is None:
                continue
            u_mae.append(u.mae)
            u_rmse.append(u.rmse)
            u_mape.append(u.mape)
            a_mae.append(a.mae)
            a_rmse.append(a.rmse)
            a_mape.append(a.mape)
        m1, s1, n = agg(u_mae)
        m2, s2, _ = agg(u_rmse)
        m3, s3, _ = agg(u_mape)
        m4, s4, _ = agg(a_mae)
        m5, s5, _ = agg(a_rmse)
        m6, s6, _ = agg(a_mape)
        md.append(f"| {label} | {n} | {fmt_pm(m1, s1)} | {fmt_pm(m2, s2)} | {fmt_pm(m3, s3)} | {fmt_pm(m4, s4)} | {fmt_pm(m5, s5)} | {fmt_pm(m6, s6)} |")
    md.append("")

    md.append("## 按种子分项")
    md.append("")
    for seed in SEEDS:
        md.append(f"### 种子 `{seed}`")
        md.append("")
        md.append("| 配置 | 日志路径 | SCRIPT DURATION | tst_u MAE | tst_u RMSE | tst_u MAPE | tst_a MAE | tst_a RMSE | tst_a MAPE |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, label in CONFIGS:
            p = logroot / "METRLA" / str(seed) / "est" / f"{key}_s{seed}.log"
            st, u, a = resolve(logroot, seed, key)
            rel = f"`{os.path.relpath(p, root).replace(os.sep, '/')}`"
            md.append(f"| {label} | {rel} | {u.duration or ('—' if st == '未开始' else st)} | {fmt(u.mae)} | {fmt(u.rmse)} | {fmt(u.mape)} | {fmt(a.mae)} | {fmt(a.rmse)} | {fmt(a.mape)} |")
        md.append("")

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
