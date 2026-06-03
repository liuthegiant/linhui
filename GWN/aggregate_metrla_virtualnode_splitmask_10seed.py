#!/usr/bin/env python3
"""Aggregate METRLA split virtual-node mask estimation logs."""
from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

SEEDS = (100, 42, 999, 555, 250, 88, 66, 233, 38, 432)
CONFIGS = (
    ("A_no_pretrain", "无预训练"),
    ("A_topo_only", "TOPO"),
    ("A_geo_only", "GEO"),
    ("A_geo_topo", "GEO+TOPO"),
)
LOGROOT_DEFAULT = "logs_topomoe/est_metrla_virtualnode_splitmask_seed10"
OUT_DEFAULT = "METRLA_virtualnode_splitmask_Estimation.md"

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[uav]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)


@dataclass
class Metrics:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    duration: Optional[str] = None


def fmt(x: Optional[float], nd: int = 6) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_pm(xs: list[Optional[float]], nd: int = 4) -> tuple[str, int]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return "—", 0
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}", 1
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}", len(vals)


def parse_text(txt: str) -> dict[str, Metrics]:
    out = {"tst_u": Metrics(), "tst_v": Metrics(), "tst_a": Metrics()}
    for split, mae, rmse, mape in _RE_EST.findall(txt):
        out[split] = Metrics(float(mae), float(rmse), float(mape))
    dm = _RE_DURATION.search(txt)
    if dm:
        dur = dm.group(1).strip()
        for m in out.values():
            m.duration = dur
    return out


def resolve(logroot: Path, seed: int, key: str) -> tuple[str, dict[str, Metrics]]:
    p = logroot / "METRLA" / str(seed) / "est" / f"{key}_s{seed}.log"
    if not p.is_file():
        return "未开始", parse_text("")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    metrics = parse_text(txt)
    if "SCRIPT DURATION" in txt:
        return "完成", metrics
    return "进行中", metrics


def metric_table(logroot: Path, split: str) -> list[str]:
    rows = [
        f"## `{split}` 汇总（均值 ± 样本标准差）",
        "",
        "| 配置 | n | MAE | RMSE | MAPE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in CONFIGS:
        maes, rmses, mapes = [], [], []
        for seed in SEEDS:
            st, ms = resolve(logroot, seed, key)
            if st != "完成":
                continue
            maes.append(ms[split].mae)
            rmses.append(ms[split].rmse)
            mapes.append(ms[split].mape)
        mae_txt, n = fmt_pm(maes)
        rmse_txt, _ = fmt_pm(rmses)
        mape_txt, _ = fmt_pm(mapes)
        rows.append(f"| {label} | {n} | {mae_txt} | {rmse_txt} | {mape_txt} |")
    rows.append("")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=Path(LOGROOT_DEFAULT))
    ap.add_argument("--out", type=Path, default=Path(OUT_DEFAULT))
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    logroot = (root / args.logroot).resolve() if not args.logroot.is_absolute() else args.logroot
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    md: list[str] = []
    md.append("# METRLA Virtual-Node SplitMask Estimation（10 seeds，无 SCPT）")
    md.append("")
    md.append("- **任务**：Estimation 掩码预测")
    md.append("- **训练掩码**：只使用原始随机点掩码，不固定整节点")
    md.append("- **测试汇报**：`tst_u` 排除 fixed virtual nodes；`tst_v` 只在 fixed virtual nodes 上算；`tst_a` 为 all-node 随机点掩码")
    md.append("- **配置**：`无预训练 / TOPO only / GEO only / GEO+TOPO`")
    md.append("- **BASE（argv[1]–[20]）**：`1 0.7 0 <SEED> 1.0 METRLA -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320`")
    md.append("- **额外参数（argv[21]–[29]）**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（n=10）")
    md.append("- **运行脚本**：`./run_metrla_virtualnode_splitmask_7gpu.sh`")
    md.append(f"- **报告更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    done = running = pending = 0
    md.append("## 进度")
    md.append("")
    md.append("| 配置 | 种子 | 状态 | tst_u MAE | tst_v MAE | tst_a MAE | 耗时 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in CONFIGS:
        for seed in SEEDS:
            st, ms = resolve(logroot, seed, key)
            done += int(st == "完成")
            running += int(st == "进行中")
            pending += int(st == "未开始")
            md.append(
                f"| {label} | {seed} | {st} | {fmt(ms['tst_u'].mae)} | {fmt(ms['tst_v'].mae)} | {fmt(ms['tst_a'].mae)} | {ms['tst_u'].duration or '—'} |"
            )
    md.append("")
    md.append(f"**进度**：完成 **{done}/40**，进行中 **{running}**，未开始 **{pending}**")
    md.append("")

    for split in ("tst_u", "tst_v", "tst_a"):
        md.extend(metric_table(logroot, split))

    md.append("## 按种子分项")
    md.append("")
    for seed in SEEDS:
        md.append(f"### 种子 `{seed}`")
        md.append("")
        md.append("| 配置 | 日志路径 | tst_u MAE/RMSE/MAPE | tst_v MAE/RMSE/MAPE | tst_a MAE/RMSE/MAPE |")
        md.append("| --- | --- | --- | --- | --- |")
        for key, label in CONFIGS:
            p = logroot / "METRLA" / str(seed) / "est" / f"{key}_s{seed}.log"
            _, ms = resolve(logroot, seed, key)
            rel = f"`{os.path.relpath(p, root).replace(os.sep, '/')}`"
            def trio(m: Metrics) -> str:
                return f"{fmt(m.mae)} / {fmt(m.rmse)} / {fmt(m.mape)}"
            md.append(f"| {label} | {rel} | {trio(ms['tst_u'])} | {trio(ms['tst_v'])} | {trio(ms['tst_a'])} |")
        md.append("")

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
