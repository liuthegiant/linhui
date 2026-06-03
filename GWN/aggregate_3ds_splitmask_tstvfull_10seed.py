#!/usr/bin/env python3
"""Aggregate splitmask logs + eval-only full-graph tst_v_full results.

We keep tst_u/tst_a from the original training logs, and replace tst_v with tst_v_full
read from each run's saved prediction_scores file.
"""
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
BASE_ARGV = "1 0.7 0 <SEED> 1.0 {dataset} -1 1 0.0 1 1 2 64 0.01 100 100 0 0.001 1 320"
EXTRA_TOPO = "topo_moe 64 16 2 1.0 0.001 0.001 0.0 1"

LOGROOTS_DEFAULT = {
    "METRLA": "logs_topomoe/est_metrla_virtualnode_splitmask_seed10",
    "PEMSBAY": "logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10",
    "PEMSD7M": "logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10",
}

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)
# Two possible formats:
#  - printed to stdout: "GraphWaveNet, tst_v_full, Masked MAE: x, RMSE: y, MAPE: z"
#  - appended to prediction_scores.txt: "GraphWaveNet, tst_v_full, Masked MAE, x, RMSE, y, MAPE, z"
_RE_TSTVFULL_PRINT = re.compile(
    r"GraphWaveNet,\s*(tst_v_full),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)
_RE_TSTVFULL_FILE = re.compile(
    r"GraphWaveNet,\s*(tst_v_full),\s*Masked MAE,\s*([0-9.]+),\s*RMSE,\s*([0-9.]+),\s*MAPE,\s*([0-9.]+)"
)
_RE_PATH_LINE = re.compile(r"^PATH\s+(?P<path>\S.+?)\s*$", re.MULTILINE)
_RE_OUTDIR = re.compile(r"\[TopoMoE\]\s+output dir:\s+(?P<path>.+)\s*$", re.MULTILINE)
_RE_SAVE_MASKPOLICY = re.compile(r"\[splitmask\]\s+saved\s+(?P<path>.+?/mask_policy\.json)\s*$", re.MULTILINE)


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


def parse_train_log(txt: str) -> dict[str, Metrics]:
    out = {"tst_u": Metrics(), "tst_a": Metrics()}
    for split, mae, rmse, mape in _RE_EST.findall(txt):
        out[split] = Metrics(float(mae), float(rmse), float(mape))
    dm = _RE_DURATION.search(txt)
    if dm:
        dur = dm.group(1).strip()
        for m in out.values():
            m.duration = dur
    return out


def extract_run_dir(root: Path, txt: str) -> Optional[Path]:
    m = _RE_OUTDIR.search(txt)
    if m:
        p = m.group("path").strip()
        return Path(p) if os.path.isabs(p) else (root / p).resolve()
    m2 = _RE_SAVE_MASKPOLICY.search(txt)
    if m2:
        p = m2.group("path").strip()
        return (root / Path(p).parent).resolve() if not os.path.isabs(p) else Path(p).resolve().parent
    m3 = _RE_PATH_LINE.search(txt)
    if m3:
        p = m3.group("path").strip()
        return (root / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
    return None


def parse_tstvfull_from_dir(run_dir: Path) -> Metrics:
    # Prefer model prediction_scores (written by testModel...).
    cands = list(run_dir.glob("*_prediction_scores.txt"))
    if not cands:
        # fallback: any txt file may contain the print line
        cands = list(run_dir.glob("*.txt"))
    best = Metrics()
    for p in cands:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = _RE_TSTVFULL_FILE.search(txt) or _RE_TSTVFULL_PRINT.search(txt)
        if m:
            _, mae, rmse, mape = m.groups()
            best = Metrics(float(mae), float(rmse), float(mape))
    return best


def resolve(root: Path, logroot: Path, dataset: str, seed: int, key: str) -> tuple[str, dict[str, Metrics]]:
    p = logroot / dataset / str(seed) / "est" / f"{key}_s{seed}.log"
    if not p.is_file():
        return "未开始", {"tst_u": Metrics(), "tst_v_full": Metrics(), "tst_a": Metrics()}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    train_ms = parse_train_log(txt)
    run_dir = extract_run_dir(root, txt)
    tv = parse_tstvfull_from_dir(run_dir) if run_dir else Metrics()
    out = {"tst_u": train_ms["tst_u"], "tst_v_full": tv, "tst_a": train_ms["tst_a"]}
    if "SCRIPT DURATION" in txt:
        return "完成", out
    return "进行中", out


def metric_table(root: Path, logroot: Path, dataset: str, split: str) -> list[str]:
    rows = [
        f"## `{split}` 汇总（均值 ± 样本标准差）",
        "",
        "| 配置 | n | MAE | RMSE | MAPE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in CONFIGS:
        maes: list[Optional[float]] = []
        rmses: list[Optional[float]] = []
        mapes: list[Optional[float]] = []
        for seed in SEEDS:
            st, ms = resolve(root, logroot, dataset, seed, key)
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
    ap.add_argument("--dataset", required=True, choices=("METRLA", "PEMSBAY", "PEMSD7M"))
    ap.add_argument("--logroot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dataset = args.dataset
    root = Path(__file__).resolve().parent
    logroot = (
        (root / args.logroot).resolve()
        if args.logroot is not None and not args.logroot.is_absolute()
        else (args.logroot.resolve() if args.logroot is not None else (root / LOGROOTS_DEFAULT[dataset]).resolve())
    )
    out = (
        (root / args.out).resolve()
        if args.out is not None and not args.out.is_absolute()
        else (args.out.resolve() if args.out is not None else (root / f"{dataset}_virtualnode_splitmask_TSTVFULL_Estimation.md").resolve())
    )

    md: list[str] = []
    md.append(f"# {dataset} Virtual-Node SplitMask Estimation（10 seeds，无 SCPT；tst_v_full 全图预测）")
    md.append("")
    md.append("- **任务**：Estimation 掩码预测")
    md.append("- **训练掩码**：只使用原始随机点掩码，不固定整节点")
    md.append("- **测试汇报**：`tst_u`（unseen 且排除 V）；`tst_v_full`（V 节点全时刻 mask，但用全图 forward，只在 V 上计分）；`tst_a`（全图随机点掩码）")
    md.append("- **配置**：`无预训练 / TOPO only / GEO only / GEO+TOPO`")
    md.append(f"- **BASE（argv[1]–[20]）**：`{BASE_ARGV.format(dataset=dataset)}`")
    md.append(f"- **额外参数（argv[21]–[29]）**：`{EXTRA_TOPO}`")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **种子**：`{', '.join(map(str, SEEDS))}`（n=10）")
    md.append("- **tst_v_full 重跑**：`./run_eval_tstv_full_3ds_7gpu.sh`（eval-only, load existing weights）")
    md.append(f"- **报告更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    done = running = pending = 0
    md.append("## 进度")
    md.append("")
    md.append("| 配置 | 种子 | 状态 | tst_u MAE | tst_v_full MAE | tst_a MAE | 耗时 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in CONFIGS:
        for seed in SEEDS:
            st, ms = resolve(root, logroot, dataset, seed, key)
            done += int(st == "完成")
            running += int(st == "进行中")
            pending += int(st == "未开始")
            md.append(
                f"| {label} | {seed} | {st} | {fmt(ms['tst_u'].mae)} | {fmt(ms['tst_v_full'].mae)} | {fmt(ms['tst_a'].mae)} | {ms['tst_u'].duration or '—'} |"
            )
    md.append("")
    md.append(f"**进度**：完成 **{done}/40**，进行中 **{running}**，未开始 **{pending}**")
    md.append("")

    for split in ("tst_u", "tst_v_full", "tst_a"):
        md.extend(metric_table(root, logroot, dataset, split))

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()

