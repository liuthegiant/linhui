#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DATASETS = ("METRLA", "PEMSBAY", "PEMSD7M")
SEEDS = (42, 88, 250, 555, 999)
FEATURES = (2, 3, 4, 5)

DEFAULT_LOGROOT = "logs_topomoe/est_3ds_virtualnode_splitmask_global5seed_geotopo_geoF2to5"
DEFAULT_OUT = "VirtualNode_splitmask_Global5seed_GeoTopo_GeoFeature2to5.md"

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]|tst_v_full),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)
_RE_TSTVFULL_FILE = re.compile(
    r"GraphWaveNet,\s*(tst_v_full),\s*Masked MAE,\s*([0-9.]+),\s*RMSE,\s*([0-9.]+),\s*MAPE,\s*([0-9.]+)"
)
_RE_OUTDIR = re.compile(r"\[TopoMoE\]\s+output dir:\s+(?P<path>.+)\s*$", re.MULTILINE)
_RE_SAVE_MASKPOLICY = re.compile(r"\[splitmask\]\s+saved\s+(?P<path>.+?/mask_policy\.json)\s*$", re.MULTILINE)
_RE_PATH_LINE = re.compile(r"^PATH\s+(?P<path>\S.+?)\s*$", re.MULTILINE)


@dataclass
class Metrics:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    duration: Optional[str] = None


def fmt_pm(vals: list[float], nd: int = 4) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def parse_train_log(txt: str) -> dict[str, Metrics]:
    out = {"tst_u": Metrics(), "tst_v_full": Metrics(), "tst_a": Metrics()}
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


def parse_tstvfull_from_dir(run_dir: Optional[Path]) -> Metrics:
    if run_dir is None or not run_dir.exists():
        return Metrics()
    cands = list(run_dir.glob("*_prediction_scores.txt"))
    if not cands:
        cands = list(run_dir.glob("*.txt"))
    for p in cands:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = _RE_TSTVFULL_FILE.search(txt)
        if m:
            _, mae, rmse, mape = m.groups()
            return Metrics(float(mae), float(rmse), float(mape))
    return Metrics()


def resolve_one(root: Path, logroot: Path, ds: str, seed: int, feat: int) -> tuple[str, dict[str, Metrics]]:
    p = logroot / ds / str(seed) / "est" / f"A_geo_topo_f{feat}_s{seed}.log"
    empty = {"tst_u": Metrics(), "tst_v_full": Metrics(), "tst_a": Metrics()}
    if not p.is_file():
        return "未开始", empty
    txt = p.read_text(encoding="utf-8", errors="ignore")
    ms = parse_train_log(txt)
    run_dir = extract_run_dir(root, txt)
    tv = parse_tstvfull_from_dir(run_dir)
    if tv.mae is not None:
        ms["tst_v_full"] = tv
    if "SCRIPT DURATION" in txt:
        return "完成", ms
    return "进行中", ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    logroot = (
        (root / args.logroot).resolve()
        if args.logroot is not None and not args.logroot.is_absolute()
        else (args.logroot.resolve() if args.logroot is not None else (root / DEFAULT_LOGROOT).resolve())
    )
    out = (
        (root / args.out).resolve()
        if args.out is not None and not args.out.is_absolute()
        else (args.out.resolve() if args.out is not None else (root / DEFAULT_OUT).resolve())
    )

    md: list[str] = []
    md.append("# Virtual-Node SplitMask Estimation — 全局5种子 GEO+TOPO（Geo Feature=2..5）")
    md.append("")
    md.append("- **任务**：`geo+topo` estimation（splitmask：`tst_u` / `tst_v_full` / `tst_a`）")
    md.append("- **统一5种子**：`42, 88, 250, 555, 999`（三数据集统一）")
    md.append("- **Geo 参数变体**：`FEATURES`（`argv[12]`）=`2,3,4,5`")
    md.append("- **GPU 计划**：`0,2,3,5`")
    md.append("- **固定其余参数**：`topo_moe 64 16 2 1.0 0.001 0.001 0.0 1`，`MOE_EXPERTS=geo,topo`")
    md.append(f"- **日志根目录**：`{os.path.relpath(logroot, root).replace(os.sep, '/')}`")
    md.append(f"- **报告更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    done = running = pending = 0
    md.append("## 进度（3 数据集 × 5 seeds × 4 features = 60）")
    md.append("")
    md.append("| 数据集 | Seed | Feature | 状态 | tst_u MAE | tst_v_full MAE | tst_a MAE |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ds in DATASETS:
        for s in SEEDS:
            for f in FEATURES:
                st, ms = resolve_one(root, logroot, ds, s, f)
                done += int(st == "完成")
                running += int(st == "进行中")
                pending += int(st == "未开始")
                mu = "—" if ms["tst_u"].mae is None else f"{ms['tst_u'].mae:.4f}"
                mv = "—" if ms["tst_v_full"].mae is None else f"{ms['tst_v_full'].mae:.4f}"
                ma = "—" if ms["tst_a"].mae is None else f"{ms['tst_a'].mae:.4f}"
                md.append(f"| {ds} | {s} | {f} | {st} | {mu} | {mv} | {ma} |")
    md.append("")
    md.append(f"**进度**：完成 **{done}/60**，进行中 **{running}**，未开始 **{pending}**")
    md.append("")

    md.append("## 每数据集每 Feature 汇总（完成项，5-seed）")
    md.append("")
    md.append("| 数据集 | Feature | tst_u MAE | tst_v_full MAE | tst_a MAE | 完成种子数 |")
    md.append("| --- | --- | --- | --- | --- | --- |")

    best_by_ds: dict[str, tuple[int, float]] = {}
    for ds in DATASETS:
        for f in FEATURES:
            u_vals: list[float] = []
            v_vals: list[float] = []
            a_vals: list[float] = []
            done_n = 0
            for s in SEEDS:
                st, ms = resolve_one(root, logroot, ds, s, f)
                if st != "完成":
                    continue
                done_n += 1
                if ms["tst_u"].mae is not None:
                    u_vals.append(ms["tst_u"].mae)
                if ms["tst_v_full"].mae is not None:
                    v_vals.append(ms["tst_v_full"].mae)
                if ms["tst_a"].mae is not None:
                    a_vals.append(ms["tst_a"].mae)

            md.append(
                f"| {ds} | {f} | {fmt_pm(u_vals)} | {fmt_pm(v_vals)} | {fmt_pm(a_vals)} | {done_n}/5 |"
            )

            if v_vals:
                v_mean = statistics.mean(v_vals)
                if ds not in best_by_ds or v_mean < best_by_ds[ds][1]:
                    best_by_ds[ds] = (f, v_mean)
    md.append("")

    md.append("## 按 `tst_v_full` 选最佳 Feature（仅基于已完成项）")
    md.append("")
    md.append("| 数据集 | 最佳 Geo Feature | `tst_v_full` MAE 均值 |")
    md.append("| --- | --- | --- |")
    for ds in DATASETS:
        if ds in best_by_ds:
            bf, bv = best_by_ds[ds]
            md.append(f"| {ds} | {bf} | {bv:.4f} |")
        else:
            md.append(f"| {ds} | — | — |")
    md.append("")

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
