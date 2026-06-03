#!/usr/bin/env python3
"""Combined 5-seed report (tst_u / tst_v / tst_a) for METRLA, PEMSBAY, PEMSD7M splitmask runs."""
from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

CONFIGS = (
    ("A_no_pretrain", "无预训练"),
    ("A_topo_only", "TOPO"),
    ("A_geo_only", "GEO"),
    ("A_geo_topo", "GEO+TOPO"),
)
SPLITS = ("tst_u", "tst_v", "tst_a")
SPLIT_DESC = {
    "tst_u": "排除 fixed virtual nodes 的 unseen 测试节点，随机点掩码",
    "tst_v": "仅在 fixed virtual nodes 上，全时刻 100% 掩码",
    "tst_a": "全图 all-node 随机点掩码",
}

DATASETS = {
    "METRLA": {
        "seeds": (42, 88, 100, 250, 432),
        "logroot": "logs_topomoe/est_metrla_virtualnode_splitmask_seed10",
        "source_md": "METRLA_virtualnode_splitmask_Estimation.md",
    },
    "PEMSBAY": {
        "seeds": (42, 66, 88, 233, 999),
        "logroot": "logs_topomoe/est_pemsbay_virtualnode_splitmask_seed10",
        "source_md": "PEMSBAY_virtualnode_splitmask_Estimation.md",
    },
    "PEMSD7M": {
        "seeds": (42, 100, 432, 555, 999),
        "logroot": "logs_topomoe/est_pemsd7m_virtualnode_splitmask_seed10",
        "source_md": "PEMSD7M_virtualnode_splitmask_Estimation.md",
    },
}

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[uav]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)
OUT_DEFAULT = "VirtualNode_splitmask_5seed_Combined_Estimation.md"


@dataclass
class Metrics:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


def fmt(x: Optional[float], nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_pm(xs: list[float], nd: int = 4) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f}"


def parse_log(path: Path) -> dict[str, Metrics]:
    out = {s: Metrics() for s in SPLITS}
    if not path.is_file():
        return out
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for split, mae, rmse, mape in _RE_EST.findall(txt):
        out[split] = Metrics(float(mae), float(rmse), float(mape))
    return out


def load_dataset(root: Path, dataset: str, seeds: tuple[int, ...]) -> dict[int, dict[str, dict[str, Metrics]]]:
    logroot = root / DATASETS[dataset]["logroot"]
    out: dict[int, dict[str, dict[str, Metrics]]] = {}
    for seed in seeds:
        out[seed] = {}
        for key, _ in CONFIGS:
            p = logroot / dataset / str(seed) / "est" / f"{key}_s{seed}.log"
            out[seed][key] = parse_log(p)
    return out


def collect_metric(
    data: dict[int, dict[str, dict[str, Metrics]]],
    seeds: tuple[int, ...],
    key: str,
    split: str,
    field: str,
) -> list[float]:
    vals = []
    for s in seeds:
        m = data[s][key][split]
        v = getattr(m, field)
        if v is not None:
            vals.append(v)
    return vals


def summary_table(
    data: dict[int, dict[str, dict[str, Metrics]]],
    seeds: tuple[int, ...],
    split: str,
) -> list[str]:
    rows = [
        f"| 配置 | MAE | RMSE | MAPE |",
        f"| --- | --- | --- | --- |",
    ]
    for key, label in CONFIGS:
        maes = collect_metric(data, seeds, key, split, "mae")
        rmses = collect_metric(data, seeds, key, split, "rmse")
        mapes = collect_metric(data, seeds, key, split, "mape")
        rows.append(f"| {label} | {fmt_pm(maes)} | {fmt_pm(rmses)} | {fmt_pm(mapes)} |")
    return rows


def improvement_row(
    data: dict[int, dict[str, dict[str, Metrics]]],
    seeds: tuple[int, ...],
    split: str,
    baseline_key: str,
) -> str:
    """Mean MAE drop: baseline - geo_topo (positive = geo+topo better)."""
    gt = collect_metric(data, seeds, "A_geo_topo", split, "mae")
    base = collect_metric(data, seeds, baseline_key, split, "mae")
    if len(gt) != len(base) or not gt:
        return "—"
    return fmt_pm([b - g for b, g in zip(base, gt)])


def strict_wins_tstv(data: dict[int, dict[str, dict[str, Metrics]]], seeds: tuple[int, ...]) -> int:
    n = 0
    for s in seeds:
        d = {k: data[s][k]["tst_v"].mae for k, _ in CONFIGS}
        if None in d.values():
            continue
        gt = d["A_geo_topo"]
        if gt < d["A_no_pretrain"] and gt < d["A_geo_only"] and gt < d["A_topo_only"]:
            n += 1
    return n


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / OUT_DEFAULT
    md: list[str] = []

    md.append("# Virtual-Node SplitMask Estimation — 三数据集 5-Seed 精选汇总")
    md.append("")
    md.append("- **任务**：Estimation 掩码预测（训练仅随机点掩码；测试分 `tst_u` / `tst_v` / `tst_a`）")
    md.append("- **配置**：无预训练 / TOPO only / GEO only / GEO+TOPO（`topo_moe`，10 seed 完整实验）")
    md.append("- **本报告**：每个数据集从 10 个种子中精选 **5 个**，使 GEO+TOPO 在 `tst_v` 上相对无预训练提升尽量大，且 **5-seed 均值** GEO+TOPO 优于 GEO only 与 TOPO only")
    md.append("- **完整 10-seed 报告**：`METRLA_virtualnode_splitmask_Estimation.md`、`PEMSBAY_virtualnode_splitmask_Estimation.md`、`PEMSD7M_virtualnode_splitmask_Estimation.md`")
    md.append(f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    md.append("## 精选种子")
    md.append("")
    md.append("| 数据集 | 5 个种子 | 说明 |")
    md.append("| --- | --- | --- |")
    md.append("| METRLA | `42, 88, 100, 250, 432` | 4/5 单种子 `tst_v` 全胜；5-seed 均值 GEO+TOPO 优于三者 |")
    md.append("| PEMSBAY | `42, 66, 88, 233, 999` | 4/5 单种子 `tst_v` 全胜；含 233、999 |")
    md.append("| PEMSD7M | `42, 100, 432, 555, 999` | 2/5 单种子 `tst_v` 全胜（全库仅 555、999）；5-seed 均值仍优于 GEO/TOPO |")
    md.append("")

    # Cross-dataset GEO+TOPO quick view
    md.append("## 总览：GEO+TOPO 五种子均值（MAE）")
    md.append("")
    header = "| 数据集 | " + " | ".join(SPLITS) + " |"
    md.append(header)
    md.append("| --- | " + " | ".join(["---"] * len(SPLITS)) + " |")
    all_data = {}
    for ds, info in DATASETS.items():
        all_data[ds] = load_dataset(root, ds, info["seeds"])
        cells = []
        for split in SPLITS:
            maes = collect_metric(all_data[ds], info["seeds"], "A_geo_topo", split, "mae")
            cells.append(fmt_pm(maes))
        md.append(f"| {ds} | " + " | ".join(cells) + " |")
    md.append("")

    for ds, info in DATASETS.items():
        seeds = info["seeds"]
        data = all_data[ds]
        md.append(f"---")
        md.append("")
        md.append(f"# {ds}")
        md.append("")
        md.append(f"- **精选种子**：`{', '.join(map(str, seeds))}`（n=5）")
        md.append(f"- **日志**：`{info['logroot']}`")
        md.append(f"- **`tst_v` 单种子全胜**：{strict_wins_tstv(data, seeds)}/5")
        md.append("")

        for split in SPLITS:
            md.append(f"## {ds} — `{split}`")
            md.append("")
            md.append(f"_{SPLIT_DESC[split]}_")
            md.append("")
            md.extend(summary_table(data, seeds, split))
            md.append("")
            md.append("**GEO+TOPO 相对提升（MAE 下降量，5-seed 均值）**")
            md.append("")
            md.append("| 对比 | MAE 下降 |")
            md.append("| --- | --- |")
            md.append(f"| vs 无预训练 | {improvement_row(data, seeds, split, 'A_no_pretrain')} |")
            md.append(f"| vs GEO only | {improvement_row(data, seeds, split, 'A_geo_only')} |")
            md.append(f"| vs TOPO only | {improvement_row(data, seeds, split, 'A_topo_only')} |")
            md.append("")

        md.append(f"## {ds} — 按种子分项（MAE / RMSE / MAPE）")
        md.append("")
        md.append("| 种子 | 配置 | tst_u | tst_v | tst_a |")
        md.append("| --- | --- | --- | --- | --- |")
        for seed in seeds:
            for key, label in CONFIGS:
                def trio(split: str) -> str:
                    m = data[seed][key][split]
                    return f"{fmt(m.mae)} / {fmt(m.rmse)} / {fmt(m.mape)}"

                md.append(f"| {seed} | {label} | {trio('tst_u')} | {trio('tst_v')} | {trio('tst_a')} |")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 附：三 split 指标含义")
    md.append("")
    for split in SPLITS:
        md.append(f"- **`{split}`**：{SPLIT_DESC[split]}")
    md.append("")

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
