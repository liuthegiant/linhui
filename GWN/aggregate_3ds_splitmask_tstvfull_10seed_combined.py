#!/usr/bin/env python3
"""Write a single combined MD for 3 datasets (10 seeds) with tst_u/tst_a from train logs and tst_v_full from eval-only runs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from aggregate_3ds_splitmask_tstvfull_10seed import (
    CONFIGS,
    SEEDS,
    LOGROOTS_DEFAULT,
    fmt_pm,
    resolve,
)


def mean_mae(root: Path, logroot: Path, ds: str, key: str, split: str) -> str:
    vals = []
    for s in SEEDS:
        st, ms = resolve(root, logroot, ds, s, key)
        if st != "完成":
            continue
        v = ms[split].mae
        vals.append(v)
    return fmt_pm(vals)[0]


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "Splitmask_3DS_10seed_TSTVFULL_Combined.md"
    md: list[str] = []
    md.append("# SplitMask Estimation（三数据集，10 seeds）— `tst_v_full` 全图预测汇总")
    md.append("")
    md.append("- `tst_u`、`tst_a`：来自原 splitmask 训练日志")
    md.append("- `tst_v_full`：eval-only 重跑（V 节点全 mask，但全图 forward，仅在 V 上计分）")
    md.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## 报告文件")
    md.append("")
    md.append("- `METRLA_virtualnode_splitmask_TSTVFULL_Estimation.md`")
    md.append("- `PEMSBAY_virtualnode_splitmask_TSTVFULL_Estimation.md`")
    md.append("- `PEMSD7M_virtualnode_splitmask_TSTVFULL_Estimation.md`")
    md.append("")

    md.append("## 三数据集总览（MAE，均值 ± 样本标准差）")
    md.append("")
    md.append("| 数据集 | split | 无预训练 | TOPO | GEO | GEO+TOPO |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for ds in ("METRLA", "PEMSBAY", "PEMSD7M"):
        logroot = (root / LOGROOTS_DEFAULT[ds]).resolve()
        for split in ("tst_u", "tst_v_full", "tst_a"):
            row = [ds, f"`{split}`"]
            for key, label in CONFIGS:
                row.append(mean_mae(root, logroot, ds, key, split))
            md.append("| " + " | ".join(row) + " |")
    md.append("")

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()

