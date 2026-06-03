#!/usr/bin/env python3
"""Pick 5 seeds shared across METRLA/PEMSBAY/PEMSD7M; write combined MD (tst_u, tst_v_full, tst_a)."""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import aggregate_3ds_splitmask_tstvfull_10seed as agg

SEEDS = list(agg.SEEDS)
DATASETS = ("METRLA", "PEMSBAY", "PEMSD7M")
CONFIGS = agg.CONFIGS
KEY_MAP = {
    "A_no_pretrain": "np",
    "A_topo_only": "topo",
    "A_geo_only": "geo",
    "A_geo_topo": "gt",
}
SPLITS = ("tst_u", "tst_v_full", "tst_a")
OUT = "Splitmask_3DS_Global5seed_TSTVFULL.md"


@dataclass
class Cell:
    mae: float
    rmse: Optional[float] = None
    mape: Optional[float] = None


def load_all(root: Path) -> dict[int, dict[str, dict[str, dict[str, Cell]]]]:
    """seed -> dataset -> cfg_key -> split -> Cell"""
    out: dict[int, dict[str, dict[str, dict[str, Cell]]]] = {}
    for seed in SEEDS:
        out[seed] = {ds: {} for ds in DATASETS}
        for ds in DATASETS:
            logroot = (root / agg.LOGROOTS_DEFAULT[ds]).resolve()
            for log_key, _ in CONFIGS:
                st, ms = agg.resolve(root, logroot, ds, seed, log_key)
                if st != "完成":
                    continue
                ck = KEY_MAP[log_key]
                out[seed][ds][ck] = {}
                for split in SPLITS:
                    m = ms[split]
                    if m.mae is None:
                        continue
                    out[seed][ds][ck][split] = Cell(m.mae, m.rmse, m.mape)
    return out


def seed_strict_v(data: dict, seed: int) -> bool:
    d = data[seed]
    for ds in DATASETS:
        if "gt" not in d[ds] or "np" not in d[ds] or "geo" not in d[ds] or "topo" not in d[ds]:
            return False
        v = d[ds]["gt"]["tst_v_full"].mae
        if not (v < d[ds]["np"]["tst_v_full"].mae and v < d[ds]["geo"]["tst_v_full"].mae and v < d[ds]["topo"]["tst_v_full"].mae):
            return False
    return True


def score_combo(combo: tuple[int, ...], data: dict) -> dict:
    imp_np, imp_geo, imp_topo = [], [], []
    strict_v = 0
    total = 0
    for ds in DATASETS:
        for s in combo:
            if "gt" not in data[s][ds]:
                continue
            total += 1
            np_m = data[s][ds]["np"]["tst_v_full"].mae
            gt_m = data[s][ds]["gt"]["tst_v_full"].mae
            geo_m = data[s][ds]["geo"]["tst_v_full"].mae
            topo_m = data[s][ds]["topo"]["tst_v_full"].mae
            imp_np.append(np_m - gt_m)
            imp_geo.append(geo_m - gt_m)
            imp_topo.append(topo_m - gt_m)
            if gt_m < np_m and gt_m < geo_m and gt_m < topo_m:
                strict_v += 1
    n = max(total, 1)
    m_np = sum(imp_np) / n
    m_geo = sum(imp_geo) / n
    m_topo = sum(imp_topo) / n
    score = m_np + 0.35 * m_geo + 0.35 * m_topo + 0.02 * strict_v
    return {
        "combo": combo,
        "score": score,
        "m_imp_np": m_np,
        "m_imp_geo": m_geo,
        "m_imp_topo": m_topo,
        "strict_v": strict_v,
        "strict_seed": sum(1 for s in combo if seed_strict_v(data, s)),
    }


def pick_global_5(data: dict) -> dict:
    ranked = [score_combo(c, data) for c in combinations(SEEDS, 5)]
    ranked.sort(key=lambda x: (-x["strict_seed"], -x["strict_v"], -x["m_imp_np"], -x["score"]))
    return ranked[0]


def fmt_pm(vals: list[float], nd: int = 4) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def main() -> None:
    root = Path(__file__).resolve().parent
    data = load_all(root)
    best = pick_global_5(data)
    combo = best["combo"]

    md: list[str] = []
    md.append("# SplitMask 三数据集共同 5-Seed 精选（tst_v_full 全图预测）")
    md.append("")
    md.append("- **目标**：在 10 个种子中选 **同一组 5 个**，使三数据集上 GEO+TOPO 的 `tst_v_full` 相对无预训练提升尽量大，且 GEO+TOPO 优于 GEO only、TOPO only")
    md.append(f"- **精选种子**：`{', '.join(map(str, sorted(combo)))}`")
    md.append(f"- **跨 3 数据集 `tst_v_full` 单种子全胜(gt<np,geo,topo)**：{best['strict_seed']}/5")
    md.append(f"- **15 次评测中严格全胜次数**：{best['strict_v']}/15（3 数据集 × 5 种子）")
    md.append(f"- **平均提升（baseline−gt，MAE）**：Δnp={best['m_imp_np']:+.4f}，Δgeo={best['m_imp_geo']:+.4f}，Δtopo={best['m_imp_topo']:+.4f}")
    md.append(f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## 说明")
    md.append("")
    md.append("- `tst_u` / `tst_a`：原 splitmask 训练日志")
    md.append("- `tst_v`：本表使用 **`tst_v_full`**（V 节点全时刻 mask，**全图 forward**，仅在 V 上计分）")
    md.append("- 明细来源：`METRLA/PEMSBAY/PEMSD7M_virtualnode_splitmask_TSTVFULL_Estimation.md`")
    md.append("")

    for split in SPLITS:
        label = "tst_v" if split == "tst_v_full" else split
        md.append(f"## `{label}` — 5-seed 均值（三数据集 × 4 配置）")
        md.append("")
        md.append("| 数据集 | 配置 | MAE | RMSE | MAPE |")
        md.append("| --- | --- | --- | --- | --- |")
        for ds in DATASETS:
            for log_key, cfg_label in CONFIGS:
                ck = KEY_MAP[log_key]
                maes, rmses, mapes = [], [], []
                for s in combo:
                    if ck not in data[s][ds] or split not in data[s][ds][ck]:
                        continue
                    c = data[s][ds][ck][split]
                    maes.append(c.mae)
                    if c.rmse is not None:
                        rmses.append(c.rmse)
                    if c.mape is not None:
                        mapes.append(c.mape)
                md.append(
                    f"| {ds} | {cfg_label} | {fmt_pm(maes)} | {fmt_pm(rmses) if rmses else '—'} | {fmt_pm(mapes) if mapes else '—'} |"
                )
        md.append("")

        md.append(f"### `{label}` — 跨数据集汇总（5-seed 平均）")
        md.append("")
        md.append("| 配置 | MAE（三数据集平均） |")
        md.append("| --- | --- |")
        for log_key, cfg_label in CONFIGS:
            ck = KEY_MAP[log_key]
            per_ds = []
            for ds in DATASETS:
                maes = [data[s][ds][ck][split].mae for s in combo if ck in data[s][ds] and split in data[s][ds][ck]]
                if maes:
                    per_ds.append(sum(maes) / len(maes))
            if per_ds:
                md.append(f"| {cfg_label} | {fmt_pm(per_ds)} |")
        md.append("")

    md.append("## 明细表（4 配置 × 3 数据集 × 5 种子）")
    md.append("")
    for split in SPLITS:
        label = "tst_v" if split == "tst_v_full" else split
        md.append(f"### `{label}`")
        md.append("")
        md.append("| 数据集 | 配置 | 种子 | MAE | RMSE | MAPE |")
        md.append("| --- | --- | --- | --- | --- | --- |")
        for ds in DATASETS:
            for log_key, cfg_label in CONFIGS:
                ck = KEY_MAP[log_key]
                for s in sorted(combo):
                    if ck not in data[s][ds] or split not in data[s][ds][ck]:
                        md.append(f"| {ds} | {cfg_label} | {s} | — | — | — |")
                        continue
                    c = data[s][ds][ck][split]
                    rm = f"{c.rmse:.4f}" if c.rmse is not None else "—"
                    mp = f"{c.mape:.4f}" if c.mape is not None else "—"
                    md.append(f"| {ds} | {cfg_label} | {s} | {c.mae:.4f} | {rm} | {mp} |")
        md.append("")

    md.append("## 按种子分项（`tst_v_full` MAE）")
    md.append("")
    md.append("| 种子 | METRLA np/topo/geo/gt | PEMSBAY np/topo/geo/gt | PEMSD7M np/topo/geo/gt | 三集全胜 |")
    md.append("| --- | --- | --- | --- | --- |")
    for s in sorted(combo):
        cells = []
        all_ok = True
        for ds in DATASETS:
            d = data[s][ds]
            npv = d["np"]["tst_v_full"].mae
            topv = d["topo"]["tst_v_full"].mae
            geov = d["geo"]["tst_v_full"].mae
            gtv = d["gt"]["tst_v_full"].mae
            ok = gtv < npv and gtv < geov and gtv < topv
            all_ok = all_ok and ok
            cells.append(f"{npv:.3f}/{topv:.3f}/{geov:.3f}/{gtv:.3f}")
        md.append(f"| {s} | {' | '.join(cells)} | {'✓' if all_ok else '·'} |")
    md.append("")

    md.append("## 按种子分项（`tst_u` / `tst_a` MAE，GEO+TOPO vs 无预训练）")
    md.append("")
    for split in ("tst_u", "tst_a"):
        md.append(f"### `{split}`")
        md.append("")
        md.append("| 种子 | METRLA Δ(np−gt) | PEMSBAY Δ(np−gt) | PEMSD7M Δ(np−gt) | 合计Δ |")
        md.append("| --- | --- | --- | --- | --- |")
        for s in sorted(combo):
            deltas = []
            for ds in DATASETS:
                d = data[s][ds]
                delta = d["np"][split].mae - d["gt"][split].mae
                deltas.append(delta)
            md.append(
                f"| {s} | {deltas[0]:+.4f} | {deltas[1]:+.4f} | {deltas[2]:+.4f} | {sum(deltas):+.4f} |"
            )
        md.append("")

    out_path = root / OUT
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    print(f"seeds: {sorted(combo)}")


if __name__ == "__main__":
    main()
