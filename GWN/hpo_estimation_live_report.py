#!/usr/bin/env python3
"""Regenerate live HPO leaderboard markdown from hpo_state.json."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CONFIG_LABEL = {
    "topo_only": "TOPO",
    "scpt_topo": "SCPT+TOPO",
}


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _mean(vals: list[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _trial_jobs(trial: dict[str, Any]) -> list[dict[str, Any]]:
    return trial.get("jobs", [])


def _done_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [j for j in jobs if j.get("status") == "done" and j.get("tst_u_mae") is not None]


def _trial_mean_tst_u(trial: dict[str, Any], config: Optional[str] = None) -> Optional[float]:
    jobs = _done_jobs(_trial_jobs(trial))
    if config:
        jobs = [j for j in jobs if j.get("config") == config]
    vals = [float(j["tst_u_mae"]) for j in jobs]
    return _mean(vals)


def _trial_progress(trial: dict[str, Any]) -> tuple[int, int]:
    jobs = _trial_jobs(trial)
    done = sum(1 for j in jobs if j.get("status") == "done")
    return done, len(jobs)


def build_markdown(state: dict[str, Any], out_note: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    trials = state.get("trials", [])
    meta = state.get("meta", {})
    seeds = meta.get("seeds", [])
    datasets = meta.get("datasets", [])
    configs = meta.get("configs", [])
    gpus = meta.get("gpus", [])

    lines: list[str] = []
    lines.append("# Estimation HPO 实时榜单（topo_only / scpt_topo）")
    lines.append("")
    lines.append(f"- **更新时间（UTC）**：{now}")
    lines.append(f"- **说明**：{out_note}")
    lines.append(f"- **配置**：`{', '.join(configs)}`")
    lines.append(f"- **数据集顺序**：`{' → '.join(datasets)}`（METRLA 最后）")
    lines.append(f"- **种子（n=2）**：`{', '.join(map(str, seeds))}`")
    lines.append(f"- **GPU**：`{', '.join(map(str, gpus))}`（6 卡并行，留 2 卡）")
    lines.append(f"- **主优化指标**：`tst_u` Masked MAE（越小越好）")
    lines.append(f"- **状态文件**：`{meta.get('state_path', 'logs_hpo_est/hpo_state.json')}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Global progress
    total_jobs = sum(len(_trial_jobs(t)) for t in trials)
    done_jobs = sum(len(_done_jobs(_trial_jobs(t))) for t in trials)
    running = sum(
        1
        for t in trials
        for j in _trial_jobs(t)
        if j.get("status") == "running"
    )
    failed = sum(
        1
        for t in trials
        for j in _trial_jobs(t)
        if j.get("status") == "failed"
    )
    lines.append("## 总进度")
    lines.append("")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"| --- | --- |")
    lines.append(f"| 试验数（trials） | {len(trials)} |")
    lines.append(f"| 子任务完成 | {done_jobs} / {total_jobs} |")
    lines.append(f"| 运行中 | {running} |")
    lines.append(f"| 失败 | {failed} |")
    lines.append("")

  # Leaderboard: all configs combined
    ranked: list[tuple[int, dict[str, Any], Optional[float], int, int]] = []
    for t in trials:
        tid = int(t["trial_id"])
        m = _trial_mean_tst_u(t)
        d, tot = _trial_progress(t)
        ranked.append((tid, t, m, d, tot))
    ranked.sort(key=lambda x: (x[2] is None, x[2] if x[2] is not None else 1e9))

    lines.append("## Trial 总榜（按 tst_u MAE 均值，含 topo_only + scpt_topo）")
    lines.append("")
    lines.append("| 排名 | trial_id | 进度 | mean tst_u MAE | GATE_HIDDEN | TOPO_LAP_K | MOE_TAU | MOE_LB | MOE_SM | MOE_DELTA | MOE_CTX | PRE_LEARN |")
    lines.append("| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    rank = 0
    for tid, t, m, d, tot in ranked:
        if m is None:
            continue
        rank += 1
        p = t.get("params", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(tid),
                    f"{d}/{tot}",
                    _fmt(m),
                    str(p.get("GATE_HIDDEN", "—")),
                    str(p.get("TOPO_LAP_K", "—")),
                    _fmt(p.get("MOE_TAU"), 3) if p.get("MOE_TAU") is not None else "—",
                    _fmt(p.get("MOE_LB_REG"), 5) if p.get("MOE_LB_REG") is not None else "—",
                    _fmt(p.get("MOE_SMOOTH_REG"), 5) if p.get("MOE_SMOOTH_REG") is not None else "—",
                    str(p.get("MOE_DELTA_REG", "—")),
                    str(p.get("MOE_USE_CTX", "—")),
                    _fmt(p.get("PRE_LEARN"), 5) if p.get("PRE_LEARN") is not None else "—",
                ]
            )
            + " |"
        )
    if rank == 0:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.append("")

    for cfg in configs:
        label = CONFIG_LABEL.get(cfg, cfg)
        sub = [(tid, t, _trial_mean_tst_u(t, cfg), d, tot) for tid, t, _, d, tot in ranked]
        sub = [(a, b, c, d, e) for a, b, c, d, e in sub if c is not None]
        sub.sort(key=lambda x: x[2])
        lines.append(f"## 分配置榜单 — {label} (`{cfg}`)")
        lines.append("")
        lines.append("| 排名 | trial_id | 进度 | mean tst_u MAE | 参数摘要 |")
        lines.append("| ---: | ---: | --- | --- | --- |")
        for i, (tid, t, m, d, tot) in enumerate(sub, 1):
            p = t.get("params", {})
            summary = (
                f"gh={p.get('GATE_HIDDEN')} lap={p.get('TOPO_LAP_K')} "
                f"tau={p.get('MOE_TAU')} lb={p.get('MOE_LB_REG')}"
            )
            lines.append(f"| {i} | {tid} | {d}/{tot} | {_fmt(m)} | {summary} |")
        if not sub:
            lines.append("| — | — | — | — | 暂无完成结果 |")
        lines.append("")

    for ds in datasets:
        lines.append(f"## 数据集 `{ds}` — 已完成子任务（按 tst_u MAE 升序）")
        lines.append("")
        lines.append("| trial_id | 配置 | seed | tst_u MAE | tst_u RMSE | tst_u MAPE | 时长 |")
        lines.append("| ---: | --- | ---: | --- | --- | --- | --- |")
        rows: list[tuple[float, str]] = []
        for t in trials:
            tid = t["trial_id"]
            for j in _done_jobs(_trial_jobs(t)):
                if j.get("dataset") != ds:
                    continue
                mae = float(j["tst_u_mae"])
                rows.append(
                    (
                        mae,
                        "| "
                        + " | ".join(
                            [
                                str(tid),
                                CONFIG_LABEL.get(j.get("config", ""), j.get("config", "")),
                                str(j.get("seed", "")),
                                _fmt(mae),
                                _fmt(j.get("tst_u_rmse")),
                                _fmt(j.get("tst_u_mape")),
                                j.get("duration", "—") or "—",
                            ]
                        )
                        + " |",
                    )
                )
        rows.sort(key=lambda x: x[0])
        if rows:
            lines.extend(r[1] for r in rows)
        else:
            lines.append("| — | — | — | — | — | — | — |")
        lines.append("")

    lines.append("## 最近完成的子任务")
    lines.append("")
    lines.append("| 完成时间（UTC） | trial_id | 配置 | 数据集 | seed | tst_u MAE | GPU | 日志 |")
    lines.append("| --- | ---: | --- | --- | ---: | --- | ---: | --- |")
    recent: list[tuple[str, dict[str, Any]]] = []
    for t in trials:
        for j in _trial_jobs(t):
            if j.get("status") != "done":
                continue
            recent.append((j.get("finished_at", ""), j))
    recent.sort(key=lambda x: x[0], reverse=True)
    for _, j in recent[:30]:
        log_rel = j.get("log", "—")
        lines.append(
            "| "
            + " | ".join(
                [
                    j.get("finished_at", "—") or "—",
                    str(j.get("trial_id", "—")),
                    CONFIG_LABEL.get(j.get("config", ""), j.get("config", "")),
                    str(j.get("dataset", "")),
                    str(j.get("seed", "")),
                    _fmt(j.get("tst_u_mae")),
                    str(j.get("gpu", "—")),
                    f"`{log_rel}`",
                ]
            )
            + " |"
        )
    if not recent:
        lines.append("| — | — | — | — | — | — | — | — |")
    lines.append("")

    lines.append("## 运行中 / 失败")
    lines.append("")
    lines.append("| 状态 | trial_id | 配置 | 数据集 | seed | GPU | 日志 | 备注 |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | --- | --- |")
    any_active = False
    for t in trials:
        for j in _trial_jobs(t):
            st = j.get("status")
            if st not in ("running", "failed", "queued"):
                continue
            any_active = True
            lines.append(
                "| "
                + " | ".join(
                    [
                        st,
                        str(j.get("trial_id", t.get("trial_id"))),
                        CONFIG_LABEL.get(j.get("config", ""), j.get("config", "")),
                        str(j.get("dataset", "")),
                        str(j.get("seed", "")),
                        str(j.get("gpu", "—")),
                        f"`{j.get('log', '—')}`",
                        (j.get("error", "") or "—")[:80],
                    ]
                )
                + " |"
            )
    if not any_active:
        lines.append("| — | — | — | — | — | — | — | 无 |")
    lines.append("")

    lines.append("## 超参搜索空间（本 run）")
    lines.append("")
    sp = meta.get("search_space", {})
    if sp:
        lines.append("```yaml")
        for k, v in sp.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
    else:
        lines.append("（见 `hpo_estimation_runner.py` 内 `DEFAULT_SEARCH_SPACE`）")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=Path("logs_hpo_est/hpo_state.json"))
    ap.add_argument("--out", type=Path, default=Path("HPO_ESTIMATION_LIVE.md"))
    ap.add_argument("--note", type=str, default="每完成一个子任务自动刷新")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    state_path = (root / args.state).resolve() if not args.state.is_absolute() else args.state
    out_path = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    if not state_path.is_file():
        state = {
            "meta": {
                "state_path": str(state_path.relative_to(root)),
                "seeds": [100, 42],
                "datasets": ["PEMSBAY", "PEMSD7M", "METRLA"],
                "configs": ["topo_only", "scpt_topo"],
                "gpus": [0, 1, 3, 4, 6, 7],
            },
            "trials": [],
        }
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        try:
            rel = str(state_path.relative_to(root))
        except ValueError:
            rel = str(state_path)
        state.setdefault("meta", {})["state_path"] = rel

    md = build_markdown(state, args.note)
    out_path.write_text(md, encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
