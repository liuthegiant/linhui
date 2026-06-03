#!/usr/bin/env python3
"""
6-GPU parallel hyperparameter search for Estimation (topo_only, scpt_topo only).
Updates HPO_ESTIMATION_LIVE.md after every completed sub-job.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Reuse live report generator
from hpo_estimation_live_report import build_markdown

ROOT = Path(__file__).resolve().parent
PYBIN_DEFAULT = "/mnt/data728/home/shared/anaconda3/envs/AutoTSenv/bin/python"
MAIN_SCRIPT = "pred_maskpredition_GWN_scpt_geo_topomoe.py"

SEEDS = (100, 42)
DATASETS = ("PEMSBAY", "PEMSD7M", "METRLA")  # METRLA last
CONFIGS = ("topo_only", "scpt_topo")
GPUS = (0, 1, 3, 4, 6, 7)

CONFIG_MAP = {
    "topo_only": {"IS_PRETRN": 1, "MOE_EXPERTS": "topo", "MOE_TOP_K": 1},
    "scpt_topo": {"IS_PRETRN": 1, "MOE_EXPERTS": "scpt,topo", "MOE_TOP_K": 2},
}

DEFAULT_SEARCH_SPACE = {
    "GATE_HIDDEN": [32, 64, 128],
    "TOPO_LAP_K": [8, 16, 32],
    "MOE_TAU": [0.5, 1.0, 1.5, 2.0],
    "MOE_LB_REG": [1e-4, 5e-4, 1e-3, 5e-3],
    "MOE_SMOOTH_REG": [1e-4, 5e-4, 1e-3, 5e-3],
    "MOE_DELTA_REG": [0.0, 1e-4, 1e-3],
    "MOE_USE_CTX": [0, 1],
    "PRE_LEARN": [5e-4, 1e-3, 2e-3],
}

_RE_DURATION = re.compile(r"^SCRIPT DURATION\s+(.+?)\s*$", re.MULTILINE)
_RE_EST = re.compile(
    r"GraphWaveNet,\s*(tst_[ua]),\s*Masked MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+),\s*MAPE:\s*([0-9.]+)"
)


@dataclass
class JobSpec:
    trial_id: int
    config: str
    dataset: str
    seed: int
    params: dict[str, Any]
    log_path: Path
    job_key: str = field(default="")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def sample_params(rng: random.Random) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in DEFAULT_SEARCH_SPACE.items()}


def base_argv20(seed: int, dataset: str) -> list[Any]:
    return [
        1,
        0.7,
        0,
        seed,
        1.0,
        dataset,
        -1,
        1,
        0.0,
        1,
        1,
        2,
        64,
        0.01,
        100,
        100,
        0,
        0.001,
        1,
        320,
    ]


def build_argv(params: dict[str, Any], seed: int, dataset: str, config: str) -> list[str]:
    cm = CONFIG_MAP[config]
    base = base_argv20(seed, dataset)
    base[0] = cm["IS_PRETRN"]
    base[17] = params["PRE_LEARN"]
    extra = [
        "topo_moe",
        params["GATE_HIDDEN"],
        params["TOPO_LAP_K"],
        cm["MOE_TOP_K"],
        params["MOE_TAU"],
        params["MOE_LB_REG"],
        params["MOE_SMOOTH_REG"],
        params["MOE_DELTA_REG"],
        params["MOE_USE_CTX"],
    ]
    return [str(x) for x in base + extra]


def parse_est_log(text: str) -> dict[str, Optional[float | str]]:
    out: dict[str, Optional[float | str]] = {
        "tst_u_mae": None,
        "tst_u_rmse": None,
        "tst_u_mape": None,
        "tst_a_mae": None,
        "duration": None,
    }
    for m in _RE_EST.finditer(text):
        split, mae, rmse, mape = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if split == "tst_u":
            out["tst_u_mae"], out["tst_u_rmse"], out["tst_u_mape"] = mae, rmse, mape
        else:
            out["tst_a_mae"] = mae
    dm = _RE_DURATION.search(text)
    if dm:
        out["duration"] = dm.group(1).strip()
    return out


def log_is_done(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    t = log_path.read_text(encoding="utf-8", errors="ignore")
    return "SCRIPT DURATION" in t


class StateStore:
    def __init__(self, path: Path, live_md: Path) -> None:
        self.path = path
        self.live_md = live_md
        self._lock = threading.Lock()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty_state()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _empty_state(self) -> dict[str, Any]:
        return {
            "meta": {
                "created_at": utc_now(),
                "seeds": list(SEEDS),
                "datasets": list(DATASETS),
                "configs": list(CONFIGS),
                "gpus": list(GPUS),
                "search_space": DEFAULT_SEARCH_SPACE,
                "state_path": str(self.path.relative_to(ROOT)),
            },
            "trials": [],
        }

    def read(self) -> dict[str, Any]:
        with self._lock:
            return self._read_unlocked()

    def write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def update(self, fn) -> dict[str, Any]:
        with self._lock:
            state = self._read_unlocked()
            fn(state)
            self.write(state)
            self._write_live_md(state)
            return state

    def _write_live_md(self, state: dict[str, Any]) -> None:
        note = "每完成一个子任务自动刷新；主榜按 trial 的 tst_u MAE 均值排序"
        md = build_markdown(state, note)
        self.live_md.write_text(md, encoding="utf-8")


def job_key(trial_id: int, config: str, dataset: str, seed: int) -> str:
    return f"t{trial_id}|{config}|{dataset}|s{seed}"


def find_job(state: dict[str, Any], key: str) -> Optional[dict[str, Any]]:
    for t in state["trials"]:
        for j in t.get("jobs", []):
            if j.get("job_key") == key:
                return j
    return None


def ensure_trials(state: dict[str, Any], n_trials: int, rng: random.Random) -> None:
    existing = {int(t["trial_id"]) for t in state["trials"]}
    for tid in range(n_trials):
        if tid in existing:
            continue
        params = sample_params(rng)
        jobs: list[dict[str, Any]] = []
        for cfg in CONFIGS:
            for ds in DATASETS:
                for seed in SEEDS:
                    log_rel = f"logs_hpo_est/trial_{tid:04d}/{cfg}/{ds}/s{seed}.log"
                    jk = job_key(tid, cfg, ds, seed)
                    jobs.append(
                        {
                            "job_key": jk,
                            "trial_id": tid,
                            "config": cfg,
                            "dataset": ds,
                            "seed": seed,
                            "log": log_rel,
                            "status": "queued",
                            "gpu": None,
                            "started_at": None,
                            "finished_at": None,
                            "tst_u_mae": None,
                            "tst_u_rmse": None,
                            "tst_u_mape": None,
                            "tst_a_mae": None,
                            "duration": None,
                            "error": None,
                        }
                    )
        state["trials"].append(
            {
                "trial_id": tid,
                "params": params,
                "created_at": utc_now(),
                "jobs": jobs,
            }
        )
    state["trials"].sort(key=lambda t: int(t["trial_id"]))


def collect_pending_jobs(state: dict[str, Any]) -> list[JobSpec]:
    pending: list[JobSpec] = []
    for t in state["trials"]:
        tid = int(t["trial_id"])
        params = t["params"]
        for j in t["jobs"]:
            if j.get("status") in ("done", "running"):
                log_path = ROOT / j["log"]
                if j.get("status") == "done":
                    continue
                if j.get("status") == "running" and log_is_done(log_path):
                    # stale running -> will be fixed by reconcile
                    pass
                elif j.get("status") == "running":
                    continue
            log_path = ROOT / j["log"]
            if log_is_done(log_path):
                continue
            pending.append(
                JobSpec(
                    trial_id=tid,
                    config=j["config"],
                    dataset=j["dataset"],
                    seed=int(j["seed"]),
                    params=params,
                    log_path=log_path,
                    job_key=j["job_key"],
                )
            )
    return pending


def reconcile_stale_running(state: dict[str, Any]) -> None:
    """Re-queue jobs stuck in running (e.g. runner killed) if log not finished."""
    for t in state["trials"]:
        for j in t["jobs"]:
            if j.get("status") != "running":
                continue
            if log_is_done(ROOT / j["log"]):
                continue
            j["status"] = "queued"
            j["gpu"] = None


def reconcile_done_from_logs(state: dict[str, Any]) -> int:
    n = 0
    for t in state["trials"]:
        for j in t["jobs"]:
            if j.get("status") == "done":
                continue
            log_path = ROOT / j["log"]
            if not log_is_done(log_path):
                continue
            metrics = parse_est_log(log_path.read_text(encoding="utf-8", errors="ignore"))
            j["status"] = "done"
            j["finished_at"] = j.get("finished_at") or utc_now()
            j["tst_u_mae"] = metrics["tst_u_mae"]
            j["tst_u_rmse"] = metrics["tst_u_rmse"]
            j["tst_u_mape"] = metrics["tst_u_mape"]
            j["tst_a_mae"] = metrics["tst_a_mae"]
            j["duration"] = metrics["duration"]
            j["error"] = None
            n += 1
    return n


def run_training(job: JobSpec, gpu: int, pybin: str, dry_run: bool) -> tuple[int, str]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    cm = CONFIG_MAP[job.config]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MOE_EXPERTS"] = cm["MOE_EXPERTS"]
    env["MOE_TOP_K"] = str(cm["MOE_TOP_K"])
    env["MOE_RUN_TAG"] = f"hpo_t{job.trial_id:04d}_{job.config}_{job.dataset}_s{job.seed}"

    argv = build_argv(job.params, job.seed, job.dataset, job.config)
    cmd = [pybin, str(ROOT / MAIN_SCRIPT), *argv]
    if dry_run:
        return 0, "dry-run"

    with open(job.log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n[HPO] launch {utc_now()} gpu={gpu} tag={env['MOE_RUN_TAG']}\n")
        logf.write(f"[HPO] cmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode, ""


def worker_loop(
    gpu: int,
    job_q: queue.Queue,
    store: StateStore,
    pybin: str,
    stop_event: threading.Event,
    dry_run: bool,
) -> None:
    while not stop_event.is_set():
        try:
            job = job_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if job is None:
            job_q.task_done()
            break

        def mark_running(st: dict[str, Any]) -> None:
            j = find_job(st, job.job_key)
            if j:
                j["status"] = "running"
                j["gpu"] = gpu
                j["started_at"] = utc_now()
                j["error"] = None

        store.update(mark_running)

        try:
            rc, _ = run_training(job, gpu, pybin, dry_run)
            err = None
            if rc != 0:
                err = f"exit code {rc}"
            if not log_is_done(job.log_path):
                err = err or "log missing SCRIPT DURATION"

            def mark_finished(st: dict[str, Any]) -> None:
                j = find_job(st, job.job_key)
                if not j:
                    return
                if err:
                    j["status"] = "failed"
                    j["error"] = err
                    j["finished_at"] = utc_now()
                else:
                    metrics = parse_est_log(job.log_path.read_text(encoding="utf-8", errors="ignore"))
                    j["status"] = "done"
                    j["tst_u_mae"] = metrics["tst_u_mae"]
                    j["tst_u_rmse"] = metrics["tst_u_rmse"]
                    j["tst_u_mape"] = metrics["tst_u_mape"]
                    j["tst_a_mae"] = metrics["tst_a_mae"]
                    j["duration"] = metrics["duration"]
                    j["finished_at"] = utc_now()
                    j["error"] = None

            store.update(mark_finished)
            print(
                f"[gpu {gpu}] done trial={job.trial_id} {job.config} {job.dataset} s{job.seed} "
                f"status={'failed' if err else 'ok'}",
                flush=True,
            )
        except Exception as e:
            def mark_exc(st: dict[str, Any]) -> None:
                j = find_job(st, job.job_key)
                if j:
                    j["status"] = "failed"
                    j["error"] = str(e)[:500]
                    j["finished_at"] = utc_now()

            store.update(mark_exc)
            traceback.print_exc()
        finally:
            job_q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description="6-GPU Estimation HPO (topo_only, scpt_topo)")
    ap.add_argument("--n-trials", type=int, default=24, help="number of hyperparameter trials")
    ap.add_argument("--seed", type=int, default=20260516, help="RNG seed for sampling hyperparams")
    ap.add_argument("--state", type=Path, default=Path("logs_hpo_est/hpo_state.json"))
    ap.add_argument("--live-md", type=Path, default=Path("HPO_ESTIMATION_LIVE.md"))
    ap.add_argument("--pybin", type=str, default=os.environ.get("PYBIN", PYBIN_DEFAULT))
    ap.add_argument("--dry-run", action="store_true", help="enqueue only, do not train")
    ap.add_argument("--report-only", action="store_true", help="only refresh live markdown from state")
    args = ap.parse_args()

    state_path = (ROOT / args.state).resolve()
    live_md = (ROOT / args.live_md).resolve()
    store = StateStore(state_path, live_md)

    if args.report_only:
        state = store.read()
        reconcile_done_from_logs(state)
        store.write(state)
        store._write_live_md(state)
        print(f"[ok] report-only -> {live_md}")
        return

    rng = random.Random(args.seed)

    def init_state(st: dict[str, Any]) -> None:
        st.setdefault("meta", {})
        st["meta"].update(
            {
                "seeds": list(SEEDS),
                "datasets": list(DATASETS),
                "configs": list(CONFIGS),
                "gpus": list(GPUS),
                "search_space": DEFAULT_SEARCH_SPACE,
                "n_trials": args.n_trials,
                "sampler_seed": args.seed,
                "state_path": str(state_path.relative_to(ROOT)),
            }
        )
        ensure_trials(st, args.n_trials, rng)
        reconcile_stale_running(st)
        reconcile_done_from_logs(st)

    store.update(init_state)
    state = store.read()
    pending = collect_pending_jobs(state)
    print(f"[HPO] trials={args.n_trials} pending_jobs={len(pending)} gpus={GPUS}")

    if not pending:
        print("[HPO] nothing pending.")
        return

    job_q: queue.Queue = queue.Queue()
    for job in pending:
        job_q.put(job)

    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for gpu in GPUS:
        t = threading.Thread(
            target=worker_loop,
            args=(gpu, job_q, store, args.pybin, stop_event, args.dry_run),
            daemon=True,
            name=f"gpu-{gpu}",
        )
        t.start()
        threads.append(t)

    job_q.join()
    stop_event.set()
    for _ in GPUS:
        job_q.put(None)
    for t in threads:
        t.join(timeout=5)

    def final_reconcile(st: dict[str, Any]) -> None:
        reconcile_done_from_logs(st)

    store.update(final_reconcile)
    print(f"[HPO] finished. live report: {live_md}")


if __name__ == "__main__":
    main()
