"""Experiment queue in lanes: GPU training one job at a time; CPU evaluation beside it.

    python -m lcgrt.runq --campaign c1 --queue configs/campaign_c1.jsonl --lane train
    python -m lcgrt.runq --campaign c1 --queue configs/campaign_c1.jsonl --lane eval

Each queue line is {"name": ..., "module": "lcgrt.train", "args": [...], "vram": "6G"}.
A job is done when its output directory holds `complete.json`; a line with "skip" (the reason)
is never run, e.g. a run superseded by a re-queued copy.

Lanes (disjoint, so two runners never take the same job):
  train  lcgrt.train jobs. On Windows each is admitted by `vramorama run` (waits for VRAM,
         refuses a power-capped card, restarts the job after a GPU reset or a pause exit 75).
  eval   lcgrt.evaluate / lcgrt.branch jobs, run on the CPU with the model held in RAM
         (`--device cpu`), so Lean-bound evaluation no longer waits behind GPU training.
         Admitted through a 64 MiB vramorama lease with --max-commit 85, which holds a job
         back while the commit charge is high. A job whose checkpoint (or branch weights)
         is not complete yet is skipped until it is.
  all    every job, in file order, on the GPU path (the original single-lane behaviour).
The queue file is re-read before every job, so lines can be appended while it runs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config

EVAL_MODULES = ("lcgrt.evaluate", "lcgrt.branch")


def log(path: Path, **row):
    row["unix"] = int(time.time())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def lane_of(job):
    return "eval" if job.get("module", "lcgrt.train") in EVAL_MODULES else "train"


def ready(job):
    """An eval job can start only once what it reads exists."""
    args = job["args"]
    for flag, marker in (("--ckpt", "complete.json"), ("--branch-weights", None)):
        if flag in args:
            p = Path(args[args.index(flag) + 1])
            if not (p / marker).exists() if marker else not p.exists():
                return False
    return True


def job_command(job, out: Path, lane: str, threads: int):
    args = list(job["args"])
    if lane == "eval":
        args += ["--device", "cpu"]
    py = [sys.executable, "-u", "-m", job.get("module", "lcgrt.train"), *args, "--out", str(out)]
    if not (config.VRAMORAMA and Path(config.VRAMORAMA).exists()):
        return py
    if lane == "eval":
        return [str(config.VRAMORAMA), "run", "--vram", "64M", "--max-commit", "85",
                "--owner", f"lcgrt-cpu-{job['name']}", "--", *py]
    return [str(config.VRAMORAMA), "run", "--vram", job.get("vram", "6G"), "--owner", f"lcgrt-{job['name']}",
            "--restart", str(job.get("restarts", 60)), "--guard-power", "120", "--pause-exit", "75", "--", *py]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--lane", choices=("all", "train", "eval"), default="all")
    ap.add_argument("--threads", type=int, default=4, help="eval lane: torch CPU threads")
    ap.add_argument("--max-consecutive-failures", type=int, default=2)
    a = ap.parse_args(argv)
    root = config.RUN_ROOT / a.campaign
    root.mkdir(parents=True, exist_ok=True)
    qlog = root / "queue.log"
    log(qlog, event="queue_start", pid=os.getpid(), queue=a.queue, lane=a.lane)
    failures = 0
    attempted: set[str] = set()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if a.lane == "eval":
        env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    while True:
        jobs = [json.loads(l) for l in Path(a.queue).read_text(encoding="utf-8").splitlines() if l.strip()]
        todo = [j for j in jobs if not (root / j["name"] / "complete.json").exists() and j["name"] not in attempted
                and not j.get("skip") and (a.lane == "all" or lane_of(j) == a.lane)]
        if not todo:
            log(qlog, event="queue_empty", lane=a.lane); return 0
        runnable = [j for j in todo if a.lane != "eval" or ready(j)]
        if not runnable:
            time.sleep(60)
            continue
        job = runnable[0]
        out = root / job["name"]; out.mkdir(parents=True, exist_ok=True)
        (root / f"heartbeat_{a.lane}.json").write_text(json.dumps({"job": job["name"], "unix": int(time.time())}))
        log(qlog, event="job_start", job=job["name"], lane=a.lane)
        t = time.time()
        with (out / "train.log").open("a", encoding="utf-8") as f:
            rc = subprocess.run(job_command(job, out, a.lane, a.threads), stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(config.PKG_ROOT), env=env).returncode
        done = (out / "complete.json").exists()
        log(qlog, event="job_end", job=job["name"], lane=a.lane, returncode=rc, complete=done,
            minutes=round((time.time() - t) / 60, 1))
        if done:
            failures = 0
        else:
            failures += 1
            attempted.add(job["name"])
            if failures >= a.max_consecutive_failures:
                log(qlog, event="queue_stopped", lane=a.lane, reason=f"{failures} consecutive failures"); return 3


if __name__ == "__main__":
    sys.exit(main())
