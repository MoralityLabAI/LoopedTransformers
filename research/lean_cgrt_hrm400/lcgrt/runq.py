"""Sequential experiment queue: one GPU job at a time, resumable, stops on repeated failure.

    python -m lcgrt.runq --campaign c1 --queue configs/campaign_c1.jsonl

Each queue line is {"name": ..., "module": "lcgrt.train", "args": [...], "vram": "6G"}.
A job is done when its output directory holds `complete.json`. On Windows each job is
admitted by `vramorama run` (waits for VRAM, refuses a power-capped card, restarts the
job after a GPU reset or a pause exit 75); elsewhere it runs directly. The queue file is
re-read before every job, so lines can be appended while it runs.
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


def log(path: Path, **row):
    row["unix"] = int(time.time())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def job_command(job, out: Path):
    py = [sys.executable, "-u", "-m", job.get("module", "lcgrt.train"), *job["args"], "--out", str(out)]
    if config.VRAMORAMA and Path(config.VRAMORAMA).exists():
        return [str(config.VRAMORAMA), "run", "--vram", job.get("vram", "6G"), "--owner", f"lcgrt-{job['name']}",
                "--restart", str(job.get("restarts", 60)), "--guard-power", "120", "--pause-exit", "75", "--", *py]
    return py


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--max-consecutive-failures", type=int, default=2)
    a = ap.parse_args(argv)
    root = config.RUN_ROOT / a.campaign
    root.mkdir(parents=True, exist_ok=True)
    qlog = root / "queue.log"
    log(qlog, event="queue_start", pid=os.getpid(), queue=a.queue)
    failures = 0
    attempted: set[str] = set()
    while True:
        jobs = [json.loads(l) for l in Path(a.queue).read_text(encoding="utf-8").splitlines() if l.strip()]
        todo = [j for j in jobs if not (root / j["name"] / "complete.json").exists() and j["name"] not in attempted]
        if not todo:
            log(qlog, event="queue_empty"); return 0
        job = todo[0]
        out = root / job["name"]; out.mkdir(parents=True, exist_ok=True)
        (root / "heartbeat.json").write_text(json.dumps({"job": job["name"], "unix": int(time.time())}))
        log(qlog, event="job_start", job=job["name"])
        t = time.time()
        with (out / "train.log").open("a", encoding="utf-8") as f:
            rc = subprocess.run(job_command(job, out), stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(config.PKG_ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        done = (out / "complete.json").exists()
        log(qlog, event="job_end", job=job["name"], returncode=rc, complete=done, minutes=round((time.time() - t) / 60, 1))
        if done:
            failures = 0
        else:
            failures += 1
            attempted.add(job["name"])
            if failures >= a.max_consecutive_failures:
                log(qlog, event="queue_stopped", reason=f"{failures} consecutive failures"); return 3


if __name__ == "__main__":
    sys.exit(main())
