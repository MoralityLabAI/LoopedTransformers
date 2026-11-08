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
With --cpu-only (no GPU; started by `lcgrt.cpugym` under one RAM cap) jobs run directly, without
vramorama, with `--device cpu` and --threads torch threads, and a pause exit (75: battery, disk)
waits five minutes and runs the job again. Train lines run in exactly one mode: those marked
"cpu": true only under --cpu-only (a CPU-sized experiment stays on one device), the rest only
on the GPU path.
Several runners may serve one lane: each job is taken through an atomic `claim.json` in its
run directory, so no job runs twice. An eval line may carry "after": [job names] to wait for other jobs (e.g. a long CPU-only
evaluation that should not hold the lane while a shorter one waits for its checkpoints), and
any line may carry "requires": [paths] that must exist before it starts (a dataset manifest).
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

EXIT_PAUSE = 75
EVAL_MODULES = ("lcgrt.evaluate", "lcgrt.branch", "lcgrt.mathlib_eval")


def log(path: Path, **row):
    row["unix"] = int(time.time())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def lane_of(job):
    return "eval" if job.get("module", "lcgrt.train") in EVAL_MODULES else "train"


def flag_values(args, flag):
    """Values after `flag` up to the next option (`--ckpt A B C` names several runs)."""
    if flag not in args:
        return []
    i = args.index(flag) + 1
    vals = []
    while i < len(args) and not args[i].startswith("--"):
        vals.append(args[i]); i += 1
    return vals


def ready(job, root=None):
    """An eval job can start only once what it reads exists, and after the jobs named in its
    optional "after" list are complete."""
    if root is not None and any(not (root / n / "complete.json").exists() for n in job.get("after", [])):
        return False
    if any(not Path(f).exists() for f in job.get("requires", [])):     # e.g. a dataset manifest
        return False
    args = job["args"]
    for flag, marker in (("--ckpt", "complete.json"), ("--branch-weights", None)):
        for v in flag_values(args, flag):
            p = Path(v)
            if not (p / marker).exists() if marker else not p.exists():
                return False
    return True


def wanted(job, lane: str, cpu_only: bool) -> bool:
    if lane != "all" and lane_of(job) != lane:
        return False
    return lane_of(job) == "eval" or bool(job.get("cpu")) == cpu_only


def job_command(job, out: Path, lane: str, threads: int, cpu_only: bool = False):
    args = list(job["args"])
    if lane == "eval" or cpu_only:
        args += ["--device", "cpu"]
    if cpu_only and lane_of(job) == "train":
        args += ["--threads", str(threads)]
    py = [sys.executable, "-u", "-m", job.get("module", "lcgrt.train"), *args, "--out", str(out)]
    if cpu_only or not (config.VRAMORAMA and Path(config.VRAMORAMA).exists()):
        return py
    if lane == "eval":
        return [str(config.VRAMORAMA), "run", "--vram", "64M", "--max-commit", "85",
                "--owner", f"lcgrt-cpu-{job['name']}", "--", *py]
    return [str(config.VRAMORAMA), "run", "--vram", job.get("vram", "6G"), "--owner", f"lcgrt-{job['name']}",
            "--restart", str(job.get("restarts", 60)), "--guard-power", "120", "--pause-exit", "75", "--", *py]


def _alive(pid, created):
    import psutil
    try:
        return abs(psutil.Process(pid).create_time() - created) < 1.0
    except psutil.NoSuchProcess:
        return False


def claim(out: Path, lane: str) -> bool:
    """Take a job for this runner (atomic create), so several runners can share a lane. A claim
    left by a runner that has exited is taken over."""
    import psutil
    p = out / "claim.json"
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                c = json.loads(p.read_text())
            except (OSError, ValueError):
                return False                         # being written by another runner
            if _alive(c["pid"], c["created"]):
                return False
            p.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "created": psutil.Process().create_time(), "lane": lane,
                       "unix": int(time.time())}, f)
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--lane", choices=("all", "train", "eval"), default="all")
    ap.add_argument("--threads", type=int, default=4, help="eval lane, or any lane with --cpu-only: torch CPU threads")
    ap.add_argument("--cpu-only", action="store_true", help="no GPU: run jobs directly on the CPU (see above)")
    ap.add_argument("--max-consecutive-failures", type=int, default=2)
    ap.add_argument("--after-pid", type=int, default=None,
                    help="start once this process has exited: hands a lane over from a runner whose "
                         "current job is still running, without running that job twice")
    a = ap.parse_args(argv)
    root = config.RUN_ROOT / a.campaign
    root.mkdir(parents=True, exist_ok=True)
    qlog = root / "queue.log"
    if a.after_pid:
        import psutil
        try:
            log(qlog, event="queue_waiting", pid=os.getpid(), after_pid=a.after_pid, lane=a.lane)
            psutil.Process(a.after_pid).wait()      # tracks that process, not a later reuse of its id
        except psutil.NoSuchProcess:
            pass
    log(qlog, event="queue_start", pid=os.getpid(), queue=a.queue, lane=a.lane)
    failures = 0
    attempted: set[str] = set()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if a.lane == "eval" or a.cpu_only:
        env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    while True:
        jobs = [json.loads(l) for l in Path(a.queue).read_text(encoding="utf-8").splitlines() if l.strip()]
        todo = [j for j in jobs if not (root / j["name"] / "complete.json").exists() and j["name"] not in attempted
                and not j.get("skip") and wanted(j, a.lane, a.cpu_only)]
        if not todo:
            log(qlog, event="queue_empty", lane=a.lane); return 0
        runnable = [j for j in todo if ready(j, root)]
        if not runnable:
            time.sleep(60)
            continue
        job = out = None
        for j in runnable:
            o = root / j["name"]; o.mkdir(parents=True, exist_ok=True)
            if claim(o, a.lane):
                job, out = j, o
                break
        if job is None:                              # everything runnable is taken by other runners
            time.sleep(60)
            continue
        (root / f"heartbeat_{a.lane}.json").write_text(json.dumps({"job": job["name"], "pid": os.getpid(), "unix": int(time.time())}))
        log(qlog, event="job_start", job=job["name"], lane=a.lane)
        t = time.time()
        while True:
            with (out / "train.log").open("a", encoding="utf-8") as f:
                rc = subprocess.run(job_command(job, out, a.lane, a.threads, a.cpu_only), stdout=f,
                                    stderr=subprocess.STDOUT, cwd=str(config.PKG_ROOT), env=env).returncode
            if not (a.cpu_only and rc == EXIT_PAUSE):
                break
            log(qlog, event="job_paused", job=job["name"], lane=a.lane)   # vramorama restarts these on the GPU path
            time.sleep(300)
        done = (out / "complete.json").exists()
        (out / "claim.json").unlink(missing_ok=True)
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
