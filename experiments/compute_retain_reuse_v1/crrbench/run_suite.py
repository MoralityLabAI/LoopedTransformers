"""Run the whole benchmark grid with a few persistent worker processes.

    python -m crrbench.run_suite --out results/full --seeds 6 --workers 4
Cells are ordered seed-major, so an interrupted run still has complete task x regime coverage
for the first seeds.  Finished cells (results/<out>/cells/*.json) are skipped on restart.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback

from .tasks import TASKS, real_path
from .train import REGIMES

DEFAULT_TASKS = ["cyclic", "dfa", "storyworld", "sweepweave"]


def jobs(tasks, regimes, seeds):
    return [(t, r, s) for s in range(seeds) for t in tasks for r in regimes]


def worker(a):
    import torch
    from .run_cell import run_cell
    js = jobs(a.tasks, a.regimes, a.seeds)[a.worker_id::a.workers]
    for (t, r, s) in js:
        cell = f"{t}__{r}__s{s}"
        if os.path.exists(os.path.join(a.out, "cells", cell + ".json")) or                 (a.train_only and os.path.exists(os.path.join(a.out, "models", cell + ".pt"))):
            continue
        try:
            run_cell(t, r, s, a.out, a.config, a.device or ("cuda" if torch.cuda.is_available() else "cpu"), a.storyworld_dir,
                     train_only=a.train_only, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
        except Exception:
            print(f"FAILED {cell}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/full")
    ap.add_argument("--config", default="full")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=list(TASKS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worker-id", type=int, default=None)
    ap.add_argument("--storyworld-dir", default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads per worker (0 = default)")
    ap.add_argument("--train-only", action="store_true", help="train and cache models only (evaluation is rerunnable from the cache)")
    a = ap.parse_args()
    a.out = real_path(a.out)
    os.makedirs(os.path.join(a.out, "worker_logs"), exist_ok=True)
    if a.worker_id is not None:
        if a.threads:
            import torch
            torch.set_num_threads(a.threads)
        return worker(a)
    def launch(i):
        cmd = [sys.executable, "-u", "-m", "crrbench.run_suite", "--out", a.out, "--config", a.config, "--seeds", str(a.seeds),
               "--workers", str(a.workers), "--worker-id", str(i), "--tasks", *a.tasks, "--regimes", *a.regimes]
        if a.storyworld_dir:
            cmd += ["--storyworld-dir", a.storyworld_dir]
        if a.train_only:
            cmd += ["--train-only"]
        if a.device:
            cmd += ["--device", a.device]
        if a.threads:
            cmd += ["--threads", str(a.threads)]
        return subprocess.Popen(cmd, stdout=open(os.path.join(a.out, "worker_logs", f"w{i}.log"), "a"), stderr=subprocess.STDOUT)

    procs = {i: launch(i) for i in range(a.workers)}
    restarts = {i: 0 for i in procs}
    while procs:
        time.sleep(5)
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if rc == 3 and restarts[i] < 8:
                restarts[i] += 1
                time.sleep(20)
                procs[i] = launch(i)
            else:
                del procs[i]


if __name__ == "__main__":
    main()
