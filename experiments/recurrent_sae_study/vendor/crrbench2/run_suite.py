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

from .tasks import TASKS
from .train import REGIMES

DEFAULT_TASKS = ["cyclic", "dfa", "storyworld"]


def jobs(tasks, regimes, seeds):
    return [(t, r, s) for s in range(seeds) for t in tasks for r in regimes]


def worker(a):
    import torch
    from .run_cell import run_cell
    js = jobs(a.tasks, a.regimes, a.seeds)[a.worker_id::a.workers]
    for (t, r, s) in js:
        cell = f"{t}__{r}__s{s}"
        if os.path.exists(os.path.join(a.out, "cells", cell + ".json")):
            continue
        try:
            run_cell(t, r, s, a.out, a.config, "cuda" if torch.cuda.is_available() else "cpu", a.storyworld_dir,
                     log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
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
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "worker_logs"), exist_ok=True)
    if a.worker_id is not None:
        return worker(a)
    procs = []
    for i in range(a.workers):
        cmd = [sys.executable, "-u", "-m", "crrbench.run_suite", "--out", a.out, "--config", a.config, "--seeds", str(a.seeds),
               "--workers", str(a.workers), "--worker-id", str(i), "--tasks", *a.tasks, "--regimes", *a.regimes]
        if a.storyworld_dir:
            cmd += ["--storyworld-dir", a.storyworld_dir]
        procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(a.out, "worker_logs", f"w{i}.log"), "a"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
