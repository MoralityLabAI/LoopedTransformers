"""The gym without a GPU, under one hard RAM cap.

    python -m lcgrt.cpugym --campaign c1 --queue configs/campaign_c1.jsonl --ram-cap 9G

Puts itself in a Windows job object (`lcgrt.ramcap`) with the RAM cap and Idle priority, then
starts one CPU-only runner per lane (`runq --cpu-only`). Every process they start (trainers,
evaluators, Lean) inherits the job, so the cap is on the sum of the whole gym, not per job,
and anything else on the machine keeps priority over it. The train lane runs only queue lines
marked "cpu": true; GPU lines wait for a GPU runner. Killing this process (one pid) ends the
whole tree. Status (cap, peak and current commit, live pids) goes to
`<run root>/cpugym_status.json` every minute; start and end go to queue.log.

Measured on a Ryzen AI 9 365 beside a CPU-heavy job in WSL, under the cap (committed memory):
- XS training at its worst case (loop-XS at T = 12, with validation): 5.85 GiB;
- a Gate 0 evaluation: about 1.6 GiB;
- the Mathlib evaluation with 2 Lean workers: 5.83 GiB. It needs every c3-cpu checkpoint, so it
  never runs beside training.
One runner per lane therefore fits in 9 GiB.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from . import config
from .ramcap import Job, parse_size
from .runq import log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--ram-cap", default="9G")
    ap.add_argument("--priority", default="idle", choices=("idle", "below_normal", "normal"))
    ap.add_argument("--lanes", nargs="+", default=["train", "eval"], choices=("train", "eval"))
    ap.add_argument("--train-threads", type=int, default=4)
    ap.add_argument("--eval-threads", type=int, default=3)
    ap.add_argument("--status-every", type=int, default=60)
    a = ap.parse_args(argv)
    root = config.RUN_ROOT / a.campaign
    root.mkdir(parents=True, exist_ok=True)
    qlog = root / "queue.log"
    job = Job(parse_size(a.ram_cap), a.priority)
    job.assign_self()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    runners = {}
    for lane in a.lanes:
        threads = a.train_threads if lane == "train" else a.eval_threads
        out = (root / f"runq_{lane}_cpu.out").open("a", encoding="utf-8")
        runners[lane] = subprocess.Popen(
            [sys.executable, "-u", "-m", "lcgrt.runq", "--campaign", a.campaign, "--queue", a.queue, "--lane", lane,
             "--cpu-only", "--threads", str(threads)], stdout=out, stderr=subprocess.STDOUT, cwd=str(config.PKG_ROOT),
            env=env)
    log(qlog, event="cpugym_start", pid=os.getpid(), ram_cap_gib=round(job.limit / 2**30, 2), priority=a.priority,
        runners={k: p.pid for k, p in runners.items()})
    seen = 0
    while True:
        alive = {k: p.poll() is None for k, p in runners.items()}
        now = job.committed_bytes(); seen = max(seen, now)
        status = {"pid": os.getpid(), "ram_cap_gib": round(job.limit / 2**30, 2),
                  "now_gib": round(now / 2**30, 2), "max_sampled_gib": round(seen / 2**30, 2),
                  "peak_gib": round(job.peak_bytes() / 2**30, 2),     # Windows' peak; counts refused requests
                  "processes": len(job.pids()), "runners_alive": alive, "unix": int(time.time())}
        (root / "cpugym_status.json").write_text(json.dumps(status) + "\n")
        if not any(alive.values()):
            break
        time.sleep(a.status_every)
    log(qlog, event="cpugym_end", peak_gib=status["peak_gib"], max_sampled_gib=status["max_sampled_gib"],
        returncodes={k: p.returncode for k, p in runners.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
