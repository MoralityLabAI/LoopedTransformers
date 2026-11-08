"""The gym under one hard RAM cap, with no GPU or with a share of one.

    python -m lcgrt.cpugym --campaign c1 --queue configs/campaign_c1.jsonl --ram-cap 9G
    python -m lcgrt.cpugym --campaign c1 --queue configs/campaign_c1.jsonl --ram-cap 9G --gpu-vram 9G --gpu-duty 0.6

Puts itself in a Windows job object (`lcgrt.ramcap`) with the RAM cap and Idle priority, then
starts one runner per lane. Every process they start (trainers, evaluators, Lean) inherits the
job, so the cap is on the sum of the whole gym, not per job, and anything else on the machine
keeps priority over it. Killing this process (one pid) ends the whole tree. Status (cap, commit,
resident RAM, live pids) goes to `<run root>/cpugym_status.json` every minute; start and end go
to queue.log.

Without --gpu-vram both lanes are CPU-only (`runq --cpu-only`): the train lane runs only queue
lines marked "cpu": true and GPU lines wait. With --gpu-vram V the train lane is a GPU runner:
one job at a time, each admitted by a vramorama lease and capped at it, a training job kept at
--gpu-duty of the card. Under WDDM the card's memory a CUDA process allocates is charged to its
commit, so the job's commit cap becomes ram-cap + V; the RAM held (`resident_gib`, private
working sets) is what the ram-cap budgets. The eval lane stays on the CPU.

Measured on a Ryzen AI 9 365 beside a CPU-heavy job in WSL, under the cap:
- CPU: XS training at its worst case (loop-XS at T = 12, with validation) commits 5.85 GiB; a
  Gate 0 evaluation about 2.4 GiB; the Mathlib evaluation with 2 Lean workers 5.83 GiB. It needs
  every checkpoint of its experiment, so it never runs beside that experiment's training.
- GPU: loop-XS fp32 at T = 12 uses 4.7 GB of VRAM, commits 7.0 GiB and holds 1.3 GiB of RAM;
  loop-S (the c3 recipe) at T = 12 uses 8.0 GB, commits 10.4 GiB and holds 1.5 GiB.
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
    ap.add_argument("--gpu-vram", default="0", help="VRAM allowance of the GPU train lane; 0 = no GPU")
    ap.add_argument("--gpu-duty", type=float, default=0.6, help="share of the card a training job keeps busy")
    ap.add_argument("--priority", default="idle", choices=("idle", "below_normal", "normal"))
    ap.add_argument("--lanes", nargs="+", default=["train", "eval"], choices=("train", "eval"))
    ap.add_argument("--train-threads", type=int, default=4)
    ap.add_argument("--eval-threads", type=int, default=3)
    ap.add_argument("--status-every", type=int, default=60)
    a = ap.parse_args(argv)
    root = config.RUN_ROOT / a.campaign
    root.mkdir(parents=True, exist_ok=True)
    qlog = root / "queue.log"
    ram, vram = parse_size(a.ram_cap), parse_size(a.gpu_vram)
    job = Job(ram + vram, a.priority)
    job.assign_self()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    runners = {}
    for lane in a.lanes:
        gpu = lane == "train" and vram > 0
        threads = a.train_threads if lane == "train" else a.eval_threads
        mode = ["--gpu-duty", str(a.gpu_duty)] if gpu else ["--cpu-only"]
        out = (root / f"runq_{lane}_{'gpu' if gpu else 'cpu'}.out").open("a", encoding="utf-8")
        runners[lane] = subprocess.Popen(
            [sys.executable, "-u", "-m", "lcgrt.runq", "--campaign", a.campaign, "--queue", a.queue, "--lane", lane,
             *mode, "--threads", str(threads)], stdout=out, stderr=subprocess.STDOUT, cwd=str(config.PKG_ROOT),
            env=env)
    log(qlog, event="cpugym_start", pid=os.getpid(), ram_cap_gib=round(ram / 2**30, 2),
        gpu_vram_gib=round(vram / 2**30, 2), commit_cap_gib=round(job.limit / 2**30, 2), gpu_duty=a.gpu_duty,
        priority=a.priority, runners={k: p.pid for k, p in runners.items()})
    seen = held = 0
    while True:
        alive = {k: p.poll() is None for k, p in runners.items()}
        now, res = job.committed_bytes(), job.resident_bytes()
        seen, held = max(seen, now), max(held, res)
        status = {"pid": os.getpid(), "ram_cap_gib": round(ram / 2**30, 2), "commit_cap_gib": round(job.limit / 2**30, 2),
                  "resident_gib": round(res / 2**30, 2), "max_resident_gib": round(held / 2**30, 2),
                  "now_gib": round(now / 2**30, 2), "max_sampled_gib": round(seen / 2**30, 2),
                  "peak_gib": round(job.peak_bytes() / 2**30, 2),     # Windows' peak commit; counts refused requests
                  "processes": len(job.pids()), "runners_alive": alive, "unix": int(time.time())}
        (root / "cpugym_status.json").write_text(json.dumps(status) + "\n")
        if not any(alive.values()):
            break
        time.sleep(a.status_every)
    log(qlog, event="cpugym_end", peak_gib=status["peak_gib"], max_sampled_gib=status["max_sampled_gib"],
        max_resident_gib=status["max_resident_gib"], returncodes={k: p.returncode for k, p in runners.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
