"""One bounded RLT-02 lane: round-robin matched continuations and immutable snapshots.

Two separate vramorama leases run --arch loop and --arch hrm concurrently. No fresh
evaluation outcomes are read by this trainer. Checkpoints survive resource pauses.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .rlt_pilot import dump

FEEDBACKS = ("direct", "rlt", "matched_damping")


def snapshot(run, target, parent_step):
    import torch
    from .train import load_verified, sha256
    which, ck = load_verified(run, "cpu")
    if ck is None or ck["step"] != target:
        raise RuntimeError(f"missing exact snapshot: {run.name}, step {target}")
    dest = run / "snapshots" / f"step{target:05d}"
    dest.mkdir(parents=True, exist_ok=True)
    total = parent_step + target
    torch.save({"model": ck["model"], "config": ck["config"], "preset": ck["preset"],
                "step": total, "phase_step": target, "identity_hash": ck["identity_hash"]}, dest / "best.part")
    os.replace(dest / "best.part", dest / "best.pt")
    metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    val = next(r["val"]["bits_per_byte"] for r in reversed(metrics) if r["step"] == target and "val" in r)
    dump(dest / "best.json", {"step": total, "phase_step": target, "bits_per_byte": val,
                               "sha256": sha256(dest / "best.pt"), "selection": "exact milestone, not best-on-val"})
    shutil.copyfile(run / "config.json", dest / "config.json")
    dump(dest / "snapshot.json", {"phase_step": target, "total_step": total,
                                   "training_checkpoint_sha256": sha256(run / f"{which}.pt")})
    return dest


def common_milestone(runs):
    sets = [{int(p.name.removeprefix("step")) for p in (run / "snapshots").glob("step*")
             if (p / "snapshot.json").exists()} for run in runs]
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("no fully matched milestone; do not compare partial arms")
    return max(common)


def probe(paths, calibration, out, target):
    import torch
    from .mathlib_eval import proposals
    from .tactic_lm import load_model, norm_tactic
    states = [json.loads(l) for l in calibration.read_text(encoding="utf-8").splitlines()][:8]
    for path in paths:
        model, _ = load_model(path, "cuda")
        beams = proposals(model, states, "cuda", 8, 8, bs=8)
        top = [p[0] if p else "<empty>" for p in beams]
        row = {"phase_step": target, "arm": path.parents[1].name, "states": len(states),
               "top1": top, "non_simp_top1": sum(t not in ("simp", "<empty>") for t in top),
               "distinct_top1": len(set(top)),
               "reference_top1": sum(t == norm_tactic(s["tactic"]) for t, s in zip(top, states)),
               "kind": "old calibration proposal diversity; not Lean validity"}
        with (out / "diversity_probes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"event": "diversity", **row}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True, choices=("loop", "hrm"))
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--max-minutes", type=float, required=True)
    ap.add_argument("--gpu-duty", type=float, default=.5)
    ap.add_argument("--priority", choices=("normal", "above_normal"), default="normal")
    ap.add_argument("--soft-gpu-temp", type=float, default=78)
    a = ap.parse_args()
    if not os.environ.get("VRAMORAMA_LEASE"):
        ap.error("CUDA lane requires a vramorama lease")
    if min(a.batch, a.max_minutes) <= 0 or not 0 < a.gpu_duty <= 1:
        ap.error("positive batch/budget and GPU duty in (0,1] required")
    if not 0 < a.soft_gpu_temp < 85:
        ap.error("soft-gpu-temp must stay below the85C hard pause")
    from .ramcap import Job, parse_size
    job = Job(parse_size("7G"), a.priority); job.assign_self()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(a.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    if a.batch != protocol["batch"] or a.grad_ckpt != protocol["grad_ckpt"] or a.max_minutes > 230:
        raise ValueError("lane differs from the registered batch/checkpointing/budget")
    identity = {"protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "arch": a.arch, "batch": a.batch, "grad_ckpt": a.grad_ckpt,
                "phase_updates": protocol["additional_phase_horizon"], "chunk": protocol["chunk_updates"]}
    if (out / "lane.json").exists():
        manifest = json.loads((out / "lane.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("lane belongs to a different training protocol")
    else:
        manifest = {"identity": identity, "created_unix": time.time()}
        dump(out / "lane.json", manifest)
    if (out / "training_complete.json").exists():
        print(json.dumps({"event": "already_complete", "out": str(out)}), flush=True); return 0
    with (out / "resource_segments.jsonl").open("a") as f:
        f.write(json.dumps({"unix": time.time(), "pid": os.getpid(), "gpu_duty": a.gpu_duty,
                            "max_minutes": a.max_minutes, "job_cap_gib": 7, "lease_gib": 4,
                            "priority": a.priority, "soft_gpu_temp": a.soft_gpu_temp}) + "\n")
    import torch
    from .tactic_lm import train, parser
    from .train import load_verified
    torch.set_num_threads(2)
    total_mib = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    fraction = 3.6 * 1024 / total_mib
    torch.cuda.set_per_process_memory_fraction(fraction)
    runs = [out / f"{a.arch}_{feedback}_s131" for feedback in FEEDBACKS]
    exhausted = False
    for target in range(identity["chunk"], identity["phase_updates"] + 1, identity["chunk"]):
        for run, feedback in zip(runs, FEEDBACKS):
            if (run / "snapshots" / f"step{target:05d}" / "snapshot.json").exists():
                continue
            remaining = a.max_minutes - (time.time() - manifest["created_unix"]) / 60
            if remaining <= 0:
                exhausted = True; break
            parent = protocol["parents"][run.name]
            if hashlib.sha256((Path(parent["path"]) / "latest.pt").read_bytes()).hexdigest() != parent["sha256"]:
                raise RuntimeError("registered parent changed")
            cmd = ["--data", protocol["data"], "--preset", f"tlm-{a.arch}-XS", "--cfg", f"feedback={feedback}",
                   "--steps", str(identity["phase_updates"]), "--stop-at", str(target),
                   "--continue-from", parent["path"], "--batch", str(a.batch), "--seed", "131", "--lr", ".0005",
                   "--warmup", "200", "--val-items", "256", "--val-batch", "16", "--eval-every", "500",
                   "--log-every", "50", "--save-every", "100", "--threads", "2", "--device", "cuda",
                   "--cuda-memory-fraction", str(fraction), "--gpu-duty", str(a.gpu_duty), "--soft-gpu-temp", str(a.soft_gpu_temp),
                   "--max-minutes", str(remaining), "--out", str(run)]
            if a.grad_ckpt: cmd += ["--grad-ckpt"]
            dump(out / "status.json", {"pid": os.getpid(), "arm": run.name, "target": target, "unix": time.time()})
            print(json.dumps({"event": "train_chunk", "arm": run.name, "target": target,
                              "remaining_minutes": remaining}), flush=True)
            with (out / f"{run.name}.log").open("a", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    code = train(parser().parse_args(cmd))
            gc.collect(); torch.cuda.empty_cache()
            if code == 75: return code
            if code == 76:
                _, partial = load_verified(run, "cpu")
                if partial is not None and partial["step"] == target:
                    snapshot(run, target, parent["step"])
                del partial
                exhausted = True; break
            if code: return code
            snapshot(run, target, parent["step"])
            gc.collect(); torch.cuda.empty_cache()
        if exhausted: break
        print(json.dumps({"event": "matched_milestone", "arch": a.arch, "phase_step": target,
                          "total_step": target + 1000}), flush=True)
        if target % 1000 == 0:
            probe([r / "snapshots" / f"step{target:05d}" for r in runs],
                  protocol_path.parent / "calibration.jsonl", out, target)
    selected = common_milestone(runs)
    selected_paths = []
    for run in runs:
        src = run / "snapshots" / f"step{selected:05d}"
        dest = out / "selected" / run.name; dest.mkdir(parents=True, exist_ok=True)
        for file in ("best.pt", "best.json", "config.json", "snapshot.json"):
            shutil.copyfile(src / file, dest / file)
        selected_paths.append(str(dest))
    elapsed = time.time() - manifest["created_unix"]
    dump(out / "training_complete.json", {"selected_paths": selected_paths, "phase_step": selected,
                                           "total_step": selected + 1000, "elapsed_seconds": elapsed,
                                           "peak_commit_gib": job.peak_bytes() / 2**30,
                                           "budget_exhausted": exhausted, "identity": identity,
                                           "trained_presentations_at_selected": 16000 + selected * a.batch})
    print(json.dumps({"event": "lane_complete", "arch": a.arch, "selected_phase_step": selected,
                      "seconds": elapsed}), flush=True)
    return 0


if __name__ == "__main__":
    from .hw import ResourcePause
    try:
        raise SystemExit(main())
    except ResourcePause as e:
        print(json.dumps({"event": "pause", "reason": str(e)}), flush=True)
        raise SystemExit(75)
