"""Bounded six-arm feedback pilot on Mathlib val; never evaluates registered test splits.

Run under vramorama for CUDA. The Windows process tree is capped before importing torch.
Existing hash-verified trainer checkpoints make interrupted arms resumable.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def diagnostics(run, data, threads=2):
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    from . import config
    from .tactic_lm import PAD, Store, load_model

    torch.set_num_threads(threads)
    model, ck = load_model(run, "cpu")
    store = Store(config.DATA_ROOT / data / "packed_tlm" / "val.npz")
    indices = np.sort(np.random.default_rng(261116).choice(store.n, min(128, store.n), replace=False))
    model.eval()
    rows = {}
    with torch.no_grad():
        for budget in (4, 8, 16, 32):
            nll, nbytes, exact, examples = 0., 0, 0, 0
            for start in range(0, len(indices), 8):
                b = store.batch(indices[start:start + 8], "cpu")
                logits = model(b, budget)
                mask = b["target"] != PAD
                nll += float(F.cross_entropy(logits[mask], b["target"][mask], reduction="sum"))
                nbytes += int(mask.sum())
                exact += int(((logits.argmax(-1) == b["target"]) | ~mask).all(-1).sum())
                examples += len(b["enc"])
            # Four fixed val goals; paired perturbations at the initial carried state.
            b = store.batch(indices[:4], "cpu")
            x0 = model.emb(b["enc"]) + model.pos[:b["enc"].shape[1]]
            gen = torch.Generator().manual_seed(261117)
            direction = torch.randn(x0.shape, generator=gen)
            direction /= direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
            noise_scale = float(x0[b["enc_mask"]].square().mean().sqrt())
            clean = []
            model.encode(b["enc"], b["enc_mask"], budget, trace=clean)
            profiles = {}
            for epsilon in (1e-4, 1e-3, 1e-2):
                noise = epsilon * noise_scale * direction * b["enc_mask"][..., None]
                perturbed = []
                model.encode(b["enc"], b["enc_mask"], budget, trace=perturbed, initial_delta=noise)
                initial_norm = float(noise.norm())
                profile = []
                for k, (a, p) in enumerate(zip(clean, perturbed)):
                    mask = b["enc_mask"]
                    state = a["state"][mask]
                    endpoint = a["endpoint"][mask]
                    update = (a["state"] - a["input"])[mask]
                    displacement = (a["endpoint"] - a["input"])[mask]
                    state_norms, update_norms = state.norm(dim=-1), update.norm(dim=-1)
                    delta = (p["state"] - a["state"])[mask]
                    full_delta_sq = delta.square().sum()
                    if a["slow"] is not None:
                        full_delta_sq += (p["slow"] - a["slow"]).square().sum()
                    profile.append({"loop": k + 1, "hidden_norm_mean": float(state_norms.mean()),
                                    "hidden_norm_p95": float(torch.quantile(state_norms, .95)),
                                    "hidden_norm_max": float(state_norms.max()),
                                    "endpoint_norm_mean": float(endpoint.norm(dim=-1).mean()),
                                    "update_norm_mean": float(update_norms.mean()),
                                    "endpoint_displacement_norm_mean": float(displacement.norm(dim=-1).mean()),
                                    "relative_update_norm": float(update.norm() / a["input"][mask].norm().clamp_min(1e-12))
                                    if k else None,
                                    "perturbation_gain": float(delta.norm()) / initial_norm,
                                    "full_state_perturbation_gain": float(full_delta_sq.sqrt()) / initial_norm,
                                    "slow_norm_mean": float(a["slow"].norm(dim=-1).mean())
                                    if a["slow"] is not None else None})
                profiles[str(epsilon)] = profile
            rows[str(budget)] = {"states": examples, "bits_per_byte": nll / nbytes / math.log(2),
                                 "teacher_forced_exact": exact / examples, "profiles": profiles}
    result = {"checkpoint_step": ck["step"], "split": "val", "sample_seed": 261116,
              "perturbation_seed": 261117, "perturbation": "initial fast state; slow state initially unchanged",
              "accuracy_kind": "teacher-forced tactic exact match, not Lean completion", "depths": rows}
    dump(run / "diagnostics.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="mathlib-c3")
    ap.add_argument("--size", choices=("tiny", "XS"), default="XS")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=180)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--ram-cap", default="6G")
    ap.add_argument("--priority", choices=("idle", "below_normal"), default="below_normal")
    ap.add_argument("--vram-gib", type=float, default=4)
    ap.add_argument("--lean-states", type=int, default=24)
    ap.add_argument("--phase", choices=("train", "evaluate", "all"), default="all",
                    help="separate phases release the GPU lease before CPU evaluation")
    a = ap.parse_args(argv)
    if min(a.steps, a.batch, a.threads) < 1 or a.max_minutes <= 0:
        ap.error("steps, batch, threads and max-minutes must be positive")
    if a.device == "cuda" and a.phase != "evaluate" and not os.environ.get("VRAMORAMA_LEASE"):
        ap.error("CUDA pilot must run inside a vramorama lease")
    # Keep this object alive until the whole process tree has finished.
    from .ramcap import Job, parse_size
    job = Job(parse_size(a.ram_cap) + (int(a.vram_gib * 2**30)
                                     if a.device == "cuda" and a.phase != "evaluate" else 0), a.priority)
    job.assign_self()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    identity = {k: getattr(a, k) for k in ("data", "size", "steps", "batch", "seed", "device", "threads")}
    manifest_path = out / "pilot.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("pilot output belongs to a different configuration")
    else:
        manifest = {"identity": identity, "created_unix": time.time(), "split": "val",
                    "claim": "exploratory feedback comparison; one training seed; no test-set selection"}
        dump(manifest_path, manifest)
    if a.device == "cuda" and a.phase != "evaluate":
        # Query without creating a CUDA context until the training phase starts.
        total_mib = float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
        fraction = a.vram_gib * 1024 * .9 / total_mib
        if not 0 < fraction < .9:
            raise ValueError("VRAM cap must fit below 90% of physical VRAM")
    else:
        fraction = 0
    arms = []
    for arch in ("loop", "hrm"):
        for feedback in ("direct", "rlt", "matched_damping"):
            run = out / f"{arch}_{feedback}_s{a.seed}"
            preset = f"tlm-{arch}-XS" if a.size == "XS" else "tlm-tiny"
            cfg = [f"feedback={feedback}"] + ([f"arch={arch}", "fast_steps=8"] if a.size == "tiny" else [])
            arms.append(run)
            if a.phase == "evaluate" or (run / "complete.json").exists():
                continue
            remaining = a.max_minutes - (time.time() - manifest["created_unix"]) / 60
            if remaining <= 0:
                print(json.dumps({"event": "budget_exhausted", "incomplete": str(run)}), flush=True)
                return 76
            cmd = [sys.executable, "-u", "-m", "lcgrt.tactic_lm", "--data", a.data, "--preset", preset,
                   "--cfg", *cfg, "--steps", str(a.steps), "--batch", str(a.batch), "--seed", str(a.seed),
                   "--threads", str(a.threads), "--device", a.device, "--out", str(run), "--lr", "0.0005",
                   "--warmup", str(min(100, max(1, a.steps // 10))), "--log-every", "25",
                   "--save-every", "100", "--eval-every", str(a.steps), "--val-items", "256", "--val-batch", "16",
                   "--max-minutes", str(remaining), "--cuda-memory-fraction", str(fraction), "--gpu-duty", "0.85"]
            print(json.dumps({"event": "arm", "run": run.name, "command": cmd}), flush=True)
            dump(out / "status.json", {"phase": "train", "arm": run.name, "pid": os.getpid(),
                                       "commit_cap_gib": job.limit / 2**30, "unix": time.time()})
            # Reuse one torch runtime: repeated Windows CUDA imports can dominate a small pilot.
            from .tactic_lm import parser as training_parser, train
            with (out / f"{run.name}.log").open("a", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    code = train(training_parser().parse_args(cmd[4:]))
            gc.collect()
            if a.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            if code:
                print(json.dumps({"event": "arm_stopped", "run": run.name, "exit_code": code}), flush=True)
                return code
    if a.phase == "train":
        dump(out / "training_complete.json", {"arms": [r.name for r in arms],
                                              "peak_commit_gib": job.peak_bytes() / 2**30,
                                              "seconds": time.time() - manifest["created_unix"]})
        print(json.dumps({"event": "training_complete", "out": str(out)}), flush=True)
        return 0
    missing = [r.name for r in arms if not (r / "complete.json").exists()
               or json.loads((r / "complete.json").read_text())["step"] != a.steps]
    if missing:
        raise RuntimeError(f"refusing to evaluate an incomplete comparison: {missing}")
    previous_eval = out / "lean_eval" / "eval.json"
    if previous_eval.exists() and json.loads(previous_eval.read_text())["per_split"] != a.lean_states:
        raise ValueError("existing Lean evaluation uses a different --lean-states sample size")
    for run in arms:
        if not (run / "diagnostics.json").exists():
            print(json.dumps({"event": "diagnostics", "run": run.name}), flush=True)
            diagnostics(run, a.data, a.threads)
    if a.lean_states and not (out / "lean_eval" / "complete.json").exists():
        cmd = [sys.executable, "-u", "-m", "lcgrt.mathlib_eval", "--data", a.data, "--ckpt", *map(str, arms),
               "--splits", "val", "--per-split", str(a.lean_states), "--per-file", "2", "--seed", "261116",
               "--beam", "8", "--workers", "1", "--threads", str(a.threads), "--proposal-jobs", "1",
               "--out", str(out / "lean_eval")]
        code = subprocess.run(cmd).returncode
        if code:
            return code
    dump(out / "complete.json", {"arms": [r.name for r in arms], "peak_commit_gib": job.peak_bytes() / 2**30,
                                 "seconds": time.time() - manifest["created_unix"]})
    print(json.dumps({"event": "pilot_complete", "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
