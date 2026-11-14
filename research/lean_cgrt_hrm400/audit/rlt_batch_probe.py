"""Measure examples/s under a 4 GiB GPU lease before fixing the long-run batch."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--feedbacks", nargs="+", default=["direct"], choices=("direct", "rlt", "matched_damping"))
    a = ap.parse_args()
    assert os.environ.get("VRAMORAMA_LEASE"), "use vramorama"
    from lcgrt.ramcap import Job, parse_size
    job = Job(parse_size("7G"), "normal"); job.assign_self()
    print(json.dumps({"event": "probe_start", "pid": os.getpid(), "job_cap_gib": 7}), flush=True)
    started = time.time()
    import torch
    import numpy as np
    from lcgrt import config, hw
    from lcgrt.tactic_lm import Store, load_model, nll
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(3.6 * 2**30)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    store = Store(config.DATA_ROOT / "mathlib-c3" / "packed_tlm" / "train.npz")
    rows = []
    guard = hw.Guard(hw.Limits(), Path(a.out).parent)
    for arch, feedback in [(arch, feedback) for arch in ("loop", "hrm") for feedback in a.feedbacks]:
        for batch in a.batches:
            guard()
            model, _ = load_model(Path(a.parents) / f"{arch}_{feedback}_s131", "cuda")
            model.grad_ckpt = a.grad_ckpt
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
            torch.cuda.reset_peak_memory_stats()
            row = {"arch": arch, "feedback": feedback, "batch": batch, "grad_ckpt": a.grad_ckpt}
            try:
                t0 = None
                for step in range(a.steps + 2):
                    guard()
                    if step == 2:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                    b = store.batch(np.random.default_rng([812, step]).integers(0, store.n, batch), "cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = nll(model, b)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                torch.cuda.synchronize()
                row.update(seconds=time.perf_counter() - t0,
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                           loss=float(loss.detach()), status="ok")
                row["examples_per_second"] = batch * a.steps / row["seconds"]
            except torch.OutOfMemoryError:
                row["status"] = "allocator_oom"
            rows.append(row); print(json.dumps(row), flush=True)
            del model, opt
            if "b" in locals(): del b
            if "loss" in locals(): del loss
            gc.collect(); torch.cuda.empty_cache()
    # Same batch across architectures; retain at least 15% allocator headroom.
    eligible = []
    for batch in a.batches:
        pair = [r for r in rows if r["batch"] == batch]
        if len(pair) == 2 * len(a.feedbacks) and all(r["status"] == "ok" and r["peak_allocated_gib"] <= 3.06 for r in pair):
            eligible.append((min(r["examples_per_second"] for r in pair), batch))
    if not eligible:
        raise RuntimeError("no measured batch fits with headroom")
    result = {"rows": rows, "batch": max(eligible)[1], "grad_ckpt": a.grad_ckpt, "elapsed_seconds": time.time() - started,
              "peak_commit_gib": job.peak_bytes() / 2**30,
              "rule": "maximize worse-family examples/s; same batch, <=3.06GiB peak allocated"}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_batch": result["batch"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
