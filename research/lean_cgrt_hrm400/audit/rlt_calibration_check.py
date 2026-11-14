"""Interim kernel check on old calibration only; never selects final weights."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arch", choices=("loop", "hrm"), required=True)
    ap.add_argument("--phase-step", type=int, required=True)
    a = ap.parse_args()
    assert os.environ.get("VRAMORAMA_LEASE"), "use a4GiB vramorama lease"
    from lcgrt.ramcap import Job, parse_size
    job = Job(parse_size("8G"), "normal"); job.assign_self()
    root = Path(a.experiment).resolve()
    out = root / "interim-calibration" / f"{a.arch}-phase{a.phase_step}"
    paths = []
    for feedback in ("direct", "rlt", "matched_damping"):
        name = f"{a.arch}_{feedback}_s131"
        source = root / a.arch / name / "snapshots" / f"step{a.phase_step:05d}"
        meta = json.loads((source / "best.json").read_text())
        assert meta["phase_step"] == a.phase_step
        assert hashlib.sha256((source / "best.pt").read_bytes()).hexdigest() == meta["sha256"]
        dest = out / "checkpoints" / name; dest.mkdir(parents=True, exist_ok=True)
        for file in ("best.pt", "best.json", "config.json", "snapshot.json"):
            shutil.copyfile(source / file, dest / file)
        paths.append(dest)
    from lcgrt import hw
    hw.Guard(hw.Limits(min_ram_avail_gib=1.5), out)()
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.6 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from lcgrt.mathlib_eval import main as evaluate
    return evaluate(["--data", "mathlib-c3", "--ckpt", *map(str, paths),
                     "--states-file", str(root / "calibration.jsonl"), "--splits", "val", "--per-split", "24",
                     "--depths", "8", "--beam", "8", "--workers", "1", "--threads", "2", "--device", "cuda",
                     "--proposal-cache", str(out / "proposal_cache"), "--out", str(out / "lean_eval")])


if __name__ == "__main__":
    from lcgrt.hw import ResourcePause
    try:
        raise SystemExit(main())
    except ResourcePause as e:
        print(json.dumps({"event": "pause", "reason": str(e)}), flush=True)
        raise SystemExit(75)
