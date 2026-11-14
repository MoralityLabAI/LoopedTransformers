"""Evaluate both completed RLT-02 lanes once on the frozen shared cohort."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .rlt_pilot import dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    a = ap.parse_args()
    if not os.environ.get("VRAMORAMA_LEASE"):
        ap.error("evaluation requires a4GiB vramorama lease")
    from .ramcap import Job, parse_size
    job = Job(parse_size("10G"), "normal"); job.assign_self()
    root = Path(a.experiment).resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if hashlib.sha256((root / "cohort.jsonl").read_bytes()).hexdigest() != protocol["cohort_file_sha256"]:
        raise RuntimeError("frozen cohort changed")
    paths, lanes = [], {}
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for arch in ("loop", "hrm"):
        lane = json.loads((root / arch / "training_complete.json").read_text())
        assert lane["identity"]["protocol_sha256"] == digest
        assert len(lane["selected_paths"]) == 3
        ids = []
        for p in lane["selected_paths"]:
            p = Path(p)
            best = json.loads((p / "best.json").read_text())
            assert hashlib.sha256((p / "best.pt").read_bytes()).hexdigest() == best["sha256"]
            assert best["phase_step"] == lane["phase_step"] and best["step"] == lane["total_step"]
            ident = json.loads((p / "config.json").read_text())["identity"]
            ident["model"].pop("feedback", None)
            ident["continuation"].pop("checkpoint_sha256")
            ident["continuation"].pop("parent_identity_hash")
            ids.append(ident); paths.append(p)
        assert all(ident == ids[0] for ident in ids), "unmatched training recipes"
        lanes[arch] = lane
    import torch
    torch.set_num_threads(2)
    total_mib = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    torch.cuda.set_per_process_memory_fraction(3.6 * 1024 / total_mib)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from .rlt_pilot import diagnostics
    for p in paths:
        if not (p / "diagnostics.json").exists():
            print(json.dumps({"event": "diagnostics", "arm": p.name}), flush=True)
            diagnostics(p, protocol["data"], 2, "cuda")
            torch.cuda.empty_cache()
    from .mathlib_eval import main as evaluate
    out = root / "evaluation"
    if not (out / "lean_eval" / "complete.json").exists():
        code = evaluate(["--data", protocol["data"], "--ckpt", *map(str, paths),
                         "--states-file", str(root / "cohort.jsonl"), "--splits", "calibration", "fresh",
                         "--per-split", str(protocol["fresh_requested"]), "--beam", "8", "--depths", "8", "32",
                         "--workers", "2", "--threads", "2", "--proposal-jobs", "1", "--device", "cuda",
                         "--proposal-cache", str(out / "proposal_cache"), "--out", str(out / "lean_eval")])
        if code: return code
    dump(out / "evaluation_complete.json", {"lanes": lanes, "selected_paths": list(map(str, paths)),
                                            "peak_commit_gib": job.peak_bytes() / 2**30,
                                            "protocol_sha256": digest})
    print(json.dumps({"event": "evaluation_complete", "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    from .hw import ResourcePause
    try:
        raise SystemExit(main())
    except ResourcePause as e:
        print(json.dumps({"event": "pause", "reason": str(e)}), flush=True)
        raise SystemExit(75)
