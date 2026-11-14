"""Freeze a fresh, input-disjoint file-held-out cohort before the longer RLT run."""
import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parents", required=True)
    ap.add_argument("--fresh", type=int, default=64)
    a = ap.parse_args()
    from lcgrt.ramcap import Job, parse_size
    job = Job(parse_size("3G"), "normal"); job.assign_self()
    import numpy as np
    from lcgrt import config
    from lcgrt.tactic_lm import Store, goal_bytes, usable
    from lcgrt.mathlib import heldout_file
    from lcgrt.mathlib_eval import sample_states, states_sha256
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("do not replace an existing frozen protocol")
    data = config.DATA_ROOT / "mathlib-c3"
    manifest = json.loads((data / "manifest.json").read_text())
    banned = set()
    packed_counts = {}
    for split in ("train", "val"):
        store = Store(data / "packed_tlm" / f"{split}.npz")
        packed_counts[split] = store.n
        for row, length in zip(store.enc, store.el):
            banned.add(hashlib.sha256(row[:int(length)].tobytes()).digest())
        del store
    calibration = sample_states("mathlib-c3", "val", 24, 2, 261116)
    assert states_sha256(calibration) == "59dad8d538229b68b7a196488a1492671bb09d17547fa5d2c78f665825680821"
    old_files = {s["file"] for s in calibration}
    by_file = collections.defaultdict(dict)
    rejected = 0
    with gzip.open(data / "test_file.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("terminal", len(row["goals_after"]) < len(row["goals_before"])) or not usable(row):
                continue
            if row["file"] in old_files or hashlib.sha256(goal_bytes(row)).digest() in banned:
                rejected += 1; continue
            assert heldout_file(row["file"], full=True)
            by_file[row["file"]].setdefault((row["line"], row["col"]), row)
    seed = 26111402
    fresh = []
    for file in sorted(by_file, key=lambda p: hashlib.sha1(f"{seed}:{p}".encode()).hexdigest()):
        rows = [by_file[file][k] for k in sorted(by_file[file])]
        rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:{file}".encode()).hexdigest()[:8], 16))
        pick = sorted(rng.choice(len(rows), min(2, len(rows)), replace=False))
        for i in pick:
            digest = hashlib.sha256(goal_bytes(rows[i])).digest()
            if digest not in banned:
                fresh.append(rows[i]); banned.add(digest)
            if len(fresh) == a.fresh: break
        if len(fresh) == a.fresh: break
    assert len(fresh) == a.fresh, "insufficient fresh goals after input-disjoint filtering"
    rows = [s | {"source_split": "val", "split": "calibration"} for s in calibration]
    rows += [s | {"source_split": "test_file", "split": "fresh"} for s in fresh]
    text = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in rows)
    (out / "cohort.jsonl").write_text(text, encoding="utf-8")
    (out / "calibration.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in calibration), encoding="utf-8")
    parents = {}
    for arch in ("loop", "hrm"):
        for feedback in ("direct", "rlt", "matched_damping"):
            name = f"{arch}_{feedback}_s131"
            p = Path(a.parents) / name
            meta = json.loads((p / "latest.json").read_text())
            assert meta["step"] == 1000
            assert hashlib.sha256((p / "latest.pt").read_bytes()).hexdigest() == meta["sha256"]
            parents[name] = {"path": str(p.resolve()), **meta}
    protocol = {"created_unix": time.time(), "name": "RLT-02 longer matched continuation",
                "user_training_cap_hours_sum_of_lanes": 8, "data": "mathlib-c3", "seed": 131,
                "parents": parents, "split_sha256": manifest["split_sha256"], "packed_rows_excluded": packed_counts,
                "fresh_selection_seed": seed, "fresh_requested": len(fresh), "fresh_files": len({r['file'] for r in fresh}),
                "calibration_requested": len(calibration), "rejected_old_or_train_val_input_candidates": rejected,
                "cohort_states_sha256": states_sha256(rows),
                "cohort_file_sha256": hashlib.sha256((out / "cohort.jsonl").read_bytes()).hexdigest(),
                "additional_phase_horizon": 6000, "chunk_updates": 500,
                "schedule": "200-step warmup to5e-4, then cosine over6000 additional updates; AdamW moments retained",
                "selection": "largest completed common500-update milestone within each architecture; never best-on-fresh",
                "eval": {"budgets": [8, 32], "beam": 8, "workers": 2,
                         "cohort": "64 fresh test_file goals plus24 old calibration goals; score separately"},
                "claim": "longer feedback-only transfer; fresh outcomes are not used to choose weights"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"fresh_goals": len(fresh), "fresh_files": protocol["fresh_files"],
                      "calibration_goals": len(calibration), "excluded_input_rows": packed_counts,
                      "states_sha256": protocol["cohort_states_sha256"]}))


if __name__ == "__main__":
    main()
