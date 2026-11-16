"""AF-23 Track A independent eight-seed registered-gate aggregation.

Only consumes seed archives with passing separate host audit and source-bound
checkpoint/episode digests. Recomputes pooled counters and the original
first-legal comparator. Missing seed is an explicit INCONCLUSIVE outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from . import stop_eval, stop_scoring

ALL = (*stop_eval.ORIGINAL_ARMS, *stop_eval.NEW_ARMS)
SPLITS = ("held_depth4", "held_depth5")
REQUIRED = 6
TARGET = "root_tied_no_interior_stop"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(root):
    root = Path(root)
    per_seed = {}
    missing = []
    pooled = {s: {mode: {"solved": 0, "provable": 0, "initial_premature_stop": 0,
                         "initial_correct_stop": 0, "unprovable": 0,
                         "false_accepted_proofs": 0}
                  for mode in ALL} for s in SPLITS}
    for seed in stop_eval.SEEDS:
        path = root / f"af23-track-a-seed-{seed}"
        result_path = path / "af23_ablation_results.json"
        audit_path = path / "af23_independent_audit.json"
        episodes_path = path / "af23_ablation_episodes.jsonl"
        if not all(p.is_file() for p in (result_path, audit_path, episodes_path)):
            missing.append(seed)
            continue
        x = json.loads(result_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (x["seed"] != seed or audit["seed"] != seed
                or x["original_zip_sha256"] != audit["original_zip_sha256"]
                or not audit["verified"] or audit["false_accepted_proofs"] != 0
                or audit["cases_checked"] != 3 * 416
                or x["new_episodes_sha256"] != sha(episodes_path)
                or x["protocol_sha256"] != sha(stop_eval.HERE / "PROTOCOL.json")
                or x["all_modes"] != list(ALL)):
            raise ValueError(f"seed {seed}: AF-23 source/evidence mismatch")
        depths = {}
        for split in SPLITS:
            scores = x["scores"][split]
            for mode in ALL:
                rec = scores[mode]
                expect_positive = 80
                expect_negative = 112
                if rec["provable"] != expect_positive or rec["unprovable"] != expect_negative:
                    raise ValueError(f"seed {seed} / {split} / {mode}: wrong roster size")
                for k in pooled[split][mode]:
                    pooled[split][mode][k] += rec[k]
            fixed = scores["fixed"]
            oracle = scores["shortest"]
            if (fixed["solved"] != (60 if split == "held_depth4" else 64)
                    or oracle["solved"] != 80):
                raise ValueError("original fixed/oracle baseline changed")
            proposed = scores[TARGET]
            passed = (
                proposed["proof_success"] >= fixed["proof_success"] + .05 - 1e-12
                and proposed["correct_stop_rate_unprovable"] >= .90 - 1e-12
                and proposed["false_accepted_proofs"] == 0
            )
            depths[split] = {
                "target_solved": proposed["solved"],
                "fixed_solved": fixed["solved"],
                "target_correct_initial_stop_on_unprovable": proposed["initial_correct_stop"],
                "unprovable": proposed["unprovable"],
                "registered_pass": bool(passed),
            }
        per_seed[str(seed)] = {
            "eligible": True, "both_depths_pass": all(
                d["registered_pass"] for d in depths.values()
            ), "depths": depths, "audit_nodes": audit["planner_node_receipts_checked"],
            "forward_replays": audit["checkpoint_forward_replays"],
            "source_zip_sha256": x["original_zip_sha256"],
        }
    if missing:
        verdict = "INCONCLUSIVE"
    else:
        count = sum(s["both_depths_pass"] for s in per_seed.values())
        verdict = "PASS" if count >= REQUIRED else "FAIL"
    result = {
        "experiment": "AF-23", "track": "A", "scientific_verdict": verdict,
        "count_both_depths_pass": sum(s["both_depths_pass"] for s in per_seed.values()),
        "required_count": REQUIRED, "missing_seeds": missing,
        "comparison": "root_tied_no_interior_stop versus first-legal fixed, with correct STOP on unprovable inputs",
        "registered_protocol_sha256": sha(stop_eval.HERE / "PROTOCOL.json"),
        "af22_original_ci": 43788715688,
        "seeds": per_seed, "pooled": pooled,
        "original_af22_scientific_verdict": "FAIL",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scientific_verdict": result["scientific_verdict"],
                      "both_depths_pass": result["count_both_depths_pass"],
                      "required": REQUIRED, "missing": result["missing_seeds"]}))


if __name__ == "__main__":
    main()
