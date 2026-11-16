"""AF-23 Track B source-checked eight-seed registered experiment aggregate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from . import branching_gym as gym
from . import branching_train as training
from . import branching_evaluate as evaluation

DEPTHS = ("held_depth4", "held_depth5")
REQUIRED_SEEDS = 6
RANDOM = evaluation.RANDOM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(root):
    root = Path(root)
    seed_rows = {}
    missing = []
    pooled = {
        depth: {mode: {"solved": 0, "provable": 0, "unprovable": 0,
                       "initial_premature_stop": 0,
                       "initial_correct_stop": 0, "false_accepted_proofs": 0}
                for mode in evaluation.ALL}
        for depth in DEPTHS
    }
    for seed in gym.SEEDS:
        folder = root / f"af23-branch-seed-{seed}"
        tr = folder / "training/train_receipt.json"
        ev = folder / "evaluation/evaluation.json"
        au = folder / "independent_audit.json"
        episodes = folder / "evaluation/episodes.jsonl"
        if not all(p.is_file() for p in (tr, ev, au, episodes)):
            missing.append(seed)
            continue
        a, b, c = [json.loads(p.read_text()) for p in (tr, ev, au)]
        if (a["seed"], b["seed"], c["seed"]) != (seed, seed, seed):
            raise ValueError("AF23 seed source identity mismatch")
        if (a["source_sha256"] != training.source_hashes()
                or b["source_sha256"] != a["source_sha256"]
                or c["source_sha256"] != a["source_sha256"]
                or not c["verified"]
                or c["false_accepted_proofs"] != 0
                or b["episodes_sha256"] != sha(episodes)
                or b["train_receipt_sha256"] != sha(tr)
                or c["training_sha256"] != sha(tr)
                or c["evaluation_sha256"] != sha(ev)
                or a["data_manifest_sha256"] != gym.manifest()["manifest_sha256"]
                or b["manifest_sha256"] != a["data_manifest_sha256"]):
            raise ValueError(f"seed {seed}: independent source replay invalid")
        eligible = all(
            b["scores"][split]["direct"]["proof_success"] >= .65
            for split in ("train", "dev")
        )
        depth_rows = {}
        for depth in DEPTHS:
            scores = b["scores"][depth]
            for mode in evaluation.ALL:
                rec = scores[mode]
                if rec["provable"] != 80 or rec["unprovable"] != 112:
                    raise ValueError("held-out roster or truth balance changed")
                for key in pooled[depth][mode]:
                    pooled[depth][mode][key] += rec[key]
            vi = scores["vi2"]["proof_success"]
            fixed = scores["fixed"]["proof_success"]
            random_mean = sum(scores[r]["proof_success"] for r in RANDOM) / len(RANDOM)
            passed = bool(
                eligible and vi >= fixed + .10 - 1e-12
                and vi >= random_mean + .10 - 1e-12
                and vi >= .50 - 1e-12
                and scores["vi2"]["false_accepted_proofs"] == 0
            )
            depth_rows[depth] = {
                "vi2_solved": scores["vi2"]["solved"],
                "fixed_solved": scores["fixed"]["solved"],
                "random_mean_solved": sum(scores[r]["solved"] for r in RANDOM) / len(RANDOM),
                "oracle_solved": scores["shortest"]["solved"],
                "registered_pass": passed if eligible else None,
                "gain_vs_fixed": vi - fixed,
                "gain_vs_random_average": vi - random_mean,
            }
            if scores["shortest"]["solved"] != 80:
                raise ValueError("host shortest-proof control failed")
        seed_rows[str(seed)] = {
            "qualified": eligible,
            "both_depths_pass": bool(eligible and all(
                depth_rows[x]["registered_pass"] for x in DEPTHS
            )),
            "train_direct": b["scores"]["train"]["direct"]["proof_success"],
            "dev_direct": b["scores"]["dev"]["direct"]["proof_success"],
            "depths": depth_rows,
            "source_hashes": a["source_sha256"],
            "audit_proof_episodes": c["proof_episodes"],
            "audit_forward_replays": c["frozen_checkpoint_forward_replays"],
        }
    passed = sum(row["both_depths_pass"] for row in seed_rows.values())
    qualified = sum(row["qualified"] for row in seed_rows.values())
    if missing or qualified < REQUIRED_SEEDS:
        verdict = "INCONCLUSIVE"
    elif passed >= REQUIRED_SEEDS:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {
        "experiment": "AF-23", "track": "B",
        "scientific_verdict": verdict, "qualified_seeds": qualified,
        "required_seeds": REQUIRED_SEEDS,
        "both_depths_pass": passed, "missing_seeds": missing,
        "protocol_sha256": training.digest(training.HERE / "PROTOCOL.json"),
        "branching_manifest_sha256": gym.manifest()["manifest_sha256"],
        "seed_results": seed_rows, "pooled": pooled,
        "scientific_scope": "15-atom 14-rule branching synthetic positive-Horn gym only",
        "af22_scientific_verdict": "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root)
    training.write_json(args.out, result)
    print(json.dumps({"verdict": result["scientific_verdict"],
                      "pass": result["both_depths_pass"], "qualified": result["qualified_seeds"],
                      "missing": result["missing_seeds"]}))


if __name__ == "__main__":
    main()
