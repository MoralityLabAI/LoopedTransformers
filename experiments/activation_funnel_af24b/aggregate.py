"""AF-24B registered all-eight-seed adversarial transfer verdict.

Consumes independently audited exact original-AF23-checkpoint evaluations;
does not select checkpoints, tune post-hoc gates or modify old verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import gym, evaluate as ev

DEPTHS = gym.SPLITS
NEURAL = "vi2"
SYMBOLIC_COMPETITORS = ("ancestor_greedy","closure_direct","bfs_cap16","bfs_cap64")
REQUIRED_SEEDS = 6
MIN_ABSOLUTE_SOLVED = 40
MIN_ADVANTAGE_SOLVED = 8  # 10 percentage points of 80 proved goals
MIN_CORRECT_STOP = 101  # ceiling of 90% of 112 impossible cases


def sha(data):
    return hashlib.sha256(data).hexdigest()


def aggregate(root):
    root = Path(root)
    per_seed = {}
    missing = []
    pooled = {
        depth:{
            mode:{
                "solved":0,"provable":0,"unprovable":0,
                "premature_stop":0,"first_premature_stop":0,
                "correct_initial_stop":0,"false_accepted_proofs":0,
                "neural_forward_nodes":0,"symbolic_rule_examinations":0,
                "symbolic_bfs_expansions":0,
            } for mode in ev.ALL
        } for depth in DEPTHS
    }
    expected_hashes = ev.hashes()
    manifest = gym.manifest()["manifest_sha256"]
    protocol = sha((ev.HERE/"PROTOCOL.json").read_bytes())
    family_rows = {
        depth: {family: {mode: {"solved":0,"provable":0,
                                "initial_stop_on_provable":0}
                         for mode in ev.ALL}
                for family in gym.FAMILIES}
        for depth in DEPTHS
    }
    for seed in gym.SEEDS:
        folder = root/f"af24b-seed-{seed}"
        ep = folder/"episodes.jsonl"
        scored = folder/"evaluation.json"
        audit = folder/"independent_audit.json"
        if not all(x.is_file() for x in (ep,scored,audit)):
            missing.append(seed)
            continue
        rec = json.loads(scored.read_text())
        att = json.loads(audit.read_text())
        if (rec["seed"] != seed or att["seed"] != seed or
                not att["verified"] or
                rec["smoke"] or att["smoke"] or
                rec["training_updates"] != 0 or
                rec["all_arms"] != list(ev.ALL) or
                rec["registered_manifest_sha256"] != manifest or
                att["registered_manifest_sha256"] != manifest or
                rec["registered_protocol_sha256"] != protocol or
                rec["source_hashes"] != expected_hashes or
                att["source_hashes"] != expected_hashes or
                rec["episodes_sha256"] != sha(ep.read_bytes()) or
                att["episodes_sha256"] != rec["episodes_sha256"] or
                att["evaluation_sha256"] != sha(scored.read_bytes()) or
                att["false_accepted_proofs"] != 0 or
                att["original_af23_checkpoint"] != rec["original_af23_checkpoint"] or
                rec["original_af23_checkpoint"]["zip_sha256"] !=
                  ev.ORIGINAL_ZIPS[seed][1]):
            raise ValueError(f"seed {seed}: original source/receipt/evidence invalid")
        rows = [json.loads(line) for line in ep.read_text().splitlines()]
        if len(rows) != 192*2*len(ev.ALL) or len(rows) != att["episodes_audited"]:
            raise ValueError("incomplete registered proof episodes")
        for row in rows:
            if row["goal_truth"]:
                per = family_rows[row["split"]][row["family"]][row["mode"]]
                per["provable"] += 1
                per["solved"] += int(row["solved"])
                per["initial_stop_on_provable"] += int(row["initial_stop"])
        depth_rows = {}
        for depth in DEPTHS:
            scores = rec["scores"][depth]
            for mode in ev.ALL:
                vals = scores[mode]
                if vals["provable"] != 80 or vals["unprovable"] != 112:
                    raise ValueError("positive/negative roster changed")
                for key in pooled[depth][mode]:
                    pooled[depth][mode][key] += vals[key]
            if scores["shortest"]["solved"] != 80:
                raise ValueError("exact oracle upper control changed")
            original = scores[NEURAL]
            best_name = max(SYMBOLIC_COMPETITORS,
                            key=lambda name:(scores[name]["solved"],
                                             -SYMBOLIC_COMPETITORS.index(name)))
            best = scores[best_name]
            pass_depth = bool(
                original["solved"] >= MIN_ABSOLUTE_SOLVED
                and original["solved"] >= best["solved"] + MIN_ADVANTAGE_SOLVED
                and original["correct_initial_stop"] >= MIN_CORRECT_STOP
                and original["false_accepted_proofs"] == 0
            )
            depth_rows[depth] = {
                "vi2_solved":original["solved"],
                "direct_solved":scores["direct"]["solved"],
                "vi1_solved":scores["vi1"]["solved"],
                "best_symbolic_arm":best_name,
                "best_symbolic_solved":best["solved"],
                "advantage_proofs":original["solved"] - best["solved"],
                "correct_initial_stop":original["correct_initial_stop"],
                "false_accepted_proofs":original["false_accepted_proofs"],
                "registered_pass":pass_depth,
            }
        per_seed[str(seed)] = {
            "eligible":True,
            "both_depths_pass":all(x["registered_pass"] for x in depth_rows.values()),
            "depths":depth_rows,
            "original_zip_sha256":rec["original_af23_checkpoint"]["zip_sha256"],
            "original_training_sha256":rec["original_af23_checkpoint"]["training_sha256"],
            "proof_episodes_audited":att["episodes_audited"],
            "planner_nodes_audited":att["planner_nodes_reconstructed"],
            "torch_forward_replays":att["original_checkpoint_replays"],
        }
    passed = sum(v["both_depths_pass"] for v in per_seed.values())
    if missing:
        verdict = "INCONCLUSIVE"
    elif passed >= REQUIRED_SEEDS:
        verdict = "NEURAL_TRANSFER_PASS"
    else:
        verdict = "NEURAL_TRANSFER_FAIL"
    return {
        "experiment":"AF-24B",
        "scientific_verdict":verdict,
        "eligible_seeds":len(per_seed),
        "required_seeds":REQUIRED_SEEDS,
        "both_depths_pass":passed,
        "missing_seeds":missing,
        "registered_protocol_sha256":protocol,
        "structural_manifest_sha256":manifest,
        "source_hashes":expected_hashes,
        "original_af23_run":43812708916,
        "original_af23_scientific_verdict":"PASS_6_of_8_original_weak_controls",
        "original_af24_scientific_verdict":"SYMBOLIC_CONTROL_MATCH_8_of_8",
        "per_seed":per_seed,
        "pooled":pooled,
        "per_family":family_rows,
        "verification":{
            "proof_episodes":sum(v["proof_episodes_audited"] for v in per_seed.values()),
            "planner_nodes":sum(v["planner_nodes_audited"] for v in per_seed.values()),
            "torch_forward_replays":sum(v["torch_forward_replays"] for v in per_seed.values()),
            "false_accepted_proofs":0,
        },
        "scope_note":"Synthetic unseen public finite Horn grammars; no model retraining; no equal-FLOP inference comparison or Lean theorem-proving claim."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    receipt=aggregate(a.root)
    a.out.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "scientific_verdict":receipt["scientific_verdict"],
        "both_depths_pass":receipt["both_depths_pass"],
        "eligible_seeds":receipt["eligible_seeds"],
        "missing_seeds":receipt["missing_seeds"],
    }))


if __name__=="__main__":
    main()
