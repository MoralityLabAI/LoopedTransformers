"""AF-30 independent source-locked eight-seed aggregation and scientific verdict."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from . import evaluate, gym, ranker

ARMS = evaluate.ARMS
SPLITS = evaluate.SPLITS


def load(path):
    return json.loads(path.read_text())


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("AF30 immutable aggregate destination already exists")
    report = {}
    source_sha = gym.manifest()["manifest_sha256"]
    protocol_sha = digest_file(Path(__file__).parent/"PROTOCOL.json")
    train_shas = set()
    for seed in ranker.SEEDS:
        folder = root/f"af30-seed-{seed}"
        receipt = load(folder/"evaluation.json")
        audit = load(folder/"audit.json")
        model = load(folder/"model.json")
        if receipt["seed"] != seed or audit["seed"] != seed or model["seed"] != seed:
            raise AssertionError("AF30 seed artifact identity mismatch")
        if receipt["manifest_sha256"] != source_sha or \
           audit["source_manifest_sha256"] != source_sha:
            raise AssertionError("AF30 source-only manifest mismatch")
        if receipt["protocol_sha256"] != protocol_sha:
            raise AssertionError("AF30 preregistration source drift")
        for f, key in (("model.json","model_sha256"),
                       ("episodes.jsonl","episodes_sha256")):
            if digest_file(folder/f) != receipt[key] or \
               digest_file(folder/f) != audit[key]:
                raise AssertionError("AF30 source proof artifact SHA mismatch")
        if (not audit["fresh_train_refit"] or
            not audit["all_public_physical_transitions_valid"] or
            not audit["all_action_decisions_recomputed"] or
            not audit["all_scores_independently_aggregated"] or
            not audit["no_false_accepted_proofs"]):
            raise AssertionError("AF30 independent source proof audit missing")
        if receipt["episodes"] != 192*len(ARMS)*len(SPLITS):
            raise AssertionError("AF30 incomplete source host episode roster")
        gate = evaluate.gate(receipt["scores"])
        if gate != receipt["registered_seed_gate"] or gate != audit["registered_seed_gate"]:
            raise AssertionError("AF30 registered gate source drift")
        train_shas.add(model["training_row_sha256"])
        report[str(seed)] = {
            "gate":gate,"scores":receipt["scores"],
            "model_sha256":receipt["model_sha256"],
            "episodes_sha256":receipt["episodes_sha256"],
            "audit_sha256":digest_file(folder/"audit.json"),
            "episode_count":receipt["episodes"],
        }
    if len(train_shas)!=1:
        raise AssertionError("AF30 seed rankers were trained on different source rows")
    pass_count=sum(r["gate"]["both_depths_pass"] for r in report.values())
    depth_sums={}
    for split in SPLITS:
        depth_sums[split]={}
        for arm in ARMS:
            depth_sums[split][arm]={
                k:sum(report[str(seed)]["scores"][split][arm][k]
                      for seed in ranker.SEEDS)
                for k in ("positive_proofs","negative_correct_initial_stop",
                          "false_accepted_proofs","rule_examinations",
                          "symbolic_expansions","linear_scores","wall_ns")
            }
    result={
        "experiment":"AF-30",
        "source_proof_audit":"PASS_ALL_EIGHT",
        "scientific_verdict":("RULE_SELECTION_TRANSFER_PASS"
                              if pass_count >= 6 else "RULE_SELECTION_TRANSFER_FAIL"),
        "registered_required_seed_count":6,
        "registered_seed_successes":pass_count,
        "source_seeds":list(ranker.SEEDS),
        "held_positive_per_depth":96,
        "held_negative_per_depth":96,
        "note":"768 per-label per-depth across seeds repeats the same 96 matched graph cases under eight separate TRAIN-only linear initializations, not 768 independent logical grammars.",
        "source_manifest_sha256":source_sha,
        "protocol_sha256":protocol_sha,
        "training_rows_sha256":next(iter(train_shas)),
        "per_seed":report,
        "depth_sums":depth_sums,
        "all_audited_episodes":sum(r["episode_count"] for r in report.values()),
        "scientific_interpretation":"Negative STOP for _CERT arms comes from full-public Horn fixed-point certificate, not learned STOP. Learned action ranking is evaluated against same-certificate untrained and hand-coded controllers, plus public closure/BFS. All certificate and feature rule examinations are counted.",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=aggregate(a.root,a.out)
    print(json.dumps({"verdict":r["scientific_verdict"],
                      "seeds_pass":r["registered_seed_successes"],
                      "audited_episodes":r["all_audited_episodes"],
                      "source_audit":r["source_proof_audit"]}))


if __name__=="__main__":
    main()
