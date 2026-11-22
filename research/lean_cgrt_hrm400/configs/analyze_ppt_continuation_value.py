"""Audit the same-start, same-rollout continuation action-value follow-up."""
import argparse
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_deadline import stage_counts, training_deadlines, validate_witness
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_root_adaptation import source_exposure
from configs.analyze_ppt_transport_pilot import audit_proposals
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "study", "controls", "sessions", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    study = json.loads(args.study.read_bytes())
    cohort_sha = sha256_file(args.cohort)
    controls = json.loads(gzip.decompress(args.controls.read_bytes()))
    control = controls["trials"]["continuation_rl"]
    resource = json.loads((args.sessions / "campaign-resource-audit.json").read_bytes())
    arm = study["arm"]
    if (study["schema"] != "ppt.continuation-action-value.v1" or study["cohort_sha256"] != cohort_sha
            or controls["report"]["cohort_sha256"] != cohort_sha
            or set(study["controls"]) != set(controls["report"]["arms"])
            or resource["cohort"] != cohort_sha or resource["statuses"] != {arm: 0}):
        raise ValueError("missing registered control or follow-up session")
    corpus = load(args.cohort)
    cases = {case.root: case for case in corpus["dev"]}
    training = {case.root: case for case in corpus["train"]}
    reg = json.loads(args.cohort.read_bytes())["registration"]
    clusters, witness, _ = validate_witness(args.sessions / "native-witness.json", cohort_sha, cases, reg)
    unseen = {digest(["continuation-deadline-v1", shape]) for shape in reg["structural_pairs"]["dev"]
              if len(shape[0][1]) == 3}
    path = args.sessions / (arm + ".sqlite")
    trial = read_arm(path, cases, cohort_sha)
    manifest = trial["manifest"]
    protocol = [key for key in PROTOCOL_FIELDS if key != "source_sha256"]
    protocol.extend(("seed", "architecture", "head_only", "prefer_direct_fact", "adapt_continuation",
                     "arrival_transport", "proof_deadline", "initialization"))
    if (not manifest["paired_action_value"]
            or any(manifest[key] != control["manifest"][key] for key in protocol)
            or trial["training_exposure_sha256"] != control["training_exposure_sha256"]
            or any(trial["stages"]["initial"][r]["tactics"] != control["stages"]["initial"][r]["tactics"] for r in cases)):
        raise ValueError("action-value tested a different training or initial policy protocol")
    proposals = audit_proposals(path, training, manifest["rollouts"], require_both=True)
    if proposals["schedule_sha256"] != control["proposal"]["schedule_sha256"]:
        raise ValueError("action-value study changed the paired proposals")
    with ram_store.open_store(path) as db:
        counts, early = stage_counts(db, trial, clusters, witness, {key: clusters[key] for key in unseen}, attempt=2)
        training_results = training_deadlines(db)
        training_sources = source_exposure(db, training)
        comparisons = sum((row["update"] or {}).get("paired_values", 0) for row in ram_store.ledger(db))
    summary = {"scope": study["scope"], "cohort_sha256": cohort_sha, "seed": manifest["seed"],
               "arm": arm, "proofs": counts, "verifier_action_value_comparisons": comparisons,
               "deadline_training": training_results, "mixed_sources": training_sources, "memory": resource["memory"],
               "controls": {name: row["proofs"]["final"] for name, row in controls["report"]["arms"].items()},
               "confirmation_scored": False}
    trial.update(early_continuation=early, proposal=proposals)
    args.summary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical({"summary": summary, "trial": trial}).encode("utf-8"), mtime=0))
    print(summary["proofs"]["final"], "verifier action-value comparisons", comparisons)


if __name__ == "__main__":
    main()
