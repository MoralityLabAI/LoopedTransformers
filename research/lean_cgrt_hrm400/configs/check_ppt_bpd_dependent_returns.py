"""Certify both training paths while retaining their eight-check return difference."""

import argparse
import gzip
from pathlib import Path
from types import SimpleNamespace

from configs.check_ppt_bpd_dependent_rewrites import BUDGET, SPUR_SCHEMA, spurred_world
from configs.probe_ppt_bpd_dependent_policy import read_witness, training_batches, trajectory
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import EQUALITY


def audit(training_path, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure

    training = read_witness(training_path, "train")
    if training["schema"] != SPUR_SCHEMA:
        raise ValueError("expected the disjoint dead-end equality training layouts")
    batches = training_batches(training, match_search_menu=True)
    configure(tree.threads, 103)
    rows = {}
    replay_only = SimpleNamespace(architecture="hybrid")
    with LeanRepl(timeout=15) as repl:
        if repl.version != training["lean"]:
            raise ValueError("the registered training verifier changed")
        for root, row in training["rows"].items():
            case = SearchCase(**row["case"])
            generated, _, guards = spurred_world("train", tuple(row["shape"]), row["winner"])
            if generated != case:
                raise ValueError("the training theorem differs from its registered layout")
            branches = []
            for branch in row["branches"]:
                original = branch["events"]
                script = [event["tactic"] for event in original]
                if not branch["closed"]:
                    target, = original[-1]["targets"]
                    match = EQUALITY.fullmatch(target["raw"])
                    if match is None:
                        raise ValueError("the longer training path lacks its final equality obligation")
                    script.append(guards[match.groups()])
                events, closed, within, _ = trajectory(repl, case, replay_only, tree, script)
                expected_prefix = [{key: event[key] for key in ("tactic", "before", "after")} for event in original]
                if (not closed or not within or events[:BUDGET] != expected_prefix
                        or len(events) != BUDGET + int(not branch["closed"])):
                    raise ValueError("the completed training path changed its registered eight-check prefix")
                cert = certify(case, {"live_kernel": True, "accepted_script": script})
                if not cert or not cert["passed"] or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]:
                    raise ValueError("the completed training path lacks an independent certificate")
                branches.append({"choice": branch["choice"], "checks_to_certify": len(events), "events": events,
                                 "within_eight": branch["closed"], "certificate": cert})
            rows[root] = branches
    return {"schema": "ppt.dependent-rewrite-return-training.v1", "lean": training["lean"], "eight_check_budget": BUDGET,
            "prefix_sha256": sha256_file(training_path), "training_layouts": len(batches), "rows": rows,
            "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("training", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args.training, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(f"Independently certified both training paths in {len(report['rows'])} worlds.", flush=True)


if __name__ == "__main__":
    main()
