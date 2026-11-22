"""Test whether the tied proof cell carries the seed-103 transfer gain."""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.evaluate_ppt_bpd_rewrite_transfer import read_arm
from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from configs.probe_ppt_bpd_rewrite_transplant import substitute
from configs.probe_ppt_bpd_target_contrast import RewriteGraphReadout
from configs.replicate_ppt_bpd_rewrite_transfer import audit_file, inputs, seed_document
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


SEED = 103
ARMS = ("hybrid_rl", "hybrid_calibration")


def audit(args, tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    reg, doc, witness, cases = inputs(args)
    projected = seed_document(reg, doc, SEED)
    original_paths = {arm: args.replication / f"{SEED}-{arm}.json.gz" for arm in ARMS}
    originals = {arm: read_arm(path, arm, projected, cases, reg["cohort_sha256"], reg["witness_sha256"])
                 for arm, path in original_paths.items()}
    audit_path = audit_file(args.audits, SEED)
    source = json.loads(audit_path.read_bytes())
    configure(tree.threads, SEED)
    policies = {arm: model for arm, model, _, _ in audited_policies(
        args.sessions / f"hybrid-bpd-rewrite-exploration-{SEED}", audit_path,
        doc["source_cohort_sha256"], doc["reference_sha256"], selected=ARMS,
    )}
    options = SimpleNamespace(cpu_seconds=doc["registration"]["cpu_seconds"], max_macs=doc["registration"]["max_macs"])
    variants = {}
    for recipient, donor in (ARMS, ARMS[::-1]):
        model, weights = substitute(policies[recipient], policies[donor], "cell.")
        variants[f"{recipient}_with_{donor}_cell"] = model, weights, recipient
    variants["calibrated_transport_only"] = RewriteGraphReadout(policies["hybrid_calibration"], "transport"), None, "hybrid_calibration"
    results = {}
    with torch.no_grad(), LeanRepl(timeout=15) as repl:
        if (repl.version != witness["native_lean"] or
                fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != source["trials"]["hybrid_rl"]["manifest"]["verifier"]):
            raise ValueError("the verifier differs from the registered seed replicate")
        for name, (model, state, recipient) in variants.items():
            rows = {}
            for root, case in cases.items():
                visits = RewriteVisits(repl)
                policy = PreferNewEquality(model, visits)
                episode = rollout(visits, tree, case, policy, options)
                replay = replay_search(repl, case, episode)
                certificate = episode["certificate"]
                if (not replay["replay_passed"] or episode["tactic_checks"] > 8 or episode["model_macs"] > options.max_macs
                        or (episode["verified_closed"] and (not certificate["passed"] or certificate["axioms"]
                            or digest(certificate["source"]) != certificate["source_digest"]))):
                    raise ValueError("a transferred cell trajectory failed its native audit")
                rows[root] = {"certified": episode["verified_closed"] and episode["within_cpu_budget"],
                              "cpu_censored": not episode["within_cpu_budget"], "macs": episode["model_macs"],
                              "checks": episode["tactic_checks"], "script": episode["accepted_script"],
                              "certificate": certificate, "replay": replay}
            matched = originals[recipient]["rows"]
            counts = {"certified": sum(row["certified"] for row in rows.values()),
                      "cpu_censored": sum(row["cpu_censored"] for row in rows.values()),
                      "gains": sum(row["certified"] and not matched[root]["prefer_new"]["certified"] for root, row in rows.items()),
                      "losses": sum(not row["certified"] and matched[root]["prefer_new"]["certified"] for root, row in rows.items())}
            results[name] = {"weights_sha256": state, "recipient": recipient, "counts": counts, "rows": rows}
            print(name, counts, flush=True)
    return {"scope": "exploratory seed-103 tied-cell exchange on the matched native revisit readout", "seed": SEED,
            "registry_sha256": sha256_file(args.registry), "baseline_receipts": {arm: sha256_file(path) for arm, path in original_paths.items()},
            "baseline_certified": {arm: sum(row["prefer_new"]["certified"] for row in data["rows"].values())
                                   for arm, data in originals.items()},
            "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "witness", "audits", "registry", "sessions", "replication", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
