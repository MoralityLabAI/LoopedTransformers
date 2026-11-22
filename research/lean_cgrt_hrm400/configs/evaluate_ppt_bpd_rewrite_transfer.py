"""Evaluate the frozen rewrite policies on the independently verified fresh readout."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from configs.check_ppt_bpd_rewrite_transfer import load_transfer
from configs.freeze_ppt_bpd_rewrite import ARMS
from configs.probe_ppt_bpd_rewrite_revisits import PreferNewEquality, RewriteVisits, audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


MODES = ("prefer_new", "pristine")


def prepare(cohort, witness_path, source_audit):
    doc, groups = load_transfer(cohort)
    witness = json.loads(gzip.decompress(witness_path.read_bytes()))
    planned = {case.root: (valid, length) for rows in groups.values() for case, valid, length in rows}
    if (doc["source_audit_sha256"] != sha256_file(source_audit) or witness["cohort_sha256"] != sha256_file(cohort)
            or tuple(doc["registration"]["arms"]) != ARMS or set(witness["rows"]) != set(planned)):
        raise ValueError("the transfer audit, reference, and frozen layouts differ")
    for root, (valid, depth) in planned.items():
        row = witness["rows"][root]
        if row["certified"] != valid or not row["replay"]["replay_passed"]:
            raise ValueError("the paired native witness did not verify its frozen proof status")
        if valid:
            cert = row["certificate"]
            if (row["checks"] != depth or not cert["passed"] or cert["axioms"]
                    or digest(cert["source"]) != cert["source_digest"]):
                raise ValueError("the fresh layout does not have its independently certified shortest proof")
    positives = {case.root: case for rows in groups.values() for case, valid, _ in rows if valid}
    return doc, witness, positives


def one_arm(args, tree, doc, witness, cases, receipt):
    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, doc["registration"]["seed"])
    arm, model, source, _ = next(audited_policies(
        args.sessions, args.source_audit, doc["source_cohort_sha256"], doc["reference_sha256"],
        selected=(args.arm,),
    ))
    expected = doc["checkpoints"][arm]
    if source != {"session_sha256": expected["session_sha256"], "weights_sha256": expected["final_weights_sha256"]}:
        raise ValueError("the model differs from the frozen transfer checkpoint")
    source_audit = json.loads(args.source_audit.read_bytes())
    manifest = source_audit["trials"][arm]["manifest"]
    reg = doc["registration"]
    options = SimpleNamespace(cpu_seconds=reg["cpu_seconds"], max_macs=reg["max_macs"])
    if manifest["cpu_seconds"] != options.cpu_seconds or manifest["max_macs"] != options.max_macs:
        raise ValueError("the fresh readout changed a source compute budget")
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if (repl.version != witness["native_lean"]
                or fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]):
            raise ValueError("the verifier differs from the audited rewrite experiment")
        for root, case in cases.items():
            visits = RewriteVisits(repl)
            modes = (("prefer_new", visits, PreferNewEquality(model, visits)), ("pristine", repl, model))
            rows[root] = {}
            for name, verifier, policy in modes:
                episode = rollout(verifier, tree, case, policy, options)
                replay = replay_search(repl, case, episode)
                if not replay["replay_passed"] or episode["tactic_checks"] > reg["tactic_checks"]:
                    raise ValueError("a fresh rewrite proof failed its native replay")
                rows[root][name] = {
                    "certified": episode["verified_closed"] and episode["within_cpu_budget"],
                    "cpu_censored": not episode["within_cpu_budget"],
                    "checks": episode["tactic_checks"], "macs": episode["model_macs"],
                    "certificate": episode["certificate"], "native_replay": replay,
                }
    result = {
        "arm": arm, "checkpoint": source, "cohort_sha256": sha256_file(args.cohort),
        "witness_sha256": sha256_file(args.witness), "rows": rows, "memory": tree.stats(),
    }
    payload = gzip.compress(canonical(result).encode("utf-8"), mtime=0)
    with receipt.open("xb") as out:
        out.write(payload)


def read_arm(path, arm, doc, cases, cohort_sha, witness_sha):
    report = json.loads(gzip.decompress(path.read_bytes()))
    checkpoint = doc["checkpoints"][arm]
    reg = doc["registration"]
    if (report["arm"] != arm or report["cohort_sha256"] != cohort_sha or report["witness_sha256"] != witness_sha
            or report["checkpoint"]["session_sha256"] != checkpoint["session_sha256"]
            or report["checkpoint"]["weights_sha256"] != checkpoint["final_weights_sha256"]
            or set(report["rows"]) != set(cases)):
        raise ValueError("the arm receipt differs from the frozen readout")
    for modes in report["rows"].values():
        if set(modes) != set(MODES):
            raise ValueError("the arm did not evaluate the two registered inference modes")
        for row in modes.values():
            if (not row["native_replay"]["replay_passed"] or not 0 <= row["checks"] <= reg["tactic_checks"]
                    or not 0 <= row["macs"] <= reg["max_macs"]):
                raise ValueError("a fresh outcome exceeds the registered budget")
            if row["certified"]:
                cert = row["certificate"]
                if row["cpu_censored"] or not cert["passed"] or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]:
                    raise ValueError("a fresh outcome lacks an independent axiom-free certificate")
    return report


def summarize(cases, arms, reg):
    from spiral_ln.evaluation_finite_sample import evaluate_manifest

    counts = {}
    for arm, info in arms.items():
        counts[arm] = {
            name: {"certified": sum(root[name]["certified"] for root in info["rows"].values()),
                   "cpu_censored": sum(root[name]["cpu_censored"] for root in info["rows"].values())}
            for name in MODES
        }
    comparisons = {}
    for control in ARMS[1:]:
        pairs = [{"case_id": root, "cluster_id": case.cluster,
                  "candidate": int(arms["hybrid_rl"]["rows"][root]["prefer_new"]["certified"]),
                  "baseline": int(arms[control]["rows"][root]["prefer_new"]["certified"])}
                 for root, case in cases.items()]
        comparisons[control] = evaluate_manifest({
            "schema_version": 1, "experiment_id": f"bpd-fresh-rewrite-151-{control}",
            "target_population": reg["domain"], "evaluation_mode": "sampled",
            "alpha": reg["alpha"], "planned_comparisons": len(ARMS) - 1, "minimum_effect": reg["minimum_effect"],
            "hypothesis_preregistered": True, "sealed_readout": True,
            "clusters_independent": True, "exchangeable_pair_labels": True, "cases": pairs,
        })
    return counts, comparisons


def campaign(args, doc, cases):
    tree = TreeBudget(6, 2)
    cohort_sha, witness_sha = map(sha256_file, (args.cohort, args.witness))

    def run(arm):
        receipt = args.output / f"{arm}.json.gz"
        if not receipt.exists():
            command = [
                sys.executable, "-u", "-m", "configs.evaluate_ppt_bpd_rewrite_transfer",
                *(str(getattr(args, field)) for field in ("cohort", "witness", "source_audit", "sessions", "output")),
                "--arm", arm,
            ]
            with (args.output / f"{arm}.log").open("ab") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return arm, read_arm(receipt, arm, doc, cases, cohort_sha, witness_sha)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            outcomes = dict(pool.map(run, ARMS))
        counts, comparisons = summarize(cases, outcomes, doc["registration"])
        report = {
            "cohort_sha256": cohort_sha, "witness_sha256": witness_sha, "source_audit_sha256": doc["source_audit_sha256"],
            "counts": counts, "primary_paired_comparisons": comparisons, "memory": tree.stats(),
            "arm_evidence_sha256": {arm: sha256_file(args.output / f"{arm}.json.gz") for arm in ARMS},
        }
        with (args.output / "summary.json").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for arm, counts in counts.items():
            print(arm, counts, flush=True)
    finally:
        tree.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "witness", "source_audit", "sessions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    doc, witness, cases = prepare(args.cohort, args.witness, args.source_audit)
    if args.arm:
        tree = TreeBudget(2, 1)
        try:
            one_arm(args, tree, doc, witness, cases, args.output / f"{args.arm}.json.gz")
        finally:
            tree.close()
    else:
        campaign(args, doc, cases)


if __name__ == "__main__":
    main()
