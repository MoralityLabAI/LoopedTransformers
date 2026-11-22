"""Independently recompile the distinct certificates in the pending-goal transfer audit."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from configs.run_ppt_pending_recovery import frozen_cohort
from ppt_gym.data import digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


AXIOM_FREE = re.compile(r"^'([^']+)' does not depend on any axioms$")


def proof_sources(cohort, evidence_path):
    from lcgrt.certify import NAME_RE, TACTIC_RE

    registration, groups = frozen_cohort(cohort)
    cases = {case.root: case for pair in groups.values() for case in pair}
    evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
    if evidence["summary"]["cohort_sha256"] != sha256_file(cohort):
        raise ValueError("proof source and registered transfer cohort differ")
    receipts = evidence["policy_receipts"]
    if set(receipts) != {str(seed) for seed in registration["seeds"]}:
        raise ValueError("proof receipts omit a registered seed")
    accepted, occurrences = {}, 0
    for seed, models in receipts.items():
        if set(models) != set(registration["arms"]):
            raise ValueError("proof receipts omit a registered policy")
        for arm, worlds in models.items():
            if set(worlds) != set(cases):
                raise ValueError("a transfer policy is missing a registered theorem")
            conditions = registration["edits"] if arm == registration["arms"][0] else ["pristine"]
            counts = Counter()
            for root, modes in worlds.items():
                case = cases[root]
                if set(modes) != set(conditions):
                    raise ValueError("a transfer theorem is missing a registered intervention")
                for mode, row in modes.items():
                    if not row["certified"]:
                        continue
                    certificate = row["certificate"]
                    script = row["script"]
                    names = [name for field in case.binders.split(") (")
                             for name in field.lstrip("(").split(" : ", 1)[0].split()]
                    expected = [f"theorem {case.tid} : {case.statement} := by",
                                "  intro " + " ".join(names),
                                *("  " + tactic for tactic in script),
                                f"#print axioms {case.tid}"]
                    if (certificate["source"] != "\n".join(expected) + "\n" or NAME_RE.search(case.tid) or not script
                            or not all(TACTIC_RE.fullmatch(tactic) for tactic in script)
                            or not certificate["passed"] or certificate["axioms"] != []
                            or digest(certificate["source"]) != certificate["source_digest"]):
                        raise ValueError("a successful policy receipt lacks its exact theorem and tactic proof")
                    accepted[certificate["source_digest"]] = (case.tid, certificate["source"])
                    occurrences += 1
                    counts[mode] += 1
            for mode in conditions:
                condition = arm if mode == "pristine" else mode
                if counts[mode] != evidence["summary"]["counts"][seed][condition]["proofs"]:
                    raise ValueError("certificate counts differ from the audited transfer results")
    return evidence, accepted, occurrences


def check_batch(certificates, index):
    from lcgrt import config

    expected = set()
    lines = []
    for i, (theorem, source) in enumerate(certificates):
        namespace = f"PptProofBatch_{index}_{i}"
        lines.extend((f"namespace {namespace}", source.rstrip("\n"), f"end {namespace}", ""))
        expected.add(f"{namespace}.{theorem}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ProofBatch.lean"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        process = subprocess.run([str(config.LEAN_EXE), str(path)], capture_output=True, text=True, encoding="utf-8",
                                 timeout=60, env=config.lean_env())
    verified = {match[1] for line in process.stdout.splitlines() if (match := AXIOM_FREE.fullmatch(line.strip()))}
    if process.returncode != 0 or verified != expected:
        raise ValueError(f"independent Lean recheck disagreed in batch {index}: "
                         f"{process.returncode}, {len(verified)}/{len(expected)}")
    return len(verified)


def recheck(cohort, evidence_path, batch_size, tree):
    from lcgrt import config
    from ppt_gym.live_lean import fingerprint

    evidence, sources, occurrences = proof_sources(cohort, evidence_path)
    verifier = fingerprint(config.REPL_EXE, config.LEAN_EXE, "4.19.0")
    if verifier != evidence["summary"]["verifier"]:
        raise ValueError("certificate recheck requires the registered Lean executable")
    proofs = sorted(sources)
    start = time.perf_counter()
    checked = 0
    for i in range(0, len(proofs), batch_size):
        checked += check_batch([sources[key] for key in proofs[i:i + batch_size]], i // batch_size)
    return {"cohort_sha256": sha256_file(cohort), "policy_evidence_sha256": sha256_file(evidence_path),
            "verifier": verifier, "certificate_occurrences": occurrences, "unique_certificates": checked,
            "lean_invocations": (len(proofs) + batch_size - 1) // batch_size,
            "source_set_sha256": digest(proofs), "lean_wall_seconds": time.perf_counter() - start,
            "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "evidence", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--batch-size", type=int, choices=range(1, 65), default=32)
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        report = recheck(args.cohort, args.evidence, args.batch_size, tree)
        with args.output.open("x", encoding="utf-8", newline="\n") as out:
            out.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(report["unique_certificates"], "distinct sources independently checked in",
              report["lean_invocations"], "Lean invocations")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
