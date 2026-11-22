"""Combine the registered rewrite model seeds against paired native proof outcomes."""

import argparse
import gzip
import json
from pathlib import Path

from configs.freeze_ppt_bpd_rewrite import ARMS
from ppt_gym.data import digest, sha256_file


MODES = ("pristine", "prefer_new")


def read_seed(seed, audit_path, revisit_path, reference, study_sha, replication_sha, cohort_sha, reference_sha):
    audit_sha = sha256_file(audit_path)
    audit = json.loads(audit_path.read_bytes())
    revisit = json.loads(gzip.decompress(revisit_path.read_bytes()))
    if (audit["cohort_sha256"] != cohort_sha or audit["reference_sha256"] != reference_sha
            or audit["exploration_study_sha256"] != study_sha
            or audit.get("replication_study_sha256") != replication_sha
            or set(audit["trials"]) != set(ARMS) or set(revisit["results"]) != set(ARMS)
            or revisit["audit_sha256"] != audit_sha or revisit["cohort_sha256"] != cohort_sha
            or revisit["reference_sha256"] != reference_sha):
        raise ValueError(f"seed {seed} has mismatched audits")
    rows = {mode: {} for mode in MODES}
    for arm in ARMS:
        trial = audit["trials"][arm]
        outcomes = revisit["results"][arm]
        if (trial["manifest"]["seed"] != seed or set(outcomes["roots"]) != set(reference)
                or outcomes["checkpoint"]["weights_sha256"] != trial["final_weights_sha256"]
                or outcomes["checkpoint"]["session_sha256"] != trial["session_sha256"]):
            raise ValueError(f"seed {seed} checkpoint identity changed for {arm}")
        for mode in MODES:
            arm_rows = rows[mode].setdefault(arm, {})
            for root, recorded in outcomes["roots"].items():
                row = recorded[mode]
                success = row["certified"]
                cert = row["certificate"]
                if (not row["replay"]["replay_passed"] or not 0 <= row["checks"] <= 8
                        or row["macs"] > trial["manifest"]["max_macs"]
                        or (mode == "pristine" and success != trial["stages"]["final"][root]["certified"])
                        or (success and (not reference[root]["certified"] or not cert["passed"]
                                         or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]))):
                    raise ValueError(f"seed {seed} has an invalid native outcome for {arm}")
                if reference[root]["certified"]:
                    arm_rows[root] = int(success)
            if sum(arm_rows.values()) != outcomes["counts"][mode]["certified"]:
                raise ValueError(f"seed {seed} counters differ from their native outcomes for {arm}")
    return rows, {"audit_sha256": audit_sha, "revisit_sha256": sha256_file(revisit_path)}


def paired(rows, group, controls):
    candidate = rows["hybrid_rl"]
    return {
        arm: {
            "wins": sum(candidate[root] > rows[arm][root] for root in group),
            "losses": sum(candidate[root] < rows[arm][root] for root in group),
            "ties": sum(candidate[root] == rows[arm][root] for root in group),
        }
        for arm in controls
    }


def combine(args):
    study = json.loads(args.study.read_bytes())
    replication = json.loads(args.replication.read_bytes())
    cohort_sha, study_sha, replication_sha = map(sha256_file, (args.cohort, args.study, args.replication))
    reference_sha = sha256_file(args.reference)
    reference = json.loads(args.reference.read_bytes())
    if (study["cohort_sha256"] != cohort_sha or replication["cohort_sha256"] != cohort_sha
            or replication["exploration_study_sha256"] != study_sha or reference["cohort_sha256"] != cohort_sha):
        raise ValueError("the combined rewrite audits must use the registered original cohort")
    expected = [study["seed"], *replication["model_seeds"]]
    supplied = [int(seed) for seed, _, _ in args.run]
    if len(expected) != len(supplied) or set(expected) != set(supplied):
        raise ValueError("supply each registered seed exactly once")
    by_seed, sources = {}, {}
    for raw_seed, audit_path, revisit_path in args.run:
        seed = int(raw_seed)
        by_seed[seed], sources[seed] = read_seed(
            seed, Path(audit_path), Path(revisit_path), reference["rows"], study_sha,
            replication_sha if seed != study["seed"] else None, cohort_sha, reference_sha,
        )
    initial_source = sources[study["seed"]]
    if (replication["seed_151_audit_sha256"] != initial_source["audit_sha256"]
            or replication["seed_151_revisit_control_sha256"] != initial_source["revisit_sha256"]):
        raise ValueError("the replication originally registered different seed-151 evidence")

    valid = {root: row["cluster"] for root, row in reference["rows"].items() if row["certified"]}
    groups = {key: [root for root, cluster in valid.items() if cluster == key] for key in set(valid.values())}
    summary = {}
    for mode in MODES:
        spectra = {str(seed): by_seed[seed][mode] for seed in sorted(by_seed)}
        comparisons = {seed: paired(rows, valid, ARMS[1:]) for seed, rows in spectra.items()}
        comparisons_by_group = {
            key: {seed: paired(rows, group, ARMS[1:]) for seed, rows in spectra.items()}
            for key, group in groups.items()
        }
        summary[mode] = {
            "certificates_by_seed": {seed: {arm: sum(values.values()) for arm, values in rows.items()}
                                     for seed, rows in spectra.items()},
            "paired_by_seed": comparisons, "paired_by_structural_group_and_seed": comparisons_by_group,
        }
    return {
        "scope": "frozen original BPD rewrite development with repeated model seeds",
        "cohort_sha256": cohort_sha, "reference_sha256": reference_sha,
        "exploration_study_sha256": study_sha, "replication_study_sha256": replication_sha,
        "model_seeds": sorted(expected), "structural_groups": len(groups),
        "valid_declarations_per_seed": len(valid), "sources": sources, "results": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "study", "replication", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--run", nargs=3, action="append", metavar=("SEED", "AUDIT", "REVISITS"), required=True)
    args = parser.parse_args()
    report = combine(args)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report["results"]["prefer_new"]["certificates_by_seed"], sort_keys=True))


if __name__ == "__main__":
    main()
