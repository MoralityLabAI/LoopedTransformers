"""Compare the same first-action rescalings in frozen RL and imitation checkpoints."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_conjunction_calibration import read_arm
from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.check_ppt_conjunction_pilot import ProofLengthWitness
from configs.eval_ppt_conjunction_calibration import read_calibration
from configs.probe_ppt_hybrid_bank import INTERACTION_DOSES
from ppt_gym.data import canonical, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "input", "pristine", "doses", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    _, cases = read_calibration(args.registry, args.input)
    registry_sha, input_sha = sha256_file(args.registry), sha256_file(args.input)
    witness = ProofLengthWitness()
    clusters = defaultdict(list)
    for case in cases:
        clusters[case.cluster].append(case.root)
    arms = ("hybrid", "hybrid_imitation")
    statuses = {f"{arm}-{dose}": 0 for arm in arms for dose in INTERACTION_DOSES}
    resource = json.loads((args.doses / "calibration-resources.json").read_bytes())
    if (resource["statuses"], resource["registry_sha256"], resource["calibration_sha256"]) != (
        statuses, registry_sha, input_sha
    ):
        raise ValueError("native rescaling campaign did not complete its matched arms")
    results, evidence = {}, {}
    for arm in arms:
        identity, pristine, raw = read_arm(args.pristine / (arm + ".sqlite"), arm, cases,
                                           registry_sha, input_sha, witness)
        results[arm], evidence[arm] = {}, {"pristine": (identity, pristine, raw)}
        for dose in INTERACTION_DOSES:
            key = f"{arm}-{dose}"
            row = read_arm(args.doses / (key + ".sqlite"), key, cases, registry_sha, input_sha, witness)
            if row[0]["root_dose"] != dose or any(row[0][field] != identity[field] for field in (
                "checkpoint_sha256", "pilot_session_sha256", "verifier", "engine_sha256", "proof_cpu_seconds", "max_macs"
            )):
                raise ValueError("dose arm differs from the original policy or proof budget")
            evidence[arm][dose] = row
        for dose, (_, rows, _) in evidence[arm].items():
            results[arm][dose] = {**pair_counts(rows, clusters),
                                  "correct_first": sum(row["feasible_first"] for row in rows.values()),
                                  "changed_paths": sum(row["path"] != pristine[root]["path"] for root, row in rows.items())}
    summary = {"scope": "reused calibration layouts, seed 103, diagnostic rescaling after reading the pilot",
               "registry_sha256": registry_sha, "calibration_sha256": input_sha, "pair_count": len(clusters),
               "counts": results, "resources": resource}
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for name, doses in results.items():
        print(name, doses)


if __name__ == "__main__":
    main()
