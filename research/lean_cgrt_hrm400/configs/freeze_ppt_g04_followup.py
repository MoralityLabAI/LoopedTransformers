"""Freeze fresh, budget-feasible Horn roots for the registered Micro follow-up."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from lcgrt.fragment import reference_sizes
from ppt_gym.data import canonical, sha256_file
from ppt_gym.live_suite import LiveCase
from ppt_gym.ram_curriculum import freeze, load


def shortest_script(case):
    theorem = case.theorem()
    sizes = reference_sizes(theorem)
    return sum(sizes[atom] for atom in theorem.goal) + int(len(theorem.goal) > 1)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("prior_cohorts", nargs="+", type=Path)
    args = cli.parse_args()
    excluded = set()
    for path in args.prior_cohorts:
        _, groups = load(path)
        excluded.update(case.lineage for group in groups.values() for case in group)
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "candidates.json"
        freeze(path, seed=908, families=2, queries=48, rounds=1)
        candidate_sha256 = sha256_file(path)
        record = json.loads(path.read_text(encoding="utf-8"))
    for group, n in (("within", 4), ("ood", 2)):
        rows = [row for row in record["cohorts"][group] if row["lineage"] not in excluded]
        keep = []
        for family in sorted({row["family"] for row in rows}):
            selected = [row for row in rows if row["family"] == family
                        and 3 <= shortest_script(LiveCase(**row)) <= 8][:n]
            if len(selected) != n:
                raise ValueError(f"short feasible-root pool in {family}")
            keep.extend(selected)
        record["cohorts"][group] = keep
    for group in ("warmup", "support"):
        if any(row["lineage"] in excluded for row in record["cohorts"][group]):
            raise ValueError("follow-up support overlaps a prior frozen cohort")
    record["registration"].update({
        "purpose": "prospective budget-feasible follow-up pilot",
        "primary_contrast": "Micro SGD versus frozen and retrieval after 16 support roots",
        "seed": 908, "within_roots_per_family": 4, "ood_roots_per_family": 2,
        "candidate_sha256": candidate_sha256,
        "prior_cohort_sha256": sorted(sha256_file(path) for path in args.prior_cohorts),
        "excluded_lineages_sha256": hashlib.sha256(canonical(sorted(excluded)).encode()).hexdigest(),
        "query_admission": "first fresh lineages with 3..8 reference checks per family, before live scoring",
    })
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(canonical(record) + "\n")


if __name__ == "__main__":
    main()
