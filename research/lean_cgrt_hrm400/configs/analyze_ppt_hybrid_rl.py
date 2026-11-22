"""Pair completed hybrid-policy development outcomes by native theorem root."""
import argparse
import json
from collections import Counter
from pathlib import Path

from ppt_gym import ram_store
from ppt_gym.data import sha256_file
from ppt_gym.hybrid_rl.corpus import load


def read_trial(path, cases, cohort_sha):
    stages = {}
    reward = Counter()
    with ram_store.open_store(path) as db:
        manifest = ram_store.read_manifest(db)
        if manifest["cohort"] != cohort_sha:
            raise ValueError("cannot compare a changed frozen cohort")
        seen = 0
        for row in ram_store.ledger(db):
            seen += 1
            phase = row["phase"]
            update = row["update"]
            if update:
                reward[f"{phase}_updates"] += row["weights_before"] != row["weights_after"]
            for ep in row["episodes"]:
                if not 0 <= ep["tactic_checks"] <= 8 or ep["model_macs"] > manifest["max_macs"]:
                    raise ValueError("proof-search budget exceeded")
                if phase in ("train", "mathcan"):
                    reward[f"{phase}_episodes"] += 1
                    reward["archived_closures"] += int(ep.get("archived_closure", False))
                    reward["censored_replay"] += int(ep.get("reward_censored", False))
                    continue
                key = ep["root"]
                group = stages.setdefault(phase, {})
                if key not in cases or key in group or not ep["native_replay"]["replay_passed"]:
                    raise ValueError("missing or repeated replayed development root")
                certificate = ep["certificate"]
                success = ep["verified_closed"] and ep["within_cpu_budget"]
                if success and not (certificate["passed"] and certificate["axioms"] == []):
                    raise ValueError("scored closure lacks an independent axiom-free certificate")
                group[key] = {"certified": int(success), "macs": ep["model_macs"],
                              "tactic_checks": ep["tactic_checks"], "cpu_seconds": ep["search_cpu_seconds"],
                              "tactics": [e["tactic"] for e in ep["events"] if e["kind"] == "tactic"]}
        if seen != manifest["entry_count"] or any(set(items) != set(cases) for items in stages.values()):
            raise ValueError("incomplete development trial")
    return {"manifest": manifest, "stages": stages, "training": dict(reward)}


def compare(a, b, cases):
    bins = {}
    for root, case in cases.items():
        row, ref = a[root], b[root]
        difference = row["certified"] - ref["certified"]
        key = ["all", f"{case.source}/{case.family}"]
        for label in key:
            counter = bins.setdefault(label, Counter())
            counter.update(roots=1, candidate=row["certified"], control=ref["certified"],
                           wins=int(difference == 1), losses=int(difference == -1),
                           ties=int(difference == 0), changed_paths=int(row["tactics"] != ref["tactics"]))
    return {key: dict(value) for key, value in bins.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sessions", nargs="+", help="label=path; first label is the candidate")
    args = parser.parse_args()
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    trials = {}
    for spec in args.sessions:
        name, path = spec.split("=", 1)
        if name in trials:
            raise ValueError("duplicate policy comparison label")
        trials[name] = read_trial(Path(path), cases, sha256_file(args.cohort))
    candidate = next(iter(trials.values()))["stages"]
    deltas = {name: compare(candidate["final"], trial["stages"]["final"], cases)
              for name, trial in list(trials.items())[1:]}
    controls = {edit: compare(candidate["final"], candidate[edit], cases)
                for edit in ("zero", "sham") if edit in candidate}
    result = {"scope": "development only", "theorem_roots": len(cases), "paired_final": deltas,
              "matched_within_model_interventions": controls,
              "within_model_before_after": {
                  name: compare(trial["stages"]["final"], trial["stages"]["initial"], cases)
                  for name, trial in trials.items()},
              "root_custody": {root: {"source": c.source, "family": c.family, "cluster": c.cluster}
                               for root, c in cases.items()}, "trials": trials}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"paired_final": deltas, "interventions": controls}, indent=2))


if __name__ == "__main__":
    main()
