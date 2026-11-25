"""Descriptive paired LS17 comparison on the already exposed LS16 cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "evalmath" / "src"))
from evalmath_fsi_gate import score

SEEDS = tuple(range(493, 517))
ARMS = ("multistart", "beam")
VERDICTS = {"CERTIFIED", "POINT_VIOLATION", "UNRESOLVED"}


def validate_rows(raw):
    keyed = {}
    for row in raw["rows"]:
        key = row["seed"], row["arm"]
        if key in keyed or type(row["seed"]) is not int or row["dose"] != 15:
            raise ValueError("Duplicate endpoint or incorrect seed/dose")
        for endpoint in ("initial", "final"):
            if row[endpoint]["verdict"] not in VERDICTS:
                raise ValueError("Unknown exact verdict")
        keyed[key] = row
    expected = {(s, a) for s in SEEDS for a in ARMS}
    if set(keyed) != expected:
        raise ValueError("Incomplete exposed cohort")
    for seed in SEEDS:
        if keyed[seed, "beam"]["initial"]["verdict"] != keyed[seed, "multistart"]["initial"]["verdict"]:
            raise ValueError("Initial verdict differs between methods")
    return keyed


def repaired(row):
    return int(row["initial"]["verdict"] == "POINT_VIOLATION" and row["final"]["verdict"] == "CERTIFIED")


def run(certificates, outdir, registration=HERE / "registration.json"):
    payload = certificates.read_bytes()
    rows = validate_rows(json.loads(payload))
    spec = json.loads(registration.read_text())
    for key, expected in {"prespecified_seeds": list(SEEDS), "planned_comparisons": 1,
        "candidate_arm": "beam", "control_arm": "multistart", "minimum_effect": "0.25",
        "sealed_readout": False, "hypothesis_preregistered": False,
        "study_evidence_class": "EXPOSED_DEVELOPMENT_ONLY"}.items():
        if spec.get(key) != expected:
            raise ValueError(f"Altered analysis contract: {key}")
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed in SEEDS:
        candidate, baseline = rows[seed, "beam"], rows[seed, "multistart"]
        pairs.append({"case_id": f"ls17-seed-{seed}", "root_id": f"initialization-{seed}",
            "seed": seed, "candidate_correct": repaired(candidate), "control_correct": repaired(baseline),
            "candidate_initial_verdict": candidate["initial"]["verdict"],
            "candidate_final_verdict": candidate["final"]["verdict"],
            "control_initial_verdict": baseline["initial"]["verdict"],
            "control_final_verdict": baseline["final"]["verdict"]})
    paired_path = outdir / "paired_outcomes.jsonl"
    paired_path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
    result = score(paired_path, registration)
    result["candidate_only_seeds"] = [p["seed"] for p in pairs if p["candidate_correct"] and not p["control_correct"]]
    result["baseline_only_seeds"] = [p["seed"] for p in pairs if p["control_correct"] and not p["candidate_correct"]]
    result["study_evidence_class"] = "EXPOSED_DEVELOPMENT_ONLY"
    result["inferential_scope"] = "The cohort selected this ablation. Nominal test/interval computations are descriptive; exposure prevents a confirmation or population-effect claim."
    result["inherited_planning_reference_passed"] = result["candidate_correct"] >= 12 and result["candidate_correct"] - result["baseline_correct"] >= 6
    result["inherited_planning_reference_note"] = "Beam11/24 was already known; the >=12count condition was known to fail before LS17. This is no new operational gate."
    (outdir / "fsi_beam_vs_multistart.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summaries = []
    for arm in ARMS:
        items = [rows[seed, arm] for seed in SEEDS]
        summaries.append({"arm": arm, "n_units": len(SEEDS), "new_certificates": sum(repaired(r) for r in items),
            "new_seeds": [r["seed"] for r in items if repaired(r)],
            "initial_verdicts": {v: sum(r["initial"]["verdict"] == v for r in items) for v in sorted(VERDICTS)},
            "final_verdicts": {v: sum(r["final"]["verdict"] == v for r in items) for v in sorted(VERDICTS)},
            "certificate_acquisitions": sum(r["initial"]["verdict"] != "CERTIFIED" and r["final"]["verdict"] == "CERTIFIED" for r in items),
            "lost_certificates": sum(r["initial"]["verdict"] == "CERTIFIED" and r["final"]["verdict"] != "CERTIFIED" for r in items)})
    report = {"experiment": "LS-NEURON-17", "study_evidence_class": "EXPOSED_DEVELOPMENT_ONLY",
        "unit": "independently seeded initialization, already exposed in LS16", "n_units": 24,
        "n_final_model_endpoints": 48, "planned_and_actual_comparisons": 1,
        "primary_contrast": "dynamic beam minus three independent greedy paths", "physical_dose": .15,
        "scientific_variation_floor": .25, "floor_scope": "Inherited planning reference",
        "matched_sham_floor": None, "direct_power_estimate": None,
        "certificate_sha256": hashlib.sha256(payload).hexdigest(),
        "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
        "arm_summaries": summaries, "comparison": result}
    (outdir / "statistics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.certificates, args.outdir)
    print(json.dumps({"scope": result["study_evidence_class"], "counts": result["arm_summaries"],
                      "descriptive_p": result["comparison"]["p_unadjusted"]}))


if __name__ == "__main__":
    main()
