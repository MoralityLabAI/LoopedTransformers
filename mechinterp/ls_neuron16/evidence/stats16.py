"""Score the prospectively locked LS16 seed-paired endpoints with official FSI-01."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "evalmath" / "src"))
from evalmath_fsi_gate import score

SEEDS = tuple(range(493, 517))
ARMS = ("greedy", "beam")
VERDICTS = {"CERTIFIED", "POINT_VIOLATION", "UNRESOLVED"}


def verdict(row: dict, endpoint: str) -> str:
    value = row[endpoint]
    value = value["verdict"] if "verdict" in value else value["certificate"]["verdict"]
    if value not in VERDICTS:
        raise ValueError(f"Unexpected {endpoint} verdict {value!r}")
    return value


def validated_rows(raw: dict | list) -> dict:
    rows = raw["rows"] if isinstance(raw, dict) else raw
    keyed = {}
    for row in rows:
        key = (row["seed"], row["arm"])
        if key in keyed:
            raise ValueError(f"Duplicate endpoint: {key}")
        if type(row["seed"]) is not int or row.get("dose", 15) != 15:
            raise ValueError("LS16 needs integer seed IDs at the registered 15-percent dose")
        verdict(row, "initial")
        verdict(row, "final")
        keyed[key] = row
    expected = {(s, a) for s in SEEDS for a in ARMS}
    if set(keyed) != expected:
        raise ValueError(f"Endpoint coverage mismatch: missing={expected-set(keyed)}, extra={set(keyed)-expected}")
    for seed in SEEDS:
        if len({verdict(keyed[seed, arm], "initial") for arm in ARMS}) != 1:
            raise ValueError(f"Initial verdict mismatch for seed {seed}")
    return keyed


def run(certificates: Path, outdir: Path, registration: Path = HERE / "registration.json") -> dict:
    source = certificates.read_bytes()
    raw = json.loads(source)
    keyed = validated_rows(raw)
    spec = json.loads(registration.read_bytes())
    expected = {
        "prespecified_seeds": list(SEEDS), "planned_comparisons": 1,
        "minimum_effect": "0.25", "candidate_arm": "beam", "control_arm": "greedy",
        "physical_dose": "0.15", "study_evidence_class": "PILOT_ONLY",
        "operational_gate": {"candidate_new_certificates_min": 12, "candidate_minus_control_new_min": 6},
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Registration changed: {key}")
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm in ARMS:
        entries = [keyed[seed, arm] for seed in SEEDS]
        initial = [verdict(r, "initial") for r in entries]
        final = [verdict(r, "final") for r in entries]
        new_seeds = [s for s, a, b in zip(SEEDS, initial, final) if a == "POINT_VIOLATION" and b == "CERTIFIED"]
        acquired_seeds = [s for s, a, b in zip(SEEDS, initial, final) if a != "CERTIFIED" and b == "CERTIFIED"]
        lost_seeds = [s for s, a, b in zip(SEEDS, initial, final) if a == "CERTIFIED" and b != "CERTIFIED"]
        summaries.append({
            "arm": arm, "n_units": len(SEEDS),
            "initial_certificates": initial.count("CERTIFIED"),
            "initial_point_violations": initial.count("POINT_VIOLATION"),
            "initial_unresolved": initial.count("UNRESOLVED"),
            "final_certificates": final.count("CERTIFIED"),
            "final_verdicts": {v: final.count(v) for v in sorted(VERDICTS)},
            "new_certificates": len(new_seeds), "new_seeds": new_seeds,
            "certificate_acquisitions": len(acquired_seeds), "certificate_acquisition_seeds": acquired_seeds,
            "lost_certificates": len(lost_seeds), "lost_seeds": lost_seeds,
        })
    pairs = []
    for seed in SEEDS:
        candidate, baseline = keyed[seed, "beam"], keyed[seed, "greedy"]
        ci, cf = verdict(candidate, "initial"), verdict(candidate, "final")
        bi, bf = verdict(baseline, "initial"), verdict(baseline, "final")
        pairs.append({
            "case_id": f"beam-vs-greedy-seed-{seed}", "root_id": f"initialization-{seed}", "seed": seed,
            "candidate_correct": int(ci == "POINT_VIOLATION" and cf == "CERTIFIED"),
            "control_correct": int(bi == "POINT_VIOLATION" and bf == "CERTIFIED"),
            "candidate_certificate_acquired": int(ci != "CERTIFIED" and cf == "CERTIFIED"),
            "control_certificate_acquired": int(bi != "CERTIFIED" and bf == "CERTIFIED"),
            "candidate_initial_verdict": ci, "candidate_final_verdict": cf,
            "control_initial_verdict": bi, "control_final_verdict": bf,
        })
    paired_path = outdir / "paired_outcomes.jsonl"
    paired_path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
    report = score(paired_path, registration)
    report["candidate_only_seeds"] = [p["seed"] for p in pairs if p["candidate_correct"] and not p["control_correct"]]
    report["baseline_only_seeds"] = [p["seed"] for p in pairs if p["control_correct"] and not p["candidate_correct"]]
    report["operational_count_gate"] = report["candidate_correct"] >= 12 and report["candidate_correct"] - report["baseline_correct"] >= 6
    report["study_evidence_class"] = "PILOT_ONLY"
    report["support_scope"] = "Exact-real certificates repairing proven initial point violations on fresh seeds from the fixed synthetic perturbation generator"
    report["confirmation_policy"] = "The official FSI result is reported intact. The LS16 study stays PILOT_ONLY because direct beam power and matched-sham calibration were unavailable before execution."
    (outdir / "fsi_beam_vs_greedy.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    result = {
        "experiment": "LS-NEURON-16", "study_evidence_class": "PILOT_ONLY",
        "unit": "independently seeded initial perturbation", "n_units": len(SEEDS),
        "n_final_model_endpoints": len(keyed), "n_paired_cases_per_comparison": len(SEEDS),
        "planned_and_actual_comparisons": 1, "primary_contrast": "beam versus greedy",
        "primary_operational_gate_passed": report["operational_count_gate"],
        "primary_endpoint": "initial POINT_VIOLATION and final CERTIFIED",
        "physical_dose": 0.15, "scientific_variation_floor": 0.25,
        "matched_sham_floor": None,
        "matched_sham_source": "No calibrated LS16 beam sham; historical preflight supplies the universal support bound 1",
        "estimated_power": None,
        "readout_selection_exposure": spec["selection_exposure"],
        "causal_controls": spec["treatment"],
        "certificate_source_sha256": hashlib.sha256(source).hexdigest(),
        "registration_sha256": report["registration_sha256"],
        "arm_summaries": summaries, "comparisons": [report],
        "interval_scope": "Official per-arm Clopper-Pearson intervals are marginal 95-percent intervals under independent Bernoulli seed sampling; they do not jointly cover both arms at 95 percent.",
    }
    (outdir / "statistics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--registration", type=Path, default=HERE / "registration.json")
    args = parser.parse_args()
    result = run(args.certificates, args.outdir, args.registration)
    print(json.dumps({"evidence_class": result["study_evidence_class"], "primary_operational_gate_passed": result["primary_operational_gate_passed"], "n_units": result["n_units"]}))


if __name__ == "__main__":
    main()
