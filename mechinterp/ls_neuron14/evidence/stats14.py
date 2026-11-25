"""Report the locked LS14 seed-paired endpoints through the official FSI bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "evalmath" / "src"))
from evalmath_fsi_gate import score

SEEDS = tuple(range(445, 469))
ARMS = ("point_single", "bound_single", "point_paired", "bound_paired")
VERDICTS = {"CERTIFIED", "POINT_VIOLATION", "UNRESOLVED"}


def verdict(row: dict, endpoint: str) -> str:
    item = row[endpoint]
    value = item["verdict"] if "verdict" in item else item["certificate"]["verdict"]
    if value not in VERDICTS:
        raise ValueError(f"Unexpected {endpoint} verdict {value!r}")
    return value


def run(certificates: Path, outdir: Path, registration_dir: Path = HERE) -> dict:
    source = certificates.read_bytes()
    raw = json.loads(source)
    rows = raw["rows"] if isinstance(raw, dict) else raw
    keyed = {}
    for row in rows:
        key = (row["seed"], row["arm"])
        if key in keyed:
            raise ValueError(f"Duplicate endpoint: {key}")
        if row.get("dose", 15) != 15:
            raise ValueError("LS14 registered dose is 15 percent")
        keyed[key] = row
    expected = {(s, a) for s in SEEDS for a in ARMS}
    if set(keyed) != expected:
        raise ValueError(f"Endpoint coverage mismatch: missing={expected-set(keyed)}, extra={set(keyed)-expected}")
    for seed in SEEDS:
        if len({verdict(keyed[seed, arm], "initial") for arm in ARMS}) != 1:
            raise ValueError(f"Initial verdict mismatch for seed {seed}")
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm in ARMS:
        entries = [keyed[s, arm] for s in SEEDS]
        initial = [int(verdict(r, "initial") == "CERTIFIED") for r in entries]
        final = [int(verdict(r, "final") == "CERTIFIED") for r in entries]
        summaries.append({
            "arm": arm,
            "n_units": len(SEEDS),
            "initial_certificates": sum(initial),
            "final_certificates": sum(final),
            "new_certificates": sum(verdict(r, "initial") == "POINT_VIOLATION" and verdict(r, "final") == "CERTIFIED" for r in entries),
            "certificate_acquisitions": sum(b and not a for a, b in zip(initial, final)),
            "initial_unresolved": sum(verdict(r, "initial") == "UNRESOLVED" for r in entries),
            "lost_certificates": sum(a and not b for a, b in zip(initial, final)),
            "new_seeds": [seed for seed, r in zip(SEEDS, entries) if verdict(r, "initial") == "POINT_VIOLATION" and verdict(r, "final") == "CERTIFIED"],
            "certificate_acquisition_seeds": [s for s, a, b in zip(SEEDS, initial, final) if b and not a],
            "lost_seeds": [s for s, a, b in zip(SEEDS, initial, final) if a and not b],
            "final_verdicts": {v: sum(verdict(r, "final") == v for r in entries) for v in sorted(VERDICTS)},
        })
    comparisons = []
    for kind in ("single", "paired"):
        registration = registration_dir / f"registration_{kind}.json"
        registered = json.loads(registration.read_bytes())
        if registered["prespecified_seeds"] != list(SEEDS):
            raise ValueError("Registration seed cohort changed")
        if registered["planned_comparisons"] != 2 or registered["minimum_effect"] != "0.25":
            raise ValueError("Registration effect floor or comparison family changed")
        pairs = []
        for seed in SEEDS:
            candidate = keyed[seed, f"bound_{kind}"]
            baseline = keyed[seed, f"point_{kind}"]
            cinitial, cfinal = verdict(candidate, "initial"), verdict(candidate, "final")
            binitial, bfinal = verdict(baseline, "initial"), verdict(baseline, "final")
            pairs.append({
                "case_id": f"{kind}-seed-{seed}",
                "root_id": f"initialization-{seed}",
                "seed": seed,
                "candidate_correct": int(cfinal == "CERTIFIED" and cinitial == "POINT_VIOLATION"),
                "control_correct": int(bfinal == "CERTIFIED" and binitial == "POINT_VIOLATION"),
                "candidate_certificate_acquired": int(cfinal == "CERTIFIED" and cinitial != "CERTIFIED"),
                "control_certificate_acquired": int(bfinal == "CERTIFIED" and binitial != "CERTIFIED"),
                "candidate_initial_verdict": cinitial,
                "candidate_final_verdict": cfinal,
                "control_initial_verdict": binitial,
                "control_final_verdict": bfinal,
            })
        case_path = outdir / f"paired_{kind}.jsonl"
        case_path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
        report = score(case_path, registration)
        report["candidate_only_seeds"] = [p["seed"] for p in pairs if p["candidate_correct"] and not p["control_correct"]]
        report["baseline_only_seeds"] = [p["seed"] for p in pairs if p["control_correct"] and not p["candidate_correct"]]
        report["operational_count_gate"] = report["candidate_correct"] >= 12 and report["candidate_correct"] - report["baseline_correct"] >= 6
        report["study_evidence_class"] = "PILOT_ONLY"
        report["comparison_role"] = "primary" if kind == "single" else "secondary"
        report["support_scope"] = "New exact-real certificates repairing proven initial point violations on fresh seeds from the fixed synthetic perturbation generator"
        (outdir / f"fsi_{kind}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        comparisons.append(report)
    result = {
        "experiment": "LS-NEURON-14",
        "study_evidence_class": "PILOT_ONLY",
        "unit": "independently seeded initial perturbation",
        "n_units": len(SEEDS),
        "n_final_model_endpoints": len(rows),
        "n_paired_cases_per_comparison": len(SEEDS),
        "planned_and_actual_comparisons": 2,
        "primary_operational_gate_passed": comparisons[0]["operational_count_gate"],
        "primary_endpoint": "initial POINT_VIOLATION and final CERTIFIED",
        "physical_dose": 0.15,
        "scientific_variation_floor": 0.25,
        "matched_sham_floor": None,
        "matched_sham_source": "No physically matched LS14 ranking sham; historical preflight uses the universal binary-difference support bound",
        "certificate_source_sha256": hashlib.sha256(source).hexdigest(),
        "arm_summaries": summaries,
        "comparisons": comparisons,
    }
    (outdir / "statistics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--registration-dir", type=Path, default=HERE)
    args = parser.parse_args()
    result = run(args.certificates, args.outdir, args.registration_dir)
    print(json.dumps({"evidence_class": result["study_evidence_class"], "primary_operational_gate_passed": result["primary_operational_gate_passed"], "n_units": result["n_units"]}))


if __name__ == "__main__":
    main()
