"""Block public descriptive evaluation unless confirmation and source freeze hold."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def main() -> int:
    selection_path = Path("reports/evals/arc_public_descriptive_v1/selection_manifest.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    confirmation = json.loads(Path("experiments/arc_open_alphabet_spiral_v4/confirmation/summary.json").read_text(encoding="utf-8"))
    audit = json.loads(Path("reports/evals/arc_open_alphabet_spiral_v4_audit/audit.json").read_text(encoding="utf-8"))
    source_commit = subprocess.check_output(("git", "-C", "external/ARC-AGI", "rev-parse", "HEAD"), text=True).strip()
    hash_matches = all(
        sha256(Path(f"external/ARC-AGI/data/evaluation/{task_id}.json").read_bytes()).hexdigest() == digest
        for task_id, digest in selection["selected_file_sha256"].items()
    )
    checks = {
        "synthetic_confirmation_gate_passed": bool(confirmation["gate"]["all_passed"]),
        "evaluator_accepts_confirmation": audit["operational_decision"]["action"] == "accept",
        "source_commit_matches": source_commit == selection["source_commit"],
        "selected_file_hashes_match": hash_matches,
        "selected_task_count_is_24": selection["selected_tasks"] == 24,
        "public_denominator_is_400": selection["total_public_tasks"] == 400,
        "checkpoint_exists": Path("experiments/arc_open_alphabet_spiral_v4/confirmation/checkpoints/step_0320.pt").is_file(),
    }
    report = {"status": "pass" if all(checks.values()) else "block", "checks": checks, "selection_digest": selection["selection_digest"]}
    out = Path("reports/evals/arc_public_descriptive_v1/preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
