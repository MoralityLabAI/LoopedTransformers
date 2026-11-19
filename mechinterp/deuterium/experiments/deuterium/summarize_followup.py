"""Build a paper-facing evidence ledger from observed experiment receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def number(value, digits=6):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "unavailable"


def summarize(recovered: dict | None, pilot: dict | None, scored: dict | None,
              failures: list[dict]) -> dict:
    rows = []
    paragraphs = []
    hosted = bool(scored and scored.get("environment", {}).get("github_run_id"))
    completed = bool(scored and scored.get("comparison_completed"))
    location = "GitHub Actions" if hosted else "local CPU"
    if scored:
        state = f"{location}: {scored['status']}"
        description = (
            f"The frozen synthetic recurrent-gate protocol was executed on {location}. "
            f"The comparison completion flag is {completed}. "
            f"Registration commit: `{scored['registration']['registration_commit']}`. "
            f"Protocol SHA-256: `{scored['registration']['protocol_sha256']}`."
        )
        if not hosted:
            description += " GitHub Actions reproduction remains blocked by repository write access; the local freeze was recorded before held-out scoring."
    else:
        state = "PREPARED_UNSCORED; GitHub Actions unavailable"
        description = "The available receipts cover local prerequisite audits and pilot work. Reserved comparison seeds remain unscored."

    if recovered:
        rows.append({"item": "Recovered checkpoint task/replay gate",
                     "status": recovered["status"],
                     "evidence": f"{recovered['base_parameters']:,} parameters; {recovered['frozen_tensor_count']} frozen tensors checked; exact equality={recovered['all_frozen_tensors_equal']}; task performance evaluated={recovered['task_performance_evaluated']}."})
        for module in recovered["modules"]:
            rows.append({"item": f"Recovered residual: {module['module']}",
                         "status": "MATRIX AUDIT",
                         "evidence": f"{module['components']} learned components; residual/weight Frobenius norm={number(module['residual_relative_norm'])}."})
        paragraphs.append("Recovered-model missing prerequisites: " + "; ".join(recovered["missing_prerequisites"]) + ". These gaps prevent a recovered-task steering comparison.")
    if pilot:
        for seed in pilot["seed_results"]:
            calibration = seed.get("calibration_clean", {})
            rows.append({"item": f"Auxiliary pilot seed {seed['seed']}",
                         "status": seed["status"],
                         "evidence": f"Clean calibration accuracy={number(calibration.get('accuracy'))}; runtime={number(seed.get('runtime_seconds'), 2)} s; held-out evaluated={seed['heldout_evaluated']}."})
            if seed.get("decomposition"):
                d = seed["decomposition"]
                rows.append({"item": "Pilot replay diagnostics", "status": "CALIBRATION ONLY",
                             "evidence": f"Components-only KL={number(d['components_only_kl'])}; argmax agreement={number(d['components_only_argmax_agreement'])}; residual-restored maximum logit error={d['residual_restored_max_logit_error']:.3g}."})
    for receipt in failures:
        results = receipt.get("seed_results", [receipt])
        for seed in results:
            rows.append({"item": f"Retained pilot failure, seed {seed.get('seed', 'unknown')}",
                         "status": seed["status"],
                         "evidence": seed.get("exception", ", ".join(seed.get("failed_gates", []))).replace("\n", " ")})
    if scored:
        for seed in scored["seed_results"]:
            rows.append({"item": f"Registered seed {seed['seed']}", "status": seed["status"],
                         "evidence": f"Clean test accuracy={number(seed.get('heldout_clean', {}).get('accuracy'))}; runtime={number(seed.get('runtime_seconds'), 2)} s; failed gates={seed.get('failed_gates', [])}; held-out evaluated={seed['heldout_evaluated']}."})
            if seed.get("effects"):
                for contrast in ("random_minus_vpd_target_nll", "task_gradient_minus_vpd_target_nll"):
                    effect = seed["effects"][contrast]
                    rows.append({"item": f"Seed {seed['seed']}: {contrast}", "status": "PAIRED TEST EFFECT",
                                 "evidence": f"{number(effect['estimate'])}; 95% paired-row interval [{number(effect['ci95'][0])}, {number(effect['ci95'][1])}], n={effect['n']}."})
                clean = seed["metrics"]["clean"]
                for arm, metrics in seed["metrics"].items():
                    rows.append({"item": f"Seed {seed['seed']}: {arm}", "status": "TASK AND COLLATERAL",
                                 "evidence": f"Target NLL={number(metrics['target_nll'])}; target accuracy={number(metrics['target_accuracy'])}; collateral accuracy change={number(metrics['collateral_accuracy']-clean['collateral_accuracy'])}; false-commit rate on veto={number(metrics['false_commit_rate_on_veto'])}."})
        if completed:
            rows.append({"item": "Primary contrast across independent seeds", "status": scored["status"],
                         "evidence": f"Random minus VPD target NLL mean={number(scored['primary_mean'])}; 95% t interval [{number(scored['primary_ci95'][0])}, {number(scored['primary_ci95'][1])}]; VPD collateral gate={scored['collateral_gate_passed']}."})
            paragraphs.append("Positive NLL differences favor VPD. The primary interval uses three independent model/decomposition seeds and two degrees of freedom; within-seed row intervals describe a separate level of variation. Secondary outcomes retain descriptive status.")
        else:
            paragraphs.append("A failed prerequisite leaves the registered comparison incomplete. Every seed and failure remains in the receipt; successful subsets do not produce a method ranking.")
    paragraphs.append("The new experiment concerns a generated commit/veto/repair distribution with output-head edits. Its evidence is separate from the recovered arena, ARC, Wordle, governance and adapter records.")
    output = {"schema_version": 1, "status": state, "comparison_completed": completed,
              "hosted_run_observed": hosted, "summary": description, "rows": rows, "paragraphs": paragraphs}
    if completed:
        output["primary_effects"] = [
            {"seed": seed["seed"], "effect": seed["effects"]["random_minus_vpd_target_nll"]["estimate"],
             "ci95": seed["effects"]["random_minus_vpd_target_nll"]["ci95"]}
            for seed in scored["seed_results"]
        ]
        output["aggregate_effect"] = {"effect": scored["primary_mean"], "ci95": scored["primary_ci95"]}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovered", type=Path)
    p.add_argument("--pilot", type=Path)
    p.add_argument("--scored", type=Path)
    p.add_argument("--failure", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    receipts = {}
    def read(path):
        if path is None:
            return None
        raw = path.read_bytes()
        receipts[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    result = summarize(read(args.recovered), read(args.pilot), read(args.scored), [read(path) for path in args.failure])
    result["source_receipts_sha256"] = receipts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "comparison_completed": result["comparison_completed"],
                      "hosted_run_observed": result["hosted_run_observed"]}))


if __name__ == "__main__":
    main()
