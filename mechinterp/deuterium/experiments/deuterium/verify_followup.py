"""Verify saved follow-up arithmetic without training or running a model.

Reads the registered protocol, saved per-row logits, learned tensors and edits.
Replays the already specified bootstrap seed solely to verify reported intervals.
No task examples, model predictions or new experimental seeds are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t as student_t
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def verify(results: Path, protocol_path: Path, out: Path) -> dict:
    torch.set_num_threads(2)
    protocol = read_json(protocol_path)
    cfg = protocol["new_toy_lane"]
    summary_path = results / "steering_summary.json"
    summary = read_json(summary_path)
    checks = []

    def check(name: str, passed: bool, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    def close(name: str, actual: float, reported: float, atol: float = 1e-8):
        check(name, math.isfinite(actual) and abs(actual - reported) <= atol,
              recomputed=actual, reported=reported, absolute_error=abs(actual - reported), tolerance=atol)

    check("protocol_sha256", sha256(protocol_path) == summary["registration"]["protocol_sha256"])
    check("three_registered_unique_seeds", len(cfg["seeds"]) == len(set(cfg["seeds"])) == 3)
    check("summary_seeds_match_protocol", [r["seed"] for r in summary["seed_results"]] == cfg["seeds"])
    repo = Path(__file__).resolve().parents[2]
    commit = summary["registration"]["registration_commit"]
    frozen = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:experiments/deuterium/protocol.json"])
    check("registered_protocol_bytes", hashlib.sha256(frozen).hexdigest() == sha256(protocol_path))
    frozen_source = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:experiments/deuterium/steering.py"])
    check("steering_source_unchanged_after_registration", hashlib.sha256(frozen_source).hexdigest() == sha256(repo / "experiments/deuterium/steering.py"))
    seed_receipts = []
    primary = []
    collateral_deltas = []
    all_input_hashes = set()
    for seed, reported in zip(cfg["seeds"], summary["seed_results"]):
        base = results / f"seed_{seed}"
        check(f"{seed}:standalone_and_aggregate_seed_results_identical", read_json(base / "seed_result.json") == reported)
        rows_path = base / "heldout_rows.jsonl"
        check(f"{seed}:row_file_sha256", sha256(rows_path) == reported["heldout_rows_sha256"])
        check(f"{seed}:selected_edits_sha256", sha256(base / "selected_edits.pt") == reported["selected_edits_sha256"])
        learned_path = base / "vpd" / f"model_{cfg['decomposition']['steps']}.pth"
        check(f"{seed}:decomposition_sha256", sha256(learned_path) == reported["decomposition"]["checkpoint_sha256"])
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        check(f"{seed}:heldout_rows_count", len(rows) == cfg["data"]["heldout_rows"])
        check(f"{seed}:row_seed_and_index", all(r["seed"] == seed and r["example"] == i for i, r in enumerate(rows)))
        hashes = {r["input_sha256"] for r in rows}
        check(f"{seed}:input_hashes_unique_within_and_between_seeds", len(hashes) == len(rows) and not (hashes & all_input_hashes))
        all_input_hashes |= hashes
        labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        target = labels == 1
        collateral = ~target
        losses = {}
        metrics = {}
        logits_by_arm = {}
        for arm in cfg["intervention"]["arms"]:
            logits = torch.tensor([r["arms"][arm]["logits"] for r in rows], dtype=torch.float32)
            logits_by_arm[arm] = logits
            predictions = logits.argmax(-1)
            check(f"{seed}:{arm}:finite_logits", bool(torch.isfinite(logits).all()))
            check(f"{seed}:{arm}:predictions_equal_argmax", predictions.tolist() == [r["arms"][arm]["prediction"] for r in rows])
            loss = F.cross_entropy(logits, labels, reduction="none")
            saved = torch.tensor([r["arms"][arm]["nll"] for r in rows], dtype=torch.float32)
            error = float((loss - saved).abs().max())
            check(f"{seed}:{arm}:per_row_loss_replay", error <= 1e-7, maximum_absolute_error=error)
            losses[arm] = loss.numpy()
            metric = {"accuracy": float((predictions == labels).float().mean()),
                      "target_accuracy": float((predictions[target] == labels[target]).float().mean()),
                      "collateral_accuracy": float((predictions[collateral] == labels[collateral]).float().mean()),
                      "false_commit_rate_on_veto": float((predictions[target] == 0).float().mean()),
                      "target_nll": float(loss[target].mean()),
                      "counts": torch.bincount(labels, minlength=3).tolist()}
            metrics[arm] = metric
            check(f"{seed}:{arm}:class_counts", metric["counts"] == reported["metrics"][arm]["counts"])
            for name, value in metric.items():
                if name != "counts":
                    close(f"{seed}:{arm}:{name}", value, reported["metrics"][arm][name])
        check(f"{seed}:clean_sham_exact_logits", bool(torch.equal(logits_by_arm["clean"], logits_by_arm["sham"])))
        effects = {}
        for arm in ("random", "task_gradient", "clean"):
            values = (losses[arm] - losses["vpd"])[target.numpy()]
            effect_key = f"{arm}_minus_vpd_target_nll"
            effect = float(values.mean())
            effects[effect_key] = effect
            close(f"{seed}:{effect_key}", effect, reported["effects"][effect_key]["estimate"], 1e-10)
            rng = np.random.default_rng(seed + cfg["endpoints"]["bootstrap_seed"])
            boot = np.asarray([values[rng.integers(0, len(values), len(values))].mean()
                               for _ in range(cfg["endpoints"]["bootstrap_resamples"])], dtype=np.float64)
            interval = np.quantile(boot, [0.025, 0.975]).tolist()
            for i in range(2):
                close(f"{seed}:{effect_key}:ci95_{i}", interval[i], reported["effects"][effect_key]["ci95"][i], 1e-10)
        primary.append(effects["random_minus_vpd_target_nll"])
        collateral_delta = metrics["vpd"]["collateral_accuracy"] - metrics["clean"]["collateral_accuracy"]
        collateral_deltas.append(collateral_delta)
        close(f"{seed}:collateral_delta", collateral_delta, reported["collateral_accuracy_change"])

        state = torch.load(base / "base_model.pt", map_location="cpu", weights_only=True)
        learned = torch.load(learned_path, map_location="cpu", weights_only=True)
        deltas = torch.load(base / "selected_edits.pt", map_location="cpu", weights_only=True)
        check(f"{seed}:all_frozen_model_tensors_equal", all("target_model." + k in learned and torch.equal(v, learned["target_model." + k]) for k, v in state.items()))
        weight = state["head.weight"]
        expected_dose = float(weight.norm()) * cfg["intervention"]["frobenius_dose_fraction_of_base_weight"]
        edits = {}
        for arm, delta in deltas.items():
            intended_norm = float(delta.norm())
            applied_norm = float(((weight + delta) - weight).norm())
            rank = int(torch.linalg.matrix_rank(delta))
            expected_rank = 0 if arm == "sham" else 1
            dose = 0.0 if arm == "sham" else expected_dose
            check(f"{seed}:{arm}:edit_rank", rank == expected_rank == reported["edits"][arm]["rank"])
            close(f"{seed}:{arm}:intended_edit_norm", intended_norm, reported["edits"][arm]["frobenius_norm"])
            close(f"{seed}:{arm}:applied_edit_norm", applied_norm, reported["edits"][arm]["applied_frobenius_norm"])
            dose_error = 0.0 if arm == "sham" else abs(applied_norm - dose) / dose
            check(f"{seed}:{arm}:applied_dose_gate", dose_error <= cfg["gates"]["max_dose_relative_error"], relative_error=dose_error)
            edits[arm] = {"rank": rank, "intended_norm": intended_norm, "applied_norm": applied_norm,
                          "expected_dose": dose, "applied_relative_error": dose_error}
        selected = reported["selection"]["vpd"]
        u, v = learned["_components.head.U"], learned["_components.head.V"]
        index = selected["candidate_index"]
        candidate = torch.outer(u[index], v[:, index])
        expected_delta = selected["sign"] * candidate / candidate.norm() * expected_dose
        error = float((expected_delta - deltas["vpd"]).abs().max())
        check(f"{seed}:selected_vpd_edit_matches_saved_learned_component", error <= 1e-8, maximum_absolute_error=error)
        for arm in ("vpd", "random"):
            scores = reported["selection"][arm]["candidate_scores"]
            minimum = min((r["nll"], r["index"], r["sign"]) for r in scores)
            sel = reported["selection"][arm]
            check(f"{seed}:{arm}:calibration_selection_matches_logged_minimum",
                  minimum == (sel["calibration_target_nll"], sel["candidate_index"], sel["sign"]))
            check(f"{seed}:{arm}:candidate_sign_budget", len(scores) == cfg["intervention"][f"{'learned' if arm == 'vpd' else 'random'}_candidates"] * len(cfg["intervention"]["signs"]))
        seed_receipts.append({"seed": seed, "rows": len(rows), "target_rows": int(target.sum()),
                              "metrics": metrics, "effects": effects,
                              "collateral_accuracy_change": collateral_delta, "edits": edits,
                              "base_model_sha256": sha256(base / "base_model.pt")})

    values = np.asarray(primary, dtype=np.float64)
    mean = float(values.mean())
    df = len(values) - 1
    critical = float(student_t.ppf(0.975, df))
    half = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
    interval = [mean - half, mean + half]
    close("aggregate_primary_mean", mean, summary["primary_mean"], 1e-10)
    for i in range(2):
        close(f"aggregate_primary_ci95_{i}", interval[i], summary["primary_ci95"][i], 1e-10)
    collateral_pass = all(v >= -cfg["gates"]["max_collateral_accuracy_drop"] for v in collateral_deltas)
    positive = bool(all(v > 0 for v in primary) and interval[0] > 0 and collateral_pass)
    verdict = "POSITIVE" if positive else "NO_REGISTERED_ADVANTAGE"
    check("aggregate_verdict", verdict == summary["status"])
    check("collateral_gate", collateral_pass == summary["collateral_gate_passed"])
    result = {"schema_version": 1, "status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
              "scope": "Saved-data arithmetic, hash, edit-tensor and selection-log consistency. No model forward, training, or new experimental seed.",
              "registration_commit": commit, "summary_sha256": sha256(summary_path),
              "protocol_sha256": sha256(protocol_path), "verification_script_sha256": sha256(Path(__file__)),
              "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks),
              "rows_verified": sum(s["rows"] for s in seed_receipts),
              "primary_effects": primary, "primary_mean": mean, "primary_ci95": interval,
              "t_critical": critical, "degrees_of_freedom": df, "scientific_verdict": verdict,
              "seed_receipts": seed_receipts, "checks": checks,
              "limitations": ["The verifier checks consistency of saved logits; native forward fidelity remains supported by the separately logged experiment checks.",
                              "Task-gradient fitting and random-bank construction are not rerun; saved edit norms and logged calibration selection are verified.",
                              "This check provides no additional independent model or task replication."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.results, args.protocol, args.out)
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "checks_total", "rows_verified", "primary_mean", "primary_ci95", "scientific_verdict")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
