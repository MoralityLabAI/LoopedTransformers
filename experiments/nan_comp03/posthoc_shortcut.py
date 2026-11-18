"""Post-hoc, source-only NAN-COMP-03 audit. Uses frozen DEV receipts only.

This script reads existing public/host labels and saved answer-token class
log-probabilities; it never runs a model, grades the sealed 168-case ARG-03
test, modifies a frozen outcome, or reinterprets a boosted score as a
generated response. The R score bias is an explicitly exposed-data sensitivity
diagnostic, not a preregistered calibration or competence gate success.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKENDS = ("bf16", "nf4")
ARMS = ("curriculum", "balanced", "erased")
OPS = ("I", "N", "S", "R")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def first_two_classify(a: list[int], b: list[int]) -> str:
    """Exact coefficient shortcut valid on the disclosed COMP-03 DEV domain."""
    if len(a) < 3 or len(b) < 3 or a[0] != 1 or b[0] != 1 or a[1] not in (-1, 1):
        raise ValueError("outside COMP-03 shortcut domain")
    a1, a2, b1, b2 = a[1], a[2], b[1], b[2]
    if b1 == a1 and b2 == a2:
        return "I"
    if b1 == 0 and b2 == a1:
        return "S"
    if b1 == -a1:
        if b2 == a2:
            return "N"
        if b2 == a1 * a1 - a2:
            return "R"
    raise ValueError("first two coefficients do not identify a unique operator")


def predicted(scores: dict[str, float], bias_R: float = 0.0) -> str:
    if set(scores) != set(OPS):
        raise ValueError("need exactly four saved operator-class log-probabilities")
    return max(OPS, key=lambda op: (float(scores[op]) + (bias_R if op == "R" else 0.0),
                                    -OPS.index(op)))


def r_margin(scores: dict[str, float]) -> float:
    return scores["R"] - max(scores[x] for x in OPS if x != "R")


def load_dev() -> tuple[list[dict], dict]:
    public_path = ROOT / "fixtures" / "public.jsonl"
    host_path = ROOT / "fixtures" / "host_labels.jsonl"
    public = read_jsonl(public_path)
    host = {row["case_id"]: row for row in read_jsonl(host_path)}
    dev = [row for row in public if row["split"] == "development"]
    if len(dev) != 16 or len({row["root"] for row in dev}) != 8:
        raise AssertionError("expected frozen 16 DEV worlds / 8 generator roots")
    if {row["case_id"] for row in dev} - set(host):
        raise AssertionError("missing host labels")
    return dev, host


def load_scores(public: list[dict], host: dict) -> dict[str, list[dict]]:
    result = {}
    for backend in BACKENDS:
        rows = []
        for row in public:
            cid = row["case_id"]
            role = cid.rsplit(":", 1)[-1]
            path = ROOT / "receipts" / backend / ("nan-comp03_" + str(row["root"]) + "_" + role + ".json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["case_id"] != cid or payload["split"] != "development":
                raise AssertionError("model receipt identity or split mismatch")
            entry = {"root": row["root"], "case_id": cid,
                     "truth": host[cid]["operator"], "scores": {}, "generated": {}}
            for arm in ARMS:
                rec = payload["records"][arm]
                if rec["steps"][0]["decoded_prefix"] != "**" or len(rec["steps"]) != 3:
                    raise AssertionError("frozen bounded three-token wrapper changed")
                entry["scores"][arm] = rec["steps"][1]["score"]["class_logprobs"]
                entry["generated"][arm] = rec["parser"]["operator"]
            rows.append(entry)
        result[backend] = rows
    return result


def score(rows: list[dict], bias: float, arm: str) -> dict:
    root_correct = {}
    correct = r_correct = 0
    generated_agree = 0
    for row in rows:
        op = predicted(row["scores"][arm], bias)
        good = op == row["truth"]
        correct += good
        r_correct += good and row["truth"] == "R"
        root_correct.setdefault(row["root"], []).append(good)
        if bias == 0 and op == row["generated"][arm]:
            generated_agree += 1
    both = sum(len(x) == 2 and all(x) for x in root_correct.values())
    return {"correct": correct, "both_pairs": both, "reciprocal": r_correct,
            "zero_bias_agrees_with_generated": generated_agree if bias == 0 else None}


def biased_scan(rows: list[dict]) -> dict:
    grid = []
    for integer in range(-100, 301):
        bias = integer / 100
        current = score(rows, bias, "curriculum")
        erased = score(rows, bias, "erased")
        gain = current["correct"] - erased["correct"]
        entry = {"R_class_score_bias": bias, "world_correct": current["correct"],
                 "both_pairs": current["both_pairs"], "reciprocal": current["reciprocal"],
                 "erased_correct": erased["correct"], "source_use_gain": gain,
                 "passes_numerical_thresholds": (
                     current["correct"] >= 12 and current["both_pairs"] >= 6
                     and current["reciprocal"] >= 3 and gain >= 4)}
        grid.append(entry)
    selected = [x for x in grid if x["passes_numerical_thresholds"]]
    return {
        "scan_grid": "[-1,3] in 0.01 increments, all post-hoc on exposed DEV",
        "original": score(rows, 0.0, "curriculum"),
        "at_R_class_score_bias_1p25": next(x for x in grid if x["R_class_score_bias"] == 1.25),
        "at_R_class_score_bias_1p50": next(x for x in grid if x["R_class_score_bias"] == 1.5),
        "passing_bias_count": len(selected),
        "passing_bias_min": selected[0]["R_class_score_bias"] if selected else None,
        "passing_bias_max": selected[-1]["R_class_score_bias"] if selected else None,
        "retrospective_only": True,
        "free_generation_under_bias_executed": False,
    }


def margin_pairs(rows: list[dict]) -> dict:
    by_root = {}
    for row in rows:
        by_root.setdefault(row["root"], []).append(row)
    chosen = []
    for root, pair in sorted(by_root.items()):
        if len(pair) != 2 or not any(x["truth"] == "R" for x in pair):
            continue
        recip = next(x for x in pair if x["truth"] != "R")
        donor = next(x for x in pair if x["truth"] == "R")
        entry = {"root": root}
        for arm in ARMS:
            entry[arm + "_R_margin_change"] = (
                r_margin(donor["scores"][arm]) - r_margin(recip["scores"][arm]))
        chosen.append(entry)
    return {
        "n_R_counterfactual_pairs": len(chosen),
        "curriculum_mean_R_margin_change": (
            sum(x["curriculum_R_margin_change"] for x in chosen) / len(chosen)),
        "curriculum_all_positive": all(x["curriculum_R_margin_change"] > 0 for x in chosen),
        "erased_max_abs_pair_change": max(abs(x["erased_R_margin_change"]) for x in chosen),
        "pairs": chosen,
    }


def audit() -> dict:
    public, host = load_dev()
    solutions = []
    for row in public:
        truth = host[row["case_id"]]["operator"]
        predicted_labels = [first_two_classify(a, b) for a, b
                            in zip(row["inputs"], row["outputs"], strict=True)]
        if predicted_labels != [truth, truth]:
            raise AssertionError("shortcut fails independently verified development source")
        solutions.append({"case_id": row["case_id"], "two_coefficient_prediction": truth})
    scored = load_scores(public, host)
    result = {
        "schema": "nan_comp03_posthoc_shortcut_and_R_bias_v1",
        "evidence_class": "post-hoc exposed development sensitivity; no model inference",
        "sealed_168_case_test_accessed": False,
        "original_COMP03_gates_unchanged": True,
        "source_shortcut": {"development_worlds": len(public),
                            "correct_with_b1_b2": len(solutions),
                            "a0_b0_equal_one": True, "a1_nonzero_unit": True,
                            "rows": solutions},
        "backends": {},
        "source_sha256": {
            path: sha256((ROOT / path).read_bytes()).hexdigest()
            for path in ("fixtures/public.jsonl", "fixtures/host_labels.jsonl")},
    }
    for backend in BACKENDS:
        rows = scored[backend]
        original = score(rows, 0, "curriculum")
        if original["zero_bias_agrees_with_generated"] != 16:
            raise AssertionError("saved operator-class scores do not match free second-token selection")
        result["backends"][backend] = {
            "bias_scan": biased_scan(rows),
            "reciprocal_pair_R_margin": margin_pairs(rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.out:
        if any(piece.lower() in {"test", "heldout", "test_full168"} for piece in args.out.parts):
            raise ValueError("refuse sealed test output paths")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "shortcut": report["source_shortcut"]["correct_with_b1_b2"],
        "BF16": report["backends"]["bf16"]["bias_scan"]["at_R_class_score_bias_1p25"],
        "NF4": report["backends"]["nf4"]["bias_scan"]["at_R_class_score_bias_1p25"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
