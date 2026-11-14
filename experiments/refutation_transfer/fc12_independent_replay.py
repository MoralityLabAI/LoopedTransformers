"""FC-12 post-score independent Lean compiler replay of every scored action prefix.

Read-only on the registered experiment. This script parses original MathCan
declarations independently of the model's scratch renderer, regenerates each
recorded finished/incomplete proof, and performs a NEW Lean4.35 compile.
The frozen CI artifact is downloaded by the separate replay workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

MATHCAN_SHA = "c84e2025334e9a967c1297425a490f98d5b3d189"
TEST_SHA = "b9b2f48fc4cc3d921ddac5d90ca7c12995ff7f9b"
TRAIN_GOALS_SHA = "403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0"
TRAIN_EFFECTS_SHA = "bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec"
TEST_GOALS_SHA = "7c8f5765cfec8f74294a60e5da6edf16856530d07e105e58ae0334d017afdeaf"
EVAL_RUN = 43548536370
SELECTED = {
    "MathCan/Hodge.lean": (
        "endpoint_append", "pathSum_append", "closed_path_gauge",
        "edge_residual_gauge",
    ),
    "MathCan/TOEDressedMatter.lean": (
        "spectral_multiplicity", "flrw_temporal_tracefree_iff",
        "metric_vacuum_shift", "gr_lapse_limit",
    ),
    "MathCan/TOELorentzBivector.lean": (
        "diagonal_matching_iff",
        "negative_discriminant_of_positive_mixed_square",
        "matched_real_spectrum_forces_zero_mixing",
        "nonzero_raised_determinant",
    ),
    "MathCan/TOETensorCompletion.lean": (
        "positivePart_nonnegative", "mismatched_inverse_response",
        "zero_gap_of_equal_products", "homogeneous_scalar_wedge_zero",
    ),
}
EXPECTED = {(f, name) for f, names in SELECTED.items() for name in names}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jsonlines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()
            if line.strip()]


def valid_action(script: str) -> bool:
    if script in {"omega", "rfl", "simp_all", "assumption", "intro", "constructor"}:
        return True
    n = r"[A-Za-z_][A-Za-z0-9_']*"
    patterns = (
        rf"exact {n}", rf"exact {n} {n} {n}",
        rf"exact \({n} {n} {n}\)\.symm", rf"apply {n}",
        rf"simpa using {n}", rf"rw \[{n}\]", rf"rw \[← {n}\]",
        rf"rcases {n} with ⟨fc12_left, fc12_right⟩",
        rf"induction {n}",
    )
    return any(re.fullmatch(p, script) is not None for p in patterns)


def reconstruct_original(original: str, theorem: str, actions: list[str]) -> str:
    """Independent original-theorem replacement, without importing scratch.py."""
    if len(actions) > 3 or any(not valid_action(s) for s in actions):
        raise AssertionError("unregistered, injected, or over-budget recorded action")
    lines = original.splitlines()
    indices = [i for i, line in enumerate(lines)
               if re.match(r"^theorem\s+" + re.escape(theorem) + r"\b", line)]
    if len(indices) != 1:
        raise AssertionError(f"missing or duplicate theorem: {theorem}")
    begin = next((i for i in range(indices[0], min(len(lines), indices[0] + 40))
                  if ":= by" in lines[i]), None)
    if begin is None:
        raise AssertionError(f"expected by-tactic theorem: {theorem}")
    end = next((i for i in range(begin + 1, len(lines))
                if re.match(r"^(?:theorem|lemma|def|namespace|end|#print)\b",
                            lines[i])
                or lines[i].startswith(("/--", "/-!"))), len(lines))
    ending = [line for line in lines[end:]
              if re.fullmatch(r"end\s+[A-Za-z_0-9.]+", line.strip())]
    header = lines[begin].split(":= by", 1)[0] + ":= by"
    # This reproduces the exact declaration context but does not use FC-12's
    # original make_source or any reference tactic from the declaration.
    return "import Lean\n" + "\n".join(
        lines[:begin] + [header] + ["  " + s for s in actions]
        + ([ending[-1]] if ending else [])
    ) + "\n"


def compile_lean(source: str, lean: str, timeout_s: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="fc12_independent_") as td:
        target = Path(td) / "Independent.lean"
        target.write_text(source, encoding="utf8")
        start = time.perf_counter()
        try:
            proc = subprocess.run([lean, str(target)], capture_output=True,
                                  text=True, timeout=timeout_s)
            log = proc.stdout + proc.stderr
            code = proc.returncode
            timeout = False
        except subprocess.TimeoutExpired as exc:
            a = exc.stdout or b""
            b = exc.stderr or b""
            if isinstance(a, bytes):
                a = a.decode("utf8", "replace")
            if isinstance(b, bytes):
                b = b.decode("utf8", "replace")
            log = a + b
            code = 124
            timeout = True
    return {
        "compiled_sorry_free": code == 0 and "sorryAx" not in log
        and "error:" not in log and not timeout,
        "returncode": code,
        "timed_out": timeout,
        "wall_ms": round((time.perf_counter() - start) * 1000, 4),
        "source_sha256": sha(source.encode("utf8")),
        "error_excerpt": log[-350:] if code != 0 else "",
    }


def assert_frozen_report(report: dict, artifact: Path) -> None:
    assert report["source_upstream_sha"] == MATHCAN_SHA
    assert report["test_source_sha"] == TEST_SHA
    assert report["train_actual_state_sha256"] == TRAIN_GOALS_SHA
    assert report["train_probe_effect_sha256"] == TRAIN_EFFECTS_SHA
    assert report["test_actual_state_sha256"] == TEST_GOALS_SHA
    assert report["original_source_oracle_proof_count"] == 16
    assert report["heldout_theorem_count"] == 16
    assert report["max_model_selected_goal_transitions"] == 3
    assert report["max_tactic_probes_per_theorem"] == 12
    assert report["registered_learning_criterion"] is False
    assert report["registered_compute_criterion"] is False
    assert {x["seed"] for x in report["results"]} == {17, 29}
    for result in report["results"]:
        seed = result["seed"]
        for kind, k in (("model", "model_output_sha256"),
                        ("native", "native_output_sha256")):
            p = artifact / f"fc12_seed{seed}_{kind}.jsonl"
            assert sha(p.read_bytes()) == result[k], f"changed {p}"


def replay(artifact: Path, source: Path, output: Path, lean: str,
           timeout_s: int) -> dict:
    if (source / "lean-toolchain").read_text().strip() != "leanprover/lean4:v4.35.0":
        raise AssertionError("wrong original MathCan Lean toolchain")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != MATHCAN_SHA:
        raise AssertionError("original MathCan source SHA changed")
    report = json.loads((artifact / "FC12_report.json").read_text())
    assert_frozen_report(report, artifact)
    originals = {p: (source / p).read_text(encoding="utf8") for p in SELECTED}
    outputs = []
    for item in sorted(report["results"], key=lambda x: x["seed"]):
        seed = item["seed"]
        by_kind = {}
        for kind in ("model", "native"):
            rows = jsonlines(artifact / f"fc12_seed{seed}_{kind}.jsonl")
            by_key = {(r["source_file"], r["theorem"]): r for r in rows}
            if len(rows) != 16 or set(by_key) != EXPECTED:
                raise AssertionError(f"modified or duplicated cases: {seed}/{kind}")
            by_kind[kind] = by_key
        for file, names in SELECTED.items():
            for name in names:
                pair = (file, name)
                m = by_kind["model"][pair]
                n = by_kind["native"][pair]
                if abs(float(n["allowed_ms"]) - float(m["elapsed_ms"])) > 0.001:
                    raise AssertionError("symbolic baseline used a different time allowance")
                for kind, row in (("model", m), ("native", n)):
                    actions = row["chosen_actions"]
                    if not isinstance(actions, list):
                        raise AssertionError("invalid selected script container")
                    if row["candidate_probe_count"] > 12:
                        raise AssertionError("probe budget exceeded")
                    if row["sorry_free_release"] != row["accepted"]:
                        raise AssertionError("untrusted scratch accepted as full proof")
                    candidate = reconstruct_original(originals[file], name, actions)
                    proof = compile_lean(candidate, lean, timeout_s)
                    if proof["compiled_sorry_free"] != row["accepted"]:
                        raise AssertionError(
                            f"replay discrepancy {seed}/{kind}/{file}/{name}: "
                            f"recorded={row['accepted']}, replay={proof}"
                        )
                    # Independently regenerated positive source must match the
                    # original compiler-checked exact candidate source SHA.
                    if row["accepted"]:
                        approved = [x for x in row["tactic_attempts"]
                                    if x["exact_compiler_accepted"]]
                        if not approved or approved[-1]["exact_source_sha256"] != proof["source_sha256"]:
                            raise AssertionError("positive full-source SHA differs")
                    outputs.append({
                        "seed": seed, "controller": kind, "file": file,
                        "theorem": name, "recorded_certified": row["accepted"],
                        "replayed_certified": proof["compiled_sorry_free"],
                        "actions": actions, "compiler": proof,
                        "model_allowance_ms": round(float(m["elapsed_ms"]), 6),
                        "native_overrun_ms": float(n["overrun_ms"]),
                    })
        for kind in ("model", "native"):
            group = [r for r in outputs if r["seed"] == seed and r["controller"] == kind]
            count = sum(r["replayed_certified"] for r in group)
            if count != item[f"{'model' if kind == 'model' else 'native'}_proofs"]:
                raise AssertionError("aggregate proof count differs from original audit")
    output.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in outputs)
    (output / "fc12_independent_replays.jsonl").write_text(raw, encoding="utf8")
    receipt = {
        "experiment": "FC12_postscore_independent_Lean435_compiler_replay",
        "frozen_evaluation_run": EVAL_RUN,
        "source_commit": MATHCAN_SHA,
        "train_effect_sha256": TRAIN_EFFECTS_SHA,
        "heldout_source_commit": TEST_SHA,
        "replayed_cases": len(outputs),
        "verdict_disagreements": 0,
        "all_recompiled_without_training_scratch_sorry": True,
        "replayed_jsonl_sha256": sha(raw.encode("utf8")),
        "per_seed": [
            {"seed": seed,
             "model_certified": sum(r["replayed_certified"] for r in outputs if r["seed"] == seed and r["controller"] == "model"),
             "native_certified": sum(r["replayed_certified"] for r in outputs if r["seed"] == seed and r["controller"] == "native")}
            for seed in (17, 29)
        ],
        "original_report_sha256": sha((artifact / "FC12_report.json").read_bytes()),
        "scope": "new Lean recompilation of frozen recorded prefixes; no new training, search, or candidate selection",
    }
    (output / "fc12_independent_replay_audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print("FC12_INDEPENDENT_REPLAY", json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mathcan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    args = parser.parse_args()
    replay(args.artifact, args.mathcan, args.out, args.lean, args.timeout_seconds)


if __name__ == "__main__":
    main()
