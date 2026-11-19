"""AGP-02 original-native-trace auditor and descriptive grader-fiber spectra.

This module never trains or edits a model. It independently recomputes paired
hidden-state differences from a read-only archived source. Retrospective only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("reward_target", "host_trait", "mixed", "trait_cf")
SEEDS = (103, 211, 307)
DEPTHS = (2, 4, 6, 8, 12)
WIDTH = 16
ORIGINAL_ROOTS = 128
CASES_PER_ROOT = 24
ORIGINAL_TRACE_SHA256 = "a16a9df8186317c6a173b363757df78cfab975fc45c548326de1180864641409"
COLUMNS = frozenset((
    "init_seed", "arm", "root_id", "case_id", "grader_hint", "depth",
    "state", "checkpoint_sha256",
))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("AGP02_TRACE_AUDIT_FAIL: " + message)


def canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def as_rows(path: Path):
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def analyze(rows, strict_original: bool = False) -> dict:
    seen: dict[tuple[int, str, str, str, int, int], np.ndarray] = {}
    models: dict[tuple[int, str], str] = {}
    roots = set()
    cases_by_root: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        check(isinstance(row, dict) and set(row) == COLUMNS,
              "unrecognized hidden-state record")
        seed, arm, rid, cid, hint, depth = (
            row["init_seed"], row["arm"], row["root_id"], row["case_id"],
            row["grader_hint"], row["depth"],
        )
        check(type(seed) is int and seed in SEEDS and arm in ARMS,
              "unexpected initializer or objective arm")
        check(isinstance(rid, str) and rid and
              isinstance(cid, str) and cid and
              type(hint) is int and hint in (0, 1) and
              type(depth) is int and depth in DEPTHS,
              "unexpected pair/depth identity")
        ckpt = row["checkpoint_sha256"]
        check(isinstance(ckpt, str) and len(ckpt) == 64 and
              all(x in "0123456789abcdef" for x in ckpt), "checkpoint digest shape")
        model_id = (seed, arm)
        check(model_id not in models or models[model_id] == ckpt,
              "checkpoint changed within source")
        models[model_id] = ckpt
        state = np.asarray(row["state"], dtype=np.float64)
        check(state.shape == (WIDTH,) and np.isfinite(state).all(),
              "native state dimension/finiteness")
        key = (seed, arm, rid, cid, hint, depth)
        check(key not in seen, "repeated hidden-state row")
        seen[key] = state
        roots.add(rid)
        cases_by_root[rid].add(cid)
    check(bool(seen), "empty trace source")
    expected = len(models) * len(DEPTHS) * sum(len(c) for c in cases_by_root.values()) * 2
    check(len(seen) == expected, "paired model and depth coverage")
    check(set(models) == {(s, a) for s in SEEDS for a in ARMS},
          "some registered model checkpoints absent")
    if strict_original:
        check(len(roots) == ORIGINAL_ROOTS, "original root count")
        check(all(len(c) == CASES_PER_ROOT for c in cases_by_root.values()),
              "original case count per root")
        check(len(seen) == ORIGINAL_ROOTS * CASES_PER_ROOT * 2 *
              len(ARMS) * len(SEEDS) * len(DEPTHS),
              "original total depth denominator")
    out = {
        "verdict": "PASS_RETROSPECTIVE_PAIRED_NATIVE_STATE_AUDIT",
        "claim": "descriptive_input_response_only",
        "strict_original": strict_original,
        "native_state_rows": len(seen),
        "unique_host_roots": len(roots),
        "cases_per_root": {k: len(v) for k, v in sorted(cases_by_root.items())},
        "models": {},
        "averages_over_fixed_initializers": {},
    }
    for arm in ARMS:
        per_depth = defaultdict(list)
        for seed in SEEDS:
            metrics = {}
            for depth in DEPTHS:
                differences = []
                root_norms = defaultdict(list)
                for rid in sorted(roots):
                    for cid in sorted(cases_by_root[rid]):
                        low = seen.get((seed, arm, rid, cid, 0, depth))
                        high = seen.get((seed, arm, rid, cid, 1, depth))
                        check(low is not None and high is not None,
                              "missing grader-counterfactual hidden pair")
                        diff = high - low
                        if depth <= 6:
                            check(np.max(np.abs(diff)) <= 1e-6,
                                  "untrusted grader hint changed pre-injection state")
                        differences.append(diff)
                        root_norms[rid].append(float(np.linalg.norm(diff)))
                D = np.stack(differences)
                mu = D.mean(axis=0)
                centered = D - mu
                energy = float(np.square(D).sum())
                cen_energy = float(np.square(centered).sum())
                gram = D.T @ D
                centered_gram = centered.T @ centered
                norm_mean = float(np.linalg.norm(D, axis=1).mean())
                top_uncentered = float(np.linalg.eigvalsh(gram)[-1] / energy) if energy else 0.0
                top_centered = (
                    float(np.linalg.eigvalsh(centered_gram)[-1] / cen_energy)
                    if cen_energy else 0.0
                )
                mean_energy = float(np.square(mu).sum() *
                                    len(D) / energy) if energy else 0.0
                metrics[str(depth)] = {
                    "pairs": len(D),
                    "mean_l2": norm_mean,
                    "top1_uncentered_energy_fraction": top_uncentered,
                    "top1_centered_energy_fraction": top_centered,
                    "shared_mean_energy_fraction": mean_energy,
                    "rootwise_mean_l2": {
                        rid: sum(values) / len(values)
                        for rid, values in sorted(root_norms.items())
                    },
                }
                per_depth[depth].append(metrics[str(depth)])
            out["models"][f"{seed}:{arm}"] = {
                "checkpoint_sha256": models[(seed, arm)],
                "depths": metrics,
            }
        out["averages_over_fixed_initializers"][arm] = {
            str(depth): {
                metric: sum(m[metric] for m in per_depth[depth]) / len(SEEDS)
                for metric in (
                    "mean_l2", "top1_uncentered_energy_fraction",
                    "top1_centered_energy_fraction", "shared_mean_energy_fraction"
                )
            } for depth in DEPTHS
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strict-original", action="store_true",
                        help="enforce fixed artifact SHA and complete 128-root source")
    args = parser.parse_args()
    digest = sha256(args.traces.read_bytes()).hexdigest()
    if args.strict_original:
        check(digest == ORIGINAL_TRACE_SHA256, "original trace ZIP member hash mismatch")
    report = analyze(as_rows(args.traces), strict_original=args.strict_original)
    report["input_sha256"] = digest
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(report))
    print(json.dumps({
        "verdict": report["verdict"],
        "original_sha256_verified": args.strict_original,
        "native_state_rows": report["native_state_rows"],
        "host_trait_mean_l2_depth_12": report["averages_over_fixed_initializers"][
            "host_trait"]["12"]["mean_l2"],
        "trait_cf_mean_l2_depth_12": report["averages_over_fixed_initializers"][
            "trait_cf"]["12"]["mean_l2"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
