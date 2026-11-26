"""MOLT-08: conditional nonlinear sign-stability for observable sketches."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.molt_goblin_v1.observability_gym import (
    analytic, correct, cut_states, observe, sample, sha_state, train,
)
from experiments.molt_goblin_v1.observer_error import (
    PAD, directional_interval, global_curvature, spectral_segment_bound,
    stable_signs, stable_symmetric,
)
from lsa.observable_memory import Sketch

RANKS = (1, 2, 3, 4, 5)
DOSES = (0.25, 0.5, 1.0)
SEGMENTS = (1, 4, 16)
METHODS = ("linear_only", "global_curvature", *(f"interval{s}_{kind}"
           for s in SEGMENTS for kind in ("scalar", "signed")))


def same_signs(a, b):
    return bool((torch.sign(a) == torch.sign(b)).all())


@torch.no_grad()
def evaluate_world(model, x, seed, index):
    position = len(x) // 2 - 2
    clean, corrupt = cut_states(model, x, position)
    _, target, cone, _ = observe(model, x, clean, position)
    o = analytic(model, x, corrupt, position)
    sketch = Sketch.build(o, max(RANKS))
    curvature = global_curvature(model, x, position)
    rows, decisions = [], []
    root = f"molt08-{len(x)}-{index}"
    d = clean[position] - corrupt[position]
    for dose in DOSES:
        delta = dose * d
        reference = observe(model, x, corrupt, position, delta)[0][cone]
        full_correct = bool(correct(reference, target[cone]).all())
        selected = {}
        work = {m: 0.0 for m in METHODS}
        for k in RANKS:
            start = time.perf_counter()
            projected = sketch.project(delta, k)
            residual = delta - projected
            approx = observe(model, x, corrupt, position, projected)[0][cone]
            shared_seconds = time.perf_counter() - start
            actual = reference - approx
            linear_bound = sketch.tail(k) * float(residual.norm())
            nonlinear_bound = spectral_segment_bound(
                sketch.tail(k), float(residual.norm()),
                max(float(delta.norm()), float(projected.norm())), curvature,
            )
            if float(actual.norm()) > nonlinear_bound + PAD * (1 + nonlinear_bound):
                raise RuntimeError("global curvature bound violated")
            accepts = {
                "linear_only": stable_symmetric(approx, linear_bound),
                "global_curvature": stable_symmetric(approx, nonlinear_bound),
            }
            costs = {m: shared_seconds for m in METHODS}
            intervals = {}
            for s in SEGMENTS:
                start = time.perf_counter()
                lower, upper = directional_interval(model, x, corrupt, position, projected, delta, s)
                if not bool(((actual >= lower) & (actual <= upper)).all()):
                    raise RuntimeError("directional interval failed to enclose actual change")
                bound = float(torch.maximum(lower.abs(), upper.abs()).max())
                accepts[f"interval{s}_scalar"] = stable_symmetric(approx, bound)
                accepts[f"interval{s}_signed"] = stable_signs(approx, lower, upper)
                elapsed = time.perf_counter() - start
                for kind in ("scalar", "signed"):
                    costs[f"interval{s}_{kind}"] += elapsed
                intervals[str(s)] = {"lower": lower.tolist(), "upper": upper.tolist(),
                                     "symmetric_bound": bound,
                                     "mean_width": float((upper - lower).mean())}
            parity = same_signs(reference, approx)
            for method, accepted in accepts.items():
                if method != "linear_only" and accepted and not parity:
                    raise RuntimeError("conditional sign gate admitted a changed sign")
                if method not in selected:
                    work[method] += costs[method]
                    if accepted:
                        metadata = 16 * len(approx) if method.endswith("_signed") else 8
                        selected[method] = {
                            "rank": k, "fallback": False, "sign_parity": parity,
                            "task_correct": bool(correct(approx, target[cone]).all()),
                            "record_bytes": 8 * k + metadata,
                            "candidate_work_seconds": work[method],
                        }
            rows.append({
                "root": root, "length": len(x), "seed": seed, "dose": dose, "rank": k,
                "reference": reference.tolist(), "approx": approx.tolist(),
                "linear_bound": linear_bound, "curvature_bound": nonlinear_bound,
                "global_curvature": curvature, "actual_error_l2": float(actual.norm()),
                "actual_error_max": float(actual.abs().max()), "sign_parity": parity,
                "full_task_correct": full_correct,
                "approx_task_correct": bool(correct(approx, target[cone]).all()),
                "accepts": accepts, "intervals": intervals, "candidate_seconds": costs,
            })
        for method in METHODS:
            # A rejected projection cannot be recovered from discarded
            # coefficients. Full fallback is selected while the donor is live.
            decision = selected.get(method, {
                "rank": len(delta), "fallback": True, "sign_parity": True,
                "task_correct": full_correct, "record_bytes": 8 * len(delta),
                "candidate_work_seconds": work[method],
            })
            decisions.append({"root": root, "length": len(x), "seed": seed, "dose": dose,
                              "method": method, "full_task_correct": full_correct, **decision})
    return rows, decisions


def summarize(rows, decisions):
    report = []
    for dose in DOSES:
        for m in METHODS:
            selected = [r for r in decisions if r["dose"] == dose and r["method"] == m]
            accepted = [r for r in selected if not r["fallback"]]
            candidates = [r for r in rows if r["dose"] == dose]
            accepted_candidates = [r for r in candidates if r["accepts"][m]]
            report.append({
                "dose": dose, "method": m, "rows": len(selected),
                "accepted": len(accepted), "fallback": len(selected) - len(accepted),
                "false_accepts": sum(not r["sign_parity"] for r in accepted),
                "candidate_false_accepts": sum(not r["sign_parity"] for r in accepted_candidates),
                "task_correct": sum(r["task_correct"] for r in selected),
                "full_task_correct": sum(r["full_task_correct"] for r in selected),
                "rank_histogram": {str(k): sum(r["rank"] == k for r in selected)
                                   for k in (*RANKS, 16)},
                "mean_record_bytes": sum(r["record_bytes"] for r in selected) / len(selected),
                "candidate_work_seconds": sum(r["candidate_work_seconds"] for r in selected),
            })
    return report


def run(steps, roots, seeds):
    if not 1 <= steps <= 1500 or not 1 <= roots <= 32 or not 1 <= len(seeds) <= 8:
        raise ValueError("development caps")
    if len(set(seeds)) != len(seeds) or not all(0 <= s <= 10000 for s in seeds):
        raise ValueError("invalid seeds")
    torch.set_num_threads(1)
    rows, decisions, fits = [], [], []
    for seed in seeds:
        model = train(seed, steps)
        checkpoint = sha_state(model)
        fits.append({"seed": seed, "checkpoint": checkpoint, "updates": steps})
        for length in (8, 12):
            for i, x in enumerate(sample(1042000 + length, roots, length, torch.float64)):
                r, d = evaluate_world(model, x, seed, i)
                rows.extend(r)
                decisions.extend(d)
        if sha_state(model) != checkpoint:
            raise RuntimeError("frozen checkpoint changed")
    base = Path(__file__).resolve().parents[2]
    files = ("experiments/molt_goblin_v1/observer_error.py",
             "experiments/molt_goblin_v1/mms_error_study.py",
             "experiments/molt_goblin_v1/observability_gym.py",
             "lsa/observable_memory.py")
    sources = {p: hashlib.sha256((base / p).read_bytes()).hexdigest() for p in files}
    return {
        "schema": "molt08-observer-error-v1", "status": "development_only",
        "model_seeds": list(seeds), "world_roots": 2 * roots, "fits": fits,
        "sources": sources, "torch": torch.__version__, "summary": summarize(rows, decisions),
        "rows": rows, "decisions": decisions,
        "conditional_real_arithmetic_only": True, "directed_rounding_verified": False,
        "formal_certificate_authorized": False, "confirmation_ready": False,
        "promotion_authorized": False, "native_bpd": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[137, 139, 149, 151])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be fresh")
    report = run(args.steps, args.roots, tuple(args.seeds))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": report["status"], "world_roots": report["world_roots"],
                      "rows": len(report["rows"]), "decisions": len(report["decisions"]),
                      "full_dose": [r for r in report["summary"] if r["dose"] == 1.0]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
