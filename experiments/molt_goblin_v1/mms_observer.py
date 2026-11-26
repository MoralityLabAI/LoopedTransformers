"""MOLT-07 development: observer-rank sketches and exact-bound map replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from experiments.molt_goblin_v1.observability_gym import (
    CUT, LOOPS, analytic, correct, cut_states, observe, sample, sha_state, train,
)
from lsa.observable_memory import MapBinding, MapCache, Sketch, tensor_digest

RANKS = (1, 2, 3, 4, 5)
DOSES = (0.25, 0.5, 1.0)
CAPS = (0, 4096, 16384, 65536)
MAX_RANK = max(RANKS)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ("lsa/observable_memory.py", "lsa/observability.py",
             "experiments/molt_goblin_v1/observability_gym.py",
             "experiments/molt_goblin_v1/mms_observer.py")
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def fixture(model, roots):
    worlds = []
    for length in (8, 12):
        for i, x in enumerate(sample(991000 + length, roots, length, torch.float64)):
            p = length // 2 - 2
            clean, bad = cut_states(model, x, p)
            worlds.append((f"molt07-dev-{length}-{i}", x, p, clean, bad))
    return worlds


def rank_panel(model, worlds, seed):
    rows = []
    for index, (root, x, p, clean, bad) in enumerate(worlds):
        intact, targets, cone, terminal = observe(model, x, clean, p)
        o = analytic(model, x, bad, p)
        stack = Sketch.build(o, MAX_RANK)
        final = Sketch.build(o[terminal], MAX_RANK)
        rng = torch.Generator().manual_seed(770000 + seed * 1000 + index)
        random_basis, _ = torch.linalg.qr(torch.randn((o.shape[1], MAX_RANK), generator=rng,
                                                      dtype=torch.float64))
        d = clean[p] - bad[p]
        for dose in DOSES:
            delta = dose * d
            full = observe(model, x, bad, p, delta)[0]
            for arm in ("stack", "terminal", "random", "full"):
                for k in ((o.shape[1],) if arm == "full" else RANKS):
                    if arm == "stack":
                        compact = stack.project(delta, k)
                    elif arm == "terminal":
                        compact = final.project(delta, k)
                    elif arm == "random":
                        q = random_basis[:, :k]
                        compact = q @ (q.T @ delta)
                    else:
                        compact = delta
                    residual = delta - compact
                    linear_error = float((o @ residual).norm())
                    bound = stack.tail(k) * float(residual.norm()) if arm == "stack" else None
                    if bound is not None and linear_error > bound + 1e-9:
                        raise RuntimeError("spectral tail bound violated")
                    replay = observe(model, x, bad, p, compact)[0]
                    rows.append({
                        "root": root, "length": len(x), "seed": seed, "arm": arm,
                        "rank": k, "dose": dose, "linear_error": linear_error, "linear_bound": bound,
                        "actual_observer_error": float((full - replay).norm()),
                        "nonlinear_difference_norm": float((full - replay - o @ residual).norm()),
                        "cone_verified": bool(correct(replay, targets)[cone].all()),
                        "full_cone_verified": bool(correct(full, targets)[cone].all()),
                        "intact_eligible": bool(correct(intact, targets)[cone].all()),
                        "coefficient_bytes": 8 * k,
                    })
    return rows


def binding_for(checkpoint, observer, epoch, root, x, p, bad):
    return MapBinding(checkpoint=checkpoint, frame=root, epoch=epoch,
                      input_digest=tensor_digest(x), state_digest=tensor_digest(bad),
                      observer=observer, cut=CUT, position=p,
                      support=json_digest(["post_visit_recipient", bad.shape[-1], "identity", "fp64"]),
                      max_rank=MAX_RANK)


def cache_panel(model, worlds, checkpoint, observer, cap, reference=None):
    cache = MapCache(cap)
    results = []
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for e in (1, 2):
        epoch = f"fixture-epoch-{e}"
        cache.reset_scope(checkpoint, epoch)
        # Forward sweep followed by reverse sweep: fixed locality pattern.
        # The epoch changes deliberately with identical numeric inputs to test
        # conservative version invalidation, not a simulated semantic repair.
        for order in (worlds, list(reversed(worlds))):
            for root, x, p, clean, bad in order:
                binding = binding_for(checkpoint, observer, epoch, root, x, p, bad)
                sketch = cache.get(binding, lambda: Sketch.build(analytic(model, x, bad, p), MAX_RANK))
                delta = sketch.project(clean[p] - bad[p], MAX_RANK)
                y, target, cone, _ = observe(model, x, bad, p, delta)
                results.append((root, epoch, y, bool(correct(y, target)[cone].all())))
    elapsed, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    max_error = 0.0
    if reference is not None:
        if len(reference) != len(results):
            raise RuntimeError("replay roster mismatch")
        for (root, epoch, y, verified), (rr, re, expected, ev) in zip(results, reference, strict=True):
            if (root, epoch, verified) != (rr, re, ev):
                raise RuntimeError("replay decision mismatch")
            max_error = max(max_error, float((y - expected).abs().max()))
        if max_error > 1e-10:
            raise RuntimeError("cache replay output mismatch")
    report = {**cache.receipt(), "queries": len(results), "wall_seconds": elapsed,
              "process_cpu_seconds": cpu, "verified": sum(r[3] for r in results),
              "max_replay_error": max_error}
    if report["hits"] + report["misses"] != report["queries"] or report["builds"] != report["misses"]:
        raise RuntimeError("cache work ledger mismatch")
    return report, results


def rank_summary(rows):
    result = []
    for arm in ("stack", "terminal", "random", "full"):
        ranks = sorted({r["rank"] for r in rows if r["arm"] == arm})
        for k in ranks:
            for dose in DOSES:
                group = [r for r in rows if (r["arm"], r["rank"], r["dose"]) == (arm, k, dose)]
                result.append({"arm": arm, "rank": k, "dose": dose, "rows": len(group),
                               "verified": sum(r["cone_verified"] for r in group),
                               "mean_linear_error": sum(r["linear_error"] for r in group) / len(group),
                               "mean_actual_error": sum(r["actual_observer_error"] for r in group) / len(group),
                               "max_actual_error": max(r["actual_observer_error"] for r in group),
                               "max_linear_bound_excess": max(
                                   (r["linear_error"] - r["linear_bound"] for r in group
                                    if r["linear_bound"] is not None), default=None)})
    return result


def cache_summary(rows):
    result = []
    for cap in CAPS:
        group = [r for r in rows if r["cap_bytes"] == cap]
        result.append({"cap_bytes": cap, "model_count": len(group),
                       **{k: sum(r[k] for r in group) for k in
                          ("queries", "hits", "misses", "builds", "evictions", "invalidations",
                           "bypasses", "verified", "wall_seconds", "process_cpu_seconds", "build_seconds")},
                       "peak_retained_bytes": max(r["peak_retained_bytes"] for r in group),
                       "max_replay_error": max(r["max_replay_error"] for r in group)})
    return result


def process_high_water():
    # Unix process-wide lifetime peak, not a per-arm or process-tree measure.
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if platform.system() == "Darwin" else rss * 1024)


def run(steps, roots, seeds):
    if not 1 <= steps <= 1500 or not 1 <= roots <= 32 or not 1 <= len(seeds) <= 8:
        raise ValueError("development caps")
    if len(set(seeds)) != len(seeds) or not all(0 <= s <= 10000 for s in seeds):
        raise ValueError("invalid seeds")
    torch.set_num_threads(1)
    sources = source_hashes()
    observer = json_digest(["MOLT07", CUT, LOOPS, "full_supervised_suffix", sources])
    rows, caches, fits = [], [], []
    for seed in seeds:
        model = train(seed, steps)
        checkpoint = sha_state(model)
        fits.append({"seed": seed, "checkpoint": checkpoint, "updates": steps})
        with torch.no_grad():
            worlds = fixture(model, roots)
            rows.extend(rank_panel(model, worlds, seed))
            reference = None
            for cap in CAPS:
                receipt, results = cache_panel(model, worlds, checkpoint, observer, cap, reference)
                caches.append({"seed": seed, **receipt})
                if reference is None:
                    reference = results
        if sha_state(model) != checkpoint:
            raise RuntimeError("frozen checkpoint changed")
    return {"schema": "molt07-observer-memory-v1", "status": "development_only",
            "sources": sources, "observer_digest": observer, "model_seeds": list(seeds),
            "world_roots": 2 * roots, "fits": fits, "rank_summary": rank_summary(rows),
            "cache_summary": cache_summary(caches), "rank_rows": rows, "cache_rows": caches,
            "rank_rows_digest": json_digest(rows), "threads": torch.get_num_threads(),
            "torch": torch.__version__, "process_lifetime_peak_rss_bytes": process_high_water(),
            "hard_process_tree_cap_verified": False, "native_bpd": False,
            "confirmation_ready": False, "promotion_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--roots", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[89, 97, 101, 103])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be fresh")
    report = run(args.steps, args.roots, tuple(args.seeds))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    selected = [r for r in report["rank_summary"] if r["dose"] == 1.0]
    print(json.dumps({"status": report["status"], "rank_rows": len(report["rank_rows"]),
                      "rank_rows_digest": report["rank_rows_digest"], "full_dose": selected,
                      "cache": report["cache_summary"],
                      "process_lifetime_peak_rss_bytes": report["process_lifetime_peak_rss_bytes"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
