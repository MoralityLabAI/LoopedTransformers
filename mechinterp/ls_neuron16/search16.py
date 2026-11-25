"""Frozen LS16 breadth-versus-depth pilot with a shared bootstrap expansion."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
import torch

import repair13 as parent
import search15 as inherited

SEEDS = tuple(range(493, 517))
ARMS = ("greedy", "beam")
ROUNDS = 12
WIDTH = 3
EPS = 1e-12
HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")


def select(pool, width):
    """Raw-score order; stable ties, then at most one state per codebook."""
    kept, seen = [], set()
    for node in sorted(pool, key=lambda n: n["scores"]["bound"]):
        code = tuple(node["codes"])
        if code in seen:
            continue
        seen.add(code)
        kept.append(node)
        if len(kept) == width:
            break
    if len(kept) != width:
        raise ValueError("Insufficient distinct codebooks for the registered width")
    return kept


def state(codes, scales, offset, db, identity, parent_id):
    value = inherited.state(codes, scales, offset, db)
    value.update(id=identity, parent_id=parent_id)
    if not all(np.isfinite(x) for x in value["scores"].values()):
        raise ValueError("Nonfinite candidate score")
    return value


def lp_fit(codes, scales, offset, db):
    """Inherited conditional bound LP, with sparse dual vectors retained."""
    matrix, rhs = inherited.lp_matrix(codes, scales, offset, db, True)
    objective = np.array([0., 0., 0., 1.])
    bounds = [(parent.LO, parent.HI), (parent.LO, parent.HI), (-8., 8.), (0., None)]
    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds,
                     method="highs-ds", options={"dual_feasibility_tolerance": 1e-8,
                     "primal_feasibility_tolerance": 1e-8, "maxiter": 1000})
    record = {"status": int(result.status), "nit": int(result.nit), "success": bool(result.success)}
    if not result.success:
        return scales.copy(), offset, record
    x = result.x
    updated = scales.copy()
    updated[[0, 3]] = x[:2]
    y, lower, upper = result.ineqlin.marginals, result.lower.marginals, result.upper.marginals
    dual = (rhs @ y + parent.LO * (lower[0] + lower[1]) - 8 * lower[2]
            + parent.HI * (upper[0] + upper[1]) + 8 * upper[2])
    record.update(
        x=x.tolist(), objective=float(result.fun),
        inequality_duals=[[int(i), float(y[i])] for i in np.flatnonzero(y)],
        lower_duals=lower.tolist(), upper_duals=upper.tolist(),
        primal_violation=float(max(0., (matrix @ x - rhs).max(), parent.LO - x[0],
            parent.LO - x[1], x[0] - parent.HI, x[1] - parent.HI, abs(x[2]) - 8, -x[3])),
        dual_stationarity=float(abs(objective - matrix.T @ y - lower - upper).max()),
        dual_gap=float(result.fun - dual),
    )
    return updated, float(x[2]), record


def expand(node, initial_scales, db, identity, round_number, position):
    codes = np.asarray(node["codes"], dtype=np.int64)
    candidates, labels = inherited.candidates(codes)
    winners, anchors = [], []
    nfev = 0
    for name, scales in (("current", np.asarray(node["scales"])), ("initial", initial_scales)):
        ranks, coefficients, offsets = parent.profile(candidates, scales)
        fits = []
        for slot in inherited.shortlist(candidates, ranks):
            start = scales.copy()
            start[[0, 3]] = coefficients[slot]
            q = candidates[slot]
            fitted, offset, fit = parent.fit(q, start, float(offsets[slot]))
            nfev += fit["nfev"]
            prefix = f"{identity}:{name}:{slot}"
            first = state(q, fitted, offset, db, prefix + ":ls", node["id"])
            refined, off2, secondary = lp_fit(q, fitted, offset, db)
            second = state(q, refined, off2, db, prefix + ":lp", node["id"])
            use_second = second["scores"]["bound"] < first["scores"]["bound"] - EPS
            winner = second if use_second else first
            winners.append(winner)
            fits.append({"slot": slot, "move": labels[slot], "start_scales": start.tolist(),
                         "start_offset": float(offsets[slot]), "fit": fit, "first": first,
                         "secondary": secondary, "second": second, "selected_second": use_second})
        anchors.append({"name": name, "profile": ranks.tolist(), "fits": fits})
    record = {"id": identity, "round": round_number, "position": position,
              "parent_id": node["id"], "anchors": anchors,
              "winner_ids": [s["id"] for s in winners]}
    return record, winners, nfev


def all_nodes(common, rounds=()):
    nodes = {common["initial_fit"]["state"]["id"]: common["initial_fit"]["state"]}
    expansions = [common["bootstrap"]]
    expansions.extend(expansion for row in rounds for expansion in row["expansions"])
    for expansion in expansions:
        for anchor in expansion["anchors"]:
            for fit in anchor["fits"]:
                for key in ("first", "second"):
                    node = fit[key]
                    if node["id"] in nodes:
                        raise ValueError("Duplicate node identity")
                    nodes[node["id"]] = node
    return nodes


def bootstrap(initial):
    started = time.monotonic()
    c, s = np.asarray(initial["codes"]), np.asarray(initial["scales"])
    fitted, offset, fit = parent.fit(c, s, initial["offset"])
    node = state(c, fitted, offset, initial["down_bias"], "s0", None)
    expansion, winners, nfev = expand(node, s, initial["down_bias"], "b0", 0, 0)
    retained = select([node] + winners, WIDTH)
    return {"schema_version": 1, "seed": initial["seed"], "initial_sha256": digest(initial),
            "initial_fit": {"fit": fit, "state": node}, "bootstrap": expansion,
            "retained": [s["id"] for s in retained],
            "budgets": {"expansions": 1, "profile_slots": 354, "ls_calls": 7, "lp_calls": 6},
            "ls_nfev": nfev + fit["nfev"], "seconds": time.monotonic() - started}


def search(initial, common, arm, common_sha256):
    if arm not in ARMS:
        raise ValueError(arm)
    started = time.monotonic()
    nodes = all_nodes(common)
    retained = [nodes[i] for i in common["retained"][:1 if arm == "greedy" else WIDTH]]
    rounds, nfev = [], common["ls_nfev"]
    initial_scales, db = np.asarray(initial["scales"]), initial["down_bias"]
    for r in range(1, ROUNDS + 1):
        parents = retained.copy()
        expansions, pool = [], parents.copy()
        for position in range(WIDTH):
            node = retained[0] if arm == "greedy" else parents[position]
            expansion, winners, count = expand(node, initial_scales, db, f"r{r:02d}:e{position}", r, position)
            nfev += count
            if arm == "greedy":
                retained = select(retained + winners, 1)
                expansion["selected_after"] = [s["id"] for s in retained]
            else:
                pool.extend(winners)
            expansions.append(expansion)
        if arm == "beam":
            retained = select(pool, WIDTH)
        rounds.append({"round": r, "parents": [s["id"] for s in parents],
                       "expansions": expansions, "retained": [s["id"] for s in retained],
                       "best": retained[0]["id"]})
    final = retained[0]
    weights = parent.decoded(np.asarray(final["codes"]), np.asarray(final["scales"]), final["offset"], db)
    return {"schema_version": 1, "seed": initial["seed"], "arm": arm,
            "initial_sha256": digest(initial), "common_sha256": common_sha256,
            "rounds": rounds, "final_id": final["id"], "final": final | {"weights": weights},
            "budgets": {"expansions": 37, "profile_slots": 13098, "ls_calls": 223, "lp_calls": 222},
            "ls_nfev": nfev, "hamming": int(np.count_nonzero(np.asarray(final["codes"]) != initial["codes"])),
            "seconds": time.monotonic() - started}


def common_worker(job):
    root, seed = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = json.loads((Path(root) / "initial" / f"{seed}.json").read_text())
    result = bootstrap(initial)
    write_json(Path(root) / "common" / f"{seed}.json", result)
    return seed


def worker(job):
    root, seed, arm = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root)
    initial = json.loads((root / "initial" / f"{seed}.json").read_text())
    common_path = root / "common" / f"{seed}.json"
    common = json.loads(common_path.read_text())
    result = search(initial, common, arm, file_sha(common_path))
    write_json(root / "results" / f"{arm}-{seed}.json", result)
    print(json.dumps({"seed": seed, "arm": arm, "final_bound": result["final"]["scores"]["bound"],
                      "seconds": result["seconds"]}), flush=True)
    return seed, arm


def verify_lock():
    lock = HERE / "lock.sha256"
    for line in lock.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if file_sha(HERE / relative) != expected:
            raise ValueError(f"Frozen source mismatch: {relative}")
    receipt = json.loads((HERE / "freeze_receipt.json").read_text())
    if receipt["lock_sha256"] != file_sha(lock) or len(receipt["commit_sha"]) != 40:
        raise ValueError("Missing matching publication receipt")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--development-seed", type=int)
    args = parser.parse_args()
    if args.development_seed is None:
        freeze = verify_lock()
        seeds = SEEDS
    else:
        if not 433 <= args.development_seed < 493:
            raise ValueError("Development runs must use an already exposed seed 433..492")
        if args.out.resolve() == HERE:
            raise ValueError("Development receipts need a separate output directory")
        seeds, freeze = (args.development_seed,), None
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.out.resolve()
    for name in ("initial", "common", "results"):
        (root / name).mkdir(parents=True, exist_ok=False)
    manifest = {"experiment": "LS-NEURON-16", "seeds": list(seeds), "arms": list(ARMS),
                "development": args.development_seed is not None, "started_utc": stamp(),
                "freeze": freeze, "workers": args.workers}
    write_json(root / "run_manifest.json", manifest)
    for seed in seeds:
        write_json(root / "initial" / f"{seed}.json", inherited.init(seed))
    manifest["initials_generated_utc"] = stamp()
    write_json(root / "run_manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(common_worker, [(str(root), s) for s in seeds], chunksize=1))
        list(executor.map(worker, [(str(root), s, a) for s in seeds for a in ARMS], chunksize=1))
    manifest["all_searches_completed_utc"] = stamp()
    manifest["actual_compute_budget"] = {"expansions": len(seeds) * 73,
        "profile_slots": len(seeds) * 25842, "ls_calls": len(seeds) * 439, "lp_calls": len(seeds) * 438}
    write_json(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
