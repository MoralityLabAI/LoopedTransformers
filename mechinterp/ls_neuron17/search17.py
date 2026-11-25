"""LS17 exposed-development control: three independent greedy continuations."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

import search16 as inherited

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(493, 517))
ROUNDS = 12
WIDTH = 3


def validate_parent(root):
    manifest = json.loads((root / "parent_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        if inherited.file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Changed parent receipt: {relative}")
    return manifest


def verify_lock():
    lock = HERE / "lock.sha256"
    for line in lock.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if inherited.file_sha(HERE / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    publication = json.loads((HERE / "freeze_receipt.json").read_text())
    if publication["lock_sha256"] != inherited.file_sha(lock) or len(publication["commit_sha"]) != 40:
        raise ValueError("Missing matching source publication receipt")
    return publication


def search(initial, common, common_sha256):
    started = time.monotonic()
    if common["initial_sha256"] != inherited.digest(initial):
        raise ValueError("Shared bootstrap belongs to a different initialization")
    nodes = inherited.all_nodes(common)
    branches = [nodes[node_id] for node_id in common["retained"]]
    if len(branches) != WIDTH or len({tuple(s["codes"]) for s in branches}) != WIDTH:
        raise ValueError("The shared bootstrap must supply three distinct codebooks")
    initial_branch_ids = [s["id"] for s in branches]
    initial_scales, db = np.asarray(initial["scales"]), initial["down_bias"]
    rounds, nfev = [], common["ls_nfev"]
    for round_number in range(1, ROUNDS + 1):
        parents = branches.copy()
        expansions = []
        for position in range(WIDTH):
            parent_node = branches[position]
            receipt, children, evaluations = inherited.expand(
                parent_node, initial_scales, db, f"r{round_number:02}:p{position}", round_number, position)
            nfev += evaluations
            # Only this branch's parent and children may compete for this slot.
            branches[position] = inherited.select([parent_node] + children, 1)[0]
            receipt["selected_after"] = [branches[position]["id"]]
            expansions.append(receipt)
        best = inherited.select(branches, 1)[0]
        rounds.append({"round": round_number, "parents": [s["id"] for s in parents],
                       "expansions": expansions, "retained": [s["id"] for s in branches],
                       "best": best["id"]})
    final = inherited.select(branches, 1)[0]
    weights = inherited.parent.decoded(np.asarray(final["codes"]), np.asarray(final["scales"]),
                                       final["offset"], db)
    return {"schema_version": 1, "experiment": "LS-NEURON-17", "seed": initial["seed"],
            "arm": "multistart", "cohort_exposure": "EXPOSED_DEVELOPMENT_ONLY",
            "initial_sha256": inherited.digest(initial), "common_sha256": common_sha256,
            "initial_branch_ids": initial_branch_ids, "rounds": rounds,
            "branch_final_ids": [s["id"] for s in branches], "final_id": final["id"],
            "winning_branch": next(i for i, s in enumerate(branches) if s["id"] == final["id"]),
            "final": final | {"weights": weights},
            "budgets": {"expansions": 37, "profile_slots": 13098, "ls_calls": 223, "lp_calls": 222},
            "new_compute_budget": {"expansions": 36, "profile_slots": 12744, "ls_calls": 216, "lp_calls": 216},
            "ls_nfev": nfev, "hamming": int(np.count_nonzero(np.asarray(final["codes"]) != initial["codes"])),
            "seconds": time.monotonic() - started}


def worker(job):
    root_text, seed = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_text)
    initial = json.loads((root / "initial" / f"{seed}.json").read_text())
    common_path = root / "common" / f"{seed}.json"
    common = json.loads(common_path.read_text())
    result = search(initial, common, inherited.file_sha(common_path))
    inherited.write_json(root / "results" / f"multistart-{seed}.json", result)
    print(json.dumps({"seed": seed, "arm": "multistart", "bound": result["final"]["scores"]["bound"],
                      "winning_branch": result["winning_branch"], "seconds": result["seconds"]}), flush=True)
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--development-seed", type=int, choices=(469,))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.development_seed is None:
        freeze = verify_lock()
        seeds = SEEDS
    else:
        if root == HERE:
            raise ValueError("Old-seed smoke requires its separate development directory")
        freeze, seeds = None, (args.development_seed,)
    parent = validate_parent(root)
    actual_initials = {p.name for p in (root / "initial").glob("*.json")}
    if actual_initials != {f"{seed}.json" for seed in seeds}:
        raise ValueError("Input cohort differs from the fixed exposed cohort")
    (root / "results").mkdir(exist_ok=False)
    manifest = {"experiment": "LS-NEURON-17", "seeds": list(seeds), "arms": ["multistart"],
                "comparison_arms": ["beam", "multistart"], "development": True,
                "cohort_exposure": "EXPOSED_DEVELOPMENT_ONLY", "smoke": args.development_seed is not None,
                "started_utc": inherited.stamp(), "freeze": freeze, "workers": args.workers,
                "parent_commit": parent["parent_commit"], "initials_reused_without_generation": True}
    inherited.write_json(root / "run_manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(worker, [(str(root), seed) for seed in seeds], chunksize=1))
    manifest["all_searches_completed_utc"] = inherited.stamp()
    manifest["actual_new_compute_budget"] = {"expansions": 36 * len(seeds),
        "profile_slots": 12744 * len(seeds), "ls_calls": 216 * len(seeds), "lp_calls": 216 * len(seeds)}
    inherited.write_json(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
