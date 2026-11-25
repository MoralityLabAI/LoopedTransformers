"""Replay independent LS17 branches and reverify the archived beam endpoints.

The reused LS16 auditor independently reconstructs FP32 weights, interval
scores and LP residuals. This module validates the distinct branch allocation.
All cohorts in this comparison were exposed before LS17 was designed.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

import audit16 as base

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(493, 517))
ARMS = ("beam", "multistart")
CHARGED_BUDGET = {"expansions": 37, "profile_slots": 13098, "ls_calls": 223, "lp_calls": 222}
NEW_BUDGET = {"expansions": 36, "profile_slots": 12744, "ls_calls": 216, "lp_calls": 216}


def check_multistart(initial: dict, common: dict, result: dict, common_sha: str) -> dict:
    assert result["schema_version"] == 1
    assert result["experiment"] == "LS-NEURON-17"
    assert result["arm"] == "multistart" and result["seed"] == initial["seed"]
    assert result["cohort_exposure"] == "EXPOSED_DEVELOPMENT_ONLY"
    assert result["initial_sha256"] == base.value_sha(initial)
    assert result["common_sha256"] == common_sha
    nodes, boot, totals = base.check_common(initial, common)
    branches = boot.copy()
    assert result["initial_branch_ids"] == [node["id"] for node in boot]
    assert len(result["rounds"]) == 12
    expanded_codes = [tuple(common["initial_fit"]["state"]["codes"])]
    retained_ids = {node["id"] for node in boot}
    retained_codes = {tuple(node["codes"]) for node in boot}
    best = base.keep(branches, 1)[0]
    best_updates, selection_changes = [], []
    for round_index, round_record in enumerate(result["rounds"], 1):
        assert round_record["round"] == round_index
        assert round_record["parents"] == [node["id"] for node in branches]
        assert len(round_record["expansions"]) == 3
        old_best = best
        for position, expansion in enumerate(round_record["expansions"]):
            expected = (f"r{round_index:02d}:p{position}", round_index, position)
            assert (expansion["id"], expansion["round"], expansion["position"]) == expected
            parent_node = branches[position]
            assert expansion["parent_id"] == parent_node["id"]
            expanded_codes.append(tuple(parent_node["codes"]))
            children = base.check_expansion(expansion, parent_node, initial, nodes, totals)
            chosen = base.keep([parent_node] + children, 1)[0]
            assert expansion["selected_after"] == [chosen["id"]]
            assert chosen["scores"]["bound"] <= parent_node["scores"]["bound"]
            branches[position] = chosen
            retained_ids.add(chosen["id"])
            retained_codes.add(tuple(chosen["codes"]))
        assert round_record["retained"] == [node["id"] for node in branches]
        best = base.keep(branches, 1)[0]
        assert round_record["best"] == best["id"]
        assert best["scores"]["bound"] <= old_best["scores"]["bound"]
        if best["id"] != old_best["id"]:
            change = {"round": round_index, "from": old_best["id"], "to": best["id"],
                      "code_changed": old_best["codes"] != best["codes"]}
            selection_changes.append(change)
            if best["scores"]["bound"] < old_best["scores"]["bound"]:
                best_updates.append(change)
    base.check_budgets(result, totals, (37, 13098, 223, 222))
    assert result["budgets"] == CHARGED_BUDGET
    assert result["new_compute_budget"] == NEW_BUDGET
    assert result["branch_final_ids"] == [node["id"] for node in branches]
    winner = next(i for i, node in enumerate(branches) if node["id"] == best["id"])
    assert type(result["winning_branch"]) is int and result["winning_branch"] == winner
    assert result["final_id"] == best["id"]
    weights = base.check_state(best, initial["down_bias"])
    assert result["final"] == best | {"weights": weights}
    hamming = sum(a != b for a, b in zip(best["codes"], initial["codes"]))
    assert result["hamming"] == hamming
    branch_diagnostics = []
    for index, terminal in enumerate(branches):
        path = base.lineage(terminal["id"], nodes)
        origin = boot[index]["id"]
        origins = [i for i, item in enumerate(path) if item["id"] == origin]
        assert len(origins) == 1
        after_boot = path[origins[0] + 1:]
        assert all(item["delta_parent_bound"] <= 0 for item in after_boot)
        assert len(after_boot) <= 12
        branch_diagnostics.append({"branch": index, "bootstrap_id": origin,
            "terminal_id": terminal["id"], "terminal_bound": terminal["scores"]["bound"],
            "path_depth": len(path) - 1, "accepted_steps_after_bootstrap": len(after_boot),
            "worsening_after_bootstrap": 0, "ancestry": path})
    final_path = branch_diagnostics[winner]["ancestry"]
    bootstrap_worse = []
    for node in boot:
        if node["parent_id"] is not None:
            ancestor = nodes[node["parent_id"]]
            delta = node["scores"]["bound"] - ancestor["scores"]["bound"]
            if delta > base.TOL:
                bootstrap_worse.append({"round": 0, "id": node["id"],
                                        "parent_id": ancestor["id"], "increase": delta})
    unique_weights = {base.value_sha(base.decode(n["codes"], n["scales"], n["offset"],
                      initial["down_bias"])) for n in nodes.values()}
    return totals | {"scores": best["scores"], "weights": weights, "hamming": hamming,
        "bootstrap_best_bound": boot[0]["scores"]["bound"],
        "bootstrap_bound_pass": boot[0]["scores"]["bound"] <= base.TARGET,
        "retained_worsening_edges": [], "bootstrap_retained_worsening_edges": bootstrap_worse,
        "total_retained_worsening_edges": len(bootstrap_worse),
        "final_ancestry": final_path, "final_path_depth": len(final_path) - 1,
        "final_ancestry_worsening_edges": sum(item["delta_parent_bound"] is not None
            and item["delta_parent_bound"] > base.TOL for item in final_path),
        "winning_branch": winner, "branch_diagnostics": branch_diagnostics,
        "best_updates": best_updates, "best_selection_changes": selection_changes,
        "unique_codebooks_visited": len({tuple(n["codes"]) for n in nodes.values()}),
        "unique_decoded_weights_visited": len(unique_weights),
        "retained_node_count_all_selections": len(retained_ids),
        "retained_codebook_count_all_selections": len(retained_codes),
        "expanded_codebook_count": len(set(expanded_codes)),
        "repeated_expanded_codebook_count": len(expanded_codes) - len(set(expanded_codes)),
        "any_visited_bound_pass": any(n["scores"]["bound"] <= base.TARGET for n in nodes.values()),
        "seconds": result["seconds"], "new_compute_budget": NEW_BUDGET,
        "new_ls_nfev": totals["ls_nfev"] - common["ls_nfev"]}


def check_coverage(root: Path, seeds: tuple[int, ...]) -> None:
    for folder in ("initial", "common"):
        actual = {p.name for p in (root / folder).glob("*.json")}
        assert actual == {f"{seed}.json" for seed in seeds}, folder
    for folder, arm in (("beam", "beam"), ("results", "multistart")):
        actual = {p.name for p in (root / folder).glob("*.json")}
        assert actual == {f"{arm}-{seed}.json" for seed in seeds}, folder


def check_parent_manifest(root: Path, seeds: tuple[int, ...]) -> dict:
    manifest = json.loads((root / "parent_manifest.json").read_text())
    required = {"baseline_certificates.json"}
    for seed in seeds:
        required.update((f"initial/{seed}.json", f"common/{seed}.json", f"beam/beam-{seed}.json"))
    assert required <= set(manifest["files"])
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"], relative
        assert base.sha(path) == expected["sha256"], relative
    return manifest


def archived_rows(root: Path, seeds: tuple[int, ...]) -> tuple[dict, dict]:
    archive = json.loads((root / "baseline_certificates.json").read_text())
    selected = [row for row in archive["rows"] if row["arm"] == "beam"]
    rows = {row["seed"]: row for row in selected}
    assert len(rows) == len(selected) == len(seeds) and set(rows) == set(seeds)
    return rows, archive["positive_control"]


def audit_seed(job: tuple[int, str, dict]) -> list[dict]:
    seed, root_text, archived = job
    root = Path(root_text)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial_path, common_path = root / "initial" / f"{seed}.json", root / "common" / f"{seed}.json"
    initial, common = json.loads(initial_path.read_text()), json.loads(common_path.read_text())
    assert initial == base.reconstruct_initial(seed)
    initial_certificate = base.verify_scalar(initial["weights"])
    initial_eval = base.evaluate(initial["weights"], seed)
    assert initial_certificate == archived["initial"] and initial_eval == archived["initial_eval"]
    rows = []
    for arm in ARMS:
        path = root / ("beam" if arm == "beam" else "results") / f"{arm}-{seed}.json"
        raw = json.loads(path.read_text())
        assert raw["arm"] == arm
        checked = (base.check_result(initial, common, raw, base.sha(common_path)) if arm == "beam"
                   else check_multistart(initial, common, raw, base.sha(common_path)))
        certificate = base.verify_scalar(checked["weights"])
        evaluation = base.evaluate(checked["weights"], seed)
        if arm == "beam":
            assert base.sha(path) == archived["receipt_sha256"]
            assert checked["weights"] == archived["weights"]
            assert certificate == archived["final"] and evaluation == archived["final_eval"]
        rows.append({"seed": seed, "arm": arm, "dose": 15,
            "cohort_exposure": "EXPOSED_DEVELOPMENT_ONLY", "initial": initial_certificate,
            "final": certificate, "initial_eval": initial_eval, "final_eval": evaluation,
            "initial_sha256": base.sha(initial_path), "common_sha256": base.sha(common_path),
            "receipt_sha256": base.sha(path), "baseline_replay": arm == "beam", **checked})
        print(json.dumps({"seed": seed, "arm": arm, "audit": "PASS", "final": certificate["verdict"]}), flush=True)
    return rows


def summarize(rows: list[dict], positive: dict, n: int) -> dict:
    summaries = []
    for arm in ARMS:
        items = [row for row in rows if row["arm"] == arm]
        repaired = [row["seed"] for row in items if row["initial"]["verdict"] == "POINT_VIOLATION"
                    and row["final"]["verdict"] == "CERTIFIED"]
        info = {"arm": arm, "n_seeds": n, "new_certificates": len(repaired), "new_seeds": repaired,
            "initial_verdicts": dict(Counter(row["initial"]["verdict"] for row in items)),
            "final_verdicts": dict(Counter(row["final"]["verdict"] for row in items)),
            "median_bound": float(np.median([row["scores"]["bound"] for row in items])),
            "bound_certificate_agreement": sum((row["scores"]["bound"] <= base.TARGET)
                == (row["final"]["verdict"] == "CERTIFIED") for row in items),
            "expanded_codebook_count": sum(row["expanded_codebook_count"] for row in items),
            "median_expanded_codebooks": float(np.median([row["expanded_codebook_count"] for row in items])),
            "repeated_expanded_codebook_count": sum(row["repeated_expanded_codebook_count"] for row in items),
            "median_final_path_depth": float(np.median([row["final_path_depth"] for row in items])),
            "rollout": {str(h): {k: sum(row["final_eval"][str(h)][k] for row in items) / n
                        for k in ("accuracy", "all_prefix")} for h in base.HORIZONS},
            "seconds_scope": "Archived LS16 search worker time" if arm == "beam" else "New LS17 search worker time"}
        for key in ("expansions", "profile_slots", "ls_calls", "lp_calls", "ls_nfev", "lp_successes",
                    "secondary_improvements", "secondary_bound_crossings", "checked_states", "seconds"):
            info[key] = sum(row[key] for row in items)
        for key in ("lp_max_primal", "lp_max_dual_sign", "lp_max_stationarity", "lp_max_duality_gap", "lp_max_complementarity"):
            info[key] = max(row[key] for row in items)
        if arm == "multistart":
            info["winning_branch_counts"] = dict(Counter(str(row["winning_branch"]) for row in items))
            info["new_ls_nfev"] = sum(row["new_ls_nfev"] for row in items)
        summaries.append(info)
    certified = [row for row in rows if row["final"]["verdict"] == "CERTIFIED"]
    return {"status": "PASS", "experiment": "LS-NEURON-17", "evidence_class": "EXPOSED_DEVELOPMENT_ONLY",
        "confirmation_eligible": False, "n_units": n, "n_final_endpoints": len(rows),
        "exact_verifier_calls": 1 + 3 * n, "numerical_horizons": list(base.HORIZONS),
        "numerical_horizon_conditions": 3 * len(base.HORIZONS) * n,
        "positive_control_verdict": positive["verdict"], "summary": summaries,
        "actual_new_search_compute": {key: value * n for key, value in NEW_BUDGET.items()},
        "archived_beam_endpoints_reverified": n, "initial_endpoints_reverified": n,
        "distinct_final_weight_sets": len({base.value_sha(row["weights"]) for row in rows}),
        "distinct_certified_weight_sets": len({base.value_sha(row["weights"]) for row in certified}),
        "scope": "Numerical decoder, segment-bound and LP audit uses unchanged independent LS16 code; independent branch retention is replayed here. Archived beam certificates and rollouts must reproduce exactly. LS optimization trajectories are outside replay scope."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--development-seed", type=int, choices=(469,))
    args = parser.parse_args()
    root = args.root.resolve()
    seeds = (args.development_seed,) if args.development_seed is not None else SEEDS
    check_coverage(root, seeds)
    parent = check_parent_manifest(root, seeds)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["experiment"] == "LS-NEURON-17" and manifest["seeds"] == list(seeds)
    assert manifest["arms"] == ["multistart"] and manifest["comparison_arms"] == list(ARMS)
    assert manifest["development"] is True and manifest["cohort_exposure"] == "EXPOSED_DEVELOPMENT_ONLY"
    assert manifest["smoke"] is (args.development_seed is not None)
    assert manifest["initials_reused_without_generation"] is True
    assert manifest["parent_commit"] == parent["parent_commit"]
    assert isinstance(manifest.get("all_searches_completed_utc"), str) and manifest["all_searches_completed_utc"]
    assert manifest["actual_new_compute_budget"] == {key: value * len(seeds) for key, value in NEW_BUDGET.items()}
    if args.development_seed is None:
        lock = HERE / "lock.sha256"
        for line in lock.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            assert base.sha(HERE / relative) == expected, relative
        publication = json.loads((HERE / "freeze_receipt.json").read_text())
        assert publication["lock_sha256"] == base.sha(lock)
        assert len(publication["commit_sha"]) == 40 and manifest["freeze"] == publication
    baseline, archived_positive = archived_rows(root, seeds)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    positive = base.verify_scalar(base.scalar_weights(4)[0])
    assert positive["verdict"] == "CERTIFIED" and positive == archived_positive
    with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as executor:
        jobs = [(seed, str(root), baseline[seed]) for seed in seeds]
        rows = [row for group in executor.map(audit_seed, jobs, chunksize=1) for row in group]
    certificates = {"experiment": "LS-NEURON-17", "cohort_exposure": "EXPOSED_DEVELOPMENT_ONLY",
                    "positive_control": positive, "rows": rows}
    path = root / "certificates.json"
    path.write_text(json.dumps(certificates, indent=2, sort_keys=True) + "\n")
    report = summarize(rows, positive, len(seeds))
    report["certificate_sha256"] = base.sha(path)
    report["parent_manifest_sha256"] = base.sha(root / "parent_manifest.json")
    (root / "audit_results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
