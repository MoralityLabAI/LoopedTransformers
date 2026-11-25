"""Independent numerical reconstruction and receipt replay for LS16.

The exact terminal theorem uses the inherited rational verifier. Numerical
scores, decoded weights, LP KKT residuals, and search selection are checked here
without calling the search implementation's corresponding routines.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

import repair13 as parent
from audit import verify_scalar
from construction import scalar_weights
from parent_train import NAMES

SEEDS = tuple(range(493, 517))
ARMS = ("greedy", "beam")
HORIZONS = (8, 128)
TARGET = 3 / 32
LO, HI = 1e-6, 128.0
TOL = 1e-12
CASES = np.asarray(list(itertools.product(range(4), (0, 1), (0, 1), (0, 1))))
GRID = np.linspace(-1 / 8, 1 / 8, 17)
FEATURES = np.empty((32, 17, 4), dtype=np.float64)
FEATURES[:, :, 0] = CASES[:, 0, None] - 2 + GRID
FEATURES[:, :, 1:] = CASES[:, None, 1:]
TARGETS = ((CASES[:, 0] + CASES[:, 1] + CASES[:, 2]) % 4 - 2)[:, None]


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(a, b, *, atol=1e-9, rtol=1e-11) -> None:
    assert np.allclose(a, b, atol=atol, rtol=rtol), (a, b)


def decode(codes, scales, offset, down_bias) -> dict:
    """Recreate exactly the stored FP32 products and offset subtraction."""
    c = np.asarray(codes)
    s = np.asarray(scales)
    return {
        "skip": (c[:4].astype(np.float32) * np.float32(s[0])).tolist(),
        "skip_bias": [float(np.float32(offset - 0.2 * down_bias))],
        "up": (c[4:36].reshape(8, 4).astype(np.float32) * np.float32(s[1])).tolist(),
        "up_bias": (c[36:44].astype(np.float32) * np.float32(s[2])).tolist(),
        "down": (c[44:52].astype(np.float32) * np.float32(s[3])).tolist(),
        "down_bias": [float(np.float32(down_bias))],
    }


def segment_curvature(preactivations: np.ndarray) -> np.ndarray:
    """Maximum |GELU''| on each affine image of a grid segment."""
    left, right = preactivations[:, :-1], preactivations[:, 1:]
    lo, hi = np.minimum(left, right), np.maximum(left, right)

    def curvature(z):
        return np.abs(2 - z * z) * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    maxima = np.maximum(curvature(lo), curvature(hi))
    for extremum in (-2.0, 0.0, 2.0):
        inside = (lo <= extremum) & (extremum <= hi)
        maxima = np.where(inside, np.maximum(maxima, curvature(extremum)), maxima)
    return maxima


def metric(weights: dict) -> dict:
    """Evaluate segment endpoint bounds independently of search15.metric."""
    up = np.asarray(weights["up"], dtype=float)
    down = np.asarray(weights["down"], dtype=float)
    pre = FEATURES @ up.T + np.asarray(weights["up_bias"], dtype=float)
    output = (
        FEATURES[:, :, 0]
        + FEATURES @ np.asarray(weights["skip"], dtype=float)
        + weights["skip_bias"][0]
        + (np.asarray(pre * ndtr(pre)) @ down + weights["down_bias"][0]) / 5
    )
    error = output - TARGETS
    maximum_second = np.sum(segment_curvature(pre) * np.abs(down) * up[:, 0] ** 2, axis=-1) / 5
    pads = maximum_second * (GRID[1] - GRID[0]) ** 2 / 8
    segment_bound = np.maximum(np.abs(error[:, :-1]), np.abs(error[:, 1:])) + pads
    return {
        "mean": float(np.mean(error ** 2 + 10 * np.maximum(np.abs(error) - TARGET, 0) ** 2)),
        "max": float(np.max(np.abs(error))),
        "bound": float(np.max(segment_bound)),
    }


def check_state(state: dict, down_bias: float) -> dict:
    c = np.asarray(state["codes"])
    s = np.asarray(state["scales"], dtype=float)
    o = state["offset"]
    assert c.shape == (52,) and np.isfinite(c).all()
    assert np.array_equal(c, c.astype(np.int64)) and np.max(np.abs(c)) <= 7
    assert s.shape == (4,) and np.isfinite(s).all()
    assert np.all(s >= LO - 1e-12) and np.all(s <= HI + 1e-10)
    assert math.isfinite(o) and abs(o) <= 8 + 1e-10
    weights = decode(c, s, o, down_bias)
    actual = metric(weights)
    assert set(state["scores"]) == {"mean", "max", "bound"}
    for key in actual:
        close(actual[key], state["scores"][key])
    return weights


def lp_matrix(first: dict, down_bias: float) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the conditional four-variable minimax problem."""
    c = np.asarray(first["codes"])
    weights = decode(c, first["scales"], first["offset"], down_bias)
    up = np.asarray(weights["up"], dtype=float)
    pre = FEATURES @ up.T + np.asarray(weights["up_bias"], dtype=float)
    a = (FEATURES @ c[:4]).ravel()
    b = ((pre * ndtr(pre)) @ c[44:52]).ravel() / 5
    local = np.sum(segment_curvature(pre) * np.abs(c[44:52]) * up[:, 0] ** 2, axis=-1)
    local *= (GRID[1] - GRID[0]) ** 2 / 40
    # Each endpoint participates in at most two adjacent segment inequalities.
    endpoint_pad = np.zeros((32, 17))
    endpoint_pad[:, :-1] = local
    endpoint_pad[:, 1:] = np.maximum(endpoint_pad[:, 1:], local)
    k = endpoint_pad.ravel()
    residual = (FEATURES[:, :, 0] - TARGETS).ravel()
    plus = np.column_stack((a, b + k, np.ones(544), -np.ones(544)))
    minus = np.column_stack((-a, -b + k, -np.ones(544), -np.ones(544)))
    return np.concatenate((plus, minus)), np.concatenate((-residual, residual))


def check_lp(first: dict, second: dict, receipt: dict, down_bias: float) -> dict:
    assert isinstance(receipt["nit"], int) and 0 <= receipt["nit"] <= 1000
    assert first["codes"] == second["codes"]
    assert first["scales"][1:3] == second["scales"][1:3]
    if not receipt["success"]:
        assert receipt["status"] != 0
        assert first["scales"] == second["scales"] and first["offset"] == second["offset"]
        return {"success": False}
    assert receipt["status"] == 0
    x = np.asarray(receipt["x"], dtype=float)
    assert x.shape == (4,) and np.isfinite(x).all()
    matrix, rhs = lp_matrix(first, down_bias)
    lower_bounds = np.asarray([LO, LO, -8, 0])
    upper_bounds = np.asarray([HI, HI, 8, np.inf])
    primal = max(0.0, float(np.max(matrix @ x - rhs)), float(np.max(lower_bounds - x)), float(np.max(x[:3] - upper_bounds[:3])))
    assert primal <= 1e-6, primal
    close(x[3], receipt["objective"], atol=1e-9, rtol=0)
    close([second["scales"][0], second["scales"][3], second["offset"]], x[:3], atol=1e-12, rtol=0)
    y = np.zeros(len(rhs))
    sparse = receipt["inequality_duals"]
    used = set()
    for index, value in sparse:
        assert isinstance(index, int) and 0 <= index < len(y) and index not in used
        assert math.isfinite(value)
        used.add(index)
        y[index] = value
    lower = np.asarray(receipt["lower_duals"], dtype=float)
    upper = np.asarray(receipt["upper_duals"], dtype=float)
    assert lower.shape == upper.shape == (4,) and np.isfinite(lower).all() and np.isfinite(upper).all()
    dual_sign = max(0.0, float(y.max()), float(-lower.min()), float(upper.max()), abs(float(upper[3])))
    objective = np.asarray([0.0, 0.0, 0.0, 1.0])
    stationarity = float(np.max(np.abs(objective - matrix.T @ y - lower - upper)))
    dual_value = float(rhs @ y + lower_bounds @ lower + upper_bounds[:3] @ upper[:3])
    gap = float(x[3] - dual_value)
    complementary = max(float(np.max(np.abs(y * (matrix @ x - rhs)))), float(np.max(np.abs(lower * (x - lower_bounds)))), float(np.max(np.abs(upper[:3] * (x[:3] - upper_bounds[:3])))))
    assert dual_sign <= 1e-7, dual_sign
    assert stationarity <= 1e-6, stationarity
    assert abs(gap) <= 1e-6, gap
    assert complementary <= 1e-6, complementary
    for key, value in (("primal_violation", primal), ("dual_stationarity", stationarity), ("dual_gap", gap)):
        if key in receipt:
            close(value, receipt[key], atol=1e-7, rtol=1e-8)
    return {"success": True, "primal_violation": primal, "dual_sign_violation": dual_sign, "stationarity": stationarity, "duality_gap": gap, "complementarity": complementary}


def evaluate(weights: dict, seed: int) -> dict:
    tensors = [torch.tensor(weights[key], dtype=torch.float32)[None] for key in NAMES]
    result = parent.old.evaluate(tensors, seed)
    assert set(result) == {str(horizon) for horizon in HORIZONS}
    return result


def keep(pool: list[dict], width: int) -> list[dict]:
    """Stable raw-bound ranking followed by full codebook deduplication."""
    selected, seen = [], set()
    for node in sorted(pool, key=lambda item: item["scores"]["bound"]):
        codes = tuple(node["codes"])
        if codes not in seen:
            seen.add(codes)
            selected.append(node)
        if len(selected) == width:
            break
    assert len(selected) == width
    return selected


def proposals(codes: list) -> tuple[np.ndarray, list[str]]:
    c = np.asarray(codes)
    candidates, names = parent.proposals(c, True)
    assert len(candidates) <= 177
    count = 177 - len(candidates)
    return np.concatenate((candidates, np.repeat(c[None], count, axis=0))), names + ["padding"] * count


def shortlist(candidates: np.ndarray, rank: np.ndarray) -> list[int]:
    found, seen = [0], {tuple(candidates[0])}
    for index in sorted(range(len(rank)), key=lambda i: rank[i]):
        code = tuple(candidates[index])
        if code not in seen:
            seen.add(code)
            found.append(index)
        if len(found) == 3:
            return found
    raise AssertionError("Fewer than two edited candidate codebooks")


def empty_totals() -> dict:
    return {
        "expansions": 0, "profile_slots": 0, "ls_calls": 0, "lp_calls": 0,
        "ls_nfev": 0, "lp_successes": 0, "secondary_improvements": 0,
        "secondary_bound_crossings": 0, "checked_states": 0,
        "lp_max_primal": 0.0, "lp_max_dual_sign": 0.0,
        "lp_max_stationarity": 0.0, "lp_max_duality_gap": 0.0,
        "lp_max_complementarity": 0.0,
    }


def check_fit_record(record: dict) -> None:
    assert isinstance(record["nfev"], int) and 1 <= record["nfev"] <= 40
    assert isinstance(record["njev"], int) and 1 <= record["njev"] <= record["nfev"]
    assert record["status"] in (0, 1, 2, 3, 4)
    assert math.isfinite(record["loss"]) and record["loss"] >= 0


def check_expansion(receipt: dict, parent_node: dict, initial: dict,
                    nodes: dict[str, dict], totals: dict) -> list[dict]:
    assert receipt["parent_id"] == parent_node["id"]
    candidates, names = proposals(parent_node["codes"])
    assert len(receipt["anchors"]) == 2
    winners = []
    totals["expansions"] += 1
    for anchor, anchor_name, scales in zip(
        receipt["anchors"], ("current", "initial"),
        (parent_node["scales"], initial["scales"]),
    ):
        assert anchor["name"] == anchor_name
        scales = np.asarray(scales, dtype=float)
        ranks, coefficients, offsets = parent.profile(candidates, scales)
        assert len(anchor["profile"]) == 177
        close(ranks, anchor["profile"], atol=1e-10, rtol=1e-10)
        assert [item["slot"] for item in anchor["fits"]] == shortlist(candidates, ranks)
        totals["profile_slots"] += 177
        for fit in anchor["fits"]:
            slot = fit["slot"]
            assert fit["move"] == names[slot]
            start_scales = scales.copy()
            start_scales[[0, 3]] = coefficients[slot]
            close(start_scales, fit["start_scales"], atol=1e-12, rtol=1e-12)
            close(offsets[slot], fit["start_offset"], atol=1e-12, rtol=1e-12)
            check_fit_record(fit["fit"])
            first, second = fit["first"], fit["second"]
            assert first["codes"] == second["codes"] == candidates[slot].tolist()
            prefix = f"{receipt['id']}:{anchor_name}:{slot}"
            for node, suffix in ((first, "ls"), (second, "lp")):
                assert node["id"] == f"{prefix}:{suffix}"
                assert node["parent_id"] == parent_node["id"]
                assert node["id"] not in nodes
                check_state(node, initial["down_bias"])
                nodes[node["id"]] = node
                totals["checked_states"] += 1
            checked = check_lp(first, second, fit["secondary"], initial["down_bias"])
            totals["ls_calls"] += 1
            totals["lp_calls"] += 1
            totals["ls_nfev"] += fit["fit"]["nfev"]
            totals["lp_successes"] += checked["success"]
            if checked["success"]:
                for out, source in (
                    ("lp_max_primal", "primal_violation"),
                    ("lp_max_dual_sign", "dual_sign_violation"),
                    ("lp_max_stationarity", "stationarity"),
                    ("lp_max_duality_gap", "duality_gap"),
                    ("lp_max_complementarity", "complementarity"),
                ):
                    totals[out] = max(totals[out], abs(checked[source]))
            choose_second = second["scores"]["bound"] < first["scores"]["bound"] - TOL
            assert fit["selected_second"] is choose_second
            totals["secondary_improvements"] += choose_second
            totals["secondary_bound_crossings"] += first["scores"]["bound"] > TARGET >= second["scores"]["bound"]
            winners.append(second if choose_second else first)
    assert receipt["winner_ids"] == [node["id"] for node in winners]
    return winners


def check_budgets(receipt: dict, totals: dict, expected: tuple[int, int, int, int]) -> None:
    names = ("expansions", "profile_slots", "ls_calls", "lp_calls")
    assert tuple(receipt["budgets"][key] for key in names) == expected
    assert tuple(totals[key] for key in names) == expected
    assert receipt["ls_nfev"] == totals["ls_nfev"]


def reconstruct_initial(seed: int) -> dict:
    c, s, o, db = parent.setup(seed, 15)
    return {"seed": seed, "codes": c.tolist(), "scales": s.tolist(),
            "offset": o, "down_bias": db, "weights": decode(c, s, o, db)}


def check_common(initial: dict, common: dict) -> tuple[dict, list[dict], dict]:
    assert common["schema_version"] == 1 and common["seed"] == initial["seed"]
    assert common["initial_sha256"] == value_sha(initial)
    initial_fit = common["initial_fit"]
    check_fit_record(initial_fit["fit"])
    node = initial_fit["state"]
    assert node["id"] == "s0" and node["parent_id"] is None
    assert node["codes"] == initial["codes"]
    check_state(node, initial["down_bias"])
    nodes = {"s0": node}
    totals = empty_totals()
    totals.update(ls_calls=1, ls_nfev=initial_fit["fit"]["nfev"], checked_states=1)
    receipt = common["bootstrap"]
    assert (receipt["id"], receipt["round"], receipt["position"]) == ("b0", 0, 0)
    children = check_expansion(receipt, node, initial, nodes, totals)
    retained = keep([node] + children, 3)
    assert common["retained"] == [item["id"] for item in retained]
    check_budgets(common, totals, (1, 354, 7, 6))
    return nodes, retained, totals


def lineage(final_id: str, nodes: dict) -> list[dict]:
    result, seen = [], set()
    current = final_id
    while current is not None:
        assert current not in seen
        seen.add(current)
        node = nodes[current]
        ancestor = node["parent_id"]
        result.append({"id": current, "parent_id": ancestor,
                       "bound": node["scores"]["bound"],
                       "delta_parent_bound": None if ancestor is None else node["scores"]["bound"] - nodes[ancestor]["scores"]["bound"],
                       "code_hamming_from_parent": None if ancestor is None else sum(a != b for a, b in zip(node["codes"], nodes[ancestor]["codes"]))})
        current = ancestor
    return result[::-1]


def check_result(initial: dict, common: dict, result: dict, common_file_sha: str) -> dict:
    assert result["schema_version"] == 1
    assert result["seed"] == initial["seed"] and result["arm"] in ARMS
    assert result["initial_sha256"] == value_sha(initial)
    assert result["common_sha256"] == common_file_sha
    nodes, boot, totals = check_common(initial, common)
    width = 1 if result["arm"] == "greedy" else 3
    retained = boot[:width]
    assert len(result["rounds"]) == 12
    retained_worse = []
    bootstrap_worse = []
    for node in retained:
        if node["parent_id"] is not None:
            ancestor = nodes[node["parent_id"]]
            delta = node["scores"]["bound"] - ancestor["scores"]["bound"]
            if delta > TOL:
                bootstrap_worse.append({"round": 0, "id": node["id"], "parent_id": ancestor["id"], "increase": delta})
    best_updates = []
    visited = set(node["id"] for node in retained)
    all_retained_nodes = set(visited)
    retained_codes = {tuple(node["codes"]) for node in retained}
    expanded_codes = [tuple(common["initial_fit"]["state"]["codes"])]
    for round_index, round_record in enumerate(result["rounds"], 1):
        assert round_record["round"] == round_index
        assert round_record["parents"] == [node["id"] for node in retained]
        assert len(round_record["expansions"]) == 3
        old_best = retained[0]
        old_ids = {node["id"] for node in retained}
        if width == 3:
            frozen_parents = retained.copy()
            pool = retained.copy()
        for position, expansion in enumerate(round_record["expansions"]):
            assert (expansion["id"], expansion["round"], expansion["position"]) == (f"r{round_index:02}:e{position}", round_index, position)
            expanded = retained[0] if width == 1 else frozen_parents[position]
            expanded_codes.append(tuple(expanded["codes"]))
            children = check_expansion(expansion, expanded, initial, nodes, totals)
            if width == 1:
                retained = keep(retained + children, 1)
                assert expansion["selected_after"] == [node["id"] for node in retained]
                all_retained_nodes.update(node["id"] for node in retained)
                retained_codes.update(tuple(node["codes"]) for node in retained)
            else:
                assert "selected_after" not in expansion
                pool.extend(children)
        if width == 3:
            retained = keep(pool, 3)
            all_retained_nodes.update(node["id"] for node in retained)
            retained_codes.update(tuple(node["codes"]) for node in retained)
        assert round_record["retained"] == [node["id"] for node in retained]
        assert round_record["best"] == retained[0]["id"]
        assert retained[0]["scores"]["bound"] <= old_best["scores"]["bound"]
        if retained[0]["id"] != old_best["id"]:
            best_updates.append({"round": round_index, "from": old_best["id"], "to": retained[0]["id"], "code_changed": old_best["codes"] != retained[0]["codes"]})
        for node in retained:
            visited.add(node["id"])
            if node["id"] not in old_ids and node["parent_id"] is not None:
                ancestor = nodes[node["parent_id"]]
                delta = node["scores"]["bound"] - ancestor["scores"]["bound"]
                if delta > TOL:
                    retained_worse.append({"round": round_index, "id": node["id"], "parent_id": ancestor["id"], "increase": delta})
    check_budgets(result, totals, (37, 13098, 223, 222))
    final = retained[0]
    assert result["final_id"] == final["id"]
    weights = check_state(final, initial["down_bias"])
    assert result["final"] == final | {"weights": weights}
    hamming = sum(a != b for a, b in zip(final["codes"], initial["codes"]))
    assert result["hamming"] == hamming
    path = lineage(final["id"], nodes)
    unique_weights = {value_sha(decode(n["codes"], n["scales"], n["offset"], initial["down_bias"])) for n in nodes.values()}
    return totals | {
        "scores": final["scores"], "weights": weights, "hamming": hamming,
        "bootstrap_best_bound": boot[0]["scores"]["bound"],
        "bootstrap_bound_pass": boot[0]["scores"]["bound"] <= TARGET,
        "retained_worsening_edges": retained_worse,
        "retained_worsening_edges_scope": "Comparison rounds 1..12 only; bootstrap is reported separately",
        "bootstrap_retained_worsening_edges": bootstrap_worse,
        "total_retained_worsening_edges": len(bootstrap_worse) + len(retained_worse),
        "final_ancestry": path,
        "final_path_depth": len(path) - 1,
        "final_ancestry_worsening_edges": sum(item["delta_parent_bound"] is not None and item["delta_parent_bound"] > TOL for item in path),
        "best_updates": best_updates,
        "unique_codebooks_visited": len({tuple(node["codes"]) for node in nodes.values()}),
        "unique_decoded_weights_visited": len(unique_weights),
        "retained_unique_node_count": len(visited),
        "retained_unique_node_count_scope": "Bootstrap and comparison round boundaries",
        "retained_node_count_all_selections": len(all_retained_nodes),
        "retained_codebook_count_all_selections": len(retained_codes),
        "expanded_codebook_count": len(set(expanded_codes)),
        "repeated_expanded_codebook_count": len(expanded_codes) - len(set(expanded_codes)),
        "any_visited_bound_pass": any(node["scores"]["bound"] <= TARGET for node in nodes.values()),
        "seconds": result["seconds"],
    }


def audit_seed(job: tuple[int, str, bool]) -> list[dict]:
    seed, root_text, exact = job
    root = Path(root_text)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial_path, common_path = root / "initial" / f"{seed}.json", root / "common" / f"{seed}.json"
    initial = json.loads(initial_path.read_text())
    common = json.loads(common_path.read_text())
    assert initial == reconstruct_initial(seed)
    initial_certificate = verify_scalar(initial["weights"]) if exact else {"verdict": "DEVELOPMENT_SKIP"}
    initial_eval = evaluate(initial["weights"], seed) if exact else None
    rows = []
    for arm in ARMS:
        result_path = root / "results" / f"{arm}-{seed}.json"
        raw = json.loads(result_path.read_text())
        assert raw["arm"] == arm
        checked = check_result(initial, common, raw, sha(common_path))
        weights = checked["weights"]
        final_certificate = verify_scalar(weights) if exact else {"verdict": "DEVELOPMENT_SKIP"}
        final_eval = evaluate(weights, seed) if exact else None
        rows.append({"seed": seed, "arm": arm, "dose": 15, "initial": initial_certificate,
                     "final": final_certificate, "initial_eval": initial_eval, "final_eval": final_eval,
                     "initial_sha256": sha(initial_path), "common_sha256": sha(common_path),
                     "receipt_sha256": sha(result_path), **checked})
        print(json.dumps({"seed": seed, "arm": arm, "audit": "PASS", "final": final_certificate["verdict"]}), flush=True)
    return rows


def check_coverage(root: Path, seeds: tuple[int, ...]) -> None:
    for folder in ("initial", "common"):
        expected = {f"{seed}.json" for seed in seeds}
        actual = {path.name for path in (root / folder).glob("*.json")}
        assert actual == expected, (folder, expected - actual, actual - expected)
    expected = {f"{arm}-{seed}.json" for seed in seeds for arm in ARMS}
    actual = {path.name for path in (root / "results").glob("*.json")}
    assert actual == expected, ("results", expected - actual, actual - expected)


def summarize(rows: list[dict], positive_control: dict, n: int) -> dict:
    summary = []
    for arm in ARMS:
        items = [row for row in rows if row["arm"] == arm]
        new = [row["seed"] for row in items if row["initial"]["verdict"] == "POINT_VIOLATION" and row["final"]["verdict"] == "CERTIFIED"]
        info = {
            "arm": arm, "n_seeds": len(items), "new_certificates": len(new), "new_seeds": new,
            "initial_verdicts": dict(Counter(row["initial"]["verdict"] for row in items)),
            "final_verdicts": dict(Counter(row["final"]["verdict"] for row in items)),
            "median_bound": float(np.median([row["scores"]["bound"] for row in items])),
            "bound_passes": sum(row["scores"]["bound"] <= TARGET for row in items),
            "bound_certificate_agreement": sum((row["scores"]["bound"] <= TARGET) == (row["final"]["verdict"] == "CERTIFIED") for row in items),
            "bootstrap_bound_passes": sum(row["bootstrap_bound_pass"] for row in items),
            "retained_worsening_edges": sum(len(row["retained_worsening_edges"]) for row in items),
            "bootstrap_retained_worsening_edges": sum(len(row["bootstrap_retained_worsening_edges"]) for row in items),
            "total_retained_worsening_edges": sum(row["total_retained_worsening_edges"] for row in items),
            "final_ancestries_with_worsening_edge": sum(row["final_ancestry_worsening_edges"] > 0 for row in items),
            "certified_final_ancestries_with_worsening_edge": sum(row["final_ancestry_worsening_edges"] > 0 and row["final"]["verdict"] == "CERTIFIED" for row in items),
        }
        for key in ("expansions", "profile_slots", "ls_calls", "lp_calls", "ls_nfev", "lp_successes", "secondary_improvements", "secondary_bound_crossings", "checked_states", "seconds"):
            info[key] = sum(row[key] for row in items)
        for key in ("lp_max_primal", "lp_max_dual_sign", "lp_max_stationarity", "lp_max_duality_gap", "lp_max_complementarity"):
            info[key] = max(row[key] for row in items)
        if items[0]["final_eval"] is not None:
            info["rollout"] = {str(horizon): {metric_name: sum(row["final_eval"][str(horizon)][metric_name] for row in items) / len(items) for metric_name in ("accuracy", "all_prefix")} for horizon in HORIZONS}
        summary.append(info)
    common_totals = {"expansions": n, "profile_slots": 354 * n, "ls_calls": 7 * n, "lp_calls": 6 * n}
    actual = {key: sum(row[key] for row in rows) - common_totals[key] for key in common_totals}
    certified = [row for row in rows if row["final"]["verdict"] == "CERTIFIED"]
    return {
        "status": "PASS", "experiment": "LS-NEURON-16", "evidence_class": "PILOT_ONLY",
        "n_units": n, "n_final_endpoints": len(rows), "exact_verifier_calls": 1 + 3 * n,
        "numerical_horizons": list(HORIZONS),
        "numerical_horizon_conditions": 3 * len(HORIZONS) * n,
        "numerical_initial_horizon_conditions": len(HORIZONS) * n,
        "numerical_final_horizon_conditions": 2 * len(HORIZONS) * n,
        "positive_control_verdict": positive_control["verdict"],
        "summary": summary, "actual_compute_with_common_once": actual,
        "distinct_certified_weight_sets": len({value_sha(row["weights"]) for row in certified}),
        "distinct_final_weight_sets": len({value_sha(row["weights"]) for row in rows}),
        "scope": "Independent numerical metric, FP32 weight decoding, LP residual/dual audit and selection replay; LS solver trajectories are not rerun; exact-real endpoint certificates use the inherited rational verifier.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--development-seed", type=int, choices=(469, 475, 491))
    parser.add_argument("--skip-exact", action="store_true", help="Only valid for old-seed development fixtures")
    args = parser.parse_args()
    assert not args.skip_exact or args.development_seed is not None
    seeds = (args.development_seed,) if args.development_seed is not None else SEEDS
    root = args.root.resolve()
    check_coverage(root, seeds)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["seeds"] == list(seeds) and manifest["arms"] == list(ARMS)
    assert manifest["development"] is (args.development_seed is not None)
    assert isinstance(manifest.get("all_searches_completed_utc"), str) and manifest["all_searches_completed_utc"]
    if args.development_seed is None:
        for line in (root / "lock.sha256").read_text().splitlines():
            digest, relative_path = line.split(maxsplit=1)
            assert sha(root / relative_path) == digest, relative_path
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    positive = verify_scalar(scalar_weights(4)[0]) if not args.skip_exact else {"verdict": "DEVELOPMENT_SKIP"}
    assert positive["verdict"] == ("DEVELOPMENT_SKIP" if args.skip_exact else "CERTIFIED")
    jobs = [(seed, str(root), not args.skip_exact) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        rows = [row for group in executor.map(audit_seed, jobs, chunksize=1) for row in group]
    certificates = {"positive_control": positive, "rows": rows}
    suffix = "_development" if args.development_seed is not None else ""
    cert_path = root / f"certificates{suffix}.json"
    cert_path.write_text(json.dumps(certificates, indent=2, sort_keys=True) + "\n")
    result = summarize(rows, positive, len(seeds))
    if args.skip_exact:
        result["exact_verifier_calls"] = result["numerical_horizon_conditions"] = 0
        result["numerical_initial_horizon_conditions"] = result["numerical_final_horizon_conditions"] = 0
    result["certificate_sha256"] = sha(cert_path)
    (root / f"audit_results{suffix}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

