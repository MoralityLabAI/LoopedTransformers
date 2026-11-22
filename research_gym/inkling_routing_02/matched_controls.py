"""Replay pass-3 K with magnitude- and query-matched development controls."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ..inkling_routing_01.checkpoint import MODEL_REVISION, TRANSFORMERS_VERSION, file_sha256
from ..inkling_routing_01.core import HF_READER_LAYER, TARGET_COORD, WIDTH
from ..inkling_routing_01.hooks import CORE_END, CORE_START, as_numpy
from ..inkling_routing_01.local import RECONSTRUCTION_LIMIT, score_reader, write_json
from .native import DOSE_TOLERANCE, development_roots, digest, headroom, seeded
from .receipts import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, RESOLUTION, verified_run

K = 1
CALIPERS = (0.10, 0.25, 0.50)
CONTROLS_PER_ROOT = 4


def coordinate_matches(x, q, realized_delta, target=TARGET_COORD):
    """Rank before looking at reader outcomes, using signed loading toward zero."""
    dose = abs(x[target])
    loading = q[target] * np.sign(x[target])
    if dose == 0 or loading == 0:
        raise ValueError("target needs nonzero magnitude and K query loading")
    magnitude_error = abs(abs(x) / dose - 1)
    loading_error = abs(q * np.sign(x) - loading) / abs(loading)
    eligible = (x != 0) & (abs(abs(realized_delta) / dose - 1) <= DOSE_TOLERANCE)
    eligible[target] = False
    coords = np.flatnonzero(eligible)
    worst_error = np.maximum(magnitude_error, loading_error)
    order = coords[np.lexsort((coords, magnitude_error[coords] + loading_error[coords], worst_error[coords]))]
    return order, magnitude_error, loading_error, worst_error


def matched_vector(x, q, seed, target=TARGET_COORD):
    """Match dose plus the two projections that determine the post-edit score."""
    mask = np.arange(x.size) != target
    dose = abs(x[target])
    rows = np.stack((x[mask], q[mask]))
    desired = -x[target] * np.array([x[target], q[target]])
    scale = np.linalg.norm(rows, axis=1)
    if np.any(scale == 0):
        return None, None
    a = rows / scale[:, None]
    b = desired / scale
    _, singular, right = np.linalg.svd(a, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
    solution = np.linalg.lstsq(a, b, rcond=1e-12)[0]
    if not np.allclose(a @ solution, b, rtol=1e-10, atol=1e-10):
        return None, None
    ratio = float(np.linalg.norm(solution) / dose)
    if ratio > 1 + 1e-12:
        return None, ratio
    basis = right[:rank]
    direction = np.random.default_rng(seed).standard_normal(mask.sum())
    direction -= basis.T @ (basis @ direction)
    if np.linalg.norm(direction) < 1e-10:
        return None, ratio
    direction /= np.linalg.norm(direction)
    delta = np.zeros_like(x)
    delta[mask] = solution + direction * dose * np.sqrt(max(0, 1 - ratio**2))
    return delta, ratio


def measure(reader, archive, root_id):
    import torch

    x, block2, q = archive["original"][0], archive["bank2"][0], archive["query"]
    source = torch.tensor(x, dtype=torch.bfloat16)
    if not np.array_equal(as_numpy(source), x):
        raise ValueError("block-3 bank does not round-trip through BF16")
    dose = abs(x[TARGET_COORD])
    intended = x - np.sign(x) * dose
    rounded = torch.tensor(intended, dtype=torch.bfloat16)
    order, magnitude, loading, joint = coordinate_matches(x, q[K], as_numpy(rounded) - x)
    nearest = order[:CONTROLS_PER_ROOT]
    edits = [("clean", None, source), ("zero_2027", TARGET_COORD, source.clone())]
    edits[1][2][TARGET_COORD] = 0
    for coord in nearest:
        patched = source.clone()
        patched[coord] = rounded[coord]
        edits.append(("coordinate_control", int(coord), patched))
    vector, required = matched_vector(x, q[K], seeded("matched-control", root_id))
    if vector is not None:
        edits.append(("projection_matched_vector", None, torch.tensor(x + vector, dtype=torch.bfloat16)))

    n = len(edits)
    index = HF_READER_LAYER - CORE_START
    reader.query[index].copy_(torch.tensor(q, dtype=torch.float32))
    bank2 = torch.tensor(block2, dtype=torch.float32).expand(n, 1, 1, WIDTH).contiguous()
    bank3 = torch.stack([edited for _, _, edited in edits]).unsqueeze(1)
    messages = as_numpy(reader(bank2, bank3, index)[:, 0])
    captured = archive["message"][0]
    error = np.linalg.norm(messages[0] - captured) / np.linalg.norm(captured)
    if error >= RECONSTRUCTION_LIMIT:
        raise RuntimeError(f"{root_id}: isolated reader failed clean reconstruction")
    x_dot = -x[TARGET_COORD] ** 2
    q_dot = -x[TARGET_COORD] * q[K, TARGET_COORD]
    coverage = {str(c): int(np.count_nonzero(joint[order] <= c)) for c in CALIPERS}
    rows = []
    for (label, coord, edited), message in zip(edits, messages):
        after = as_numpy(edited)
        delta = after - x
        realized = float(np.linalg.norm(delta))
        fit = score_reader(block2, after, message, q)
        dose_error = abs(realized / dose - 1) if label != "clean" else None
        dx_error = float(abs(np.dot(x, delta) / x_dot - 1))
        dq_error = float(abs(np.dot(q[K], delta) / q_dot - 1))
        if dose_error is not None and dose_error > DOSE_TOLERANCE:
            raise RuntimeError(f"{root_id}: {label} failed dose matching")
        if label.endswith(("control", "vector")) and delta[TARGET_COORD] != 0:
            raise RuntimeError(f"{root_id}: {label} changed coordinate 2027")
        rows.append({
            "root_id": root_id, "arm": label, "coordinate": coord,
            "target_magnitude": float(dose), "target_query_loading": float(q[K, TARGET_COORD] * np.sign(x[TARGET_COORD])),
            "coordinate_before": float(x[coord]) if coord is not None else None,
            "coordinate_after": float(after[coord]) if coord is not None else None,
            "coordinate_oriented_query_loading": float(q[K, coord] * np.sign(x[coord])) if coord is not None else None,
            "magnitude_relative_error": float(magnitude[coord]) if coord is not None else None,
            "loading_relative_error": float(loading[coord]) if coord is not None else None,
            "eligible_coordinates": len(order), "caliper_counts": coverage,
            "vector_minimum_dose_ratio": required,
            "dose": realized, "dose_relative_error": dose_error,
            "bank_projection_relative_error": dx_error, "query_projection_relative_error": dq_error,
            "projection_match": label == "projection_matched_vector" and max(dx_error, dq_error) <= DOSE_TOLERANCE,
            "clean_capture_relative_error": float(error), "reader_scores": fit,
            "k_message_l2_from_zero": float(np.linalg.norm(message[K] - messages[1, K])),
        })
    return rows


def describe(rows_by_root):
    groups = {"nearest_coordinates": [], "caliper_coordinates": [], "projection_vectors": []}
    supports = dict.fromkeys(map(str, CALIPERS), 0)
    nearest_error = {"magnitude": [], "loading": [], "joint": []}
    ratios, vector_residuals = [], []
    within_resolution = 0
    vector_feasible = 0
    for rows in rows_by_root:
        _, zero, *controls = rows
        for caliper, count in zero["caliper_counts"].items():
            supports[caliper] += count > 0
        if zero["vector_minimum_dose_ratio"] is not None:
            ratios.append(zero["vector_minimum_dose_ratio"])
        at = lambda row: row["reader_scores"]["beta3_qkvm"][K]
        coords = [r for r in controls if r["arm"] == "coordinate_control"]
        tight = [r for r in coords if max(r["magnitude_relative_error"], r["loading_relative_error"]) <= CALIPERS[0]]
        if coords:
            mag, load = coords[0]["magnitude_relative_error"], coords[0]["loading_relative_error"]
            for key, value in zip(nearest_error, (mag, load, max(mag, load))):
                nearest_error[key].append(value)
        for name, arms in (("nearest_coordinates", coords), ("caliper_coordinates", tight),
                           ("projection_vectors", [r for r in controls if r["projection_match"]])):
            if arms:
                groups[name].append(float(at(zero) - np.mean([at(row) for row in arms])))
        vectors = [r for r in controls if r["arm"] == "projection_matched_vector"]
        vector_feasible += bool(vectors)
        for vector in vectors:
            if vector["projection_match"]:
                vector_residuals.append(abs(at(zero) - at(vector)))
                within_resolution += vector_residuals[-1] <= RESOLUTION

    def interval(values):
        n = len(values)
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(BOOTSTRAP_DRAWS, n))
        return np.quantile(np.asarray(values)[draws].mean(axis=1), [0.025, 0.975]).tolist()

    return {
        "development_only": True, "sealed_evaluations_accessed": False, "roots": len(rows_by_root),
        "rows": sum(map(len, rows_by_root)), "roots_with_support_by_caliper": supports,
        "median_best_coordinate_relative_error": {k: float(np.median(v)) if v else None for k, v in nearest_error.items()},
        "contrasts": {
            name: {"roots": len(values), "mean_zero_minus_control": float(np.mean(values)) if values else None,
                   "descriptive_95_root_bootstrap_interval": interval(values) if len(values) > 1 else None}
            for name, values in groups.items()
        },
        "vector_feasible_roots": vector_feasible,
        "median_vector_minimum_dose_ratio": float(np.median(ratios)) if ratios else None,
        "projection_matched_roots_within_beta_resolution": within_resolution,
        "max_projection_matched_absolute_beta_gap": max(vector_residuals, default=None),
        "max_reconstruction": max(max(r["reader_scores"]["reconstruction_relative_error"]) for rows in rows_by_root for r in rows),
        "max_clean_capture_error": max(rows[0]["clean_capture_relative_error"] for rows in rows_by_root),
    }


def run(receipts, roster_path, out, count):
    import torch
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopMessageAttentionResidual

    roster = development_roots(roster_path)
    manifest, _, failures = verified_run(receipts)
    if (failures or manifest["schema"] != "ink-route-02.native-reader-development.v1"
            or manifest["checkpoint_revision"] != MODEL_REVISION or manifest["development_only"] is not True
            or manifest["sealed_evaluations_accessed"] is not False
            or manifest["roster_sha256"] != file_sha256(roster_path)
            or sorted(sum(manifest["groups"].values(), [])) != sorted(root["root_id"] for root in roster)):
        raise ValueError("expected the verified native public development banks")
    if not 1 <= count <= len(roster) or importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError("use the pinned Transformers version and a valid development root count")
    torch.set_num_threads(2)
    reader = Inkling2LoopMessageAttentionResidual(SimpleNamespace(
        loop_layer_index_start=CORE_START, loop_layer_index_end=CORE_END, hidden_size=WIDTH,
        _attn_implementation="eager",
    )).eval()
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "development_only": True, "sealed_evaluations_accessed": False, "reader_pass": 3, "query_channel": "K",
        "controls_per_root": CONTROLS_PER_ROOT, "calipers": CALIPERS, "dose_tolerance": DOSE_TOLERANCE,
        "root_ids": [root["root_id"] for root in roster[:count]], "checkpoint_revision": MODEL_REVISION,
        "source_manifest_sha256": file_sha256(receipts / "sha256.json"), "roster_sha256": file_sha256(roster_path),
        "transformers": TRANSFORMERS_VERSION, "torch": torch.__version__, "numpy": np.__version__,
        "code_sha256": file_sha256(Path(__file__)),
    }
    config["manifest_sha256"] = digest(config)
    write_json(out / "manifest.json", config)
    failures_path = out / "failures.jsonl"
    failures_path.touch()
    started = time.perf_counter()
    measured = []
    root_id = None
    try:
        with torch.inference_mode(), (out / "records.jsonl").open("w", encoding="utf-8") as records:
            for root_id in config["root_ids"]:
                headroom("cpu")
                with np.load(receipts / f"reader_{root_id}_pass3.npz", allow_pickle=False) as archive:
                    rows = measure(reader, archive, root_id)
                measured.append(rows)
                for row in rows:
                    records.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(out / "summary.json", describe(measured))
    except Exception as error:
        failures_path.write_text(json.dumps({"root_id": root_id, "error": repr(error)}) + "\n", encoding="utf-8")
        raise
    finally:
        write_json(out / "timing.json", {"elapsed_seconds": time.perf_counter() - started})
        write_json(out / "sha256.json", {p.name: file_sha256(p) for p in out.iterdir()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--public-roster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    run(args.receipts, args.public_roster, args.out, args.count)
