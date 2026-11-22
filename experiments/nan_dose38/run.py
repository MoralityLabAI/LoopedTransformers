"""Calibrate two-sided post-BF16 dose on the existing CAL-only physical coordinate system."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readpath37 import run as native


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
READPATH_RESULT_SHA256 = "59187786540694adbb0c6165b7067b8375e0c6ad07ecf6772d5dd2d0a4c7c08c"
READPATH_MANIFEST_SHA256 = "4b6b8dce06a203335d405d5349a061c81273796233414f2798c4465bda8614db"
MIN_COS = 0.98
STEPS = 36
MAX_MULTIPLE = 4


def seal():
    paths = [HERE / name for name in ("__init__.py", ".gitattributes", "PROTOCOL.md", "run.py")]
    paths += [native.MANIFEST, native.HERE / "run.py", REPO / "experiments/nan_phase35/run.py"]
    return {"experiment": "NAN-DOSE-38",
            "files_sha256": {path.relative_to(REPO).as_posix(): native.digest(path) for path in paths},
            "readpath_result_sha256": READPATH_RESULT_SHA256,
            "minimum_direction_cosine": MIN_COS, "bisection_steps": STEPS, "maximum_input_scale_multiple": MAX_MULTIPLE}


def displacement(h, unit, scales, np):
    return native.bf16(h + scales[:, None] * unit, np).astype(np.float64) - h


def lengths(h, unit, scales, np):
    return np.linalg.norm(displacement(h, unit, scales, np), axis=1)


def geometry(h, unit, scales, target, np):
    delta = displacement(h, unit, scales, np)
    dose = np.linalg.norm(delta, axis=1)
    dot = np.sum(delta * unit, axis=1)
    cosine = np.divide(dot, dose, out=np.zeros_like(dose), where=dose > 0)
    return abs(dose / target - 1.0), cosine


def calibrate(h, signed_unit, target, np):
    size = signed_unit.shape[0]
    low = np.zeros(size, dtype=np.float64)
    high = np.full(size, target * MAX_MULTIPLE, dtype=np.float64)
    reached_high = lengths(h, signed_unit, high, np)
    bracketed = reached_high >= target
    for _ in range(STEPS):
        middle = (low + high) * .5
        stays_low = lengths(h, signed_unit, middle, np) < target
        low = np.where(stays_low, middle, low)
        high = np.where(stays_low, high, middle)
    attained_low = lengths(h, signed_unit, low, np)
    attained_high = lengths(h, signed_unit, high, np)
    choose_high = abs(attained_high - target) < abs(attained_low - target)
    chosen = np.where(choose_high, high, low)
    max_gap = float(np.max((attained_high - attained_low) / target))
    return geometry(h, signed_unit, chosen, target, np), bracketed, max_gap


def test(np):
    h = np.array([256.0], np.float64)
    unit = np.array([[1.0]], np.float64)
    (e, c), ok, _ = calibrate(h, unit, 1.6, np)
    assert ok[0] and c[0] == 1 and abs(e[0] - .25) < 1e-11
    h = np.zeros(3, np.float64)
    unit = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    (e, c), ok, _ = calibrate(h, unit, .25, np)
    assert all(ok) and all(e == 0) and all(c == 1)


def process(backend, cfg, np, check):
    pin = cfg["backends"][backend]
    p = "experiments/nan_comp14/fixtures/public.jsonl"
    rows = [json.loads(line) for line in native.bound(REPO / p, cfg["common_pins"][p]).splitlines()]
    place = native.LOCAL / backend
    audit = json.loads(native.bound(native.LOCAL / "audits" / f"{backend}.json", pin["audit"]))
    assert audit["status"] == "passed" and audit["worker_status"] == "completed"
    folder = native.FIT / backend / "t2_layer21_down_proj_rank8"
    with np.load(BytesIO(native.bound(folder / "final.npz", pin["factor_bank"])), allow_pickle=False) as data:
        writes = data["U"].astype(np.float64)
    assert writes.shape == (cfg["rank"], cfg["width"])
    writes /= np.linalg.norm(writes, axis=1, keepdims=True)
    random = np.linalg.qr(np.random.default_rng(cfg["random_seed"]).normal(
        size=(cfg["width"], cfg["rank"])), mode="reduced")[0].T
    coordinate = np.zeros((3, cfg["width"]), dtype=np.float64)
    coordinate[np.arange(3), [2278, 2279, 2280]] = 1
    directions = np.concatenate((writes, random, coordinate))
    assert np.isfinite(directions).all() and np.allclose(np.linalg.norm(directions, axis=1), 1)
    results = []
    for i, row in enumerate(rows):
        if row["split"] != "calibration" or row["arm"] not in cfg["arms"]:
            continue
        check()
        verified = audit["records"][i]
        assert all(verified[k] == row[k] for k in ("id", "split", "world", "root", "arm"))
        case = json.loads(native.bound(place / f"case_{i:04d}.json", verified["case_sha256"]))
        assert case["state_file"] == f"case_{i:04d}_states.npz" and case["state_file_sha256"] == verified["state_sha256"]
        payload = native.bound(place / case["state_file"], verified["state_sha256"])
        states = native.case_states(case, payload, np)
        for phase in (1, 2):
            h = states[phase, "residual_after_layer"].astype(np.float64)
            assert np.array_equal(native.bf16(h, np), h)
            for fraction in cfg["dose_fractions"]:
                target = fraction * float(np.linalg.norm(h))
                proposal = [geometry(h, sign * directions, np.full(len(directions), target), target, np)
                            for sign in (1, -1)]
                adjusted = [calibrate(h, sign * directions, target, np) for sign in (1, -1)]
                original_error = np.maximum(*[g[0] for g in proposal])
                fixed_error = np.maximum(*[g[0][0] for g in adjusted])
                original_cos = np.minimum(*[g[1] for g in proposal])
                fixed_cos = np.minimum(*[g[0][1] for g in adjusted])
                assert (fixed_error <= original_error + 1e-9).all()
                results.append({
                    "root": row["root"], "arm": row["arm"], "phase": phase, "fraction": fraction,
                    "one_shot_error": original_error.tolist(), "one_shot_cos": original_cos.tolist(),
                    "calibrated_error": fixed_error.tolist(), "calibrated_cos": fixed_cos.tolist(),
                    "both_sides_bracketed": np.logical_and(*[g[1] for g in adjusted]).tolist(),
                    "max_relative_bracket_gap": max(g[2] for g in adjusted),
                })
    assert len(results) == 192
    return results


def summarize(records, cfg):
    cells = defaultdict(list)
    for row in records:
        cells[row["arm"], row["phase"], row["fraction"]].append(row)
    groups = [("IO18_T2", range(8)), ("orthonormal_random", range(8, 16)),
              ("channel_2278", [16]), ("channel_2279", [17]), ("channel_2280", [18])]
    result = []
    for (arm, phase, dose), rows in sorted(cells.items()):
        for group, indices in groups:
            assert len(rows) == 16 and len({r["root"] for r in rows}) == 8
            root_counts = defaultdict(list)
            for row in rows:
                root_counts[row["root"]].append(row)
            metrics = {}
            for label in ("one_shot", "calibrated"):
                errors = [r[label + "_error"][i] for r in rows for i in indices]
                cosines = [r[label + "_cos"][i] for r in rows for i in indices]
                def valid(r, i):
                    return r[label + "_error"][i] <= cfg["dose_relative_tolerance"] and r[label + "_cos"][i] >= MIN_COS
                metrics[label] = {
                    "qualified_world_directions": sum(valid(r, i) for r in rows for i in indices),
                    "complete_roots": sum(all(valid(r, i) for r in pair for i in indices) for pair in root_counts.values()),
                    "mean_max_relative_dose_error": fmean(errors),
                    "mean_min_direction_cosine": fmean(cosines),
                }
            result.append({"arm": arm, "phase": phase, "dose": dose, "direction_group": group,
                           "world_directions": len(rows) * len(indices), "roots": 8,
                           "both_sides_bracketed": sum(r["both_sides_bracketed"][i] for r in rows for i in indices),
                           **metrics})
    return result


def main():
    if not __debug__:
        raise RuntimeError("Enable assertions")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        native.write_new(MANIFEST, seal())
        print(native.digest(MANIFEST))
        return
    assert args.output and json.loads(MANIFEST.read_bytes()) == seal()
    assert native.digest(native.MANIFEST) == READPATH_MANIFEST_SHA256
    assert native.digest(REPO / "runs/nan_readpath37/results.json") == READPATH_RESULT_SHA256
    cfg = json.loads((native.HERE / "registration.json").read_bytes())
    with lease(cfg) as (check, samples):
        import numpy as np

        test(np)
        groups = {}
        for backend in cfg["backends"]:
            records = process(backend, cfg, np, check)
            native.write_new(args.output.with_name(f"{backend}_private.json"), records)
            groups[backend] = summarize(records, cfg)
            print(f"Calibrated {backend}: {len(records)} dose/state rows", flush=True)
        assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    payload = {"status": "complete", "experiment": "NAN-DOSE-38", "source_manifest_sha256": native.digest(MANIFEST),
               "scope": "CAL phase coordinate dose feasibility, no model or readout calculation",
               "roots_per_backend": 8, "dose_state_rows_per_backend": 192,
               "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
               "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
               "per_backend": groups, "new_forwards": 0, "sealed_test_accessed": False}
    native.write_new(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status","source_manifest_sha256","max_sampled_private_bytes",
                                                     "min_sampled_available_RAM_bytes")}), flush=True)


if __name__ == "__main__":
    main()
