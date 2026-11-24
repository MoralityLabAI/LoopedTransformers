"""Replay ATTAIN101 head diagnostics and test its amplified chord at the clean state."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from math import sqrt
from pathlib import Path
import sys

from experiments.nan_attain101 import head as prior_head
from experiments.nan_attain101 import run as prior_geom
from experiments.nan_attain101.geometry import attained, median_or_none, sphere_pair
from experiments.nan_bankdose99 import run as dose
from experiments.nan_bankdose99.geometry import CURRENT_DRIFT_LIMIT, bind_banks
from experiments.nan_bankgap100.rounding import adjacent, assess, joint_round
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bf16, bound, digest, joined, write_new
from experiments.nan_refit49 import run as refit
from experiments.nan_vocab81.run import LABELS, native_controls, stream_tiles


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
OLD_HEAD = refit.REPO / "reports/nanbeige_bpd_gym/nan_attain101_head_results.json"
BUDGET = prior_head.BUDGET
EXPERIMENT = "NAN-COMMONHEAD-102"


def registration():
    previous = prior_head.registration()
    if previous != json.loads(prior_head.HEAD_MANIFEST.read_bytes()):
        raise ValueError("The previous full-head replay source differs")
    for name, pin in previous["files_sha256"].items():
        bound(refit.REPO / name, pin)
    receipt = json.loads(OLD_HEAD.read_bytes())
    if receipt["head_source_sha256"] != digest(prior_head.HEAD_MANIFEST):
        raise ValueError("The previous first-pass receipt differs")
    paths = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "run.py")]
    paths += [OLD_HEAD, prior_head.HEAD_MANIFEST, prior_geom.MANIFEST, prior_head.AGGREGATE]
    return {"experiment": EXPERIMENT, "budget": BUDGET, "files_sha256": {
        p.relative_to(refit.REPO).as_posix(): digest(p) for p in paths}}


def centered(before, gap, cfg, np):
    nominal = float(np.linalg.norm(gap))/2
    ideals = [-gap/2, gap/2]
    pair, bound_value = joint_round(before + ideals[0], before + ideals[1], gap, np)
    audit = assess(before, pair, ideals, nominal, cfg, np, lower_bound=bound_value)
    lower = [nearest_error(before + aim, nominal, np) for aim in ideals]
    tol, cosine = cfg["dose_relative_tolerance"], cfg["minimum_direction_cosine"]
    upper = sqrt(tol**2 + 2*(1 + tol)*(1 - cosine))
    if audit["qualified"] and max(lower) > upper + 1e-10:
        raise ValueError("A qualified pair contradicted the BF16 endpoint bound")
    return pair, audit | {"endpoint_nearest_relative_lower_bounds": lower,
                          "endpoint_allowed_relative_error_upper_bound": upper,
                          "endpoint_lattice_obstruction": max(lower) > upper + 1e-10}


def nearest_error(ideal, radius, np):
    """Exact separable squared error to the finite BF16 lattice in this range."""
    seed = bf16(ideal, np).astype(np.float64)
    if not np.isfinite(seed).all():
        raise ValueError("Requested endpoint exceeds the finite BF16 range")
    lo, hi = adjacent(seed, np)
    lower = np.minimum.reduce([(candidate - ideal)**2 for candidate in (lo, seed, hi)])
    return sqrt(float(lower.sum())) / radius


def setup(snapshot_roster, locks, known, base, cfg, check, np):
    fits = refit.load_fits(base, np)
    states, ancestors, groups, panels = [], [], [], {}
    for b, report in locks.items():
        chosen = report["locked"]
        if chosen is None:
            continue
        banks = bind_banks(fits[f"{b}/T1"], fits[f"{b}/T2"], np)
        selected = prior_head.add_panel(b, base, chosen, known[b], snapshot_roster, banks, cfg, states, ancestors, check, np)
        panels[b] = selected
        incoming = prior_geom.sources(b, base, "development", snapshot_roster, banks, check, np)
        for slot, record in selected.items():
            if not record["edits"]:
                continue
            row = snapshot_roster[slot]
            before, dirs = incoming[row["root"]][row["world"]]
            _, amount, _ = attained(before, dirs["svd"], chosen["seed_fraction"], cfg, np)
            (a, e), _ = sphere_pair(dirs["t1_fit"], dirs["t2_fit"], chosen["gain"], np)
            for i, sign in enumerate((1, -1)):
                gap = (sign * amount * e) - (sign * amount * a)
                pair, audit = centered(before, gap, cfg, np)
                left, right = record["edits"][2*i:2*i + 2]
                groups.append({"backend": b, "slot": slot, "sign": sign, "clean": record["clean"],
                               "original": [left, right], "centered": [len(states), len(states)+1], "geometry": audit})
                states.extend(pair)
                ancestors.extend([record["clean"]] * 2)
    return np.stack(states), ancestors, groups, panels


def pair_components(left, right, base, np):
    a, b = left - base, right - base
    common, half = (a+b)/2, (b-a)/2
    worst = np.maximum(abs(a), abs(b))
    if not np.allclose(worst, abs(common) + abs(half), atol=1e-12, rtol=0):
        raise ValueError("The pointwise first-pass decomposition failed")
    return abs(common), abs(half), worst


def replay(head, states, ancestors, groups, labels, check, np):
    clean = [g["clean"] for g in groups]
    left, right = zip(*(g["original"] for g in groups), strict=True)
    size = len(groups)
    maxima = np.zeros((size, 3))
    at_worst = np.zeros((size, 2))
    ix = np.arange(size)

    def scan():
        for tile in stream_tiles(head, states, check, np):
            common, half, worst = pair_components(tile[list(left)], tile[list(right)], tile[clean], np)
            index = worst.argmax(axis=1)
            improve = worst.max(axis=1) > maxima[:, 2]
            at_worst[improve, 0] = common[ix, index][improve]
            at_worst[improve, 1] = half[ix, index][improve]
            maxima[:] = np.maximum(maxima, np.stack([a.max(axis=1) for a in (common, half, worst)], axis=1))
            yield tile

    read = prior_head.read_tiles(scan(), labels, len(states), ancestors, np)
    if read[-1] != head.vocab or not np.allclose(at_worst.sum(axis=1), maxima[:, 2], atol=1e-12, rtol=0):
        raise ValueError("The streamed decomposition missed the worst-drift token")
    return read, maxima.tolist(), at_worst.tolist()


def evaluate(groups, panels, roster, prior, read, maxima, at_worst, cfg):
    top, _, drift, bare, _ = read
    joined_roots = {b: {} for b in panels}
    peer = {(p["root"], p["world"]): p["answer"] for p in roster.values()}
    for g, (common, half, worst), (shared, differential) in zip(groups, maxima, at_worst, strict=True):
        source = roster[g["slot"]]
        root, world = source["root"], source["world"]
        p = joined_roots[g["backend"]].setdefault(root, {}).setdefault(world, [])
        a, b = [LABELS.index(x) for x in (source["answer"], peer[root, "B" if world == "A" else "A"])]
        clean = g["clean"]
        first, second = g["original"]
        if worst != max(drift[first], drift[second]):
            raise ValueError("The original drift differs from the current-head reduction")

        def gate(i):
            change = (bare[i,a] - bare[i,b]) - (bare[clean,a] - bare[clean,b])
            return bool(top[i] == top[clean] and max(drift[i], abs(change)) <= CURRENT_DRIFT_LIMIT)
        p.append({"sign": g["sign"], "common_peak": common, "half_difference_peak": half, "original_peak": worst,
                  "common_at_original_peak": shared, "half_difference_at_original_peak": differential,
                  "centered_geometry": g["geometry"], "centered_head_pass": all(gate(i) for i in g["centered"]),
                  "centered_head_drift": max(float(drift[i]) for i in g["centered"])})
    report = {}
    for backend, roots in joined_roots.items():
        if len(roots) != prior[backend]["geometry_qualified_roots"]:
            raise ValueError("The common-head audit missed a geometry-qualified root")
        flat = [s for pair in roots.values() for signs in pair.values() for s in signs]
        counts = Counter()
        for pair in roots.values():
            geom, heads = [], []
            for signs in pair.values():
                if {s["sign"] for s in signs} != {1, -1}:
                    raise ValueError("A centered world lost a contrast sign")
                edits = [s["centered_geometry"] for s in signs]
                lengths = [w["length"] for row in edits for w in row["writes"]]
                geom.append(all(c["qualified"] for c in edits) and (
                    max(lengths) - min(lengths) <= cfg["max_contrast_dose_gap_fraction"] * edits[0]["nominal"]))
                heads.append(all(s["centered_head_pass"] for s in signs))
            counts["centered_geometry_roots"] += len(geom) == 2 and all(geom)
            counts["centered_current_head_roots"] += len(heads) == 2 and all(heads)
            counts["centered_joint_roots"] += len(heads) == 2 and all(geom) and all(heads)
        report[backend] = dict(counts) | {
            "roots": len(roots), "signed_pairs": len(flat),
            "pairs_common_dominant_at_original_peak": sum(r["common_at_original_peak"] > r["half_difference_at_original_peak"] for r in flat),
            "pairs_logit_common_only_bound_refused": sum(r["half_difference_peak"] > CURRENT_DRIFT_LIMIT for r in flat),
            "median_common_at_original_peak": median_or_none(s["common_at_original_peak"] for s in flat),
            "median_half_difference_at_original_peak": median_or_none(s["half_difference_at_original_peak"] for s in flat),
            "median_centered_diff_cosine": median_or_none(s["centered_geometry"]["differential"]["cosine"] for s in flat),
            "median_centered_worst_logit_drift": median_or_none(s["centered_head_drift"] for s in flat),
            "centered_signed_head_pairs_passing": sum(s["centered_head_pass"] for s in flat),
            "centered_signed_geometry_pairs_passing": sum(s["centered_geometry"]["qualified"] for s in flat),
            "centered_differential_pairs_passing": sum(s["centered_geometry"]["differential"]["qualified"] for s in flat),
            "centered_endpoint_lattice_obstructed_pairs": sum(s["centered_geometry"]["endpoint_lattice_obstruction"] for s in flat),
            "median_worst_endpoint_nearest_relative_error": median_or_none(
                max(s["centered_geometry"]["endpoint_nearest_relative_lower_bounds"]) for s in flat),
        }
    return report, joined_roots


def fixture(cfg, np):
    a, b, worst = pair_components(np.array([[.25, -.5, 0.]]), np.array([[.5, .5, .25]]), np.zeros((1, 3)), np)
    np.testing.assert_array_equal(a, [[.375, 0., .125]])
    np.testing.assert_array_equal(b, [[.125, .5, .125]])
    np.testing.assert_array_equal(worst, a+b)
    original = np.array([2., -.5])
    (left, right), info = centered(original, np.array([0., .25]), cfg, np)
    np.testing.assert_array_equal(left, [2., -.625])
    np.testing.assert_array_equal(right, [2., -.375])
    assert info["qualified"]
    assert info["endpoint_nearest_relative_lower_bounds"] == [0., 0.]
    _, unresolved = centered(np.array([2., 2.]), np.array([.001, 0.]), cfg, np)
    assert unresolved["endpoint_lattice_obstruction"]
    return {"coordinatewise_common_difference_identity": True, "exact_centered_bf16_pair": True,
            "all_lattice_endpoint_bound_fixture": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--locked", type=Path)
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    seal = registration()
    if args.freeze:
        write_new(MANIFEST, seal)
        return
    if not all((args.locked, args.snapshot, args.output)) or seal != json.loads(MANIFEST.read_bytes()):
        cli.error("Require the frozen replay, locked carrier records, pinned snapshot and fresh output")
    archive = json.loads(prior_head.AGGREGATE.read_bytes())
    past = json.loads(OLD_HEAD.read_bytes())
    if past["geometry_lock_sha256"] != archive["calibration_lock_sha256"]:
        raise ValueError("The historical head and carrier lock differ")
    lock = json.loads(bound(args.locked / "calibration_lock.json", archive["calibration_lock_sha256"]))["calibration"]
    known = json.loads(bound(args.locked / "private_dev.json", archive["private_dev_sha256"]))
    base = json.loads(dose.BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("The head snapshot differs from COMP14")
    cfg = prior_geom.registered()["dose"]
    with lease(BUDGET) as (check, samples):
        if "torch" in sys.modules or (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Require the pinned host-only runtime")
        import numpy as np
        from tokenizers import Tokenizer

        proof = fixture(cfg, np)
        path = args.snapshot / "tokenizer.json"
        bound(path, base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(path))
        ids = [tok.encode(c, add_special_tokens=False).ids for c in LABELS]
        if any(len(t) != 1 for t in ids):
            raise ValueError("Bare labels changed token length")
        source, _ = joined(base)
        roster = {i: p for i, p in source.items() if p["arm"] == "focused" and p["split"] == "development"}
        states, parents, groups, panels = setup(roster, lock, known, base, cfg, check, np)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            read, maxima, at_worst = replay(head, states, parents, groups, [t[0] for t in ids], check, np)
        for b, panel in panels.items():
            native = native_controls(b, base, roster, tok, check, np)
            checked, _ = prior_head.reduce_panel(panel, roster, native, read)
            if checked != past["per_backend"][b]:
                raise ValueError("Parent first-pass readout counts did not close exactly")
        result, private = evaluate(groups, panels, roster, past["per_backend"], read, maxima, at_worst, cfg)
    assert "torch" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=False)
    witness = args.output / "private_pairs.json"
    write_new(witness, private)
    write_new(args.output / "aggregate.json", {
        "experiment": EXPERIMENT, "status": "current_head_decomposition_complete", "source_manifest_sha256": digest(MANIFEST),
        "private_pairs_sha256": digest(witness), "geometry_lock_sha256": archive["calibration_lock_sha256"],
        "per_backend": result, "full_head_rows": len(states), "vocabulary_rows": read[-1], "fixture": proof,
        "parent_first_head_counts_exact": True, "pretrained_decoder_forwards": 0, "formal_comparisons": 0,
        "sealed_test_accessed": False, "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
