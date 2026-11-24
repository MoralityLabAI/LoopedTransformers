"""Select a reachable carrier rule on COMP14 CAL and replay its locked DEV condition."""

import argparse
from collections import Counter
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import sys

from experiments.nan_bankdose99.geometry import bind_banks, materialize
from experiments.nan_bankdose99 import run as dose
from experiments.nan_bankgap100 import run as previous
from experiments.nan_bankgap100.run import world_pass
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bound, digest, joined, write_new
from experiments.nan_refit49 import run as refit

from .geometry import GAIN_LADDER, MIN_CAL, REACHED_FRACTIONS, SEED_FRACTIONS, attained, measure, median_or_none, select, unit_directions
from .verify import fixture


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PRIOR = refit.REPO / "reports/nanbeige_bpd_gym/nan_bankgap100_results.json"
EXPERIMENT = "NAN-ATTAIN-101"
BUDGET = dose.BUDGET | {"maximum_seconds": 240}


def registered():
    seal = previous.registered()
    if (seal != json.loads(previous.MANIFEST.read_bytes())
            or json.loads(PRIOR.read_bytes())["source_manifest_sha256"] != digest(previous.MANIFEST)):
        raise ValueError("The audited BANKGAP100 parent differs from the source freeze")
    for name, pin in seal["files_sha256"].items():
        bound(refit.REPO / name, pin)
    paths = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "geometry.py", "run.py", "verify.py")]
    paths += [previous.MANIFEST, PRIOR]
    return {
        "experiment": EXPERIMENT, "seed_fractions": list(SEED_FRACTIONS), "gains": list(GAIN_LADDER),
        "reached_fraction_range": list(REACHED_FRACTIONS), "minimum_cal_roots": MIN_CAL, "dose": seal["dose"],
        "resource_budget": BUDGET, "files_sha256": {p.relative_to(refit.REPO).as_posix(): digest(p) for p in paths},
    }


def sources(backend, base, split, roster, banks, check, np):
    captured = dose.captured(backend, base, {i: p for i, p in roster.items() if p["split"] == split}, check, np)
    if len(captured) != base["roots"][split] or any(key[0] != split for key in captured):
        raise ValueError("The split-specific native archive differs from COMP14")
    out = {}
    for (_, root), pair in sorted(captured.items()):
        worlds = out.setdefault(root, {})
        for label, peer in (("A", "B"), ("B", "A")):
            check()
            x, before = pair[label]
            direction = materialize(pair[peer][0] - x, banks, root, label, np)
            worlds[label] = before.astype(np.float64), unit_directions(direction, np)
    return out


def summarize(roots, cfg):
    report = {}
    worlds = [w for pair in roots.values() for w in pair.values()]
    for method in ("joint", "nearest"):
        vote = []
        records = []
        for pair in roots.values():
            votes = []
            for w in pair.values():
                if w["status"] == "measured":
                    sides = list(w["signs"].values())
                    records.extend(row[method] for row in sides)
                    votes.append(world_pass([{"methods": row} for row in sides], method, cfg))
            vote.append((len(votes) == 2 and all(d for d, _ in votes),
                         len(votes) == 2 and all(v for _, v in votes)))
        report[method] = {
            "dose_roots": sum(d for d, _ in vote), "faithful_roots": sum(f for _, f in vote),
            "signed_cases": len(records), "faithful_signs": sum(w["qualified"] for w in records),
            "exact_alias_signs": sum(w["exact_state_alias"] for w in records),
            "stencil_refused_signs": sum(w["stencil_certified_incompatible"] for w in records),
            "median_differential_error": median_or_none(w["differential_relative_error"] for w in records),
            "median_differential_cosine": median_or_none(w["differential"]["cosine"] for w in records),
        }
    return report | {"total_roots": len(roots), "world_status": dict(Counter(w["status"] for w in worlds)),
                     "median_reached_fraction": median_or_none(w.get("reached_fraction") for w in worlds),
                     "median_amplification_midpoint_shift_over_dose": median_or_none(
                         w.get("ideal_midpoint_shift_over_dose") for w in worlds)}


def trial(roots, anchors, seed, gain, cfg, check, np):
    out = {}
    for key, pair in roots.items():
        check()
        out[key] = {world: measure(before, dirs, anchors[key][world], gain, cfg, np)
                    for world, (before, dirs) in pair.items()}
    return summarize(out, cfg) | {"seed_fraction": seed, "gain": gain}, out


def carriers(roots, seed, cfg, np):
    return {key: {world: attained(before, dirs["svd"], seed, cfg, np) if dirs is not None else None
                  for world, (before, dirs) in pair.items()} for key, pair in roots.items()}


def calibration(roots, cfg, check, np):
    grid = {seed: carriers(roots, seed, cfg, np) for seed in SEED_FRACTIONS}
    rows, witnesses = [], {}
    for seed, gain in product(SEED_FRACTIONS, GAIN_LADDER):
        summary, raw = trial(roots, grid[seed], seed, gain, cfg, check, np)
        rows.append(summary)
        witnesses[f"{seed}/{gain}"] = raw
    return {"locked": select(rows), "candidates": rows,
            "best_gain_one_cal_roots": max(r["joint"]["faithful_roots"] for r in rows if r["gain"] == 1)}, witnesses


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    source = registered()
    if args.freeze:
        write_new(MANIFEST, source)
        return
    if not args.output or json.loads(MANIFEST.read_bytes()) != source:
        cli.error("Require the frozen CAL-only replay and a fresh output directory")
    base = json.loads(dose.BASE.read_bytes())
    cfg = source["dose"]
    with lease(BUDGET) as (check, samples):
        if "torch" in sys.modules or version("numpy") != "2.5.5":
            raise ValueError("The pinned host-only runtime is required")
        import numpy as np

        verify = fixture(cfg, np)
        selected, _ = joined(base)
        # No outcome labels are passed to the carrier or selector.
        roster = {i: p for i, p in selected.items() if p["arm"] == "focused"}
        fits = refit.load_fits(base, np)
        banks = {b: bind_banks(fits[f"{b}/T1"], fits[f"{b}/T2"], np) for b in base["backends"]}
        cal, private = {}, {}
        for b, physical in banks.items():
            roots = sources(b, base, "calibration", roster, physical, check, np)
            cal[b], private[b] = calibration(roots, cfg, check, np)

        args.output.mkdir(parents=True, exist_ok=False)
        witness = args.output / "private_cal.json"
        write_new(witness, private)
        lockfile = args.output / "calibration_lock.json"
        write_new(lockfile, {"calibration": cal, "private_cal_sha256": digest(witness),
                             "source_manifest_sha256": digest(MANIFEST)})
        stamp = digest(lockfile)
        lock = json.loads(bound(lockfile, stamp))["calibration"]
        dev, raw = {}, {}
        for b, physical in banks.items():
            chosen = lock[b]["locked"]
            if chosen is None:
                dev[b] = {"status": "calibration_refused", "scored_roots": 0}
                continue
            roots = sources(b, base, "development", roster, physical, check, np)
            seed, gain = chosen["seed_fraction"], chosen["gain"]
            summary, raw[b] = trial(roots, carriers(roots, seed, cfg, np), seed, gain, cfg, check, np)
            dev[b] = summary | {"status": "locked_archive_replay"}
        if digest(lockfile) != stamp or "torch" in sys.modules:
            raise ValueError("CAL lock changed during replay or Torch loaded unexpectedly")
        check()
    evidence = args.output / "private_dev.json"
    write_new(evidence, raw)
    write_new(args.output / "aggregate.json", {
        "experiment": EXPERIMENT, "source_manifest_sha256": digest(MANIFEST), "status": "geometry_complete",
        "calibration_lock_sha256": stamp, "private_dev_sha256": digest(evidence), "fixture": verify,
        "calibration": {b: {k: v for k, v in record.items() if k != "candidates"} for b, record in lock.items()},
        "cal_faithful_joint_root_grid": {b: {f"{r['seed_fraction']}/{r['gain']}": r["joint"]["faithful_roots"]
                                               for r in record["candidates"]} for b, record in lock.items()},
        "development": dev, "pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
        "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
        "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
    })


if __name__ == "__main__":
    main()
