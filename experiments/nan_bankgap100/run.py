"""Recompute full-fit geometric resolution from the hash-bound COMP14 archive."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_bankdose99.geometry import bind_banks, materialize
from experiments.nan_bankdose99 import run as parent
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bound, digest, joined, write_new
from experiments.nan_refit49 import run as refit

from .rounding import FULL, GAINS, METHODS, compare
from .verify import fixture


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
OLD = refit.REPO / "reports/nanbeige_bpd_gym/nan_bankdose99_results.json"
REPAIR = refit.REPO / "experiments/nan_shellrepair90/SOURCE_MANIFEST.json"
EXPERIMENT = "NAN-BANKGAP-100"
BUDGET = parent.BUDGET | {"maximum_seconds": 240}


def registered():
    source = parent.registered()
    if source != json.loads(parent.MANIFEST.read_bytes()):
        raise ValueError("BANKDOSE99 source differs from its frozen archive receipt")
    if json.loads(OLD.read_bytes())["source_manifest_sha256"] != digest(parent.MANIFEST):
        raise ValueError("The old full-fit dose witness changed")
    for file, pin in json.loads(REPAIR.read_bytes())["files_sha256"].items():
        bound(refit.REPO / file, pin)
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "rounding.py", "run.py", "verify.py")]
    paths += [OLD, REPAIR, parent.MANIFEST, parent.HERE / "CPU_VERIFICATION.json"]
    return {"experiment": EXPERIMENT, "gains": list(GAINS), "dose": source["dose"], "budget": BUDGET,
            "files_sha256": {p.relative_to(refit.REPO).as_posix(): digest(p) for p in paths}}


def world_pass(rows, key, cfg):
    candidates = [row["methods"][key] for row in rows]
    lengths = [w["length"] for row in candidates for w in row["writes"]]
    dose = all(c["dose_qualified"] for c in candidates) and (
        max(lengths) - min(lengths) <= candidates[0]["nominal"] * cfg["max_contrast_dose_gap_fraction"])
    return dose, dose and all(c["qualified"] for c in candidates)


def summarize(roots, cfg):
    def center(rows):
        collected = [v for v in rows if v is not None]
        return median(collected) if collected else None

    detail = {}
    for key in METHODS:
        records = []
        dose_roots = faithful_roots = 0
        for worlds in roots.values():
            all_worlds = []
            for world in worlds.values():
                sides = list(world.get("signs", {}).values())
                records.extend(s["methods"][key] for s in sides if key in s["methods"])
                if len(sides) == 2 and all(key in s["methods"] for s in sides):
                    all_worlds.append(world_pass(sides, key, cfg))
            dose_roots += len(all_worlds) == 2 and all(a for a, _ in all_worlds)
            faithful_roots += len(all_worlds) == 2 and all(b for _, b in all_worlds)
        detail[key] = {
            "signed_worlds": len(records), "dose_matched_both_world_roots": dose_roots,
            "faithful_both_world_roots": faithful_roots,
            "exact_alias_signed_worlds": sum(r["exact_state_alias"] for r in records),
            "faithful_signed_worlds": sum(r["qualified"] for r in records),
            "stencil_incompatible_signed_worlds": sum(r["stencil_certified_incompatible"] for r in records),
            "median_differential_cosine": center(r["differential"]["cosine"] for r in records),
            "median_differential_relative_error": center(r["differential_relative_error"] for r in records),
            "median_common_midpoint_relative_error": center(r["midpoint_error_relative_to_nominal"] for r in records),
        }
    statuses = Counter(w["status"] for worlds in roots.values() for w in worlds.values())
    carriers = Counter(s["carrier_mode"] for worlds in roots.values() for w in worlds.values()
                       for s in w.get("signs", {}).values())
    obstructions = {name: sum(s["coordinate_obstructions"][name] is not None for worlds in roots.values()
                              for w in worlds.values() for s in w.get("signs", {}).values()) for name in FULL}
    return {"roots": len(roots), "world_status": statuses, "signed_carrier_status": carriers,
            "coordinate_obstructed_signed_worlds": obstructions, "methods": detail}


def evaluate(backend, base, banks, roster, cfg, prior, check, np):
    private = {}
    for (split, root), pair in sorted(parent.captured(backend, base, roster, check, np).items()):
        group = private.setdefault(split, {}).setdefault(root, {})
        for world, peer in (("A", "B"), ("B", "A")):
            check()
            x, before = pair[world]
            direction = materialize(pair[peer][0] - x, banks, root, world, np)
            group[world] = compare(before, direction, cfg, np)
    out = {}
    for split, roots in private.items():
        panel = summarize(roots, cfg)
        control = panel["methods"]["original_rays"]["dose_matched_both_world_roots"]
        if (control != prior[split]["joint_dose_roots"]["t2_fit_vs_t1_fit"]
                or len(roots) != base["roots"][split]):
            raise ValueError("The original T1/T2 dose controls or source roster differ")
        out[split] = panel | {"original_bankdose99_dose_count_exact": True}
    return out, private


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    seal = registered()
    if args.freeze:
        write_new(MANIFEST, seal)
        return
    if not args.output or json.loads(MANIFEST.read_bytes()) != seal:
        cli.error("Require the frozen geometry replay and a fresh output")
    base = json.loads(parent.BASE.read_bytes())
    with lease(BUDGET) as (check, samples):
        if "torch" in sys.modules or version("numpy") != "2.5.5":
            raise ValueError("The pinned archive-only runtime is required")
        import numpy as np

        cfg = seal["dose"]
        tests = fixture(cfg, np)
        prior = json.loads(OLD.read_bytes())["per_backend"]
        selected, _ = joined(base)
        roster = {i: p for i, p in selected.items() if p["arm"] == "focused"}
        fits = refit.load_fits(base, np)
        aggregate, private = {}, {}
        for backend in base["backends"]:
            banks = bind_banks(fits[f"{backend}/T1"], fits[f"{backend}/T2"], np)
            aggregate[backend], private[backend] = evaluate(backend, base, banks, roster, cfg, prior[backend], check, np)
    assert "torch" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=False)
    witness = args.output / "private_geometry.json"
    write_new(witness, private)
    write_new(args.output / "aggregate.json", {
        "experiment": EXPERIMENT, "status": "archive_geometry_complete", "source_manifest_sha256": digest(MANIFEST),
        "private_geometry_sha256": digest(witness), "per_backend": aggregate, "fixture": tests,
        "pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
        "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
        "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
    })


if __name__ == "__main__":
    main()
