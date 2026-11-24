"""Bound any BF16 edit's alignment to the original ROUND54 route contrast."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

from experiments.nan_dose51.route import SIGNS, directions
from experiments.nan_lattice53.run import exact_infeasible, route_states, screen
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bf16, bound, digest, write_new

from .run import DOSE, HERE, MANIFEST, READPATH, REPO, calibrations, rosters, seal


LOCK = HERE / "FEASIBILITY_MANIFEST.json"
PROTOCOL = HERE / "FEASIBILITY.md"


def freeze():
    files = [Path(__file__), PROTOCOL, MANIFEST, REPO / "experiments/nan_lattice53/SOURCE_MANIFEST.json"]
    return {path.relative_to(REPO).as_posix(): digest(path) for path in files}


def geometric_support_bound(h, raw, cfg, np):
    """Exact support relaxation ignoring the need to combine reachable coordinates."""
    from experiments.nan_lattice53.run import next_neighbors

    lower, upper = next_neighbors(h.astype(np.float32), np)
    gaps = np.where(raw > 0, upper.astype(np.float64) - h, h - lower.astype(np.float64))
    ratio = Fraction(str(cfg["phase1_dose_fraction"])) * (1 + Fraction(str(cfg["dose_relative_tolerance"])))
    max_radius2 = ratio**2 * sum(Fraction(x)**2 for x in h.tolist())
    weights = [Fraction(x)**2 for x in raw.tolist()]
    usable = sum(weight for weight, gap in zip(weights, gaps.tolist()) if Fraction(gap)**2 <= max_radius2)
    limit = Fraction(str(cfg["minimum_direction_cosine"]))
    return usable < limit**2 * sum(weights)


def audit(panel, backend, base, cfg, fit, source, saved, check, masks, np):
    bank, ci, eps, anchor = fit
    identities = {(row["root"], row["world"]): row for row in saved[panel] if row["backend"] == backend}
    out = []
    for found, (h, x, y) in route_states(backend, base, source, check, np):
        origin = h.astype(np.float64)
        if not np.array_equal(origin, bf16(origin, np)):
            raise ValueError("Native residual is not a finite BF16 vector")
        effects, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
        d = effects["learned"] - effects["static"]
        old = identities[found["root"], found["world"]]
        if float(np.linalg.norm(d)) != old["route_contrast_l2"]:
            raise ValueError("The recomputed contrast differs from ROUND54")
        for sign, polarity in SIGNS.items():
            check()
            signed = d * polarity
            support = geometric_support_bound(origin, signed, cfg, np)
            cosine_bound, scale = screen(origin, signed, cfg, np)
            lattice = cosine_bound < cfg["minimum_direction_cosine"] and exact_infeasible(origin, signed, scale, cfg, np)
            out.append({"root": found["root"], "world": found["world"], "sign": sign,
                        "support_certificate": support, "lattice_certificate": lattice})
    return {"registered": len(out),
            "support_certificates": sum(item["support_certificate"] for item in out),
            "lattice_certificates": sum(item["lattice_certificate"] for item in out),
            "either_certificate": sum(item["support_certificate"] or item["lattice_certificate"] for item in out),
            "all_four_signs_certified_roots": sum(all(item["support_certificate"] or item["lattice_certificate"]
                for item in out if item["root"] == root) for root in {r["root"] for r in out})}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--round-archive", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(LOCK, freeze())
        return
    if not (args.round_archive and args.output):
        cli.error("--round-archive and --output required")
    if not __debug__ or json.loads(LOCK.read_bytes()) != freeze() or json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("ROUND54 feasibility source freeze changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    registration = json.loads((HERE / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes())
    cfg = dose | {"minimum_direction_cosine": registration["minimum_reached_contrast_cosine"]}
    receipt = args.round_archive / "aggregate.json"
    previous = json.loads(receipt.read_bytes())
    if previous["source_manifest_sha256"] != digest(MANIFEST):
        raise ValueError("Parent calibration was run under a different freeze")
    saved = json.loads(bound(args.round_archive / "private.json", previous["private_sha256"]))
    with lease(registration) as (check, samples):
        import numpy as np
        from experiments.nan_fit07.pointwise import masks

        h = np.ones(2)
        assert geometric_support_bound(h, np.array([1., 0.]), cfg, np)
        assert not geometric_support_bound(h, np.array([1., 0.]), cfg | {"phase1_dose_fraction": .01}, np)
        fits = calibrations(base, masks, np)
        results = {panel: {backend: audit(panel, backend, base, cfg, fit, roster, saved, check, masks, np)
                           for backend, fit in fits.items()}
                   for panel, roster in zip(("fit", "check"), rosters(base))}
        if "torch" in sys.modules:
            raise RuntimeError("Geometric audit imported Torch")
        check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {"experiment": "NAN-ROUND-54-FEASIBILITY", "source_sha256": digest(LOCK),
              "round54_aggregate_sha256": digest(receipt), "panel_results": results,
              "phase1_dose_fraction": cfg["phase1_dose_fraction"], "target_cosine": cfg["minimum_direction_cosine"],
              "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
              "peak_private_bytes": max(item["private_bytes"] for item in samples)}
    write_new(args.output, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
