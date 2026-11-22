"""Certify finite BF16 angular infeasibility independently of the ray search."""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

from experiments.nan_dose51.route import SIGNS, directions, patches
from experiments.nan_dose51.run import HERE as DOSE, focused
from experiments.nan_gauge42.run import ARCHIVE, READPATH
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bf16, bound, digest, write_new
from experiments.nan_route50.run import fit_packet, npz_fields


HERE = Path(__file__).resolve().parent
SCALES = (.5, .55, .65, .75, .85, .95, 1., 1.05, 1.15, 1.3, 1.5, 2.)


def next_neighbors(pivot, np):
    code = (pivot.view(np.uint32) >> 16).astype(np.int32)
    negative = code >= 0x8000
    zero = (code & 0x7fff) == 0
    down = np.where(zero, 0x8001, np.where(negative, code + 1, code - 1))
    up = np.where(zero, 1, np.where(negative, code - 1, code + 1))

    def decode(bits):
        return (bits.astype(np.uint32) << 16).view(np.float32)

    return decode(down), decode(up)


def squared(z):
    return sum(v * v for v in z)


def exact_infeasible(h, d, scale, cfg, np):
    center_approx = bf16(h + scale * d, np)
    lo, hi = next_neighbors(center_approx, np)
    if not np.isfinite([lo, center_approx, hi]).all():
        raise ValueError("Nonfinite BF16 probe neighborhood")
    s = Fraction(scale)
    R = Fraction(str(cfg["phase1_dose_fraction"]))
    e = Fraction(str(cfg["dose_relative_tolerance"]))
    c = Fraction(str(cfg["minimum_direction_cosine"]))
    hq, dq = [list(map(Fraction, v.tolist())) for v in (h, d)]
    h2, d2 = squared(hq), squared(dq)
    m = Fraction(0)
    for base, direction, left, mid, right in zip(hq, dq, lo.tolist(), center_approx.tolist(), hi.tolist()):
        a, b, c_hi = map(Fraction, (left, mid, right))
        center = base + s * direction
        if not a <= center <= c_hi:
            raise ValueError("Exact BF16 probe escaped its neighboring values")
        m += min((center - value)**2 for value in (a, b, c_hi))
    k = s*s*d2 - m
    if k < 0:
        raise ValueError("The clean BF16 state must be a reachable comparator")
    radii_sq = ((R*(1-e))**2*h2, (R*(1+e))**2*h2)
    return all((r2+k)**2 < 4*s*s*r2*d2*c*c for r2 in radii_sq)


def screen(h, d, cfg, np):
    radius = float(np.linalg.norm(h)) * cfg["phase1_dose_fraction"]
    norm = float(np.linalg.norm(d))
    if not radius or not norm:
        raise ValueError("No finite nonzero direction and radius")
    e = cfg["dose_relative_tolerance"]
    edges = [radius*(1-e), radius*(1+e)]
    scales = np.asarray(SCALES) * (radius / norm)
    center = h[None, :] + scales[:, None] * d
    m = ((bf16(center, np) - center)**2).sum(axis=1)
    k = np.maximum(0., (scales * norm)**2 - m)
    upper = np.maximum(*((r*r + k) / (2*scales*r*norm) for r in edges))
    idx = int(upper.argmin())
    return float(upper[idx]), float(scales[idx])


def route_states(backend, base, roster, check, np):
    pins = base["backends"][backend]
    location = ARCHIVE / backend
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pins["audit"]))
    manifest = json.loads(bound(location / "manifest.json", pins["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["contexts_checked"], audit["missing_contexts"])
            != ("passed", "completed", 240, 0) or manifest["target_parameter_sha256"] != pins["target"]
            or manifest["checkpoint_revision"] != base["revision"]):
        raise ValueError("Unaudited COMP14 archive")
    sites = ("residual_after_layer", "down_proj_input", "down_proj_output")
    for slot, row in roster.items():
        check()
        lock = audit["records"][slot]
        case = json.loads(bound(location / f"case_{slot:04d}.json", lock["case_sha256"]))
        if any(row[k] != case[k] or row[k] != lock[k] for k in ("root", "world", "split", "arm", "id")):
            raise ValueError("COMP14 row identity differs")
        state = f"case_{slot:04d}_states.npz"
        if case["state_file"] != state or case["state_file_sha256"] != lock["state_sha256"]:
            raise ValueError("COMP14 state binding differs")
        keys = {entry["site"]: entry["array_key"] for entry in case["state_index"]
                if (entry["loop"], entry["layer_zero_based"]) == (1, 21) and entry["site"] in sites}
        values = npz_fields(location / state, lock["state_sha256"], [keys[k] for k in sites], np)
        yield row, [values[keys[k]].astype(np.float32) for k in sites]


def verify_math(cfg, np):
    h = np.array([1., 1.], dtype=np.float64)
    d = np.array([1., 1.], dtype=np.float64)
    bound_value, scale = screen(h, d, cfg, np)
    assert bound_value < .98 and exact_infeasible(h, d, scale, cfg, np)
    feasible = cfg | {"phase1_dose_fraction": .00552427}
    d = np.array([1., 0.])
    bound_value, scale = screen(h, d, feasible, np)
    delta = np.array([1/128, 0.])
    assert abs(math.sqrt(float(delta @ delta)) / (math.sqrt(2) * .00552427) - 1) < .03
    assert bound_value >= 1. - 1e-12 and not exact_infeasible(h, d, scale, feasible, np)
    pivot = np.array([-1., -0., 0., 1.], dtype=np.float32)
    lo, hi = next_neighbors(pivot, np)
    assert np.all(lo < pivot) and np.all(pivot < hi)
    assert float(lo[0]) == -1-1/128 and float(hi[-1]) == 1+1/128


def analyze(backend, source, base, cfg, archive, prior, check, masks, np):
    _, bank, ci, eps, _, _ = fit_packet(backend, base, np)
    anchor = np.fromiter(map(float, prior["archive_and_route_closure"][backend]["selected_CAL_anchor"].split(":")[1]),
                         dtype=np.float64)
    counts = Counter()
    by_arm = {"learned": Counter(), "static": Counter()}
    by_root = Counter()
    for row, (h, x, y) in route_states(backend, base, source, check, np):
        effects, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
        _, got, _ = patches(h, effects, cfg, np)
        saved = archive[backend, row["root"], row["world"]]["arms"]
        for arm, effect in effects.items():
            for sign, order in SIGNS.items():
                check()
                key = f"{arm}/{sign}"
                if any(value != saved[key][name] for name, value in got[key].items()):
                    raise ValueError("Physical edit did not reproduce DOSE51")
                counts["signed_edits"] += 1
                by_arm[arm]["signed_edits"] += 1
                if got[key]["direction_cosine"] >= cfg["minimum_direction_cosine"]:
                    counts["observed_ray_angle_pass"] += 1
                    by_arm[arm]["observed_ray_angle_pass"] += 1
                    continue
                counts["observed_ray_angle_failure"] += 1
                by_arm[arm]["observed_ray_angle_failure"] += 1
                d = effect * order
                upper, scale = screen(h.astype(np.float64), d, cfg, np)
                if upper < cfg["minimum_direction_cosine"] and exact_infeasible(h.astype(np.float64), d, scale, cfg, np):
                    counts["certified_impossible"] += 1
                    by_arm[arm]["certified_impossible"] += 1
                    by_root[row["root"]] += 1
    return {"counts": counts, "by_arm": by_arm,
            "roots_with_certificate": len(by_root), "roots_all_eight_signs_certified": sum(v == 8 for v in by_root.values())}


def source_lock():
    files = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "run.py")]
    files += [DOSE / name for name in ("SOURCE_MANIFEST.json", "route.py", "run.py", "registration.json")]
    repo = HERE.parents[1]
    return {p.relative_to(repo).as_posix(): digest(p) for p in files}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--dose-archive", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    manifest = HERE / "SOURCE_MANIFEST.json"
    if args.freeze:
        write_new(manifest, source_lock())
        return
    if not args.dose_archive or not args.output:
        cli.error("--dose-archive and --output are required")
    if not __debug__ or json.loads(manifest.read_bytes()) != source_lock():
        raise ValueError("LATTICE53 source freeze changed")
    cfg = json.loads((DOSE / "registration.json").read_bytes())
    base = json.loads((READPATH / "registration.json").read_bytes())
    receipt_path = args.dose_archive / "aggregate.json"
    prior = json.loads(receipt_path.read_bytes())
    if prior["source_manifest_sha256"] != digest(DOSE / "SOURCE_MANIFEST.json"):
        raise ValueError("DOSE51 archive is for a different analysis freeze")
    raw = json.loads(bound(args.dose_archive / "private.json", prior["private_receipt_sha256"]))
    source = focused(base)
    archive = {(r["backend"], r["root"], r["world"]): r for r in raw}
    if not len(raw) == len(archive) == 2 * len(source):
        raise ValueError("Unexpected DOSE51 row count")
    with lease(cfg) as (check, samples):
        import numpy as np
        from experiments.nan_fit07.pointwise import masks

        verify_math(cfg, np)
        results = {}
        for backend in base["backends"]:
            results[backend] = analyze(backend, source, base, cfg, archive, prior, check, masks, np)
            print(backend, results[backend]["counts"], flush=True)
        if "torch" in sys.modules:
            raise RuntimeError("Geometry audit imported Torch")
        check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, {
        "experiment": "NAN-LATTICE-53", "per_backend": results, "source_manifest_sha256": digest(manifest),
        "dose51_aggregate_sha256": digest(receipt_path), "new_pretrained_forwards": 0, "formal_comparisons": 0,
        "sealed_test_accessed": False, "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
        "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
    })


if __name__ == "__main__":
    main()
