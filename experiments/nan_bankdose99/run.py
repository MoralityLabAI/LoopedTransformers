"""Check registered rounded bank geometry on closed, exposed COMP14 states."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_packedcal97 import lane
from experiments.nan_packedcal97.source import config
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bf16, bound, case_states, digest, joined, write_new
from experiments.nan_refit49.geometry import cos
from experiments.nan_refit49 import run as refit

from .geometry import METHODS, bind_banks, materialize, plan


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
BASE = refit.READPATH / "registration.json"
EXPERIMENT = "NAN-BANKDOSE-99"
BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 384 * 2**20, "maximum_seconds": 180}


def registered():
    runtime = lane.verified()
    lane.commissioned(runtime)
    prior = refit.registration()
    if prior != json.loads(refit.MANIFEST.read_bytes()):
        raise ValueError("The IO18 REFIT49 freeze differs")
    for name, hash_value in prior["files_sha256"].items():
        bound(refit.REPO / name, hash_value)
    local = (".gitattributes", "__init__.py", "PROTOCOL.md", "geometry.py", "instrument.py", "run.py", "verify.py")
    paths = [HERE / name for name in local] + [refit.MANIFEST, lane.MANIFEST, lane.PROOF]
    cfg = config()
    dose = {key: cfg[key] for key in ("phase1_dose_fraction", "dose_relative_tolerance", "minimum_direction_cosine",
                                     "max_contrast_dose_gap_fraction")}
    return {"experiment": EXPERIMENT, "dose": dose, "archive_budget": BUDGET,
            "files_sha256": {p.relative_to(refit.REPO).as_posix(): digest(p) for p in paths}}


def captured(backend, base, roster, check, np):
    pin = base["backends"][backend]
    folder = refit.SOURCE / backend
    audit = json.loads(bound(refit.SOURCE / "audits" / f"{backend}.json", pin["audit"]))
    manifest = json.loads(bound(folder / "manifest.json", pin["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"])
            != ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or manifest["target_parameter_sha256"] != pin["target"] or manifest["checkpoint_revision"] != base["revision"]):
        raise ValueError("COMP14 state archive lacks the native terminal binding")
    roots = {}
    for i, row in roster.items():
        check()
        key = f"case_{i:04d}"
        verified = audit["records"][i]
        case = json.loads(bound(folder / f"{key}.json", verified["case_sha256"]))
        if (any(case[k] != row[k] or verified[k] != row[k] for k in ("id", "root", "split", "arm", "world"))
                or case["state_file"] != f"{key}_states.npz" or case["state_file_sha256"] != verified["state_sha256"]):
            raise ValueError("A captured physical state differs from the frozen source")
        state = case_states(case, bound(folder / case["state_file"], verified["state_sha256"]), np)
        pair = roots.setdefault((row["split"], row["root"]), {})
        if row["world"] in pair:
            raise ValueError("Repeated captured counterfactual world")
        pair[row["world"]] = (state[1, "down_proj_input"].astype(np.float64), state[1, "residual_after_layer"])
    if any(set(worlds) != {"A", "B"} for worlds in roots.values()):
        raise ValueError("Missing captured counterfactual state")
    return roots


def natural_reach(before, directions, np):
    """Diagnose the unscaled update at the same physical BF16 boundary."""
    result = {}
    for name in ("t1_update", "t2_update"):
        action = directions[name]
        exact = float(np.linalg.norm(action))
        pos, neg = (bf16(before.astype(np.float64) + sign * action, np).astype(np.float64) for sign in (1, -1))
        central = (pos - neg) / 2
        result[name] = {"positive_is_noop": np.array_equal(pos, before), "negative_is_noop": np.array_equal(neg, before),
                        "half_vector_coordinates": int(np.count_nonzero(central)),
                        "half_vector_cosine": cos(central, action, np),
                        "half_vector_norm_over_ideal": float(np.linalg.norm(central) / exact) if exact > 0 else None}
    return result


def evaluate(backend, base, banks, roster, cfg, check, np):
    def reached_median(values):
        accepted = [v for v in values if v is not None]
        return median(accepted) if accepted else None

    protected = {}
    summary = {}
    for (split, root), pair in sorted(captured(backend, base, roster, check, np).items()):
        cells = protected.setdefault(split, {}).setdefault(root, {})
        for world, peer in (("A", "B"), ("B", "A")):
            check()
            x, before = pair[world]
            directions = materialize(pair[peer][0] - x, banks, root, world, np)
            record, _ = plan(before, directions, cfg, np)
            cells[world] = record | {"natural_reach": natural_reach(before, directions, np)}
    for split, roots in protected.items():
        if len(roots) != base["roots"][split]:
            raise ValueError("The finite geometry replay changed the exposed source roster")
        worlds = [arm for pair in roots.values() for arm in pair.values()]
        gain = {method: reached_median(cell["cases"][method]["positive"]["nominal_over_natural"] for cell in worlds)
                for method in METHODS[:-1]}
        names = next(iter(worlds))["qualified_panels"]
        natural = {}
        for method in ("t1_update", "t2_update"):
            rows = [w["natural_reach"][method] for w in worlds]
            natural[method] = {
                "zero_signed_edits": sum(x[k] for x in rows for k in ("positive_is_noop", "negative_is_noop")),
                "zero_half_vectors": sum(x["half_vector_coordinates"] == 0 for x in rows),
                "median_half_vector_coordinates": reached_median(x["half_vector_coordinates"] for x in rows),
                "median_defined_half_vector_cosine": reached_median(x["half_vector_cosine"] for x in rows),
                "median_half_vector_norm_over_ideal": reached_median(x["half_vector_norm_over_ideal"] for x in rows),
            }
        summary[split] = {
            "roots": len(roots), "worlds": len(worlds),
            "joint_dose_roots": {k: sum(all(w["qualified_panels"][k] for w in pair.values()) for pair in roots.values())
                                 for k in names},
            "median_nominal_gain": gain,
            "median_t1_t2_full_cosine": reached_median(w["full_fit_direction_cosine"] for w in worlds),
            "median_t1_t2_update_cosine": reached_median(w["learned_direction_cosine"] for w in worlds),
            "signed_eligible_edits": sum(c["eligible"] for w in worlds for m in w["cases"].values() for c in m.values()),
            "unique_eligible_edits": sum(w["distinct_rounded_edits"] for w in worlds),
            "exploratory_unscaled_update_rounding": natural,
        }
    return summary, protected


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
        cli.error("Require a verified source freeze and a fresh output directory")
    base = json.loads(BASE.read_bytes())
    with lease(BUDGET) as (check, samples):
        if "torch" in sys.modules or version("numpy") != "2.5.5":
            raise ValueError("The pinned archive NumPy runtime is required")
        import numpy as np

        selected, _ = joined(base)
        roster = {i: p for i, p in selected.items() if p["arm"] == "focused"}
        fits = refit.load_fits(base, np)
        out, private = {}, {}
        for backend in base["backends"]:
            banks = bind_banks(fits[f"{backend}/T1"], fits[f"{backend}/T2"], np)
            out[backend], private[backend] = evaluate(backend, base, banks, roster, seal["dose"], check, np)
    assert "torch" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=False)
    witness = args.output / "private_geometry.json"
    write_new(witness, private)
    write_new(args.output / "aggregate.json", {"experiment": EXPERIMENT, "status": "geometry_verified",
              "source_manifest_sha256": digest(MANIFEST), "private_geometry_sha256": digest(witness),
              "per_backend": out, "formal_comparisons": 0, "pretrained_forwards": 0, "sealed_test_accessed": False,
              "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
              "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples)})


if __name__ == "__main__":
    main()
