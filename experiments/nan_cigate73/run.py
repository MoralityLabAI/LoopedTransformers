"""Replay frozen pointwise BPD gates on immutable native attention states."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_attnspan67.run import SITES, make_pair, read_archive, read_banks
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .account import TERMS, route, verify


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIT = REPO / "runs/nan_ranksite13"
ATTN = REPO / "experiments/nan_attnspan67/registration.json"
BASE = REPO / "experiments/nan_readpath37/registration.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
MODES = ("dynamic", "static_CAL")
EPS = 1e-6


def freeze():
    files = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md", "account.py", "run.py")]
    files += [BASE, ATTN] + [REPO / n for n in (
        "experiments/nan_readpath37/run.py", "experiments/nan_carry63/algebra.py",
        "experiments/nan_attnspan67/run.py", "experiments/nan_phase35/run.py",
        "experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py",
        "experiments/nan_fit07/pointwise.py", "mechinterp/nanbeige_bpd_gym/native_components.py")]
    return {"files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in files},
            "experiment": "NAN-CIGATE-73"}


def controller(name, audited_sha, independent_factors, np, masks):
    folder = FIT / "conditions" / name
    audit = json.loads(bound(FIT / "audit" / f"{name}.json", audited_sha))
    rec = json.loads(bound(folder / "receipt.json", audit["fit_receipt_sha256"]))
    depth = rec["passes"]
    recipe = rec["training_contract"]["ablation"]
    cfg = rec["final_checkpoint"]
    actual = (recipe["mode"], recipe["sphere_rescale"], recipe["eps"])
    want = ("zero", False, EPS) if depth == 1 else ("pass_mean", True, EPS)
    if (depth not in (1, 2) or actual != want or rec["target_parameter_sha256_unchanged"] is None
            or not rec["native_all_one_boundary_states_bit_exact"]):
        raise ValueError("Unexpected audited ablation or native reference contract")
    with np.load(BytesIO(bound(folder / cfg["file"], audit["checkpoint_sha256"])), allow_pickle=False) as archive:
        fields = ("V", "U", "mean_comp_acts", "mean_write_rms", "calibrated_passes",
                  "ci__proj_in.weight", "ci__proj_in.bias", "ci__out_proj.weight", "ci__out_proj.bias")
        bank = {key: archive[key].copy() for key in fields}
    shapes = ((6144, 8), (8, 3072), (depth, 8), (depth,), (depth,), (32, 6144), (32,), (8, 32), (8,))
    if (cfg["sha256"] != audit["checkpoint_sha256"] or
            any(bank[key].shape != shape or not np.isfinite(bank[key]).all()
                for key, shape in zip(fields, shapes, strict=True)) or not bank["calibrated_passes"].all()
            or any(not np.array_equal(bank[k], match) for k, match in zip(("V", "U"), independent_factors))):
        raise ValueError("Controller bank differs from its audited factors")
    for k in fields:
        bank[k] = bank[k].astype(np.float64)
    with np.load(BytesIO(bound(folder / "CI_replay_local.npz", rec["CI_replay_sha256"])), allow_pickle=False) as replay:
        computed = masks(replay["X"], bank, cfg["CI_config"])
        np.testing.assert_allclose(computed, replay["masks"], atol=1e-5, rtol=2e-5)
        max_error = float(np.max(abs(computed - replay["masks"])))
    diagnostic = json.loads(bound(folder / "diagnostic_local.json", rec["diagnostic_artifact_sha256"]["diagnostic_local.json"]))
    static = np.asarray(diagnostic["static_CAL_mask"], dtype=np.float64)
    if (static.shape != (depth, 8) or not np.isfinite(static).all() or static.min() < 0 or static.max() > 1):
        raise ValueError("Static FIT07 CAL controller is malformed")
    custody = {"fit_receipt_sha256": audit["fit_receipt_sha256"],
               "checkpoint_sha256": audit["checkpoint_sha256"], "archived_CI_max_abs_error": max_error,
               "fit_steps": rec["optimizer_steps_completed"], "fit_depth": depth}
    return bank, cfg["CI_config"], static, custody


def evaluate(worlds, pair, bundle, seed_factors, erased, np, masks):
    bank, cfg, static, meta = bundle
    depth = meta["fit_depth"]
    v, u = bank["V"], bank["U"]
    seed_v, seed_u = seed_factors
    a, b = [worlds[key][2] for key in ("A", "B")]
    permutation = [7, 3, 6, 2, 5, 1, 4, 0]
    gauge = np.array([.125, .25, .5, 1., 2., 4., 8., 16.])
    results = []
    for t, record in enumerate(pair["phase"]):
        phase = t + 1
        x, native = (np.stack([source[phase, 21, site] for source in (a, b)]).astype(np.float64) for site in SITES[1:])
        direction = record["direction"]
        gate = masks(x, bank, cfg)
        if gate.shape != (2, 8) or not np.isfinite(gate).all():
            raise RuntimeError("Controller did not emit finite paired physical masks")
        mean = bank["mean_comp_acts"][t] if depth == 2 else np.zeros(8)
        rms = bank["mean_write_rms"][t] if depth == 2 else None
        original = np.broadcast_to(static[min(t, depth - 1)], gate.shape)
        noop, null = route(x, native, np.ones_like(gate), v, u, mean, rms, EPS, np)
        np.testing.assert_array_equal(noop, native)
        assert all(not np.count_nonzero(i) for i in null.values())
        output = {}
        for name, m in zip(MODES, (gate, original), strict=True):
            reached, terms = route(x, native, m, v, u, mean, rms, EPS, np)
            transformed, other = route(x, native, m[:, permutation], v[:, permutation] * gauge,
                                       u[permutation] / gauge[:, None], mean[permutation] * gauge, rms, EPS, np)
            np.testing.assert_array_equal(transformed, reached)
            for key in TERMS:
                np.testing.assert_allclose(other[key], terms[key], atol=1e-11, rtol=1e-11)
            added = reached - native
            direct = float((added[0] - added[1]) @ direction)
            projected = {key: float(part @ direction) for key, part in terms.items()}
            if not np.isclose(sum(projected.values()), direct, atol=1e-10, rtol=1e-10):
                raise RuntimeError("Projected reader contrast failed four-term closure")
            if erased and (not np.array_equal(reached[0], reached[1]) or any(projected.values())):
                raise RuntimeError("Erased twin generated a controller contrast")
            baseline = np.linalg.norm(native, axis=1)
            output[name] = {
                "edit": direct, "terms": projected,
                "mean_world_relative_write_dose": float(np.mean(np.linalg.norm(added, axis=1)/np.maximum(baseline, 1e-12))),
                "world_gate_mean_abs_difference": float(np.mean(abs(m[0] - m[1]))),
                "gate_zero": int((m == 0).sum()), "gate_one": int((m == 1).sum()),
            }
        results.append({"out_of_fit_depth": phase > depth, "native_attention": record["direct_write"],
                        "initial_VU": float(((record["x"] @ seed_v) @ seed_u) @ direction),
                        "learned_VU": float(((record["x"] @ v) @ u) @ direction), **output})
    return results


def growth(rows, getter):
    return fmean(getter(r[1]) - getter(r[0]) for r in rows)


def summarize(rows):
    out = {key + "_gain": growth(rows, lambda p: p[key]) for key in ("native_attention", "initial_VU", "learned_VU")}
    for name in MODES:
        out[name] = {
            "edit_gain": growth(rows, lambda p: p[name]["edit"]),
            "mean_edit_by_pass": [fmean(p[t][name]["edit"] for p in rows) for t in (0, 1)],
            "edit_gain_by_term": {k: growth(rows, lambda p: p[name]["terms"][k]) for k in TERMS},
            **{key+"_by_pass": [fmean(p[t][name][key] for p in rows) for t in (0, 1)]
               for key in ("mean_world_relative_write_dose", "world_gate_mean_abs_difference")},
            **{key+"_count_by_pass": [sum(p[t][name][key] for p in rows) for t in (0, 1)]
               for key in ("gate_zero", "gate_one")},
        }
    out["dynamic_minus_static_gain"] = out["dynamic"]["edit_gain"] - out["static_CAL"]["edit_gain"]
    out["roots"] = len(rows)
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable Python assertions")
    if args.freeze:
        write_new(MANIFEST, freeze())
        return
    if not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes()) != freeze():
        cli.error("Supply the frozen source, snapshot and fresh output directory")
    cfg, base = (json.loads(p.read_bytes()) for p in (ATTN, BASE))
    if args.snapshot.name != base["revision"] or base["revision"] != cfg["revision"]:
        raise ValueError("Checkpoint identity differs from the archive")
    with lease(cfg) as (check, samples):
        assert (version("numpy"), version("tokenizers")) == ("2.5.5", "0.20.3")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(np)
        config = json.loads(bound(args.snapshot/"config.json", base["snapshot_pins"]["config.json"]))
        if config["attention_bias"]:
            raise ValueError("The projection accounting requires a bias-free attention module")
        bound(args.snapshot/"tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot/"tokenizer.json"))
        labels = {s: tokenizer.encode(s, add_special_tokens=False).ids for s in "ABCDEFGHIJKLMNOP"}
        assert all(len(v) == 1 for v in labels.values()) and len({v[0] for v in labels.values()}) == 16
        labels = {name: ids[0] for name,ids in labels.items()}
        roster, _ = joined(base)
        args.work.mkdir(parents=True, exist_ok=False)
        banks = read_banks("bf16", 21, cfg, base, np)
        fits = {phase: controller(f"t{phase}_layer21_o_proj_rank8", cfg["ranksite_audits"][str(phase)],
                                  banks[f"attn_bpd_t{phase}"], np, masks) for phase in (1, 2)}
        with (args.snapshot/SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            archive, error = read_archive("bf16", (21,), base, roster, head, check, np)
            summaries = {}
            for phase, fit in fits.items():
                groups = defaultdict(list)
                rows = []
                for (split, arm, root), worlds in sorted(archive.items()):
                    check()
                    erased = arm == "focused_erased"
                    pair = make_pair(worlds, 21, labels, head, erased, np)
                    result = evaluate(worlds, pair, fit, banks["svd8"], erased, np, masks)
                    rows.append({"split": split, "arm": arm, "root": root, "phases": result})
                    groups[split, arm].append(result)
                write_new(args.work/f"T{phase}_private.json", rows)
                summaries[f"T{phase}"] = fit[-1] | {"groups": {"/".join(key): summarize(group) for key,group in groups.items()}}
        assert "torch" not in sys.modules
    assert json.loads(MANIFEST.read_bytes()) == freeze()
    write_new(args.work/"aggregate.json", {
        "experiment": "NAN-CIGATE-73", "status": "passed", "source_manifest_sha256": digest(MANIFEST),
        "archive_pins": {k: base["backends"]["bf16"][k] for k in ("audit", "manifest", "target")},
        "native_head_max_abs_error": error, "roots_by_split": base["roots"], "archived_worlds": len(roster),
        "fits": summaries, "pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_resident_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })
    print({name: fit["groups"]["development/focused"] for name, fit in summaries.items()}, flush=True)


if __name__ == "__main__":
    main()
