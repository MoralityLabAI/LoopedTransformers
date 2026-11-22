"""Audit continuous joint constraints and exact rounded NF4 heads on CAL."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dose51.route import SIGNS, directions, patches
from experiments.nan_drift57.geometry import pair_head
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_lowhead59.geometry import fine_basis, head_spectrum
from experiments.nan_lowhead59.run import ACTIVE_AGG, HERE as LOW, PRIOR as SHAM56
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations, rosters
from experiments.nan_sham56.minimax import atlas, derivative_columns
from experiments.nan_sham56.run import HERE as SHAM, physical_head_closure

from .geometry import solve, verify


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
METHOD = "dual_primal"
PARENT = REPO / "reports/nanbeige_bpd_gym/nan_headnull60_results.json"
PARENT_LOCK = REPO / "experiments/nan_headnull60/SOURCE_MANIFEST.json"


def seal():
    files = [HERE / p for p in (".gitattributes", "__init__.py", "geometry.py", "registration.json", "run.py", "PROTOCOL.md")]
    files += [LOW / p for p in ("geometry.py", "run.py", "registration.json", "SOURCE_MANIFEST.json")]
    files += [PARENT, PARENT_LOCK, REPO / "experiments/nan_headnull60/optimize.py", SHAM56, ACTIVE_AGG, SHAM / "minimax.py", SHAM / "run.py", SHAM / "MINIMAX_MANIFEST.json",
              SHAM / "registration.json", ROUND / "registration.json", READPATH / "registration.json",
              DOSE / "registration.json", REPO / "experiments/nan_drift57/geometry.py"]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in files}


def measure(source, model, reference, baseline_sham, prior_null, tokens, head, base, cfg, low_cfg, sham_cfg, round_cfg, dose,
            spectral, check, masks, np):
    bank, ci, eps, anchor = model
    directory = ARCHIVE / "nf4"
    locked = json.loads(bound(ARCHIVE / "audits/nf4.json", base["backends"]["nf4"]["audit"]))
    slots = {(r["root"], r["world"]): i for i,r in source.items()}
    packets, columns = [], []
    for row, (h, x, y) in route_states("nf4", base, source, check, np):
        h = h.astype(np.float64)
        root, world = row["root"], row["world"]
        slot = slots[root, world]
        twin = source[slots[root, "B" if world == "A" else "A"]]
        pair = [tokens[twin["answer"]], tokens[row["answer"]]]
        case = json.loads(bound(directory / f"case_{slot:04d}.json", locked["records"][slot]["case_sha256"]))
        original_head = case["initial_phases"][0]
        if original_head["loop"] != 1 or pair[0] == pair[1]:
            raise ValueError("Wrong native readout")
        parts, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
        raw = parts["learned"] - parts["static"]
        basis_gradient = head.gamma.astype(np.float64) * (
            head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
        gradient = head_gradient(h, basis_gradient, head.eps, np)
        radius = float(np.linalg.norm(h)) * dose["phase1_dose_fraction"]
        tag = f"nf4/{root}/{world}"
        dense, original, _ = make_pair(h, raw, gradient, radius, None, tag, round_cfg, np)
        active, recorded, admitted = patches(h, {"contrast": dense["contrast"]}, dose, np)
        old = reference[root, world]["recipes"]["0.0075/dense"]["arm_dose"]
        if not admitted or any(any(old[k][f] != v for f,v in branch.items()) for k,branch in recorded.items()):
            raise ValueError("Previously qualified active state did not replay")
        axes, count = fine_basis(h, original, gradient, [p.after-p.before for p in active.values()],
                                 radius, tag, spectral, low_cfg, sham_cfg, np)
        columns.append(derivative_columns(head, h, axes, radius, np))
        active_profile = baseline_sham[root, world]["recipes"]["ambient_peak_minimax"]["arm_dose"]
        target = sum(active_profile[f"contrast/{s}"]["rms_logit_shift"] for s in SIGNS) / 2.
        packets.append((root, world, h, pair, original_head["score"], dense["contrast"], original, axes, count, active, target))
    derivative = atlas(head, columns, dose["block_rows"], check, np)
    width = low_cfg["spectral_axes"] + low_cfg["ambient_axes"]
    rows, states, refs, labels, comparisons, addresses = [], [], [], [], [], []

    def add(value, pair, reference=None):
        pos = len(states)
        refs.append(pos if reference is None else reference)
        states.append(value)
        labels.append(pair)
        return pos

    for i, (root, world, h, pair, saved, contrast, original, axes, support, active, target) in enumerate(packets):
        check()
        seeds = prior_null[root, world]["prediction"]["excluded_rows"]
        sham, design = solve(derivative[:, i*width:(i+1)*width].copy(), seeds, target, cfg, check, np)
        clean = add(h, pair)
        active_idx = {key: add(p.after, pair, clean) for key,p in active.items()}
        edits, entry = match(h, {"contrast": contrast, "sham": axes @ sham}, original, round_cfg, dose, np)
        if any(not np.array_equal(edits[key].after, p.after) for key,p in active.items()):
            raise ValueError("The sham changed a fixed active edit")
        entry["support"] = support
        if entry["physical_gate"]:
            entry["head_indices"] = active_idx | {k: add(edit.after, pair, clean)
                                                 for k,edit in edits.items() if k not in active_idx}
            entry["pairs"] = {}
            for sign in SIGNS:
                comparisons.append((clean, active_idx[f"contrast/{sign}"], entry["head_indices"][f"sham/{sign}"]))
                addresses.append((entry, sign))
        rows.append({"root": root, "world": world, "clean": clean, "citation": saved,
                     "fixed_active_head_indices": active_idx, "prediction": design,
                     "recipes": {METHOD: entry}})
    del derivative
    live, pairs = pair_head(head, np.stack(states), np.asarray(refs), np.asarray(labels),
                            np.asarray(comparisons, dtype=np.int64).reshape(-1, 3), dose["block_rows"], check, np)
    for (entry, sign), record in zip(addresses, pairs, strict=True):
        entry["pairs"][sign] = record
    custody = physical_head_closure(rows, live, reference, dose, base)
    for row in rows:
        entry = row["recipes"][METHOD]
        entry["rms_match"] = entry["full_pass"] and all(
            abs(v["signed_rms_gap"]) <= cfg["rms_gap_limit"] for v in entry["pairs"].values())
    return rows, custody


def summary(rows):
    roots = {r["root"] for r in rows}
    if len(rows) != 8 or len(roots) != 4 or any({r["world"] for r in rows if r["root"] == root} != {"A", "B"} for root in roots):
        raise ValueError("Incomplete fixed CAL roster")
    options = [r["recipes"][METHOD] for r in rows]
    physical = [p for p in options if p["physical_gate"]]
    passed = [p for p in physical if p["full_pass"]]
    signed = [v for p in passed for v in p["pairs"].values()]
    pred = [r["prediction"] for r in rows]
    return {"worlds": len(options), "physical_gate": len(physical), "full_pass": len(passed),
            "rms_matched_worlds": sum(p["rms_match"] for p in options),
            "complete_root_full_gate": sum(all(r["recipes"][METHOD]["full_pass"] for r in rows if r["root"] == root)
                                           for root in roots),
            "complete_root_rms_match": sum(all(r["recipes"][METHOD]["rms_match"] for r in rows if r["root"] == root)
                                           for root in roots),
            "continuous_joint_witnesses": sum(r["maximum_constraint_ratio"] <= 1. for r in pred),
            "continuous_certified_infeasible_worlds": sum(r["certified_continuous_infeasible"] for r in pred),
            "median_predicted": {k: median(r[k] for r in pred) for k in ("maximum_constraint_ratio",
                "predicted_peak", "predicted_rms", "rms_upper", "dual_numeric_lower_rms")},
            "rounded_medians_among_full_pass_signs": {k: median(values) if (
                values := [r[k] for r in signed if r[k] is not None]) else None
                for k in ("active_rms", "sham_rms", "signed_rms_gap", "head_shift_cosine", "head_shift_rms_distance")},
            "max_rounded_sham_peak_on_pass": max((p["arm_dose"][f"sham/{s}"]["max_abs_logit_delta"]
                                                 for p in passed for s in SIGNS), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--active", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--null-parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("DUALHEAD61 freeze changed")
    if not all((args.snapshot,args.active,args.prior,args.null_parent,args.output)):
        parser.error("--snapshot, --active, --prior, --null-parent, and --output required")
    if digest(args.active / "aggregate.json") != digest(ACTIVE_AGG) or digest(args.prior / "aggregate.json") != digest(SHAM56):
        raise ValueError("Original active or sham reference changed")
    if digest(args.null_parent / "aggregate.json") != digest(PARENT):
        raise ValueError("The fixed worst-row parent changed")
    prior = json.loads(PARENT.read_bytes())
    if prior["source_manifest_sha256"] != digest(PARENT_LOCK):
        raise ValueError("The worst-row freeze does not match its receipt")
    previous = json.loads(SHAM56.read_bytes())
    active = json.loads(ACTIVE_AGG.read_bytes())
    reference = json.loads(bound(args.active / "private.json", active["private_sha256"]))
    cfg = json.loads((HERE / "registration.json").read_bytes())
    low = json.loads((LOW / "registration.json").read_bytes())
    sham = json.loads((SHAM / "registration.json").read_bytes())
    rounded = json.loads((ROUND / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes()) | {"phase1_dose_fraction": sham["phase1_dose_fraction"]}
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Tokenizer snapshot mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    with lease(cfg) as (check, samples):
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(np)
        weights = calibrations(base, masks, np)["nf4"]
        path = args.snapshot / "tokenizer.json"
        bound(path, base["snapshot_pins"]["tokenizer.json"])
        tk = Tokenizer.from_file(str(path))
        encoded = {c: tk.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(v) != 1 for v in encoded.values()) or len({x[0] for x in encoded.values()}) != 16:
            raise ValueError("Bare token ids changed")
        tokens = {k: v[0] for k,v in encoded.items()}
        outcomes = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            spectrum, metrics = head_spectrum(head, low, dose["block_rows"], check, np)
            bank_path = args.output / "spectral_proposal_private.npy"
            with bank_path.open("xb") as stream:
                np.save(stream, spectrum, allow_pickle=False)
            print("Finished frozen vocabulary spectrum", flush=True)
            for panel, sources in zip(("fit", "check"), rosters(base)):
                refs = {(r["root"], r["world"]): r for r in reference if r["panel"] == panel}
                baseline = json.loads(bound(args.prior / f"{panel}_private.json", previous["panels"][panel]["private_sha256"]))
                baseline = {(r["root"], r["world"]): r for r in baseline}
                raw = json.loads(bound(args.null_parent / f"{panel}_private.json", prior["panels"][panel]["private_sha256"]))
                observed = {(r["root"], r["world"]): r for r in raw}
                rows, custody = measure(sources, weights, refs, baseline, observed, tokens, head, base, cfg, low, sham,
                                        rounded, dose, spectrum, check, masks, np)
                file = args.output / f"{panel}_private.json"
                write_new(file, rows)
                outcomes[panel] = summary(rows) | {"custody": custody, "private_sha256": digest(file)}
                print(panel, outcomes[panel], flush=True)
        if "torch" in sys.modules:
            raise RuntimeError("Archive-only manipulation imported Torch")
        check()
    write_new(args.output / "aggregate.json", {"experiment": cfg["experiment"], "panels": outcomes, "status": "completed",
        "source_manifest_sha256": digest(MANIFEST), "headnull_parent_sha256": digest(PARENT), "head_design": metrics, "spectral_proposal_sha256": digest(bank_path),
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
        "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples)})


if __name__ == "__main__":
    main()
