"""Measure finite NF4 controls from a low-sensitivity vocabulary proposal."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dose51.route import SIGNS, directions, patches
from experiments.nan_drift57.geometry import pair_head
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations, rosters
from experiments.nan_sham56.minimax import atlas, derivative_columns, optimize
from experiments.nan_sham56.run import ACTIVE_AGG, HERE as SHAM, physical_head_closure

from .geometry import designs, fine_basis, head_spectrum, verify


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PRIOR = REPO / "reports/nanbeige_bpd_gym/nan_sham56_minimax_results.json"
POLICIES = ("rms_targeted", "low_spectrum_minimum")


def seal():
    files = [HERE / f for f in ("geometry.py", "run.py", "__init__.py", ".gitattributes", "PROTOCOL.md",
                               "registration.json")]
    files += [PRIOR, ACTIVE_AGG, SHAM / "MINIMAX_MANIFEST.json", SHAM / "SOURCE_MANIFEST.json",
              SHAM / "minimax.py", SHAM / "geometry.py", SHAM / "run.py", SHAM / "registration.json",
              ROUND / "registration.json", ROUND / "geometry.py", ROUND / "run.py",
              READPATH / "registration.json", DOSE / "registration.json"]
    files += [REPO / p for p in ("experiments/nan_drift57/geometry.py", "experiments/nan_dose51/route.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_dark44/head.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_readpath37/run.py", "experiments/nan_phase35/run.py")]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in files}


def replay(source, base, cfg, sham_cfg, round_cfg, dose, bank_pack, head, tokens, spectral,
           previous, prior, check, masks, np):
    bank, ci, eps, anchor = bank_pack
    location = ARCHIVE / "nf4"
    original_audit = json.loads(bound(ARCHIVE / "audits/nf4.json", base["backends"]["nf4"]["audit"]))
    slots = {(r["root"], r["world"]): slot for slot, r in source.items()}
    columns, packets = [], []
    for src, (h, x, y) in route_states("nf4", base, source, check, np):
        h = h.astype(np.float64)
        root, world = src["root"], src["world"]
        foil = source[slots[root, "B" if world == "A" else "A"]]
        pair = [tokens[foil["answer"]], tokens[src["answer"]]]
        key = slots[root, world]
        case = json.loads(bound(location / f"case_{key:04d}.json", original_audit["records"][key]["case_sha256"]))
        clean = case["initial_phases"][0]
        if clean["loop"] != 1 or pair[0] == pair[1]:
            raise ValueError("Wrong native phase or ambiguous bare labels")
        effects, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
        contrast = effects["learned"] - effects["static"]
        normal = head.gamma.astype(np.float64) * (
            head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
        grad = head_gradient(h, normal, head.eps, np)
        radius = float(np.linalg.norm(h)) * dose["phase1_dose_fraction"]
        tag = f"nf4/{root}/{world}"
        dense, raw, _ = make_pair(h, contrast, grad, radius, None, tag, round_cfg, np)
        active, sampled, admitted = patches(h, {"contrast": dense["contrast"]}, dose, np)
        receipt = previous[root, world]["recipes"]["0.0075/dense"]["arm_dose"]
        if not admitted or any(any(receipt[k][f] != val for f, val in record.items()) for k,record in sampled.items()):
            raise ValueError("The fixed active edit changed from the archive")
        axes, support = fine_basis(h, raw, grad, [p.after-p.before for p in active.values()],
                                   radius, tag, spectral, cfg, sham_cfg, np)
        columns.append(derivative_columns(head, h, axes, radius, np))
        packets.append((src, h, pair, clean["score"], dense["contrast"], raw, axes, support, active))
    width = cfg["spectral_axes"] + cfg["ambient_axes"]
    derivative = atlas(head, columns, dose["block_rows"], check, np)
    rows, states, refs, labels, pairs, address = [], [], [], [], [], []
    slots = {}

    def add(state, pair, reference=None):
        vec = np.asarray(state, dtype=np.float32)
        if reference is not None:
            key = reference, vec.tobytes()
            if key in slots:
                return slots[key]
            slots[key] = len(states)
        i = len(states)
        states.append(vec)
        refs.append(i if reference is None else reference)
        labels.append(pair)
        return i

    for i, (src, h, pair, baseline, contrast, original, axes, support, active) in enumerate(packets):
        # Target the pinned rounded active magnitude at the same native world.
        check()
        root, world = src["root"], src["world"]
        receipt = prior[root, world]["recipes"]["ambient_peak_minimax"]["arm_dose"]
        clean = add(h, pair)
        active_ids = {k: add(p.after, pair, clean) for k,p in active.items()}
        row = {"root": root, "world": world, "clean": clean, "citation": baseline,
               "fixed_active_head_indices": active_ids, "recipes": {}}
        rows.append(row)
        local = derivative[:, i*width:(i+1)*width].copy()
        peak, _, _ = optimize(local, np)
        target = sum(receipt[f"contrast/{sign}"]["rms_logit_shift"] for sign in SIGNS) / 2.
        choices, prediction = designs(local, peak, target, cfg["predicted_peak_limit"], np)
        row["prediction"] = prediction
        for method, coefficients in choices.items():
            edits, entry = match(h, {"contrast": contrast, "sham": axes @ coefficients}, original, round_cfg, dose, np)
            if any(not np.array_equal(edit.after, edits[k].after) for k,edit in active.items()):
                raise ValueError("A new sham changed the fixed active contrast")
            row["recipes"][method] = entry | {"sham_support": support}
            entry = row["recipes"][method]
            if entry["physical_gate"]:
                entry["pairs"] = {}
                scanned = {k: add(edit.after, pair, clean) for k,edit in edits.items()}
                entry["head_indices"] = scanned
                for sign in SIGNS:
                    pairs.append((clean, scanned[f"contrast/{sign}"], scanned[f"sham/{sign}"]))
                    address.append((entry, sign))
    del derivative
    live, measurements = pair_head(head, np.stack(states), np.asarray(refs), np.asarray(labels),
                                    np.asarray(pairs, dtype=np.int64).reshape((-1, 3)), dose["block_rows"], check, np)
    for (entry, sign), data in zip(address, measurements, strict=True):
        entry["pairs"][sign] = data
    closure = physical_head_closure(rows, live, previous, dose, base)
    for row in rows:
        for entry in row["recipes"].values():
            entry["rms_match"] = entry["full_pass"] and all(
                abs(v["signed_rms_gap"]) <= cfg["realized_rms_gap_limit"] for v in entry["pairs"].values())
    return rows, closure


def summarize(rows, prior):
    roots = {r["root"] for r in rows}
    if len(rows) != 8 or len(roots) != 4 or any({r["world"] for r in rows if r["root"] == root} != {"A", "B"} for root in roots):
        raise ValueError("The paired CAL roster changed")
    records = {}
    for key in POLICIES:
        entries = [r["recipes"][key] for r in rows]
        passed = [v for v in entries if v["full_pass"]]
        linked = [d for v in passed for d in v["pairs"].values()]
        only_physical = [v for v in entries if v["physical_gate"]]
        records[key] = {
            "worlds": len(entries), "physical_gate": len(only_physical), "full_pass": len(passed),
            "both_worlds_pass_roots": sum(all(r["recipes"][key]["full_pass"] for r in rows if r["root"] == root)
                                         for root in roots),
            "rms_matched_worlds": sum(v["rms_match"] for v in entries),
            "both_worlds_rms_matched_roots": sum(all(r["recipes"][key]["rms_match"] for r in rows if r["root"] == root)
                                               for root in roots),
            "worst_sham_peak_on_pass": max((v["arm_dose"][f"sham/{s}"]["max_abs_logit_delta"]
                                           for v in passed for s in SIGNS), default=None),
            "medians_among_full_pass_signs": {f: median(values) if (
                values := [p[f] for p in linked if p[f] is not None]) else None
                for f in ("active_rms", "sham_rms", "signed_rms_gap", "head_shift_cosine", "head_shift_rms_distance")},
        }
    previous = [r["recipes"]["ambient_peak_minimax"] for r in prior.values()]
    if len(previous) != 8 or not all(p["full_pass"] for p in previous):
        raise ValueError("Prior comparison roster changed")
    gap = lambda p: max(abs(p["arm_dose"][f"sham/{s}"]["rms_logit_shift"] -
                            p["arm_dose"][f"contrast/{s}"]["rms_logit_shift"]) for s in SIGNS)
    predictions = [item["prediction"] for item in rows]
    return {"controls": records, "prior_sham56_median_worst_sign_rms_gap": median(gap(p) for p in previous),
            "median_continuous_minimum_rms": median(v["predicted_minimum_rms"] for v in predictions),
            "median_active_target_rms": median(v["target_active_rms"] for v in predictions),
            "median_targeted_continuous_rms": median(v["targeted_predicted_rms"] for v in predictions),
            "targeted_worlds_predicted_peak_met": sum(v["predicted_peak_gate_met"] for v in predictions)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--active", type=Path)
    cli.add_argument("--prior", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("LOWHEAD59 freeze changed")
    if not (args.snapshot and args.active and args.prior and args.output):
        cli.error("--snapshot, --active, --prior, and --output required")
    if digest(args.active / "aggregate.json") != digest(ACTIVE_AGG) or digest(args.prior / "aggregate.json") != digest(PRIOR):
        raise ValueError("The active or peak-aware reference receipt changed")
    past = json.loads(PRIOR.read_bytes())
    parent = json.loads(ACTIVE_AGG.read_bytes())
    active = json.loads(bound(args.active / "private.json", parent["private_sha256"]))
    cfg = json.loads((HERE / "registration.json").read_bytes())
    sham_cfg = json.loads((SHAM / "registration.json").read_bytes())
    round_cfg = json.loads((ROUND / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes()) | {"phase1_dose_fraction": sham_cfg["phase1_dose_fraction"]}
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Pinned checkpoint revision changed")
    args.output.mkdir(parents=True, exist_ok=False)
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned numerical runtime changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(np)
        bank = calibrations(base, masks, np)["nf4"]
        path = args.snapshot / "tokenizer.json"
        bound(path, base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(path))
        encoded = {c: tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in encoded.values()) or len({v[0] for v in encoded.values()}) != 16:
            raise ValueError("Bare token bindings changed")
        tokens = {c: v[0] for c,v in encoded.items()}
        panels = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            spectral, spec = head_spectrum(head, cfg, dose["block_rows"], check, np)
            print(f"Head spectral proposal from {spec['sampled_vocabulary_rows']} rows", flush=True)
            for panel, roster in zip(("fit", "check"), rosters(base)):
                parent_rows = {(r["root"], r["world"]): r for r in active if r["panel"] == panel}
                archived = json.loads(bound(args.prior / f"{panel}_private.json", past["panels"][panel]["private_sha256"]))
                prior = {(r["root"], r["world"]): r for r in archived}
                rows, closure = replay(roster, base, cfg, sham_cfg, round_cfg, dose, bank, head,
                                       tokens, spectral, parent_rows, prior, check, masks, np)
                if {(r["root"], r["world"]) for r in rows} != set(prior):
                    raise ValueError("New and prior comparators have different worlds")
                private = args.output / f"{panel}_private.json"
                write_new(private, rows)
                panels[panel] = summarize(rows, prior) | {"custody": closure, "private_sha256": digest(private)}
                print(panel, panels[panel], flush=True)
        if "torch" in sys.modules:
            raise RuntimeError("Archive spectral audit imported Torch")
        check()
    write_new(args.output / "aggregate.json", {"experiment": cfg["experiment"], "status": "completed",
        "source_manifest_sha256": digest(MANIFEST), "sham56_aggregate_sha256": digest(PRIOR),
        "active_reference_sha256": digest(ACTIVE_AGG), "head_design": spec, "panels": panels,
        "rms_gap_limit": cfg["realized_rms_gap_limit"], "new_pretrained_forwards": 0, "formal_comparisons": 0,
        "sealed_test_accessed": False, "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
        "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples),
        "max_sampled_working_set_bytes": max(x["working_set_bytes"] for x in samples)})


if __name__ == "__main__":
    main()
