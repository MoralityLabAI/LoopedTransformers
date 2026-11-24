"""Compare the original peak-aware NF4 sham with a fixed RMS refinement on DEV."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dose51.route import SIGNS, directions, patches
from experiments.nan_dose51.run import focused
from experiments.nan_frontier55.run import check_head
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations
from experiments.nan_sham56.minimax import WIDTH, atlas, derivative_columns, form_basis, optimize
from experiments.nan_sham56.run import HERE as SHAM

from .geometry import pair_head, refine, verify


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PARENT = REPO / "reports/nanbeige_bpd_gym/nan_sham56_minimax_results.json"
ARMS = ("peak", "rms_refined")


def seal():
    files = [HERE / p for p in (".gitattributes", "__init__.py", "registration.json", "geometry.py", "run.py", "PROTOCOL.md")]
    files += [PARENT, SHAM / "MINIMAX_MANIFEST.json", SHAM / "minimax.py", SHAM / "geometry.py",
              SHAM / "registration.json", DOSE / "registration.json", READPATH / "registration.json",
              ROUND / "registration.json", ROUND / "geometry.py", ROUND / "run.py"]
    files += [REPO / name for name in ("experiments/nan_dose51/run.py", "experiments/nan_dose51/route.py",
        "experiments/nan_dark44/head.py", "experiments/nan_frontier55/run.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_phase35/run.py", "experiments/nan_readpath37/run.py")]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in files}


def panels(base, per_panel):
    all_rows = focused(base)
    roots = sorted({r["root"] for r in all_rows.values()})
    groups = [set(roots[i:i+per_panel]) for i in range(0, len(roots), per_panel)]
    return [{i: r for i, r in all_rows.items() if r["root"] in group} for group in groups]


def replay(source, base, cfg, sham_cfg, round_cfg, dose, bank_pack, head, tokens, check, masks, np):
    bank, ci, eps, anchor = bank_pack
    folder = ARCHIVE / "nf4"
    audit = json.loads(bound(ARCHIVE / "audits/nf4.json", base["backends"]["nf4"]["audit"]))
    ids = {(r["root"], r["world"]): slot for slot, r in source.items()}
    states, refs, labels, rows, packets, columns = [], [], [], [], [], []
    indexed, head_pairs, pair_destinations = {}, [], []

    def add(h, pair, ref=None):
        state = np.asarray(h, dtype=np.float32)
        index = len(states)
        if ref is not None:
            identity = (ref, state.tobytes())
            if identity in indexed:
                return indexed[identity]
            indexed[identity] = index
        states.append(state)
        refs.append(index if ref is None else ref)
        labels.append(pair)
        return index

    for src, (h, x, y) in route_states("nf4", base, source, check, np):
        h = h.astype(np.float64)
        root, world = src["root"], src["world"]
        slot = ids[root, world]
        foil = source[ids[root, "B" if world == "A" else "A"]]
        pair = [tokens[foil["answer"]], tokens[src["answer"]]]
        case = json.loads(bound(folder / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
        retained = case["initial_phases"][0]
        if retained["loop"] != 1 or pair[0] == pair[1]:
            raise ValueError("Invalid original head scope")
        row = {"root": root, "world": world, "clean": add(h, pair), "citation": retained["score"],
               "recipes": {key: {"physical_gate": False} for key in ("active", *ARMS)}}
        rows.append(row)
        effects, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
        raw = effects["learned"] - effects["static"]
        weights = head.gamma.astype(np.float64) * (head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
        g = head_gradient(h, weights, head.eps, np)
        nominal = float(np.linalg.norm(h)) * dose["phase1_dose_fraction"]
        tag = f"nf4/{root}/{world}"
        dense, original, _ = make_pair(h, raw, g, nominal, None, tag, round_cfg, np)
        if dense is None:
            row["active_refusal"] = "degenerate_route"
            continue
        active, fields, matched = patches(h, {"contrast": dense["contrast"]}, dose, np)
        for sign, polarity in SIGNS.items():
            name = f"contrast/{sign}"
            edit = active[name]
            fields[name]["original_signed_cosine"] = (
                float((edit.after - edit.before) @ original) * polarity / edit.realized_l2 if edit.realized_l2 else 0.)
        if not matched or any(r["original_signed_cosine"] < round_cfg["minimum_reached_contrast_cosine"] for r in fields.values()):
            row["active_refusal"] = "active_dose_or_alignment"
            continue
        row["recipes"]["active"] = {"physical_gate": True, "arm_dose": fields,
                                    "head_indices": {k: add(p.after, pair, row["clean"]) for k, p in active.items()}}
        axes, support = form_basis(h, original, g, [p.after-p.before for p in active.values()], nominal, tag, sham_cfg, np)
        columns.append(derivative_columns(head, h, axes, nominal, np))
        packets.append((row, h, pair, dense["contrast"], original, support, axes, active))
    if packets:
        matrix = atlas(head, columns, dose["block_rows"], check, np)
        for i, (row, h, pair, contrast, original, support, axes, active) in enumerate(packets):
            check()
            world_derivative = matrix[:, i*WIDTH:(i+1)*WIDTH].copy()
            q, _, _ = optimize(world_derivative, np)
            refined, continuous = refine(world_derivative, q, np)
            row["continuous"] = continuous
            for key, coeff in zip(ARMS, (q, refined), strict=True):
                sham = axes @ coeff.astype(np.float64)
                edits, result = match(h, {"contrast": contrast, "sham": sham}, original, round_cfg, dose, np)
                if any(not np.array_equal(edits[name].after, p.after) for name, p in active.items()):
                    raise ValueError("Sham changed the active edit")
                result["sham_support"] = support
                row["recipes"][key] = result
                if not result["physical_gate"]:
                    continue
                item = {k: add(p.after, pair, row["clean"]) for k, p in edits.items()}
                result["head_indices"] = item
                result["pairs"] = {}
                for sign in SIGNS:
                    head_pairs.append((row["clean"], item[f"contrast/{sign}"], item[f"sham/{sign}"]))
                    pair_destinations.append((result, sign))
        del matrix
    live, info = pair_head(head, np.stack(states), np.asarray(refs), np.asarray(labels),
                           np.asarray(head_pairs, dtype=np.int64).reshape((-1, 3)), dose["block_rows"], check, np)
    for (record, sign), pair in zip(pair_destinations, info, strict=True):
        record["pairs"][sign] = pair
    custody = check_head(rows, live, dose, base["native_logit_abs_tol"])
    for row in rows:
        for key in ARMS:
            option = row["recipes"][key]
            option["matched_realized_head_rms"] = option["full_pass"] and all(
                abs(p["signed_rms_gap"]) <= cfg["rms_gap_limit"] for p in option["pairs"].values())
    return rows, custody


def summarize(rows):
    roots = {r["root"] for r in rows}
    if len(roots) != 16 or len(rows) != 32 or any({r["world"] for r in rows if r["root"] == k} != {"A", "B"} for k in roots):
        raise ValueError("The full exposed DEV roster is required")
    result = {}
    for key in ("active", *ARMS):
        options = [r["recipes"][key] for r in rows]
        passed = [v for v in options if v["full_pass"]]
        pair = [item for record in passed for item in record.get("pairs", {}).values()]
        entry = {"worlds": len(options), "physical_gate": sum(v["physical_gate"] for v in options),
                 "full_pass": len(passed), "both_worlds_pass_roots": sum(
                    all(r["recipes"][key]["full_pass"] for r in rows if r["root"] == root) for root in roots)}
        if key != "active":
            entry["rms_matched_both_signs_worlds"] = sum(v["matched_realized_head_rms"] for v in options)
            entry["medians_among_full_pass_signs"] = {
                field: median(values) if (values := [v[field] for v in pair if v[field] is not None]) else None
                for field in ("active_rms", "sham_rms", "signed_rms_gap", "head_shift_cosine", "head_shift_rms_distance")}
        result[key] = entry
    both = [r for r in rows if all(r["recipes"][key]["full_pass"] for key in ARMS)]
    gaps = [[abs(pair["signed_rms_gap"]) for pair in r["recipes"][key]["pairs"].values()]
            for key in ARMS for r in both]
    pre = gaps[:len(both)]
    post = gaps[len(both):]
    continuous = [r["continuous"] for r in rows if "continuous" in r]
    return {"roots": len(roots), "by_recipe": result, "paired_full_gate_worlds": len(both),
            "paired_rms_gap_worst_sign_median": {
                "peak": median(map(max, pre)) if pre else None, "refined": median(map(max, post)) if post else None},
            "paired_rms_gap_worst_sign_improved_worlds": sum(max(b) < max(a) for a,b in zip(pre, post, strict=True)),
            "continuous_atlas": {"worlds": len(continuous), **{field: median(v[field] for v in continuous)
                for field in ("continuous_minimum_rms_in_atlas", "peak_recipe_continuous_rms", "refined_continuous_rms",
                              "peak_recipe_continuous_peak", "refined_continuous_peak")}} if continuous else {"worlds": 0}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("DRIFT57 source freeze changed")
    if not args.snapshot or not args.output:
        cli.error("--snapshot and --output are required")
    ref = json.loads(PARENT.read_bytes())
    if ref["source_sha256"] != digest(SHAM / "MINIMAX_MANIFEST.json"):
        raise ValueError("SHAM56 derivative design provenance changed")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    sham_cfg = json.loads((SHAM / "registration.json").read_bytes())
    round_cfg = json.loads((ROUND / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes()) | {"phase1_dose_fraction": sham_cfg["phase1_dose_fraction"]}
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Snapshot revision changed")
    args.output.mkdir(parents=True, exist_ok=False)
    output, custody = [], []
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned numerical runtime changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(np)
        banks = calibrations(base, masks, np)["nf4"]
        path = args.snapshot / "tokenizer.json"
        bound(path, base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(path))
        encoded = {c: tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(v) != 1 for v in encoded.values()) or len({v[0] for v in encoded.values()}) != 16:
            raise ValueError("Bare label tokens changed")
        tokens = {c: v[0] for c, v in encoded.items()}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for i, source in enumerate(panels(base, cfg["roots_per_panel"]), 1):
                print(f"Replaying DEV block {i} with {len(source)} worlds", flush=True)
                rows, block = replay(source, base, cfg, sham_cfg, round_cfg, dose, banks, head, tokens, check, masks, np)
                part = args.output / f"block{i}_private.json"
                write_new(part, rows)
                custody.append(block | {"block": i, "private_sha256": digest(part)})
                output += rows
        if "torch" in sys.modules:
            raise RuntimeError("Archive-only replay imported Torch")
        check()
    result = summarize(output)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "completed", "analysis": result, "blocks": custody,
        "rms_gap_limit": cfg["rms_gap_limit"], "source_manifest_sha256": digest(MANIFEST),
        "parent_cal_result_sha256": digest(PARENT), "formal_comparisons": 0, "sealed_test_accessed": False,
        "new_pretrained_forwards": 0, "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
