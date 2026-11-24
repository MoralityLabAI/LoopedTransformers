"""Audit equal-dose route directions on the pinned native COMP14 DEV archive."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dark44.head import full_head, verify_reference
from experiments.nan_gauge42.run import ARCHIVE, READPATH, read_archive
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new
from experiments.nan_route50.masks import mass_matched, relative_write_error
from experiments.nan_route50.run import fit_packet, npz_fields

from .route import ARMS, SIGNS, directions, patches, separation


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
ROUTE = REPO / "reports/nanbeige_bpd_gym/nan_route50_results.json"


def seal():
    paths = [HERE / name for name in (
        ".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json", "run.py", "route.py",
        "native.py", "verify.py")]
    paths += [ROUTE] + [REPO / name for name in (
        "experiments/nan_route50/SOURCE_MANIFEST.json", "experiments/nan_route50/run.py",
        "experiments/nan_route50/masks.py", "experiments/nan_gated46/SOURCE_MANIFEST.json",
        "experiments/nan_gated46/gated.py", "experiments/nan_gated46/verify.py",
        "experiments/nan_delay39/contrasts.py", "experiments/nan_delay39/instrument.py",
        "experiments/nan_dark44/head.py", "experiments/nan_dark44/run.py",
        "experiments/nan_gauge42/run.py", "experiments/nan_readpath37/registration.json",
        "experiments/nan_readpath37/run.py", "experiments/nan_fit07/pointwise.py",
        "experiments/nan_phase35/run.py")]
    return {"experiment": "NAN-DOSE-51",
            "files_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in paths}}


def focused(base):
    rows, _ = joined(base)
    rows = {i: row for i, row in rows.items() if (row["split"], row["arm"]) == ("development", "focused")}
    if len(rows) != 32 or len({r["root"] for r in rows.values()}) != 16:
        raise ValueError("Focused DEV roster changed")
    return rows


def write_packet(backend, slot, audit, np):
    lock = audit["records"][slot]
    path = ARCHIVE / backend
    case = json.loads(bound(path / f"case_{slot:04d}.json", lock["case_sha256"]))
    fields = [record["array_key"] for record in case["state_index"]
              if (record["loop"], record["layer_zero_based"], record["site"]) == (1, 21, "down_proj_output")]
    if len(fields) != 1 or case["state_file_sha256"] != lock["state_sha256"]:
        raise ValueError("Archived route write has changed")
    state_file = f"case_{slot:04d}_states.npz"
    if case["state_file"] != state_file:
        raise ValueError("Archived route write file changed")
    write = npz_fields(path / state_file, lock["state_sha256"], fields, np)[fields[0]].astype(np.float32)
    if write.ndim != 1 or not np.isfinite(write).all():
        raise ValueError("Invalid archived native write")
    phase = case["initial_phases"][0]
    if phase["loop"] != 1:
        raise ValueError("Archived write or head is for the wrong pass")
    return write, phase["score"]


def collect(base, cfg, head, tokens, masks, check, np):
    report = json.loads(ROUTE.read_bytes())
    roster = focused(base)
    slots = {(row["root"], row["world"]): i for i, row in roster.items()}
    records, states, references, labels, closure = [], [], [], [], {}

    def add(h, token_pair, reference=None):
        i = len(states)
        states.append(h)
        references.append(i if reference is None else reference)
        labels.append(token_pair)
        return i

    for backend, pin in base["backends"].items():
        twins, reads, writes, replay = read_archive(backend, base, roster, head, check, np)
        _, bank, ci, eps, fixture, _ = fit_packet(backend, base, np)
        if not np.array_equal(bank["V"], reads) or not np.array_equal(bank["U"], writes):
            raise ValueError("T2 route and archived factor banks differ")
        ci_error = float(abs(masks(fixture["X"], bank, ci) - fixture["masks"]).max())
        if ci_error > 2e-5:
            raise ValueError("Pointwise controller differs from the native fixture")
        selected = report["per_backend"][backend]["passes"]["1"]["static_prefit"]["selected_anchor"]
        prefix, binary = selected.split(":")
        if prefix != "binary" or len(binary) != base["rank"] or set(binary) != {"0", "1"}:
            raise ValueError("Phase-1 CAL static anchor changed")
        anchor = np.fromiter(map(float, binary), dtype=np.float64)
        audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
        maximum_write_error = 0.
        for (_, _, root), worlds in sorted(twins.items()):
            if set(worlds) != {"A", "B"} or worlds["A"][0] == worlds["B"][0]:
                raise ValueError("Incomplete or ambiguous source twin")
            for world, other in (("A", "B"), ("B", "A")):
                check()
                answer, vectors, _ = worlds[world]
                pair = [tokens[worlds[other][0]], tokens[answer]]
                pick = lambda site: vectors[1, site].astype(np.float32)
                h, x = (pick(site) for site in ("residual_after_layer", "down_proj_input"))
                y, archived_score = write_packet(backend, slots[root, world], audit, np)
                learned = masks(x[None], bank, ci)[0]
                effects, budget_error = directions(x, y, learned, anchor, bank, eps, np)
                norm = float(np.linalg.norm(y.astype(np.float64)))
                if norm <= 1e-12:
                    raise ValueError("Archived native write norm is undefined")
                ref_errors = relative_write_error(
                    np.stack([learned, mass_matched(anchor, float(learned.sum()), np)]),
                    y, x @ bank["V"], bank, 0, eps, np)
                observed = np.asarray([np.linalg.norm(effects[arm]) / norm for arm in ARMS])
                maximum_write_error = max(maximum_write_error, float(np.max(abs(observed - ref_errors))))
                signed, summaries, matched = patches(h, effects, cfg, np)
                baseline = add(h, pair)
                record = {
                    "backend": backend, "root": root, "world": world, "clean": baseline,
                    "archive": archived_score,
                    "original_write_relative_l2": dict(zip(ARMS, observed.tolist())),
                    "soft_mass_absolute_error": budget_error,
                    "route_direction_before_dose": separation(effects, *ARMS, np),
                    "matched_all_four": matched, "separation": {}, "arms": summaries,
                }
                for key, patch in signed.items():
                    summaries[key]["index"] = add(patch.after, pair, baseline) if patch.eligible else None
                for sign in SIGNS:
                    keys = [f"{arm}/{sign}" for arm in ARMS]
                    if all(signed[k].eligible for k in keys):
                        actual = {k: signed[k].after - signed[k].before for k in keys}
                        metrics = separation(actual, *keys, np)
                        metrics["distance_as_nominal_dose"] = metrics["l2_distance"] / signed[keys[0]].requested_l2
                        record["separation"][sign] = metrics
                records.append(record)
        if maximum_write_error > 1e-10:
            raise ValueError("Finite write direction does not reproduce ROUTE50 relative write arithmetic")
        closure[backend] = {"selected_CAL_anchor": selected, "native_selected_abs_logit_error": replay,
                            "native_controller_abs_error": ci_error,
                            "route50_write_formula_abs_error": maximum_write_error}
    return records, np.stack(states), np.array(references), np.asarray(labels), closure


def evaluate(rows, live, cfg, gate):
    peak_top = peak_logz = 0.
    bare = live["bare"][:, 0] - live["bare"][:, 1]
    for row in rows:
        index = row.pop("clean")
        saved = row.pop("archive")
        if (live["winner"][index] != saved["free_top_token_id"] or live["vocab"] != saved["vocabulary_size"]):
            raise ValueError("Unedited full-head winner differs from native COMP14")
        peak_top = max(peak_top, abs(float(live["top_logit"][index]) - saved["free_top_token_logit"]))
        peak_logz = max(peak_logz, abs(float(live["logz"][index]) - saved["full_log_normalizer"]))
        for arm in row["arms"].values():
            variant = arm.pop("index")
            if variant is None:
                continue
            arm["max_abs_logit_delta"] = float(live["max_abs_delta"][variant])
            arm["abs_margin_delta"] = float(abs(bare[variant] - bare[index]))
            arm["greedy_same"] = bool(live["winner"][variant] == live["winner"][index])
            arm["gate"] = bool(arm["greedy_same"] and arm["max_abs_logit_delta"] <= cfg["maximum_head_delta"]
                               and arm["abs_margin_delta"] <= cfg["maximum_margin_delta"])
        row["matched_and_head_preserved"] = row["matched_all_four"] and all(
            arm.get("gate", False) for arm in row["arms"].values())
        row["distinct_both_signs"] = all(
            sign in row["separation"] and not row["separation"][sign]["exactly_equal"] for sign in SIGNS)
    if max(peak_top, peak_logz) > gate:
        raise ValueError("Unedited full-vocabulary replay exceeded the native archive tolerance")
    return {"max_native_top_logit_error": peak_top, "max_native_log_normalizer_error": peak_logz}


def summarize(rows):
    roots = defaultdict(list)
    for row in rows:
        roots[row["root"]].append(row)
    if len(roots) != 16 or any({r["world"] for r in group} != {"A", "B"} or len(group) != 2
                              for group in roots.values()):
        raise ValueError("Incomplete two-world coverage")
    matched = [r for r in rows if r["matched_all_four"]]
    preserved = [r for r in matched if r["matched_and_head_preserved"]]
    qualified = [r for r in preserved if r["distinct_both_signs"]]
    usable = {(r["root"], r["world"]) for r in qualified}
    usable_roots = {root for root, _ in usable}
    before = [r["route_direction_before_dose"]["cosine"] for r in rows]
    return {
        "roots": len(roots), "worlds": len(rows), "all_four_dose_matched_worlds": len(matched),
        "all_four_dose_matched_and_full_head_preserved_worlds": len(preserved),
        "dose_head_and_distinct_both_signs_worlds": len(qualified),
        "at_least_one_qualified_world_roots": len(usable_roots),
        "both_qualified_worlds_roots": sum((root, "A") in usable and (root, "B") in usable for root in roots),
        "max_soft_mask_mass_absolute_error": max(r["soft_mass_absolute_error"] for r in rows),
        "original_write_relative_l2_median": {arm: median(r["original_write_relative_l2"][arm] for r in rows)
                                               for arm in ARMS},
        "original_route_direction_cosine_median": median(v for v in before if v is not None),
        "by_sign": {sign: {
            "same_sign_exact_collisions_in_matched_worlds": sum(r["separation"][sign]["exactly_equal"] for r in matched),
            "rounded_direction_cosine_median": median(r["separation"][sign]["cosine"] for r in matched)
            if matched else None,
            "route_distance_as_nominal_dose_median": median(
                r["separation"][sign]["distance_as_nominal_dose"] for r in matched) if matched else None,
        } for sign in SIGNS},
        "arms": {key: {
            "eligible": sum(r["arms"][key]["eligible"] for r in rows),
            "eligible_and_head_preserved": sum(r["arms"][key].get("gate", False) for r in rows),
        } for key in rows[0]["arms"]},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Use assertions for archive custody")
    if args.freeze:
        write_new(MANIFEST, seal())
        print(digest(MANIFEST))
        return
    if not (args.snapshot and args.output):
        cli.error("--snapshot and --output are required")
    if json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("DOSE51 source freeze changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Snapshot revision mismatch")
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned replay packages changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        fixture_rows = verify_reference(np)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        tokens = {c: tokenizer.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in tokens.values()) or len({ids[0] for ids in tokens.values()}) != 16:
            raise ValueError("Bare-label token identity changed")
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            rows, states, refs, labels, closure = collect(
                base, cfg, head, {c: ids[0] for c, ids in tokens.items()}, masks, check, np)
            print(f"Replaying {len(states)} full-vocabulary residual rows", flush=True)
            live = full_head(head, states, refs, labels, cfg["block_rows"], check, np)
            full_closure = evaluate(rows, live, cfg, base["native_logit_abs_tol"])
        if "torch" in sys.modules:
            raise RuntimeError("Archive replay imported Torch")
        check()
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private.json"
    write_new(private, rows)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "private_receipt_sha256": digest(private), "vocabulary_rows": live["vocab"], "scored_states": len(states),
        "fixture_streaming_reduction_rows": fixture_rows, "archive_and_route_closure": closure, **full_closure,
        "per_backend": {b: summarize([r for r in rows if r["backend"] == b]) for b in base["backends"]},
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)},
    })


if __name__ == "__main__":
    main()
