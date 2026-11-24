"""Fit and check a BF16-aware route contrast on disjoint COMP14 CAL panels."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dark44.head import full_head, verify_reference
from experiments.nan_dark44.run import roster as dark_roster
from experiments.nan_dose51.route import directions
from experiments.nan_dose51.run import HERE as DOSE, ROUTE
from experiments.nan_gauge42.run import ARCHIVE, READPATH
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new
from experiments.nan_route50.run import fit_packet

from .geometry import make_pair, match, verify


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DARK = REPO / "experiments/nan_dark44"
MANIFEST = HERE / "SOURCE_MANIFEST.json"


def seal():
    files = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "geometry.py",
                                      "registration.json", "run.py")]
    files += [ROUTE, DARK / "registration.json", DOSE / "registration.json", DOSE / "SOURCE_MANIFEST.json",
              READPATH / "registration.json"]
    files += [REPO / name for name in (
        "experiments/nan_dose51/run.py", "experiments/nan_dose51/route.py", "experiments/nan_dark44/run.py",
        "experiments/nan_dark44/head.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_route50/run.py", "experiments/nan_fit07/pointwise.py")]
    return {"experiment": "NAN-ROUND-54", "sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in files}}


def rosters(base):
    rows, _ = joined(base)
    cal = {slot: row for slot, row in rows.items() if (row["split"], row["arm"]) == ("calibration", "focused")}
    fit = dark_roster(base, json.loads((DARK / "registration.json").read_bytes()))
    fit_roots = {r["root"] for r in fit.values()}
    split = {slot: row for slot, row in cal.items() if row["root"] not in fit_roots}
    check_roots = {r["root"] for r in split.values()}
    if len(fit) != 8 or len(split) != 8 or len(fit_roots) != 4 or len(check_roots) != 4:
        raise ValueError("CAL split or twins changed")
    return fit, split


def calibrations(base, masks, np):
    ref = json.loads(ROUTE.read_bytes())
    out = {}
    for backend in base["backends"]:
        _, bank, ci, eps, fixture, _ = fit_packet(backend, base, np)
        if float(abs(masks(fixture["X"], bank, ci) - fixture["masks"]).max()) > 2e-5:
            raise ValueError("Archived native controller did not reproduce")
        name, bits = ref["per_backend"][backend]["passes"]["1"]["static_prefit"]["selected_anchor"].split(":")
        if name != "binary" or any(v not in "01" for v in bits) or len(bits) != base["rank"]:
            raise ValueError("Unexpected frozen CAL static anchor")
        out[backend] = (bank, ci, eps, np.fromiter(map(float, bits), dtype=np.float64))
    return out


def capture_stage(roster, policies, banks, base, cfg, dose, head, tokens, check, masks, np):
    ids = {(row["root"], row["world"]): slot for slot, row in roster.items()}
    states, refs, labels, records = [], [], [], []
    raw_closure = 0.

    def add(state, pair, ref=None):
        index = len(states)
        states.append(state)
        refs.append(index if ref is None else ref)
        labels.append(pair)
        return index

    for backend, (bank, ci, eps, anchor) in banks.items():
        pin = base["backends"][backend]
        audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
        for row, (h, x, y) in route_states(backend, base, roster, check, np):
            root, world = row["root"], row["world"]
            h = h.astype(np.float64)
            other = "B" if world == "A" else "A"
            foil = roster[ids[root, other]]
            pair = [tokens[foil["answer"]], tokens[row["answer"]]]
            if pair[0] == pair[1]:
                raise ValueError("Reference foil and recipient labels coincide")
            slot = ids[root, world]
            case = json.loads(bound(ARCHIVE / backend / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
            first = case["initial_phases"][0]
            if first["loop"] != 1:
                raise ValueError("Archived logits do not belong to pass 1")
            saved = first["score"]
            selected = {int(k): v for k, v in saved["selected_logits"].items()}
            err = float(abs(head.logits(h, list(selected)) - list(selected.values())).max())
            raw_closure = max(raw_closure, err)
            local = add(h, pair)
            effects, mass = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
            raw = effects["learned"] - effects["static"]
            w = head.gamma.astype(np.float64) * (head.row(pair[0]).astype(np.float64) - head.row(pair[1]))
            gradient = head_gradient(h, w, head.eps, np)
            radius = float(np.linalg.norm(h)) * dose["phase1_dose_fraction"]
            record = {"backend": backend, "root": root, "world": world, "clean": local, "saved": saved,
                      "soft_mask_mass_error": mass, "route_contrast_l2": float(np.linalg.norm(raw)), "policies": {}}
            tag = f"{backend}/{root}/{world}"
            for key, threshold in policies:
                check()
                plan, target, features = make_pair(h, raw, gradient, radius, threshold, tag, cfg, np)
                if plan is None:
                    record["policies"][key] = features | {"physical_gate": False}
                    continue
                edits, gates = match(h, plan, target, cfg, dose, np)
                result = features | gates
                if result["physical_gate"]:
                    result["heads"] = {arm: add(p.after, pair, local) for arm, p in edits.items()}
                record["policies"][key] = result
            records.append(record)
    live = full_head(head, np.stack(states), np.asarray(refs), np.asarray(labels), dose["block_rows"], check, np)
    bare = live["bare"][:, 0] - live["bare"][:, 1]
    norm_error = top_error = 0.
    for row in records:
        clean, saved = row.pop("clean"), row.pop("saved")
        if live["winner"][clean] != saved["free_top_token_id"] or live["vocab"] != saved["vocabulary_size"]:
            raise ValueError("Recomputed native greedy winner differs")
        norm_error = max(norm_error, abs(float(live["logz"][clean]) - saved["full_log_normalizer"]))
        top_error = max(top_error, abs(float(live["top_logit"][clean]) - saved["free_top_token_logit"]))
        for result in row["policies"].values():
            if not result["physical_gate"]:
                result["full_pass"] = False
                continue
            for arm, index in result.pop("heads").items():
                peak = float(live["max_abs_delta"][index])
                margin = float(abs(bare[index] - bare[clean]))
                same = bool(live["winner"][index] == live["winner"][clean])
                result["arm_dose"][arm].update(max_abs_logit_delta=peak, abs_margin_delta=margin, same_winner=same,
                                              head_qualified=same and peak <= dose["maximum_head_delta"]
                                              and margin <= dose["maximum_margin_delta"])
            result["full_pass"] = all(a["head_qualified"] for a in result["arm_dose"].values())
    if max(norm_error, top_error, raw_closure) > base["native_logit_abs_tol"]:
        raise ValueError("Native baseline archive did not pass replay")
    return records, {"max_native_logz_error": norm_error, "max_native_top_error": top_error,
                     "max_native_selected_error": raw_closure, "head_rows_per_residual": live["vocab"],
                     "head_scored_residual_states": len(states)}


def summarize(rows, names, backends):
    def median_or_missing(values):
        return median(values) if values else None

    summaries = {}
    for key in names:
        per_backend, across = {}, defaultdict(bool)
        for backend in backends:
            subset = [r for r in rows if r["backend"] == backend]
            roots = defaultdict(dict)
            for row in subset:
                roots[row["root"]][row["world"]] = row["policies"][key]
            if len(roots) != 4 or any(set(worlds) != {"A", "B"} for worlds in roots.values()):
                raise ValueError("Incomplete four-root CAL subpanel")
            for root, worlds in roots.items():
                joint = all(v["full_pass"] for v in worlds.values())
                across[root] = (across[root] and joint) if root in across else joint
            pairs = [r["policies"][key] for r in subset]
            tails = [v for v in pairs if v["full_pass"]]
            per_backend[backend] = {"roots": len(roots), "worlds": len(subset),
                **{step: sum(r.get(step, False) for r in pairs) for step in (
                    "dose_matched", "contrast_retained", "sham_specific", "far_enough", "physical_gate", "full_pass")},
                "both_worlds_pass_roots": sum(all(value["full_pass"] for value in worlds.values()) for worlds in roots.values()),
                "median_support_size": median(v["support"] for v in pairs),
                "median_continuous_contrast_cosine": median_or_missing(
                    [v["continuous_contrast_cosine"] for v in pairs if "continuous_contrast_cosine" in v]),
                "median_minimum_reached_contrast_cosine_among_passes": median_or_missing([
                    min(v["arm_dose"][f"contrast/{s}"]["cosine_with_original_contrast"] for s in ("positive", "negative"))
                    for v in tails]),
            }
        summaries[key] = {"per_backend": per_backend,
                           "both_worlds_both_precisions_roots": sum(across.values()),
                           "total_passing_world_precision_rows": sum(x["full_pass"] for x in per_backend.values())}
    return summaries


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
        return
    if not (args.snapshot and args.output):
        cli.error("--snapshot and --output required")
    if json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("ROUND54 source freeze changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Checkpoint revision differs")
    candidates = [(f"step_ratio_{x}", x) for x in cfg["spacing_to_expected_step"]] + [("dense", None)]
    fit_roster, check_roster = rosters(base)
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Replay package pins differ")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(cfg, dose, np)
        verify_reference(np)
        bank = calibrations(base, masks, np)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        toks = {c: tokenizer.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in toks.values()) or len({ids[0] for ids in toks.values()}) != 16:
            raise ValueError("Bare tokenizer labels changed")
        tokens = {c: ids[0] for c, ids in toks.items()}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            fit, fit_custody = capture_stage(fit_roster, candidates, bank, base, cfg, dose, head, tokens, check, masks, np)
            tuning = summarize(fit, [key for key, _ in candidates], base["backends"])
            objective = lambda key: (tuning[key]["both_worlds_both_precisions_roots"],
                                     tuning[key]["total_passing_world_precision_rows"])
            selected = max(tuning, key=objective)
            print(f"CAL fit selected {selected} with {objective(selected)}", flush=True)
            held = [(key, t) for key, t in candidates if key in {selected, "dense"}]
            reviewed, check_custody = capture_stage(check_roster, held, bank, base, cfg, dose, head, tokens, check, masks, np)
            checking = summarize(reviewed, [key for key, _ in held], base["backends"])
        check()
        if "torch" in sys.modules:
            raise RuntimeError("CAL audit imported Torch")
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private.json"
    write_new(private, {"fit": fit, "check": reviewed})
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "source_manifest_sha256": digest(MANIFEST), "status": "completed",
        "private_sha256": digest(private), "selected": selected, "calibration_failed": not objective(selected)[1],
        "fit": tuning, "check": checking, "custody": {"fit": fit_custody, "check": check_custody},
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)},
    })
    print(f"Disjoint CAL check for {selected}: {checking[selected]['total_passing_world_precision_rows']} passing world-precision rows")


if __name__ == "__main__":
    main()
