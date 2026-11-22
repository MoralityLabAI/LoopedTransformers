"""Observe the signed physical dose/current-head tradeoff on archived COMP14 CAL."""
import argparse
from collections import defaultdict
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dark44.head import full_head, verify_reference
from experiments.nan_dose51.route import SIGNS, directions
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bf16, bound, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations, rosters

from .lower import proof_data, signed_nominal_floor, verify


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "registration.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
ROUND_AGG = REPO / "reports/nanbeige_bpd_gym/nan_round54_results.json"


def seal():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "lower.py", "run.py", "PROTOCOL.md",
                                      "registration.json")]
    paths += [ROUND_AGG, ROUND / "SOURCE_MANIFEST.json", ROUND / "run.py", ROUND / "geometry.py",
              DOSE / "registration.json", READPATH / "registration.json"]
    paths += [REPO / name for name in (
        "experiments/nan_dark44/head.py", "experiments/nan_dark44/registration.json",
        "experiments/nan_dose51/route.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_phase35/run.py", "experiments/nan_route50/run.py")]
    return {"experiment": "NAN-FRONTIER-55", "sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def recipes(cfg, prior):
    name = prior["selected"]
    if name != "step_ratio_0.8":
        raise ValueError("Parent calibration selected a different support rule")
    return [(f"{dose:.4f}/{method}", dose, method, support)
            for dose in cfg["dose_fractions"] for method, support in ((name, .8), ("dense", None))]


def prepare_panel(source, cfg, grid, dose_ref, base, banks, token_ids, head, check, masks, np):
    keys = {(row["root"], row["world"]): slot for slot, row in source.items()}
    states, refs, labels, records = [], [], [], []
    variants = {}

    def add(h, pair, ref=None):
        vec = np.asarray(h, dtype=np.float32)
        if ref is not None:
            tag = ref, vec.tobytes()
            if tag in variants:
                return variants[tag]
            variants[tag] = len(states)
        local = len(states)
        states.append(vec)
        refs.append(local if ref is None else ref)
        labels.append(pair)
        return local

    for backend, (bank, ci, eps, anchor) in banks.items():
        pin = base["backends"][backend]
        audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
        for meta, (h, x, y) in route_states(backend, base, source, check, np):
            h = h.astype(np.float64)
            if not np.array_equal(h, bf16(h, np)):
                raise ValueError("The native state must be exactly representable in BF16")
            root, world = meta["root"], meta["world"]
            foil = source[keys[root, "B" if world == "A" else "A"]]
            pair = [token_ids[foil["answer"]], token_ids[meta["answer"]]]
            slot = keys[root, world]
            case = json.loads(bound(ARCHIVE / backend / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
            phase = case["initial_phases"][0]
            if phase["loop"] != 1 or pair[0] == pair[1]:
                raise ValueError("Wrong phase or ambiguous task readout")
            scores = phase["score"]
            residual_idx = add(h, pair)
            ref = {"backend": backend, "root": root, "world": world, "clean": residual_idx,
                   "citation": {k: scores[k] for k in ("free_top_token_id", "free_top_token_logit",
                                                       "full_log_normalizer", "vocabulary_size")},
                   "floor": {}, "recipes": {}}
            delta, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
            raw = delta["learned"] - delta["static"]
            normal = head.gamma.astype(np.float64) * (
                head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
            gradient = head_gradient(h, normal, head.eps, np)
            floors = {}
            for sign, sign_scale in SIGNS.items():
                floors[sign] = signed_nominal_floor(
                    h, sign_scale * raw, cfg["minimum_reached_contrast_cosine"],
                    dose_ref["dose_relative_tolerance"], np)
                ref["floor"][sign] = proof_data(floors[sign])
            tag = f"{backend}/{root}/{world}"
            for name, nominal, _, support in grid:
                check()
                target_square = Fraction(str(nominal))**2
                excluded = any(target_square < exact for exact in floors.values())
                dose = dose_ref | {"phase1_dose_fraction": nominal}
                radius = nominal * float(np.linalg.norm(h))
                arms, original, features = make_pair(h, raw, gradient, radius, support, tag, cfg, np)
                cell = features | {"exact_support_floor_excludes": excluded, "physical_gate": False}
                ref["recipes"][name] = cell
                if arms is None:
                    continue
                edits, result = match(h, arms, original, cfg, dose, np)
                cell.update(result)
                if cell["physical_gate"]:
                    if excluded:
                        raise ValueError("A physical candidate did not respect its exact support bound")
                    cell["head_indices"] = {arm: add(p.after, pair, residual_idx) for arm, p in edits.items()}
            records.append(ref)
    return records, np.stack(states), np.asarray(refs), np.asarray(labels)


def check_head(records, live, dose, tolerance):
    maximum_top = maximum_lz = 0.
    bare = live["bare"][:, 0] - live["bare"][:, 1]
    count = len(live["winner"])
    for record in records:
        idx, old = record.pop("clean"), record.pop("citation")
        if old["free_top_token_id"] != live["winner"][idx] or old["vocabulary_size"] != live["vocab"]:
            raise ValueError("Unedited native full-vocabulary closure did not reproduce")
        maximum_top = max(maximum_top, abs(float(live["top_logit"][idx]) - old["free_top_token_logit"]))
        maximum_lz = max(maximum_lz, abs(float(live["logz"][idx]) - old["full_log_normalizer"]))
        for cell in record["recipes"].values():
            cell["full_pass"] = False
            if "head_indices" not in cell:
                continue
            for arm, head_idx in cell.pop("head_indices").items():
                peak = float(live["max_abs_delta"][head_idx])
                margin = float(abs(bare[head_idx] - bare[idx]))
                same_greedy = bool(live["winner"][head_idx] == live["winner"][idx])
                cell["arm_dose"][arm].update(
                    max_abs_logit_delta=peak, abs_margin_delta=margin, same_winner=same_greedy,
                    head_qualified=same_greedy and peak <= dose["maximum_head_delta"]
                    and margin <= dose["maximum_margin_delta"])
            cell["full_pass"] = all(g["head_qualified"] for g in cell["arm_dose"].values())
    if max(maximum_top, maximum_lz) > tolerance:
        raise ValueError("Unedited head exceeded native archive tolerance")
    return {"clean_max_top_logit_delta": maximum_top, "clean_max_log_normalizer_delta": maximum_lz,
            "vocabulary_rows_per_state": int(live["vocab"]), "unique_head_states": count}


def summarize(records, grid, dose_ref):
    by_backend = defaultdict(list)
    for item in records:
        by_backend[item["backend"]].append(item)
    floors = {}
    for backend, rows in by_backend.items():
        if len(rows) != 8:
            raise ValueError("The original CAL panel changed")
        values = [p["nominal_dose_lower_bound"] for r in rows for p in r["floor"].values()]
        floors[backend] = {"signed_targets": len(values), "min": min(values),
                           "median": median(values), "max": max(values)}
    recipes = {}
    for key, fraction, _, _ in grid:
        root_sets, stats = [], {}
        for backend, rows in by_backend.items():
            groups = defaultdict(dict)
            for row in rows:
                groups[row["root"]][row["world"]] = row["recipes"][key]
            if len(groups) != 4 or any(set(worlds) != {"A", "B"} for worlds in groups.values()):
                raise ValueError("Four complete CAL twins were required")
            cells = [row["recipes"][key] for row in rows]
            qualified = [cell for cell in cells if cell["physical_gate"]]
            max_first = [max(arm["max_abs_logit_delta"] for arm in cell["arm_dose"].values())
                         for cell in qualified]
            root_set = {root for root, worlds in groups.items() if all(c["full_pass"] for c in worlds.values())}
            root_sets.append(root_set)
            stats[backend] = {
                "worlds": len(cells), "both_worlds_roots": len(root_set),
                **{field: sum(cell.get(field, False) for cell in cells)
                   for field in ("exact_support_floor_excludes", "dose_matched", "contrast_retained",
                                 "sham_specific", "physical_gate", "full_pass")},
                "worst_logit_delta_median_across_physical_worlds": median(max_first) if max_first else None,
                "physical_worlds_peak_failed": sum(value > dose_ref["maximum_head_delta"] for value in max_first),
                "physical_worlds_margin_failed": sum(any(arm["abs_margin_delta"] > dose_ref["maximum_margin_delta"]
                                                         for arm in c["arm_dose"].values()) for c in qualified),
                "physical_worlds_greedy_changed": sum(any(not arm["same_winner"] for arm in c["arm_dose"].values())
                                                     for c in qualified),
            }
        recipes[key] = {"dose_fraction": fraction, "both_worlds_roots_in_both_precisions": len(set.intersection(*root_sets)),
                        "total_world_precision_passes": sum(x["full_pass"] for x in stats.values()),
                        "per_backend": stats}
    return {"exact_necessary_nominal_dose": floors, "recipes": recipes}


def parent_closure(panel, prior, archived):
    by_id = {(r["backend"], r["root"], r["world"]): r for r in archived[panel]}
    fields = ("dose_matched", "contrast_retained", "sham_specific", "far_enough", "physical_gate", "full_pass")
    checks = 0
    for row in prior:
        old = by_id[row["backend"], row["root"], row["world"]]["policies"]
        for method in ("step_ratio_0.8", "dense"):
            new = row["recipes"][f"0.0025/{method}"]
            if any(old[method].get(field) != new.get(field) for field in fields):
                raise ValueError("The retained 0.25% control differs from the ROUND54 result")
            checks += len(fields)
    return checks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--round-archive", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Use native archive assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not (args.snapshot and args.round_archive and args.output):
        cli.error("--snapshot, --round-archive, and --output required")
    if json.loads(MANIFEST.read_bytes()) != seal() or digest(args.round_archive / "aggregate.json") != digest(ROUND_AGG):
        raise ValueError("FRONTIER55 freeze or parent CAL custody changed")
    cfg = json.loads(CONFIG.read_bytes())
    parent = json.loads(ROUND_AGG.read_bytes())
    parent_raw = json.loads(bound(args.round_archive / "private.json", parent["private_sha256"]))
    base = json.loads((READPATH / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes())
    physical = json.loads((ROUND / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Checkpoint revision differs")
    grid = recipes(cfg, parent)
    fit, held = rosters(base)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with lease(cfg) as (check, samples):
            if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
                raise ValueError("Replay runtime pins differ")
            import numpy as np
            from tokenizers import Tokenizer
            from experiments.nan_fit07.pointwise import masks

            verify(np)
            verify_reference(np)
            banks = calibrations(base, masks, np)
            bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
            tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
            ids = {c: tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
            if any(len(v) != 1 for v in ids.values()) or len({v[0] for v in ids.values()}) != 16:
                raise ValueError("Bare label tokenizer IDs changed")
            ids = {c: v[0] for c, v in ids.items()}
            results = {}
            with (args.snapshot / SHARD).open("rb") as stream:
                head = FixedHead(stream, args.snapshot, base, np, check)
                for panel, selected_roster in (("fit", fit), ("check", held)):
                    rows, states, refs, labels = prepare_panel(
                        selected_roster, physical, grid, dose, base, banks, ids, head, check, masks, np)
                    print(f"{panel}: streaming head for {len(states)} unique residual states", flush=True)
                    live = full_head(head, states, refs, labels, dose["block_rows"], check, np)
                    custody = check_head(rows, live, dose, base["native_logit_abs_tol"])
                    closure_checks = parent_closure(panel, rows, parent_raw)
                    stats = summarize(rows, grid, dose)
                    private = args.output / f"{panel}_private.json"
                    write_new(private, rows)
                    results[panel] = stats | {"custody": custody, "round54_gate_fields_reproduced": closure_checks,
                                                "private_sha256": digest(private)}
                    if panel == "fit":
                        def criterion(item):
                            m = stats["recipes"][item]
                            nf4 = m["per_backend"]["nf4"]
                            return (m["both_worlds_roots_in_both_precisions"], nf4["both_worlds_roots"],
                                    nf4["full_pass"], m["total_world_precision_passes"])

                        chosen = max(stats["recipes"], key=criterion)
                        write_new(args.output / "fit_selection.json", {"recipe": chosen, "criterion": criterion(chosen),
                                                                        "fit_private_sha256": digest(private)})
                        print(f"Selected {chosen}: {criterion(chosen)}", flush=True)
            check()
            if "torch" in sys.modules:
                raise RuntimeError("Audit imported Torch")
    except BaseException as error:
        write_new(args.output / "terminal.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    record = {"experiment": cfg["experiment"], "status": "completed", "selected": chosen, "panels": results,
        "formal_comparisons": 0, "new_pretrained_forwards": 0, "sealed_test_accessed": False,
        "source_manifest_sha256": digest(MANIFEST), "parent_aggregate_sha256": digest(ROUND_AGG),
        "fit_selection_sha256": digest(args.output / "fit_selection.json"),
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)}}
    path = args.output / "aggregate.json"
    write_new(path, record)
    write_new(args.output / "terminal.json", {"status": "completed", "aggregate_sha256": digest(path)})


if __name__ == "__main__":
    main()
