"""Fit and replay BF16-aware shams on archived NF4 route contrasts."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dark44.head import full_head
from experiments.nan_dose51.route import SIGNS, directions, patches
from experiments.nan_frontier55.run import check_head
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations, rosters

from .geometry import basis, candidate, drive, span_fraction, verify


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
ACTIVE_AGG = REPO / "reports/nanbeige_bpd_gym/nan_frontier55_active_head.json"


def seal():
    paths = [HERE / p for p in (".gitattributes", "__init__.py", "geometry.py", "run.py", "PROTOCOL.md",
                               "registration.json")]
    paths += [ACTIVE_AGG, ROUND / "registration.json", ROUND / "geometry.py", ROUND / "run.py",
              READPATH / "registration.json", DOSE / "registration.json"]
    paths += [REPO / name for name in (
        "experiments/nan_dose51/route.py", "experiments/nan_frontier55/ACTIVE_HEAD_MANIFEST.json",
        "experiments/nan_frontier55/run.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_dark44/head.py",
        "experiments/nan_readpath37/run.py", "experiments/nan_delay39/instrument.py")]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in paths}


def plans(cfg):
    return [(f"{method}/{cap:g}", method, cap)
            for method in ("screened_bpd", "ambient") for cap in cfg["spacing_caps"]]


def physical_head_closure(rows, live, prior, dose, base):
    bare = live["bare"][:, 0] - live["bare"][:, 1]
    for row in rows:
        old = prior[row["root"], row["world"]]["recipes"]["0.0075/dense"]["arm_dose"]
        for key, idx in row.pop("fixed_active_head_indices").items():
            peak = float(live["max_abs_delta"][idx])
            margin = float(abs(bare[idx] - bare[row["clean"]]))
            same = bool(live["winner"][idx] == live["winner"][row["clean"]])
            expected = old[key]
            if (peak, margin, same) != (expected["max_abs_logit_delta"], expected["abs_margin_delta"],
                                        expected["same_winner"]):
                raise ValueError("Active-only full head changed from the FRONTIER55 receipt")
        for option in row["recipes"].values():
            for key, index in option.get("head_indices", {}).items():
                option["arm_dose"][key]["rms_logit_shift"] = float(live["rms_delta"][index])
    return check_head(rows, live, dose, base["native_logit_abs_tol"])


def collect(source, policies, archive, head, tokens, base, cfg, round_cfg, dose, model, check, masks, np):
    backend = "nf4"
    bank, controller, eps, anchor = model
    write_span = basis(bank["U"].T.astype(np.float64), np)
    location = ARCHIVE / backend
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", base["backends"][backend]["audit"]))
    slots = {(row["root"], row["world"]): slot for slot, row in source.items()}
    states, refs, labels, rows = [], [], [], []
    registered = {}

    def add(state, pair, ref=None):
        state = np.asarray(state, dtype=np.float32)
        if ref is not None:
            identity = ref, state.tobytes()
            if identity in registered:
                return registered[identity]
            registered[identity] = len(states)
        i = len(states)
        states.append(state)
        refs.append(i if ref is None else ref)
        labels.append(pair)
        return i

    for meta, (h, x, y) in route_states(backend, base, source, check, np):
        h = h.astype(np.float64)
        root, world = meta["root"], meta["world"]
        slot = slots[root, world]
        opposite = source[slots[root, "B" if world == "A" else "A"]]
        pair = [tokens[opposite["answer"]], tokens[meta["answer"]]]
        case = json.loads(bound(location / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
        phase = case["initial_phases"][0]
        if phase["loop"] != 1 or pair[0] == pair[1]:
            raise ValueError("Wrong native phase or ambiguous labels")
        clean = add(h, pair)
        axes, _ = directions(x, y, masks(x[None], bank, controller)[0], anchor, bank, eps, np)
        contrast = axes["learned"] - axes["static"]
        grad_scale = head.gamma.astype(np.float64) * (
            head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
        gradient = head_gradient(h, grad_scale, head.eps, np)
        radius = dose["phase1_dose_fraction"] * float(np.linalg.norm(h))
        tag = f"{backend}/{root}/{world}"
        dense, raw, _ = make_pair(h, contrast, gradient, radius, None, tag, round_cfg, np)
        active, status, admitted = patches(h, {"contrast": dense["contrast"]}, dose, np)
        old = archive[root, world]["recipes"]["0.0075/dense"]["arm_dose"]
        if not admitted or any(any(old[k][field] != value for field, value in info.items()) for k, info in status.items()):
            raise ValueError("FRONTIER55 active geometry did not replay")
        reached = [p.after - p.before for p in active.values()]
        seeded = drive(tag, cfg["sham_namespace"], len(h), np)
        baseline_edits = {k: add(p.after, pair, clean) for k, p in active.items()}
        row = {"root": root, "world": world, "clean": clean, "citation": phase["score"],
               "fixed_active_head_indices": baseline_edits, "recipes": {}}
        for name, method, cap in policies:
            check()
            v, count = candidate(h, bank["U"], raw, gradient, reached, radius, cap, method, seeded, np)
            option = {"physical_gate": False, "sham_support_size": count}
            row["recipes"][name] = option
            if v is None:
                continue
            edits, audit_bits = match(h, {"contrast": dense["contrast"], "sham": v}, raw, round_cfg, dose, np)
            option.update(audit_bits)
            if any(not np.array_equal(edits[k].after, p.after) for k, p in active.items()):
                raise ValueError("A sham recipe changed the fixed active edit")
            option["sham_energy_in_original_span"] = {
                sign: span_fraction(edits[f"sham/{sign}"].after - h, write_span, np) for sign in SIGNS}
            if option["physical_gate"]:
                option["head_indices"] = {k: add(p.after, pair, clean) for k, p in edits.items()}
        rows.append(row)
    if len(rows) != 8:
        raise ValueError("The fixed CAL panel changed")
    print(f"Scoring {len(states)} unique NF4 first-pass states", flush=True)
    live = full_head(head, np.stack(states), np.asarray(refs), np.asarray(labels), dose["block_rows"], check, np)
    return rows, physical_head_closure(rows, live, archive, dose, base)


def summarize(rows, policies):
    out = {}
    for name, _, _ in policies:
        recipe = [r["recipes"][name] for r in rows]
        roots = {r["root"] for r in rows}
        if len(roots) != 4 or any({r["world"] for r in rows if r["root"] == k} != {"A", "B"} for k in roots):
            raise ValueError("A source root was missing an A/B world")
        passing = [r for r in recipe if r["full_pass"]]
        shams = [r["arm_dose"][f"sham/{s}"] for r in passing for s in SIGNS]
        active = [r["arm_dose"][f"contrast/{s}"] for r in passing for s in SIGNS]
        peak_gap = [s["max_abs_logit_delta"] - a["max_abs_logit_delta"] for s, a in zip(shams, active)]
        out[name] = {
            "worlds": len(rows),
            "both_worlds_roots": sum(all(r["recipes"][name]["full_pass"] for r in rows if r["root"] == root)
                                     for root in roots),
            **{k: sum(r.get(k, False) for r in recipe)
               for k in ("dose_matched", "contrast_retained", "sham_specific", "physical_gate", "full_pass")},
            "median_sham_support_size": median(r["sham_support_size"] for r in recipe),
            "median_sham_energy_in_original_span_on_pass": median(
                r["sham_energy_in_original_span"][s] for r in passing for s in SIGNS) if passing else None,
            "max_absolute_sham_leakage_on_pass": max((abs(r["cosine_with_original_contrast"]) for r in shams), default=None),
            "max_sham_head_peak_on_pass": max((r["max_abs_logit_delta"] for r in shams), default=None),
            "median_sham_head_rms_on_pass": median(r["rms_logit_shift"] for r in shams) if shams else None,
            "median_active_head_rms_on_pass": median(r["rms_logit_shift"] for r in active) if active else None,
            "median_sham_minus_active_head_peak_on_pass": median(peak_gap) if peak_gap else None,
        }
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--active", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("SHAM56 source freeze changed")
    if not (args.snapshot and args.active and args.output):
        cli.error("--snapshot, --active, and --output are required")
    if digest(args.active / "aggregate.json") != digest(ACTIVE_AGG):
        raise ValueError("FRONTIER55 aggregate changed")
    parent = json.loads(ACTIVE_AGG.read_bytes())
    parent_rows = json.loads(bound(args.active / "private.json", parent["private_sha256"]))
    cfg = json.loads((HERE / "registration.json").read_bytes())
    round_cfg = json.loads((ROUND / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes()) | {"phase1_dose_fraction": cfg["phase1_dose_fraction"]}
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Frozen checkpoint revision changed")
    fit, held = rosters(base)
    recipes = plans(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Replay package pins changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(cfg, np)
        model = calibrations(base, masks, np)["nf4"]
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        encoded = {c: tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in encoded.values()) or len({v[0] for v in encoded.values()}) != 16:
            raise ValueError("Pinned bare labels changed")
        tokens = {c: v[0] for c, v in encoded.items()}
        outcome = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for panel, roster in (("fit", fit), ("check", held)):
                archive = {(r["root"], r["world"]): r for r in parent_rows if r["panel"] == panel}
                rows, custody = collect(roster, recipes, archive, head, tokens, base, cfg, round_cfg, dose, model,
                                        check, masks, np)
                summary = summarize(rows, recipes)
                destination = args.output / f"{panel}_private.json"
                write_new(destination, rows)
                outcome[panel] = summary | {"custody": custody, "private_sha256": digest(destination)}
                if panel == "fit":
                    selected = max(summary, key=lambda label: (summary[label]["both_worlds_roots"],
                                                               summary[label]["full_pass"]))
                    write_new(args.output / "fit_selection.json", {"selected": selected, "fit_sha256": digest(destination)})
                    print(f"Fit selected {selected}: {summary[selected]['full_pass']}/8 worlds", flush=True)
                    recipes = [r for r in recipes if r[0] == selected]
        if "torch" in sys.modules:
            raise RuntimeError("Archive replay imported Torch")
        check()
    aggregate = {"experiment": cfg["experiment"], "status": "completed", "selected": selected,
                 "panels": outcome, "source_manifest_sha256": digest(MANIFEST), "frontier_active_sha256": digest(ACTIVE_AGG),
                 "fit_selection_sha256": digest(args.output / "fit_selection.json"),
                 "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                 "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
                 "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False}
    write_new(args.output / "aggregate.json", aggregate)


if __name__ == "__main__":
    main()
