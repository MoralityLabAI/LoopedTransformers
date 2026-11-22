"""Score the finite current head of reachable NF4 edits that lacked a matched sham."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_dark44.head import full_head
from experiments.nan_delay39.contrasts import qualified_pair
from experiments.nan_dose51.route import directions
from experiments.nan_gauge42.run import ARCHIVE
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, write_new
from experiments.nan_round54.geometry import make_pair, match
from experiments.nan_round54.run import DOSE, HERE as ROUND, READPATH, REPO, calibrations, rosters

from .run import CONFIG, HERE, MANIFEST, check_head


LOCK = HERE / "ACTIVE_HEAD_MANIFEST.json"
RECIPES = (.0075, .015)


def freeze():
    files = [Path(__file__), HERE / "ACTIVE_HEAD.md", MANIFEST, HERE / "run.py",
             ROUND / "geometry.py", DOSE / "registration.json"]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in files}


def prepare(base, head, tokens, archive, round_cfg, dose, roster_pair, check, masks, np):
    backend = "nf4"
    bank, ci, eps, anchor = calibrations(base, masks, np)[backend]
    location = ARCHIVE / backend
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", base["backends"][backend]["audit"]))
    states, refs, labels, records = [], [], [], []

    def append(state, pair, ref=None):
        i = len(states)
        states.append(state)
        refs.append(i if ref is None else ref)
        labels.append(pair)
        return i

    for panel, source in zip(("fit", "check"), roster_pair):
        slots = {(r["root"], r["world"]): i for i, r in source.items()}
        for row, (h, x, y) in route_states(backend, base, source, check, np):
            h = h.astype(np.float64)
            root, world = row["root"], row["world"]
            slot = slots[root, world]
            twin = source[slots[root, "B" if world == "A" else "A"]]
            pair = [tokens[twin["answer"]], tokens[row["answer"]]]
            case = json.loads(bound(location / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
            first = case["initial_phases"][0]
            if first["loop"] != 1:
                raise ValueError("Wrong native readout phase")
            clean = append(h, pair)
            record = {"panel": panel, "root": root, "world": world, "clean": clean,
                      "citation": first["score"], "recipes": {}}
            axes, _ = directions(x, y, masks(x[None], bank, ci)[0], anchor, bank, eps, np)
            contrast = axes["learned"] - axes["static"]
            g = head.gamma.astype(np.float64) * (
                head.row(pair[0]).astype(np.float64) - head.row(pair[1]).astype(np.float64))
            gradient = head_gradient(h, g, head.eps, np)
            for nominal in RECIPES:
                check()
                name = f"{nominal:.4f}/dense"
                rule = dose | {"phase1_dose_fraction": nominal}
                radius = nominal * float(np.linalg.norm(h))
                planned, original, _ = make_pair(h, contrast, gradient, radius, None,
                                                  f"{backend}/{root}/{world}", round_cfg, np)
                old = archive[panel][root, world]["recipes"][name]
                if planned is None:
                    if "arm_dose" in old:
                        raise ValueError("The original geometry receipt differs")
                    record["recipes"][name] = {"active_pair_admitted": False, "physical_gate": False}
                    continue
                edits, result = match(h, planned, original, round_cfg, rule, np)
                dose_fields = ("eligible", "requested_l2", "realized_l2", "direction_cosine",
                               "cosine_with_original_contrast")
                for arm, item in result["arm_dose"].items():
                    if any(item.get(field) != old["arm_dose"][arm].get(field) for field in dose_fields):
                        raise ValueError("Recomputed edit differs from the parent")
                signed = ("contrast/positive", "contrast/negative")
                passed = result["contrast_retained"] and qualified_pair(
                    *(result["arm_dose"][arm] for arm in signed), rule)
                cell = {"active_pair_admitted": passed, "physical_gate": passed,
                        "arm_dose": {arm: result["arm_dose"][arm] for arm in signed}}
                if passed:
                    cell["head_indices"] = {arm: append(edits[arm].after, pair, clean) for arm in signed}
                record["recipes"][name] = cell
            records.append(record)
    return records, np.stack(states), np.asarray(refs), np.asarray(labels)


def summarize(rows, dose):
    panel_metrics = {}
    for panel in ("fit", "check"):
        stats = {}
        stage = [r for r in rows if r["panel"] == panel]
        if len(stage) != 8 or len({r["root"] for r in stage}) != 4:
            raise ValueError("The CAL evaluation roster changed")
        for name in stage[0]["recipes"]:
            cells = [r["recipes"][name] for r in stage]
            live = [c for c in cells if c["active_pair_admitted"]]
            peak = [max(a["max_abs_logit_delta"] for a in c["arm_dose"].values()) for c in live]
            count = Counter()
            for c in live:
                count["margin"] += any(a["abs_margin_delta"] > dose["maximum_margin_delta"] for a in c["arm_dose"].values())
                count["greedy"] += any(not a["same_winner"] for a in c["arm_dose"].values())
            stats[name] = {"worlds": len(cells), "admitted_active_pairs": len(live),
                           "passed_both_active_heads": sum(c["full_pass"] for c in live),
                           "peak_failure_worlds": sum(v > dose["maximum_head_delta"] for v in peak),
                           "margin_failure_worlds": count["margin"], "greedy_failure_worlds": count["greedy"],
                           "median_worst_active_peak": median(peak) if peak else None}
        panel_metrics[panel] = stats
    return panel_metrics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--frontier", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(LOCK, freeze())
        return
    if not (args.snapshot and args.frontier and args.output):
        cli.error("--snapshot, --frontier, and --output required")
    if not __debug__ or json.loads(LOCK.read_bytes()) != freeze():
        raise ValueError("Active-head source freeze changed")
    root = args.frontier
    terminal = json.loads((root / "terminal.json").read_bytes())
    receipt = root / "aggregate.json"
    previous = json.loads(bound(receipt, terminal["aggregate_sha256"]))
    if terminal["status"] != "completed" or previous["source_manifest_sha256"] != digest(MANIFEST):
        raise ValueError("Unexpected parent result status or source")
    archive = {}
    for panel in ("fit", "check"):
        verified = json.loads(bound(root / f"{panel}_private.json", previous["panels"][panel]["private_sha256"]))
        archive[panel] = {(r["root"], r["world"]): r for r in verified if r["backend"] == "nf4"}
    base = json.loads((READPATH / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes())
    cfg = json.loads(CONFIG.read_bytes())
    round_cfg = json.loads((ROUND / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Pinned snapshot revision changed")
    with lease(cfg) as (check, samples):
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        ids = {c: tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOP"}
        if any(len(v) != 1 for v in ids.values()) or len({v[0] for v in ids.values()}) != 16:
            raise ValueError("Pinned bare label IDs changed")
        tokens = {c: v[0] for c, v in ids.items()}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            rows, states, refs, labels = prepare(base, head, tokens, archive, round_cfg, dose, rosters(base), check, masks, np)
            print(f"Streaming {len(states)} NF4 active or clean head states", flush=True)
            live = full_head(head, states, refs, labels, dose["block_rows"], check, np)
            closure = check_head(rows, live, dose, base["native_logit_abs_tol"])
        check()
        if "torch" in sys.modules:
            raise RuntimeError("Auxiliary head replay imported Torch")
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private.json"
    write_new(private, rows)
    data = {"experiment": "NAN-FRONTIER-55-ACTIVE-HEAD", "source_manifest_sha256": digest(LOCK),
            "frontier_aggregate_sha256": digest(receipt), "private_sha256": digest(private),
            "panels": summarize(rows, dose), "custody": closure, "formal_comparisons": 0,
            "new_pretrained_forwards": 0, "sealed_test_accessed": False,
            "peak_private_bytes": max(item["private_bytes"] for item in samples)}
    write_new(args.output / "aggregate.json", data)
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
