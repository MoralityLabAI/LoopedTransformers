"""Audit the compensated 0.5% NF4 MLP shell and ordered head bounds."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_carrybound88.bounds import calculate, fixture, residual_box
from experiments.nan_carrybound88.run import BASE, BUDGET, LIMIT, REPO, STAGES
from experiments.nan_inverse65.lattice import add, lattice
from experiments.nan_inverse65.run import open_archive
from experiments.nan_lattice53.run import route_states
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new
from experiments.nan_routematch86.geometry import ARMS
from experiments.nan_shelldark89.run import focus_split
from experiments.nan_tangent87.worker import prepare_fit
from experiments.nan_vocab81.run import LABELS, native_controls, token_classes

from .geometry import FRACTION, fixture as repair_fixture, plan
from .ordered import fixture as ordered_fixture, refine


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PRIOR = REPO / "reports/nanbeige_bpd_gym/nan_shelldark89_results.json"
OLD = REPO / "runs/nan_shelldark89/attempt01/private.json"


def seal():
    names = (".gitattributes", "__init__.py", "geometry.py", "ordered.py", "PROTOCOL.md", "run.py")
    files = {HERE / name for name in names}
    files.update((PRIOR, BASE, REPO / "experiments/nan_cibracket75/repair.py"))
    previous = json.loads(PRIOR.read_bytes())
    for folder in ("nan_shelldark89", "nan_cibracket75"):
        path = REPO / "experiments" / folder / "SOURCE_MANIFEST.json"
        files.add(path)
        if folder == "nan_shelldark89" and digest(path) != previous["source_manifest_sha256"]:
            raise ValueError("Prior source receipt changed")
        for name, sha in json.loads(path.read_bytes())["files_sha256"].items():
            candidate = REPO / name
            bound(candidate, sha)
            files.add(candidate)
    if (previous["development"]["fraction"] != FRACTION or
            digest(OLD) != previous["private_sha256"]):
        raise ValueError("Prior dose or private audit changed")
    return {"experiment": "NAN-SHELLREPAIR-90", "dose_fraction": FRACTION, "resource_budget": BUDGET,
            "prior_private_sha256": previous["private_sha256"], "current_logit_limit": LIMIT,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(files)}}


def status(winner, peak, pair):
    if winner == "changed" or peak[0] > LIMIT or pair[0] > LIMIT or pair[1] < -LIMIT:
        return "failed"
    if winner == "preserved" and peak[1] <= LIMIT and max(map(abs, pair)) <= LIMIT:
        return "preserved"
    return "unresolved"


def panel(base, selected, before, head, routes, tokenizer, classes, canon, check, np):
    previous_geometry = {(r["root"], r["world"]): r for r in before["geometry"] if r["dose"] == FRACTION}
    previous_heads = {(r["root"], r["world"], r["arm"]): r for r in before["readouts"] if r["dose"] == FRACTION}
    states, replay_error = open_archive("nf4", base, selected, head, STAGES, check, np)
    vectors = list(route_states("nf4", base, selected, check, np))
    h, x, w = (np.stack([v[i] for _, v in vectors]) for i in range(3))
    directions = np.stack(routes(x.astype(np.float64), w.astype(np.float64)), axis=1)
    grid = lattice(np)
    site = STAGES.index((1, 21))
    geometry, arms, edits, owners, target, foil = [], [], [], [], [], []
    for i, (source, _) in enumerate(vectors):
        check()
        root, world = source["root"], source["world"]
        pair = states[source["split"], source["arm"], root]
        answer, _, pack = pair[world]
        native, attention, writer = pack[site]
        if not np.array_equal(native, h[i]) or not np.array_equal(writer, w[i]):
            raise ValueError("Independent archive states differ")
        target.append(LABELS.index(answer))
        foil.append(LABELS.index(pair["B" if world == "A" else "A"][0]))
        reached, original, after = plan(writer.astype(np.float64), directions[i], source["id"], grid, np)
        identity = {"root": root, "world": world}
        if previous_geometry[root, world] != identity | {"dose": FRACTION} | original:
            raise ValueError("The retained SHELLDARK89 geometry changed")
        if set(reached) not in (set(), set(ARMS)):
            raise ValueError("Incomplete writer shell")
        geometry.append(identity | {"original": original, "repair": after})
        _, _, c_lo, c_hi = residual_box(grid, native, attention, writer, writer, np)
        for arm, vector in reached.items():
            rounded = vector.astype(np.float32)
            edits.append((add(c_lo, rounded, np), add(c_hi, rounded, np)))
            owners.append(i)
            arms.append(identity | {"arm": arm, "mode": after["mode"]})
    box = np.asarray(edits).transpose(1, 0, 2)
    owners = np.asarray(owners)
    native = native_controls("nf4", base, selected, tokenizer, check, np)[:, 0]
    winners = native[:, 0].astype(np.int32)
    scores = calculate(head, h, box, owners, np.asarray(target), np.asarray(foil), winners,
                       classes, canon, check, np)
    error = float(np.max(abs(scores["logz"] - native[:, 2])))
    if (not np.array_equal(scores["winner"], winners) or not np.array_equal(scores["peak"], native[:, 1])
            or error > base["native_logit_abs_tol"]):
        raise ValueError("Full clean head differs from the native receipt")
    ambiguous = []
    for i, row in enumerate(arms):
        peak = scores["peak_drift"][i].tolist()
        contrast = scores["paired_margin_change"][i].tolist()
        lo, hi = scores["winner_value"][i]
        other_lo, other_hi = scores["competitor"][i]
        strict = "preserved" if lo > other_hi else "changed" if hi < other_lo else "unresolved"
        gate = status(strict, peak, contrast)
        old = previous_heads.get((row["root"], row["world"], row["arm"]))
        if old is not None:
            if (row["mode"] != "original_shell" or (peak, contrast, strict, gate) != (
                    old["peak_drift_interval"], old["paired_margin_change_interval"],
                    old["winner"], old["current_gate"])):
                raise ValueError("Original first-pass evidence changed")
        elif row["mode"] != "compensated_shell":
            raise ValueError("Unreconciled original full-head observation")
        row.update(strict_winner=strict, strict_gate=gate, ordered_winner=strict, ordered_gate=gate,
                   peak_drift_interval=peak, paired_margin_change_interval=contrast)
        if strict == "unresolved":
            ambiguous.append(i)
    if ambiguous:
        found = refine(head, box[0, ambiguous], box[1, ambiguous], winners[owners[ambiguous]],
                       scores["winner_value"][ambiguous, 0], check, np)
        for i, outcome in zip(ambiguous, found, strict=True):
            row = arms[i]
            ordered = "preserved" if outcome["greedy_token_certified"] else "unresolved"
            row.update(ordered_winner=ordered, ordered_gate=status(ordered, row["peak_drift_interval"],
                                                                   row["paired_margin_change_interval"]),
                       ordered_detail=outcome)
    return {"geometry": geometry, "heads": arms, "native_selected_logit_error": replay_error,
            "native_clean_logz_error": error, "clean_heads": len(h)}


def reduce_panel(raw):
    roots = defaultdict(list)
    worlds = defaultdict(list)
    for head in raw["heads"]:
        roots[head["root"]].append(head)
        worlds[head["root"], head["world"]].append(head)
    counts = Counter(row["repair"]["mode"] for row in raw["geometry"])
    by_root = Counter(row["root"] for row in raw["geometry"] if row["repair"]["mode"] != "unqualified")
    if (len(worlds) != counts["original_shell"] + counts["compensated_shell"]
            or any({r["arm"] for r in v} != set(ARMS) or len(v) != len(ARMS) for v in worlds.values())):
        raise ValueError("Writer and head observations did not close")
    both = {root: rows for root, rows in roots.items() if len(rows) == 2 * len(ARMS)}
    details = [r["ordered_detail"] for r in raw["heads"] if "ordered_detail" in r]
    repaired = [r["repair"] for r in raw["geometry"] if r["repair"]["mode"] == "compensated_shell"]
    return {"roots": len(raw["geometry"]) // 2, "geometry_worlds": dict(counts),
            "geometry_both_world_roots": sum(n == 2 for n in by_root.values()),
            "strict_both_world_roots": sum(all(r["strict_gate"] == "preserved" for r in v) for v in both.values()),
            "ordered_both_world_roots": sum(all(r["ordered_gate"] == "preserved" for r in v) for v in both.values()),
            "strict_head_gate": dict(Counter(r["strict_gate"] for r in raw["heads"])),
            "ordered_head_gate": dict(Counter(r["ordered_gate"] for r in raw["heads"])),
            "witness": {"ambiguous_strict_heads": len(details),
                        "ordered_certified_heads": sum(d["greedy_token_certified"] for d in details),
                        "attainable_inputs_tested": sum(d["witnesses_tested"] for d in details),
                        "original_winner_witness_heads": sum(d["original_greedy_token_witness"] for d in details),
                        "different_winner_witness_heads": sum(d["different_greedy_token_witness"] for d in details)},
            "median_peak_drift_interval": [median(r["peak_drift_interval"][i] for r in raw["heads"])
                                           for i in range(2)],
            "maximum_repaired_angular_loss": max((1 - r["measured"]["learned"]["planned_direction_cosine"]
                                                   for r in repaired), default=None),
            "maximum_repaired_dose_relative_error": max((abs(r["measured"]["learned"]["realized_l2"] /
                                                              r["nominal_l2"] - 1) for r in repaired), default=None)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    source = seal()
    if args.freeze:
        write_new(MANIFEST, source)
        return
    base = json.loads(BASE.read_bytes())
    if (not __debug__ or not args.snapshot or not args.work or args.snapshot.name != base["revision"]
            or json.loads(MANIFEST.read_bytes()) != source):
        cli.error("Require frozen sources, pinned checkpoint and new result folder")
    prior = json.loads(bound(OLD, source["prior_private_sha256"]))
    with lease(BUDGET) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned numerical environment changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        tests = fixture(np) | repair_fixture(np) | ordered_fixture(np)
        routes = prepare_fit(np, masks)
        file = args.snapshot / "tokenizer.json"
        bound(file, base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(file))
        label_ids = [tokenizer.encode(c, add_special_tokens=False).ids for c in LABELS]
        if any(len(ids) != 1 for ids in label_ids):
            raise ValueError("Canonical labels changed")
        canon = [ids[0] for ids in label_ids]
        roster, _ = joined(base)
        args.work.mkdir(parents=True, exist_ok=False)
        raw, summaries = {}, {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            classes = token_classes(tokenizer, head.vocab, np, check)
            if classes[canon].tolist() != list(range(len(LABELS))):
                raise ValueError("Canonical grammar partition changed")
            for split, roots in (("calibration", 8), ("development", 16)):
                chosen = focus_split(roster, split, roots)
                observed = panel(base, chosen, prior[split], head, routes, tokenizer, classes, canon, check, np)
                raw[split] = observed
                summaries[split] = summary = reduce_panel(observed)
                print(split, summary, flush=True)
                if split == "calibration":
                    lock = args.work / "calibration_receipt.json"
                    write_new(lock, {"source_manifest_sha256": digest(MANIFEST), "summary": summary})
                    closure = digest(lock)
        if digest(lock) != closure:
            raise ValueError("CAL receipt changed during DEV")
        check()
    assert source == seal() and "torch" not in sys.modules
    private = args.work / "private.json"
    write_new(private, raw)
    write_new(args.work / "aggregate.json", {"experiment": source["experiment"], "status": "passed",
              "dose_fraction": FRACTION, "calibration_receipt_sha256": closure, "fixture": tests,
              "source_manifest_sha256": digest(MANIFEST), "private_sha256": digest(private),
              "vocabulary_rows": head.vocab, "splits": summaries,
              "clean_head_replay": {k: {n: v[n] for n in ("clean_heads", "native_clean_logz_error",
                                                         "native_selected_logit_error")} for k, v in raw.items()},
              "formal_comparisons": 0, "pretrained_decoder_forwards": 0, "sealed_test_accessed": False,
              "resources": {"min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
                            "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                            "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples)}})


if __name__ == "__main__":
    main()
