"""Select one low-dose phase-one shell on CAL, then score exposed DEV once."""

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
from experiments.nan_routematch86.geometry import ARMS, DOSE
from experiments.nan_tangent87.worker import prepare_fit
from experiments.nan_vocab81.run import LABELS, native_controls, token_classes

from .geometry import geometry


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
DOSES = (.0025, .004, .005, .006, .0075, .01, .0125, .015, DOSE)
PRIOR = REPO / "reports/nanbeige_bpd_gym/nan_carrybound88_results.json"
INHERITED = ("nan_carrybound88", "nan_routematch86")


def seal():
    files = {HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "geometry.py", "run.py")}
    files.update((PRIOR, BASE, REPO / "experiments/nan_lattice53/run.py"))
    for package in INHERITED:
        path = REPO / "experiments" / package / "SOURCE_MANIFEST.json"
        files.add(path)
        for name, sha in json.loads(path.read_bytes())["files_sha256"].items():
            candidate = REPO / name
            bound(candidate, sha)
            files.add(candidate)
    return {"experiment": "NAN-SHELLDARK-89", "dose_fractions": list(DOSES),
            "resource_budget": BUDGET, "current_logit_limit": LIMIT,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(files)}}


def focus_split(roster, split, roots):
    selected = {slot: row for slot, row in roster.items() if row["split"] == split and row["arm"] == "focused"}
    keys = defaultdict(set)
    for row in selected.values():
        keys[row["root"]].add(row["world"])
    if len(selected) != roots * 2 or len(keys) != roots or any(worlds != {"A", "B"} for worlds in keys.values()):
        raise ValueError("Incomplete matched archive roots")
    return selected


def process(base, selected, fractions, head, routes, tokenizer, classes, canon, check, np):
    records, native_replay_error = open_archive("nf4", base, selected, head, STAGES, check, np)
    vectors = list(route_states("nf4", base, selected, check, np))
    h, x, w = (np.stack([v[i] for _, v in vectors]) for i in range(3))
    proposals = np.stack(routes(x.astype(np.float64), w.astype(np.float64)), axis=1)
    if (not np.isfinite(proposals).all() or np.any(np.linalg.norm(proposals, axis=2) <= 1e-12)
            or np.any(np.linalg.norm(w.astype(np.float64), axis=1) <= 1e-12)):
        raise ValueError("Invalid fitted phase-one route vectors")
    grid = lattice(np)
    primary = STAGES.index((1, 21))
    targets, foils, edits, owners, geometry_rows, heads = [], [], [], [], [], []
    for i, (source, _) in enumerate(vectors):
        check()
        world = source["world"]
        pair = records[source["split"], source["arm"], source["root"]]
        label, _, pack = pair[world]
        effective, attention, writer = pack[primary]
        if not np.array_equal(h[i], effective) or not np.array_equal(w[i], writer):
            raise ValueError("Audited phase-one state paths disagree")
        targets.append(LABELS.index(label))
        foils.append(LABELS.index(pair["B" if world == "A" else "A"][0]))
        _, _, lower, upper = residual_box(grid, effective, attention, writer, writer, np)
        for fraction in fractions:
            check()
            reached, diagnostic = geometry(writer.astype(np.float64), proposals[i], source["id"], fraction,
                                           grid, np)
            if bool(reached) != (diagnostic["status"] == "matched"):
                raise ValueError("Geometry and delivered shell disagree")
            geometry_rows.append({"root": source["root"], "world": world, "dose": fraction} | diagnostic)
            for arm, write in reached.items():
                rounded = write.astype(np.float32)
                edits.append((add(lower, rounded, np), add(upper, rounded, np)))
                owners.append(i)
                heads.append({"root": source["root"], "world": world, "dose": fraction, "arm": arm})
    # A diagnostic with no admissible shell still verifies its native clean head.
    if not edits:
        edits, owners = [(row, row) for row in h], list(range(len(h)))
    box = tuple(np.stack(v) for v in zip(*edits, strict=True))
    controls = native_controls("nf4", base, selected, tokenizer, check, np)[:, 0]
    winners = controls[:, 0].astype(np.int32)
    scores = calculate(head, h, box, np.asarray(owners), np.asarray(targets), np.asarray(foils),
                       winners, classes, canon, check, np)
    logz_error = float(np.max(abs(scores["logz"] - controls[:, 2])))
    if (not np.array_equal(scores["winner"], winners) or not np.array_equal(scores["peak"], controls[:, 1])
            or logz_error > base["native_logit_abs_tol"]):
        raise ValueError("Clean full vocabulary differs from the native archive")
    for i, entry in enumerate(heads):
        peak_lo, peak_hi = map(float, scores["peak_drift"][i])
        pair_lo, pair_hi = map(float, scores["paired_margin_change"][i])
        w_lo, w_hi = map(float, scores["winner_value"][i])
        o_lo, o_hi = map(float, scores["competitor"][i])
        winner = "preserved" if w_lo > o_hi else "changed" if w_hi < o_lo else "unresolved"
        proven = winner == "preserved" and peak_hi <= LIMIT and max(abs(pair_lo), abs(pair_hi)) <= LIMIT
        failed = winner == "changed" or peak_lo > LIMIT or pair_lo > LIMIT or pair_hi < -LIMIT
        entry.update(current_gate="preserved" if proven else "failed" if failed else "unresolved",
                     winner=winner, peak_drift_interval=[peak_lo, peak_hi],
                     paired_margin_change_interval=[pair_lo, pair_hi])
    data = {"geometry": geometry_rows, "readouts": heads}
    evidence = {"clean_heads": len(h), "logz_max_abs_error": logz_error,
                "native_selected_logit_max_abs_error": native_replay_error}
    return data, evidence


def summarize(data, fractions, roots):
    result = []
    for fraction in fractions:
        geometry = [r for r in data["geometry"] if r["dose"] == fraction]
        heads = [r for r in data["readouts"] if r["dose"] == fraction]
        by_root, matched = Counter(), Counter()
        good = defaultdict(list)
        for row in geometry:
            by_root[row["root"]] += 1
            matched[row["root"]] += row["status"] == "matched"
        for row in heads:
            good[row["root"]].append(row)
        if (len(by_root) != roots or set(by_root.values()) != {2}
                or any(len(good[root]) != matched[root] * len(ARMS) for root in by_root)):
            raise ValueError("Mismatched geometry and readout census")
        gate = Counter(r["current_gate"] for r in heads)
        geometry_matches = sum(matched.values())
        both_geometry = sum(n == 2 for n in matched.values())
        passed = sum(len(rows) == 2 * len(ARMS) and all(r["current_gate"] == "preserved" for r in rows)
                     for rows in good.values())
        excluded = Counter(a for r in geometry for a in r.get("exact_angle_excluded", []))
        result.append({"fraction": fraction, "roots": roots, "matched_worlds": geometry_matches,
                       "geometry_matched_both_world_roots": both_geometry, "qualified_both_world_roots": passed,
                       "geometry_status": dict(Counter(r["status"] for r in geometry)),
                       "worlds_with_exact_angular_exclusion": sum(bool(r.get("exact_angle_excluded")) for r in geometry),
                       "exact_angle_exclusions_by_arm": dict(excluded), "head_status": dict(gate),
                       "winner_status": dict(Counter(r["winner"] for r in heads)),
                       "peak_drift_lower_over_limit": sum(r["peak_drift_interval"][0] > LIMIT for r in heads),
                       "median_peak_drift_interval": [median(r["peak_drift_interval"][i] for r in heads)
                                                      for i in range(2)] if heads else None})
    return result


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
    if (not __debug__ or not args.snapshot or not args.work
            or json.loads(MANIFEST.read_bytes()) != source):
        cli.error("Require frozen source, checkpoint and a new work directory")
    base = json.loads(BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        cli.error("Checkpoint revision differs")
    with lease(BUDGET) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned numerical runtime changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        finite_fixture = fixture(np)
        routes = prepare_fit(np, masks)
        tok_file = args.snapshot / "tokenizer.json"
        bound(tok_file, base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(tok_file))
        label_ids = [tokenizer.encode(c, add_special_tokens=False).ids for c in LABELS]
        if any(len(ids) != 1 for ids in label_ids):
            raise ValueError("Canonical label tokenization changed")
        canon = [ids[0] for ids in label_ids]
        roster, _ = joined(base)
        args.work.mkdir(parents=True, exist_ok=False)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            classes = token_classes(tokenizer, head.vocab, np, check)
            if classes[canon].tolist() != list(range(len(LABELS))):
                raise ValueError("Canonical grammar classes changed")
            context = (head, routes, tokenizer, classes, canon, check, np)
            cal, cal_check = process(base, focus_split(roster, "calibration", 8), DOSES, *context)
            cells = summarize(cal, DOSES, 8)
            prior = json.loads(PRIOR.read_bytes())["splits"]["calibration"]
            last = cells[-1]
            if (last["fraction"] != DOSE or last["matched_worlds"] != 16
                    or last["peak_drift_lower_over_limit"] != 48
                    or last["head_status"] != {"failed": 48}
                    or last["geometry_matched_both_world_roots"] != prior["both_world_roots"]["registered"]):
                raise ValueError("The 2% CAL control failed to reproduce the prior audit")
            full = [row for row in cells if row["qualified_both_world_roots"] == 8]
            selected = full[-1] if full else max(cells, key=lambda row: (
                row["qualified_both_world_roots"], row["geometry_matched_both_world_roots"], -row["fraction"]))
            path = args.work / "calibration_lock.json"
            write_new(path, {"source_manifest_sha256": digest(MANIFEST), "sweep": cells,
                             "classification": "calibration_qualified" if full else "descriptive_only",
                             "selected_fraction": selected["fraction"]})
            lock = digest(path)
            print("CAL decision", selected["fraction"], "qualified roots", selected["qualified_both_world_roots"],
                  flush=True)
            dev, dev_check = process(base, focus_split(roster, "development", 16),
                                     (selected["fraction"],), *context)
            dev_cells = summarize(dev, (selected["fraction"],), 16)
            if digest(path) != lock:
                raise ValueError("Calibration decision changed during development evaluation")
        check()
    assert source == seal() and "torch" not in sys.modules
    private = args.work / "private.json"
    write_new(private, {"calibration": cal, "development": dev})
    write_new(args.work / "aggregate.json", {"experiment": source["experiment"], "status": "passed",
              "source_manifest_sha256": digest(MANIFEST), "calibration_decision_sha256": lock,
              "private_sha256": digest(private), "vocabulary_rows": head.vocab, "calibration": cells,
              "selection_classification": "calibration_qualified" if full else "descriptive_only",
              "development": dev_cells[0], "calibration_clean_replay": cal_check,
              "development_clean_replay": dev_check, "fixture": finite_fixture,
              "formal_comparisons": 0, "sealed_test_accessed": False, "pretrained_decoder_forwards": 0,
              "resources": {"min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
                            "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                            "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples)}})
    print("DEV qualification", dev_cells[0], flush=True)


if __name__ == "__main__":
    main()
