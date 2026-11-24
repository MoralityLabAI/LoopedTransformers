"""Audit current-head preservation on the hash-bound ATTAIN101 locked geometry."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_bankdose99.geometry import CURRENT_DRIFT_LIMIT, bind_banks
from experiments.nan_bankdose99 import run as dose
from experiments.nan_bankgap100.rounding import assess, joint_round
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new
from experiments.nan_refit49 import run as refit
from experiments.nan_vocab81.run import LABELS, native_controls, stream_tiles

from .geometry import SIGNS, attained, sphere_pair
from . import run as geometry


HERE = Path(__file__).resolve().parent
HEAD_MANIFEST = HERE / "HEAD_MANIFEST.json"
AGGREGATE = refit.REPO / "reports/nanbeige_bpd_gym/nan_attain101_results.json"
BUDGET = geometry.BUDGET | {"maximum_private_bytes": 512 * 2**20}
EXPERIMENT = "NAN-ATTAIN-101-CURRENT-HEAD"


def registration():
    locked = geometry.registered()
    if locked != json.loads(geometry.MANIFEST.read_bytes()):
        raise ValueError("The carrier source freeze changed")
    aggregate = json.loads(AGGREGATE.read_bytes())
    if aggregate["source_manifest_sha256"] != digest(geometry.MANIFEST):
        raise ValueError("The archived lock aggregate lacks the registered carrier")
    files = [HEAD_MANIFEST.with_name("head.py"), HEAD_MANIFEST.with_name("HEAD_PROTOCOL.md"), AGGREGATE,
             geometry.MANIFEST, dose.BASE, refit.REPO / "experiments/nan_vocab81/run.py",
             refit.REPO / "experiments/nan_readpath37/run.py"]
    return {"experiment": EXPERIMENT, "budget": BUDGET, "limit": CURRENT_DRIFT_LIMIT,
            "files_sha256": {p.relative_to(refit.REPO).as_posix(): digest(p) for p in files}}


def planned(before, dirs, chosen, cfg, known, np):
    states, nominal, fraction = attained(before, dirs["svd"], chosen["seed_fraction"], cfg, np)
    pair, _ = sphere_pair(dirs["t1_fit"], dirs["t2_fit"], chosen["gain"], np)
    if known["reached_fraction"] != fraction:
        raise ValueError("The locked reached radius did not reproduce")
    result = []
    for key, sign in SIGNS:
        ideals = [sign * nominal * v for v in pair]
        center = [states[key] + d - sign*nominal*dirs["svd"] for d in ideals]
        edits, lower = joint_round(*center, ideals[1] - ideals[0], np)
        check = assess(before, edits, ideals, nominal, cfg, np, lower_bound=lower)
        if check != known["signs"][key]["joint"] or not check["qualified"]:
            raise ValueError("The physical head candidate differs from the locked archive geometry")
        result.extend(edits)
    return result


def add_panel(backend, base, chosen, witness, roster, banks, cfg, states, baselines, check, np):
    roots = geometry.sources(backend, base, "development", roster, banks, check, np)
    if set(roots) != set(witness):
        raise ValueError("The locked DEV roster changed")
    worlds = {}
    for slot, source in roster.items():
        if source["split"] != "development":
            continue
        root, world = source["root"], source["world"]
        before, dirs = roots[root][world]
        clean = len(states)
        worlds[slot] = {"clean": clean, "edits": []}
        states.append(before)
        baselines.append(clean)
        pair = witness[root]
        if not all(w["status"] == "measured" and geometry.world_pass(
                [{"methods": m} for m in w["signs"].values()], "joint", cfg)[1] for w in pair.values()):
            continue
        edits = planned(before, dirs, chosen, cfg, pair[world], np)
        worlds[slot]["edits"] = list(range(len(states), len(states) + len(edits)))
        states.extend(edits)
        baselines.extend([clean] * len(edits))
    return worlds


def read_tiles(blocks, labels, size, baseline, np):
    offset = 0
    top = np.full(size, -1, dtype=np.int64)
    peaks = np.full(size, -np.inf)
    drift = np.zeros(size)
    bare = np.full((size, len(labels)), np.nan)
    for values in blocks:
        i = values.argmax(axis=1)
        best = values[np.arange(size), i]
        better = best > peaks
        top = np.where(better, i + offset, top)
        peaks = np.maximum(best, peaks)
        drift = np.maximum(drift, np.max(abs(values - values[baseline]), axis=1))
        for j, token in enumerate(labels):
            if offset <= token < offset + values.shape[1]:
                bare[:, j] = values[:, token - offset]
        offset += values.shape[1]
    if min(top) < 0 or not np.isfinite(bare).all():
        raise ValueError("The tiled current-head readout is incomplete")
    return top, peaks, drift, bare, offset


def fixture(np):
    dense = np.array([[2., 1., 2., -4.], [2., 1., 2.25, -4.], [2., 1., 2., -3.875]])
    top, peak, drift, bare, total = read_tiles([dense[:, :2], dense[:, 2:]], [0, 3], 3, [0, 0, 0], np)
    assert total == 4
    np.testing.assert_array_equal(top, [0, 2, 0])
    np.testing.assert_array_equal(peak, [2., 2.25, 2.])
    np.testing.assert_array_equal(drift, [0., .25, .125])
    np.testing.assert_array_equal(bare, dense[:, [0, 3]])
    return {"dense_tile_crosscheck": True, "cross_tile_lowest_token_tie": True, "off_label_drift_recovered": True}


def reduce_panel(positions_by_slot, roster, native, read):
    top, peaks, drift, bare, _ = read
    roots = {}
    for i, (slot, positions) in enumerate(positions_by_slot.items()):
        row = roster[slot]
        a = LABELS.index(row["answer"])
        pair = roots.setdefault(row["root"], {})
        c = positions["clean"]
        if (top[c], peaks[c]) != tuple(native[i, 0, :2]) or drift[c] != 0.:
            raise ValueError("Clean first-pass decoder winner or logit did not replay exactly")
        pair[row["world"]] = {"label_index": a, "positions": positions}
    counts = Counter()
    details = {}
    dose_roots = 0
    for root, worlds in roots.items():
        dose_ok = all(w["positions"]["edits"] for w in worlds.values())
        dose_roots += bool(dose_ok)
        if not dose_ok:
            continue
        output = []
        for world, peer in (("A", "B"), ("B", "A")):
            w = worlds[world]
            a, b = w["label_index"], worlds[peer]["label_index"]
            c = w["positions"]["clean"]
            tags = ("t1_positive", "t2_positive", "t1_negative", "t2_negative")
            for tag, j in zip(tags, w["positions"]["edits"], strict=True):
                shift = float((bare[j, a] - bare[j, b]) - (bare[c, a] - bare[c, b]))
                unchanged = top[j] == top[c]
                drift_ok = drift[j] <= CURRENT_DRIFT_LIMIT
                paired_ok = abs(shift) <= CURRENT_DRIFT_LIMIT
                output.append({"world": world, "arm": tag, "winner_preserved": bool(unchanged),
                               "max_vocab_drift": float(drift[j]), "paired_margin_shift": shift,
                               "passed": bool(unchanged and drift_ok and paired_ok)})
                counts["signed_heads"] += 1
                counts["winner_preserved"] += bool(unchanged)
                counts["max_vocab_drift_pass"] += bool(drift_ok)
                counts["paired_margin_pass"] += bool(paired_ok)
        details[root] = output
        counts["both_world_current_head_roots"] += all(w["passed"] for w in output)
    worst = [max(row["max_vocab_drift"] for row in values) for values in details.values()]
    return dict(counts) | {"clean_native_heads_exact": len(positions_by_slot),
                           "geometry_qualified_roots": dose_roots,
                           "median_root_max_drift": median(worst) if worst else None}, details


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--locked", type=Path)
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    seal = registration()
    if args.freeze:
        write_new(HEAD_MANIFEST, seal)
        return
    if not all((args.locked, args.snapshot, args.output)) or seal != json.loads(HEAD_MANIFEST.read_bytes()):
        cli.error("Require frozen head replay, locked carrier work, pinned snapshot, and fresh output")
    old = json.loads(AGGREGATE.read_bytes())
    lock = json.loads(bound(args.locked / "calibration_lock.json", old["calibration_lock_sha256"]))["calibration"]
    private = json.loads(bound(args.locked / "private_dev.json", old["private_dev_sha256"]))
    base = json.loads(dose.BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("The checkpoint snapshot differs from COMP14")
    with lease(BUDGET) as (check, samples):
        if "torch" in sys.modules or (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("The pinned head-only runtime is required")
        import numpy as np
        from tokenizers import Tokenizer

        proof = fixture(np)
        tok_path = args.snapshot / "tokenizer.json"
        bound(tok_path, base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(tok_path))
        ids = [tok.encode(c, add_special_tokens=False).ids for c in LABELS]
        if any(len(x) != 1 for x in ids):
            raise ValueError("Registered bare labels changed token length")
        canonical = [x[0] for x in ids]
        rows, _ = joined(base)
        roster = {i: p for i, p in rows.items() if p["arm"] == "focused" and p["split"] == "development"}
        cfg = geometry.registered()["dose"]
        fits = refit.load_fits(base, np)
        states, baselines, panels = [], [], {}
        for b, selection in lock.items():
            chosen = selection["locked"]
            if chosen is not None:
                banks = bind_banks(fits[f"{b}/T1"], fits[f"{b}/T2"], np)
                panels[b] = add_panel(b, base, chosen, private[b], roster, banks, cfg, states, baselines, check, np)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            read = read_tiles(stream_tiles(head, np.stack(states), check, np), canonical, len(states), baselines, np)
            if read[-1] != head.vocab:
                raise ValueError("Tiled replay missed registered vocabulary rows")
        result, raw = {}, {}
        for b, panel in panels.items():
            native = native_controls(b, base, roster, tok, check, np)
            result[b], raw[b] = reduce_panel(panel, roster, native, read)
            if result[b]["geometry_qualified_roots"] != old["development"][b]["joint"]["faithful_roots"]:
                raise ValueError("The current-head panel lost a qualified geometry root")
    assert "torch" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=False)
    details = args.output / "private_head.json"
    write_new(details, raw)
    write_new(args.output / "aggregate.json", {
        "experiment": EXPERIMENT, "status": "head_archive_complete", "head_source_sha256": digest(HEAD_MANIFEST),
        "geometry_lock_sha256": old["calibration_lock_sha256"], "private_head_sha256": digest(details),
        "per_backend": result, "fixture": proof, "full_head_rows": len(states), "vocabulary_tokens": read[-1],
        "pretrained_decoder_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
        "max_sampled_working_set_bytes": max(x["working_set_bytes"] for x in samples),
    })


if __name__ == "__main__":
    main()
