"""Audit head-aware matched shams on the closed Nanbeige development archive."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_chordsham106 import run as parent
from experiments.nan_chordsham106.geometry import MAX_RMS_GAP
from experiments.nan_readpath37.run import bound, digest, write_new

from .search import DESIGN_OFFSET, DESIGN_ROWS, SHORTLIST, STARTS, STEPS, design_ids, flip_gain, select


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PREVIOUS = parent.REPO / "reports/nanbeige_bpd_gym/nan_chordsham106_results.json"
EXPERIMENT = "NAN-HEADSWAP-107"


def registration():
    previous = parent.registration()
    if previous != json.loads(parent.MANIFEST.read_bytes()):
        raise ValueError("The parent sham source freeze differs")
    if json.loads(PREVIOUS.read_bytes())["source_manifest_sha256"] != digest(parent.MANIFEST):
        raise ValueError("The parent sham receipt refers to different code")
    paths = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "run.py", "search.py")]
    paths += [parent.MANIFEST, PREVIOUS]
    return {
        "experiment": EXPERIMENT, "budget": parent.BUDGET, "design_rows": DESIGN_ROWS,
        "design_offset": DESIGN_OFFSET, "starts": STARTS, "steps": STEPS, "shortlist": SHORTLIST,
        "files_sha256": {p.relative_to(parent.REPO).as_posix(): digest(p) for p in paths},
    }


def add_shams(head, design, states, refs, groups, roster, centered, check, np):
    enlarged = list(states)
    entries = []
    weights = np.asarray([head.row(i) for i in design], dtype=np.float64) * head.gamma
    for g in groups:
        check()
        row = roster[g["slot"]]
        b, r, w, sign = g["backend"], row["root"], row["world"], g["sign"]
        record, = [p for p in centered[b][r][w] if p["sign"] == sign]
        if record["centered_geometry"] != g["geometry"]:
            raise ValueError("The attained parent chord differs from its archived geometry")
        clean = states[g["clean"]]
        left, right = (states[i] for i in g["centered"])
        pair, metrics = select(clean, left, right, f"{parent.EXPERIMENT}/{b}/{r}/{w}/{sign}",
                               head, design, weights, np)
        g.update(sham_geometry=metrics, old_head=record)
        if pair is not None:
            g["sham"] = (len(enlarged), len(enlarged)+1)
            enlarged.extend(pair)
            refs.extend([g["clean"]]*2)
        entries.append({"precision": b, "root": r, "world": w, "sign": sign, "geometry": metrics})
    return np.stack(enlarged), entries


def replay(head, design, states, refs, groups, labels, check, np):
    matched = [g for g in groups if "sham" in g]
    energy = np.zeros((2, 2, len(matched), 2))
    clean = [g["clean"] for g in matched]
    pairs = [np.array([g[k] for g in matched], dtype=np.intp).reshape(-1, 2) for k in ("centered", "sham")]
    excluded = np.zeros(head.vocab, dtype=bool)
    excluded[design] = True
    heldout_rows = 0

    def project():
        nonlocal heldout_rows
        pos = 0
        for tile in parent.stream_tiles(head, states, check, np):
            end = pos + tile.shape[1]
            heldout = ~excluded[pos:end]
            pos = end
            heldout_rows += int(heldout.sum())
            for i, pair in enumerate(pairs):
                energy[0, i] += parent.pair_energy(tile, pair, clean, np)
                energy[1, i] += parent.pair_energy(tile[:, heldout], pair, clean, np)
            yield tile

    output = parent.prior_head.read_tiles(project(), labels, len(states), refs, np)
    if output[-1] != head.vocab or heldout_rows != head.vocab - len(design):
        raise ValueError("The full and held-out vocabulary counts differ")
    for i, g in enumerate(matched):
        g["pair_rms"] = np.sqrt(energy[0, :, i] / head.vocab).tolist()
        g["heldout_rms"] = np.sqrt(energy[1, :, i] / heldout_rows).tolist()
    return output, heldout_rows


def reduce(groups, roster, output, previous):
    summary, private = parent.summarize(groups, roster, output)
    for g in groups:
        row = roster[g["slot"]]
        b, r, w, sign = g["backend"], row["root"], row["world"], g["sign"]
        record, = [p for p in private[b][r][w] if p["sign"] == sign]
        old, = [p for p in previous[b][r][w] if p["sign"] == sign]
        if "sham" not in g:
            continue
        if any(abs(a-v) > 1e-12 for a, v in zip(record["active_rms"], old["active_rms"], strict=True)):
            raise ValueError("The differential or common parent head RMS differs")
        active, sham = g["heldout_rms"]
        half, shared = [abs(a-v) for a, v in zip(active, sham, strict=True)]
        record.update(heldout_active_rms=active, heldout_sham_rms=sham, heldout_half_RMS_gap=half,
                      heldout_common_RMS_gap=shared, passed_both=record["passed"] and max(half, shared) <= MAX_RMS_GAP,
                      parent_half_RMS_gap=old["half_difference_RMS_gap"])
    for b, roots in private.items():
        clusters = [[p for ps in worlds.values() for p in ps] for worlds in roots.values()]
        all_rows = [p for cluster in clusters for p in cluster]
        matched = [p for p in all_rows if "heldout_half_RMS_gap" in p]
        report = summary[b]
        if len(all_rows) != report["parent_signed_pairs"]:
            raise ValueError("The archive panel is incomplete")
        report.update(
            design_half_RMS_matched_signed_pairs=sum(
                abs(p["geometry"]["sample_active_rms"][0]-p["geometry"]["sample_sham_rms"][0]) <= MAX_RMS_GAP
                for p in matched),
            full_half_RMS_matched_signed_pairs=sum(p["half_difference_RMS_gap"] <= MAX_RMS_GAP for p in matched),
            heldout_half_RMS_matched_signed_pairs=sum(p["heldout_half_RMS_gap"] <= MAX_RMS_GAP for p in matched),
            heldout_and_full_joint_signed_pairs=sum(p["passed_both"] for p in matched),
            heldout_and_full_joint_roots=sum(all(p.get("passed_both", False) for p in cluster) for cluster in clusters),
            median_heldout_half_RMS_gap=median(p["heldout_half_RMS_gap"] for p in matched) if matched else None,
            median_parent_half_RMS_gap=median(p["parent_half_RMS_gap"] for p in matched) if matched else None,
        )
    return summary, private


def fixture(np):
    matrix = np.array([[2., 3., -1.], [-1., 4., 1.]], dtype=np.float64)
    s = np.array([1, -1, 1])
    base = matrix @ s
    flipped = np.stack([s * (1 - 2 * unit) for unit in np.eye(len(s))])
    direct = np.sum((flipped @ matrix.T)**2, axis=1) - base @ base
    np.testing.assert_array_equal(flip_gain(matrix, base, s, (matrix**2).sum(axis=0)), direct)
    ids = design_ids(166144, np)
    assert len(np.unique(ids)) == DESIGN_ROWS and min(ids) >= 0 and max(ids) < 166144

    class FixtureHead:
        eps = 0.

        def logits(self, values, ids):
            return np.asarray(values) @ weights[ids].T

    weights = np.array([[1., .2, -.1, .8], [2., -1., -.4, .3], [.1, .2, .3, .4]])
    clean = np.full(4, 2.)
    left, right = clean - .125, clean + .125
    chosen, metrics = select(clean, left, right, "exact_fixture", FixtureHead(), np.arange(3), weights, np)
    a, b = chosen
    assert metrics["status"] == "matched" and metrics["reached_abs_chord_cosine"] == 0
    np.testing.assert_array_equal(a+b, left+right)
    np.testing.assert_array_equal(abs(b-a), abs(right-left))
    return {"coordinate_flip_gain": True, "unique_design_rows": True, "exact_swap_geometry": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    for name in ("common", "locked", "previous", "snapshot", "output"):
        cli.add_argument("--"+name, type=Path)
    args = cli.parse_args()
    source = registration()
    if args.freeze:
        write_new(MANIFEST, source)
        return
    if not all((args.common, args.locked, args.previous, args.snapshot, args.output)):
        cli.error("Require both parent archives, their geometry lock, snapshot and fresh output")
    if source != json.loads(MANIFEST.read_bytes()):
        raise ValueError("The frozen head-aware swap source differs")
    old = json.loads(PREVIOUS.read_bytes())
    previous = json.loads(bound(args.previous / "private_pairs.json", old["private_pairs_sha256"]))
    center = json.loads(parent.OLD.read_bytes())
    centered = json.loads(bound(args.common / "private_pairs.json", center["private_pairs_sha256"]))
    a = json.loads(parent.prior_head.AGGREGATE.read_bytes())
    lock = json.loads(bound(args.locked / "calibration_lock.json", a["calibration_lock_sha256"]))["calibration"]
    known = json.loads(bound(args.locked / "private_dev.json", a["private_dev_sha256"]))
    if center["geometry_lock_sha256"] != a["calibration_lock_sha256"]:
        raise ValueError("The parent archives refer to different locked chords")
    base = json.loads(parent.dose.BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Wrong archived head snapshot")
    cfg = parent.prior_geom.registered()["dose"]
    with parent.lease(parent.BUDGET) as (check, samples):
        if "torch" in sys.modules or (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("The pinned CPU-only head replay is required")
        import numpy as np
        from tokenizers import Tokenizer

        proof = fixture(np)
        token_file = args.snapshot / "tokenizer.json"
        bound(token_file, base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(token_file))
        labels = [tokenizer.encode(label, add_special_tokens=False).ids for label in parent.LABELS]
        if any(len(ids) != 1 for ids in labels):
            raise ValueError("The pinned bare labels changed")
        public, _ = parent.joined(base)
        roster = {i: v for i, v in public.items() if v["arm"] == "focused" and v["split"] == "development"}
        states, refs, groups, panels = parent.common.setup(roster, lock, known, base, cfg, check, np)
        args.output.mkdir(parents=True, exist_ok=False)
        selection = args.output / "selection_lock.json"
        with (args.snapshot / parent.SHARD).open("rb") as stream:
            head = parent.FixedHead(stream, args.snapshot, base, np, check)
            ids = design_ids(head.vocab, np)
            states, entries = add_shams(head, ids, states, refs, groups, roster, centered, check, np)
            write_new(selection, {"design_token_ids": ids.tolist(), "cases": entries})
            output, heldout_rows = replay(head, ids, states, refs, groups, [v[0] for v in labels], check, np)
        expected = json.loads(parent.common.OLD_HEAD.read_bytes())["per_backend"]
        for b, panel in panels.items():
            controls = parent.native_controls(b, base, roster, tokenizer, check, np)
            current, _ = parent.prior_head.reduce_panel(panel, roster, controls, output)
            if current != expected[b]:
                raise ValueError("Original full-head controls differ from ATTAIN101")
        summary, private = reduce(groups, roster, output, previous)
        if {b: v["parent_signed_pairs"] for b, v in summary.items()} != {
                b: v["parent_signed_pairs"] for b, v in old["per_backend"].items()}:
            raise ValueError("The upgrade did not cover every parent sign")
    assert "torch" not in sys.modules
    raw = args.output / "private_pairs.json"
    write_new(raw, private)
    write_new(args.output / "aggregate.json", {
        "experiment": EXPERIMENT, "status": "archive_head_complete", "source_manifest_sha256": digest(MANIFEST),
        "selection_lock_sha256": digest(selection), "private_pairs_sha256": digest(raw),
        "fixture": proof, "per_backend": summary, "full_head_states": len(states),
        "vocabulary_rows": output[-1], "design_rows": len(ids), "heldout_rows": heldout_rows,
        "pretrained_decoder_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
