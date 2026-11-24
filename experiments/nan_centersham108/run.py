"""Verify redistributed exact-center chords on the pinned Nanbeige head."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

from experiments.nan_headswap107 import run as swap
from experiments.nan_headswap107.search import design_ids
from experiments.nan_readpath37.run import bf16, bound, digest, write_new

from .search import ANGLES, POWER_STEPS, RANDOM_AXES, SCALES, geometry, midpoint_round, proposal_planes, select


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
parent = swap.parent
PREVIOUS = parent.REPO / "reports/nanbeige_bpd_gym/nan_headswap107_results.json"
EXPERIMENT = "NAN-CENTERSHAM-108"


def registration():
    if swap.registration() != json.loads(swap.MANIFEST.read_bytes()):
        raise ValueError("The head-aware swap source freeze differs")
    if json.loads(PREVIOUS.read_bytes())["source_manifest_sha256"] != digest(swap.MANIFEST):
        raise ValueError("The head-aware swap receipt refers to different frozen bytes")
    paths = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md", "run.py", "search.py")]
    paths.extend([swap.MANIFEST, PREVIOUS])
    return {
        "experiment": EXPERIMENT, "budget": parent.BUDGET, "angles": ANGLES, "scales": list(SCALES),
        "power_iterations": POWER_STEPS, "random_axes_per_family": RANDOM_AXES,
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
        pair, metrics = select(clean, left, right, f"{EXPERIMENT}/{b}/{r}/{w}/{sign}",
                               head, design, weights, np)
        g.update(sham_geometry=metrics, old_head=record)
        if pair is not None:
            g["sham"] = (len(enlarged), len(enlarged)+1)
            enlarged.extend(pair)
            refs.extend([g["clean"]]*2)
        entries.append({"precision": b, "root": r, "world": w, "sign": sign, "geometry": metrics})
    return np.stack(enlarged), entries


def fixture(np):
    left = np.array([1., -1., 1., 1., .75])
    right = np.array([1.+1/128, -1.+1/256, 1.+2/128, 1., .75])
    mid = (left+right)/2
    half = np.array([[.06, -.09, .04, .02, -.035], [-.006, .001, -.002, -.01, .001]])
    a, b = midpoint_round(mid, half, left, right, np)
    np.testing.assert_array_equal(a+b, np.broadcast_to(left+right, a.shape))
    np.testing.assert_array_equal(bf16(a, np), a)
    np.testing.assert_array_equal(bf16(b, np), b)
    assert np.any(abs(b-a) != abs(right-left))
    a, b = midpoint_round(mid, ((right-left)/2)[None, :], left, right, np)
    np.testing.assert_array_equal(a[0], left)
    np.testing.assert_array_equal(b[0], right)
    dose, radius, cosine = geometry(mid, left, right, a, b, np)
    np.testing.assert_allclose([dose[0], radius[0], cosine[0]], [0., 0., 1.], atol=1e-14)

    active = np.array([1., 0., 0., 0.])
    carry = np.array([.5, 1., 0., 0.])
    planes = list(proposal_planes(active, carry, np.eye(4), np.random.default_rng(12), np))
    assert len(planes) == 2 * RANDOM_AXES
    for fixed, high, random in planes:
        np.testing.assert_allclose([active@fixed, active@high, active@random, high@random], 0., atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(fixed+random), np.linalg.norm(active), atol=1e-14)
    for fixed, high, random in planes[RANDOM_AXES:]:
        np.testing.assert_allclose([carry@(fixed+high), carry@(fixed+random)], carry@active, atol=1e-14)
    return {"mirrored_BF16_endpoints": True, "redistributed_coordinate_magnitude": True,
            "active_chord_identity": True, "radius_split_projection": True}


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
    old = json.loads(swap.PREVIOUS.read_bytes())
    older = json.loads(PREVIOUS.read_bytes())
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
            output, heldout_rows = swap.replay(head, ids, states, refs, groups, [v[0] for v in labels], check, np)
        expected = json.loads(parent.common.OLD_HEAD.read_bytes())["per_backend"]
        for b, panel in panels.items():
            controls = parent.native_controls(b, base, roster, tokenizer, check, np)
            current, _ = parent.prior_head.reduce_panel(panel, roster, controls, output)
            if current != expected[b]:
                raise ValueError("Original full-head controls differ from ATTAIN101")
        summary, private = swap.reduce(groups, roster, output, previous)
        for b, stats in summary.items():
            stats["headswap107_median_half_RMS_gap"] = older["per_backend"][b]["median_half_difference_RMS_gap"]
            records = [r for worlds in private[b].values() for rows in worlds.values() for r in rows]
            stats["design_both_RMS_gates"] = sum(r["geometry"].get("design_rms_matched", False) for r in records)
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
