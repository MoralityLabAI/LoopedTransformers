"""Replay exact-center swap shams against the pinned full-vocabulary head."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import sys

from experiments.nan_attain101 import head as prior_head
from experiments.nan_attain101 import run as prior_geom
from experiments.nan_attain101.geometry import median_or_none
from experiments.nan_bankdose99 import run as dose
from experiments.nan_bankdose99.geometry import CURRENT_DRIFT_LIMIT
from experiments.nan_commonhead102 import run as common
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new
from experiments.nan_refit49.run import REPO
from experiments.nan_vocab81.run import LABELS, native_controls, stream_tiles

from .geometry import MAX_ABS_COSINE, MAX_ENDPOINT_RADIUS_ERROR, MAX_RMS_GAP, TRIALS, swap_sham


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
OLD = REPO / "reports/nanbeige_bpd_gym/nan_commonhead102_results.json"
BUDGET = common.BUDGET | {"maximum_seconds": 300}
EXPERIMENT = "NAN-CHORDSHAM-106"


def registration():
    parent = common.registration()
    if json.loads(common.MANIFEST.read_bytes()) != parent:
        raise ValueError("The clean-centered source freeze differs")
    if json.loads(OLD.read_bytes())["source_manifest_sha256"] != digest(common.MANIFEST):
        raise ValueError("The clean-centered replay is not bound to the registered geometry")
    for name, pin in parent["files_sha256"].items():
        bound(REPO / name, pin)
    local = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "geometry.py", "run.py")]
    local += [OLD, common.MANIFEST]
    return {"experiment": EXPERIMENT, "budget": BUDGET, "trial_count": TRIALS, "maximum_abs_chord_cosine": MAX_ABS_COSINE,
            "maximum_relative_endpoint_radius_error": MAX_ENDPOINT_RADIUS_ERROR, "maximum_head_RMS_gap": MAX_RMS_GAP,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in local}}


def add_shams(states, ancestors, groups, roster, previous, check, np):
    enlarged = list(states)
    for g in groups:
        check()
        row = roster[g["slot"]]
        b, r, w, sign = g["backend"], row["root"], row["world"], g["sign"]
        records = [old for old in previous[b][r][w] if old["sign"] == sign]
        if len(records) != 1 or records[0]["centered_geometry"] != g["geometry"]:
            raise ValueError("The attained parent chord differs from its archived geometry")
        left, right = (states[i] for i in g["centered"])
        pair, metrics = swap_sham(states[g["clean"]], left, right, f"{EXPERIMENT}/{b}/{r}/{w}/{sign}", np)
        g["sham_geometry"] = metrics
        g["old_head"] = records[0]
        if pair is not None:
            g["sham"] = (len(enlarged), len(enlarged)+1)
            enlarged.extend(pair)
            ancestors.extend([g["clean"]]*2)
    return np.stack(enlarged)


def pair_energy(tile, pairs, clean, np):
    left, right = (tile[pairs[:, j]] for j in (0, 1))
    half_diff = (right-left)/2
    mean = (right+left)/2 - tile[clean]
    return np.stack((np.einsum("ij,ij->i", half_diff, half_diff),
                     np.einsum("ij,ij->i", mean, mean)), axis=1)


def replay(head, states, ancestors, groups, ids, check, np):
    matched = [g for g in groups if "sham" in g]
    energy = np.zeros((2, len(matched), 2))
    if matched:
        clean = np.asarray([g["clean"] for g in matched])
        active, sham = (np.asarray([g[k] for g in matched]) for k in ("centered", "sham"))

    def project():
        for tile in stream_tiles(head, states, check, np):
            if matched:
                energy[0] += pair_energy(tile, active, clean, np)
                energy[1] += pair_energy(tile, sham, clean, np)
            yield tile

    output = prior_head.read_tiles(project(), ids, len(states), ancestors, np)
    if output[-1] != head.vocab:
        raise ValueError("The full-vocabulary head replay is incomplete")
    rms = np.sqrt(energy/head.vocab)
    for i, g in enumerate(matched):
        g["pair_rms"] = rms[:, i, :].tolist()
    return output


def summarize(groups, roster, output):
    top, _, drift, bare, _ = output
    peer = {(p["root"], p["world"]): p["answer"] for p in roster.values()}
    private = {}
    for g in groups:
        row = roster[g["slot"]]
        r, w, b = row["root"], row["world"], g["backend"]
        a, f = [LABELS.index(ans) for ans in (row["answer"], peer[r, "B" if w == "A" else "A"])]
        c = g["clean"]
        margin = bare[c, a] - bare[c, f]

        def gate(pair):
            return all(top[i] == top[c] and max(drift[i], abs(bare[i, a]-bare[i, f]-margin))
                       <= CURRENT_DRIFT_LIMIT for i in pair)

        active = g["centered"]
        head_ok = gate(active)
        if (head_ok != g["old_head"]["centered_head_pass"]
                or max(drift[i] for i in active) != g["old_head"]["centered_head_drift"]):
            raise ValueError("The earlier centered head response differs")
        record = {"sign": g["sign"], "active_head_pass": head_ok, "geometry": g["sham_geometry"]}
        if "sham" in g:
            sham = g["sham"]
            head_sham = gate(sham)
            active_rms, sham_rms = g["pair_rms"]
            diff_gap, mean_gap = [abs(u-v) for u, v in zip(active_rms, sham_rms, strict=True)]

            def half_margin(pair):
                l, u = [bare[i,a]-bare[i,f] for i in pair]
                return float((u-l)/2)

            record.update(sham_head_pass=head_sham, matched_full_head=head_ok and head_sham,
                          active_rms=active_rms, sham_rms=sham_rms,
                          half_difference_RMS_gap=diff_gap, common_shift_RMS_gap=mean_gap,
                          paired_label_signed_half_gap=half_margin(sham) - half_margin(active),
                          passed=head_ok and head_sham and max(diff_gap, mean_gap) <= MAX_RMS_GAP)
        private.setdefault(b, {}).setdefault(r, {}).setdefault(w, []).append(record)

    summary = {}
    for b, roots in private.items():
        all_rows = [v for worlds in roots.values() for signs in worlds.values() for v in signs]
        matched = [v for v in all_rows if v["geometry"]["status"] == "matched"]
        counts = Counter(r["geometry"]["status"] for r in all_rows)
        checks = {k: 0 for k in ("active_head_roots", "geometry_roots", "both_head_roots", "joint_roots")}
        for worlds in roots.values():
            if set(worlds) != {"A", "B"} or any({s["sign"] for s in pair} != {1, -1} for pair in worlds.values()):
                raise ValueError("The signed counterfactual roster is incomplete")
            signs = [s for pair in worlds.values() for s in pair]
            checks["active_head_roots"] += all(s["active_head_pass"] for s in signs)
            checks["geometry_roots"] += all(s["geometry"]["status"] == "matched" for s in signs)
            checks["both_head_roots"] += all(s.get("matched_full_head", False) for s in signs)
            checks["joint_roots"] += all(s.get("passed", False) for s in signs)
        summary[b] = checks | {"parent_roots": len(roots), "parent_signed_pairs": len(all_rows),
            "geometry_by_status": dict(counts), "matched_full_head_signed_pairs": sum(v["matched_full_head"] for v in matched),
            "joint_signed_pairs": sum(v["passed"] for v in matched),
            "median_half_difference_RMS_gap": median_or_none(v["half_difference_RMS_gap"] for v in matched),
            "median_common_shift_RMS_gap": median_or_none(v["common_shift_RMS_gap"] for v in matched),
            "median_abs_paired_label_signed_half_gap": median_or_none(abs(v["paired_label_signed_half_gap"]) for v in matched),
            "median_abs_chord_cosine": median_or_none(v["geometry"]["reached_abs_chord_cosine"] for v in matched)}
    return summary, private


def fixture(np):
    a = np.array([-1., -1., -1., -1.])
    b = -a
    base = np.zeros(4)
    (x, y), report = swap_sham(base, a, b, "orthogonal_fixture", np)
    assert report["status"] == "matched" and report["reached_abs_chord_cosine"] == 0.
    assert report["max_endpoint_radius_relative_error"] == 0. and np.array_equal(x+y, a+b)
    blocked, evidence = swap_sham(base, a, np.array([1., -.75, -1., -1.]), "blocked_fixture", np)
    assert blocked is None and evidence["status"] == "dominant_coordinate_obstruction"
    data = np.array([[1., 1., 1.], [.5, 1., 1.], [1.5, 1.5, 1.]])
    got = pair_energy(data, np.array([[1, 2]]), np.array([0]), np)
    np.testing.assert_array_equal(got, [[.3125, .0625]])
    return {"exact_midpoint_and_radius_fixture": True, "dominant_coordinate_certificate_fixture": True,
            "pair_energy_dense_fixture": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    for name in ("common", "locked", "snapshot", "output"):
        cli.add_argument("--"+name, type=Path)
    args = cli.parse_args()
    source = registration()
    if args.freeze:
        write_new(MANIFEST, source)
        return
    if not all((args.common, args.locked, args.snapshot, args.output)) or source != json.loads(MANIFEST.read_bytes()):
        cli.error("Require the frozen source, both archived parents, snapshot and fresh output")
    old = json.loads(OLD.read_bytes())
    previous = json.loads(bound(args.common / "private_pairs.json", old["private_pairs_sha256"]))
    a = json.loads(prior_head.AGGREGATE.read_bytes())
    lock = json.loads(bound(args.locked / "calibration_lock.json", a["calibration_lock_sha256"]))["calibration"]
    known = json.loads(bound(args.locked / "private_dev.json", a["private_dev_sha256"]))
    if old["geometry_lock_sha256"] != a["calibration_lock_sha256"]:
        raise ValueError("The two archival stages use different locked chords")
    base = json.loads(dose.BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Wrong archived head snapshot")
    cfg = prior_geom.registered()["dose"]
    with lease(BUDGET) as (check, samples):
        if "torch" in sys.modules or (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("The pinned CPU-only head replay is required")
        import numpy as np
        from tokenizers import Tokenizer

        proof = fixture(np)
        token_file = args.snapshot / "tokenizer.json"
        bound(token_file, base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(token_file))
        ids = [tok.encode(c, add_special_tokens=False).ids for c in LABELS]
        if any(len(x) != 1 for x in ids):
            raise ValueError("Pinned bare labels changed token length")
        public, _ = joined(base)
        roster = {i: p for i, p in public.items() if p["arm"] == "focused" and p["split"] == "development"}
        states, refs, groups, panels = common.setup(roster, lock, known, base, cfg, check, np)
        states = add_shams(states, refs, groups, roster, previous, check, np)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            result = replay(head, states, refs, groups, [x[0] for x in ids], check, np)
        expected = json.loads(common.OLD_HEAD.read_bytes())["per_backend"]
        for name, panel in panels.items():
            old_tokens = native_controls(name, base, roster, tok, check, np)
            current, _ = prior_head.reduce_panel(panel, roster, old_tokens, result)
            if current != expected[name]:
                raise ValueError("Original full-head controls differ from ATTAIN101")
        report, detail = summarize(groups, roster, result)
        for name in report:
            if report[name]["parent_signed_pairs"] != old["per_backend"][name]["signed_pairs"]:
                raise ValueError("The sham replay missed a registered parent sign")
    assert "torch" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "private_pairs.json"
    write_new(raw, detail)
    write_new(args.output / "aggregate.json", {
        "experiment": EXPERIMENT, "status": "archive_head_complete", "source_manifest_sha256": digest(MANIFEST),
        "private_pairs_sha256": digest(raw), "fixture": proof, "per_backend": report,
        "full_head_rows": len(states), "vocabulary_rows": result[-1],
        "pretrained_decoder_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
