"""Compare fixed T2 attention edit directions at matched realized BF16 doses."""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean, median
import sys

from experiments.nan_attnspan67.run import SITES, make_pair, read_archive, read_banks
from experiments.nan_cigate73.run import ATTN, BASE, controller
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .geometry import ARMS, CONTRASTS, DOSE, DOSE_TOLERANCE, directions, match, verify


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"


def sources():
    files = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md", "geometry.py", "run.py")]
    global_files = (BASE, ATTN, *(REPO/n for n in (
        "experiments/nan_attnspan67/run.py", "experiments/nan_carry63/algebra.py",
        "experiments/nan_cigate73/account.py", "experiments/nan_cigate73/run.py",
        "experiments/nan_cigate73/SOURCE_MANIFEST.json", "experiments/nan_fit07/pointwise.py",
        "experiments/nan_delay39/instrument.py", "experiments/nan_dose38/run.py",
        "experiments/nan_readpath37/run.py", "experiments/nan_phase35/run.py",
        "experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py",
        "reports/nanbeige_bpd_gym/nan_cigate73_results.json")))
    return {"experiment": "NAN-CIDOSE-74", "files_sha256": {p.relative_to(REPO).as_posix(): digest(p)
                                                               for p in (*files, *global_files)}}


def angle(a, b, np):
    ab = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if ab == 0 else float(np.clip((a @ b) / ab, -1., 1.))


def replay(worlds, pair, fitter, root, erased, masks, np):
    a, b = [worlds[k][2] for k in ("A", "B")]
    bank, ci, static, _ = fitter
    points = []
    for index, observation in enumerate(pair["phase"], start=1):
        x, y = (np.stack([p[index, 21, i] for p in (a, b)]).astype(np.float64) for i in SITES[1:])
        gate = masks(x, bank, ci)
        shared = np.broadcast_to(static[index - 1], gate.shape)
        intended, parity_count = directions(x, y, gate, shared, bank, index, root, np)
        if erased and any(not np.array_equal(v[0], v[1]) for v in intended.values()):
            raise RuntimeError("Erased prompts map to different proposed directions")
        reached, rec = match(y, intended, np)
        point = rec | {"verified_reference_endpoints": parity_count,
                       "native_attention_readout": observation["direct_write"], "effects": {}}
        if intended:
            point["direction_cosines"] = {f"{a}_{b}": [angle(v, w, np) for v,w in zip(intended[a], intended[b], strict=True)]
                                           for a,b in (("dynamic_on", "static_on"), ("dynamic_off", "static_off"),
                                                       ("dynamic_on", "dynamic_radial"))}
        for arm, edited in reached.items():
            if erased and not np.array_equal(edited[0], edited[1]):
                raise RuntimeError("Erased prompts map to different BF16 writes")
            effects = (edited - y) @ observation["direction"]
            nominal_effects = [val * goal / info["dose"] for val, goal, info in
                               zip(effects, point["nominal_l2"], rec["arms"][arm], strict=True)]
            point["effects"][arm] = {"physical": float(effects[0] - effects[1]),
                                      "exact_dose_linear": float(nominal_effects[0] - nominal_effects[1])}
        points.append(point)
    return points


def summarize(pairs):
    qualified = [pair for pair in pairs if all(p["status"] == "matched" for p in pair)]
    out = {"roots_registered": len(pairs), "roots_qualified": len(qualified),
           "roots_unqualified": len(pairs) - len(qualified),
           "pass_status": dict(Counter(p["status"] for pair in pairs for p in pair)),
           "ineligible_arm_worlds": {arm: sum(not row["eligible"] for pair in pairs for p in pair
                                               for row in p["arms"].get(arm, ())) for arm in ARMS},
           "native_attention_gain": fmean(b["native_attention_readout"]-a["native_attention_readout"] for a,b in pairs)}
    if not qualified:
        return out
    out["qualified_only"] = {}
    for name in ("physical", "exact_dose_linear"):
        changes = {arm: [b["effects"][arm][name]-a["effects"][arm][name] for a,b in qualified] for arm in ARMS}
        contrasts = {key: [sum(coefficient * changes[arm][i] for arm, coefficient in rule.items())
                           for i in range(len(qualified))] for key,rule in CONTRASTS.items()}
        out["qualified_only"][name] = {"arm_mean_gains": {arm: fmean(arr) for arm,arr in changes.items()},
                                       "contrasts": {k: {"mean": fmean(v), "median_abs": median(map(abs, v)),
                                                         "range": [min(v), max(v)]}
                                                     for k,v in contrasts.items()}}
    out["qualified_only"]["pass2_physical_checks"] = {
        arm: {"median_abs_radial_share": median(abs(w["signed_radial_share"]) for _,p in qualified for w in p["arms"][arm]),
              "median_abs_norm_drift": median(abs(w["relative_norm_drift"]) for _,p in qualified for w in p["arms"][arm])}
        for arm in ARMS}
    similarities = qualified[0][1]["direction_cosines"]
    out["qualified_only"]["pass2_direction_cosine_median"] = {
        name: median(v for _,b in qualified for v in b["direction_cosines"][name] if v is not None)
        for name in similarities}
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable Python assertions")
    if args.freeze:
        write_new(MANIFEST, sources())
        return
    if not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes()) != sources():
        cli.error("Supply exact frozen source, snapshot and fresh output")
    cfg, base = (json.loads(p.read_bytes()) for p in (ATTN, BASE))
    if cfg["revision"] != base["revision"] or args.snapshot.name != base["revision"]:
        raise ValueError("The archive and checkpoint revision differ")
    with lease(cfg) as (check, samples):
        assert (version("numpy"),version("tokenizers")) == ("2.5.5","0.20.3")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        verify(np)
        config = json.loads(bound(args.snapshot/"config.json", base["snapshot_pins"]["config.json"]))
        if config["attention_bias"]:
            raise ValueError("The replay requires unbiased attention writes")
        bound(args.snapshot/"tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot/"tokenizer.json"))
        bare = {s: tokenizer.encode(s, add_special_tokens=False).ids for s in "ABCDEFGHIJKLMNOP"}
        assert all(len(v) == 1 for v in bare.values()) and len({v[0] for v in bare.values()}) == 16
        labels = {name: ids[0] for name,ids in bare.items()}
        roster, _ = joined(base)
        factors = read_banks("bf16", 21, cfg, base, np)
        fit = controller("t2_layer21_o_proj_rank8", cfg["ranksite_audits"]["2"],
                         factors["attn_bpd_t2"], np, masks)
        args.work.mkdir(parents=True, exist_ok=False)
        with (args.snapshot/SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            archive, error = read_archive("bf16", (21,), base, roster, head, check, np)
            rows = []
            grouped = defaultdict(list)
            for (split, arm, root), worlds in sorted(archive.items()):
                check()
                erased = arm == "focused_erased"
                pair = make_pair(worlds, 21, labels, head, erased, np)
                result = replay(worlds, pair, fit, root, erased, masks, np)
                grouped[split,arm].append(result)
                rows.append({"split": split, "arm": arm, "root": root, "phases": result})
        assert "torch" not in sys.modules
    assert json.loads(MANIFEST.read_bytes()) == sources()
    groups = {"/".join(k): summarize(v) for k,v in grouped.items()}
    write_new(args.work/"private.json", rows)
    write_new(args.work/"aggregate.json", {
        "experiment": "NAN-CIDOSE-74", "status": "passed", "source_manifest_sha256": digest(MANIFEST),
        "archive_pins": {k: base["backends"]["bf16"][k] for k in ("audit","manifest","target")},
        "controller": fit[-1], "roots_by_split": base["roots"], "archived_worlds": len(roster),
        "head_replay_max_abs_error": error, "dose_fraction": DOSE, "dose_tolerance": DOSE_TOLERANCE,
        "verified_reference_endpoints": sum(p["verified_reference_endpoints"] for r in rows for p in r["phases"]),
        "groups": groups, "pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_resident_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })
    print(groups["development/focused"], flush=True)


if __name__ == "__main__":
    main()
