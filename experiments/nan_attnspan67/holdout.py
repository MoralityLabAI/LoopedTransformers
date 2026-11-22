"""Secondary layer-17 test with both answer labels held entirely out of each CAL fit."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .run import HERE, READPATH, MANIFEST, cal_bases, make_pair, read_archive, score_pair, summarize

LOCK = HERE / "LABEL_HOLDOUT_MANIFEST.json"
LAYER = 17


def registration(primary):
    result = json.loads(primary.read_bytes())
    if result["status"] != "completed" or result["source_manifest_sha256"] != digest(MANIFEST):
        raise ValueError("The primary archive replay must close before the supplemental test")
    return {
        "experiment": "NAN-ATTNSPAN-67/label_holdout",
        "stage": "registered_after_primary_results_before_label_disjoint_replay",
        "condition": "all focused CAL roots containing either target answer token excluded",
        "layer": LAYER, "rank": 4, "expected_CAL_roots_per_fold": 7, "expected_DEV_roots": 16,
        "primary_aggregate_sha256": digest(primary), "primary_manifest_sha256": digest(MANIFEST),
        "code_sha256": digest(HERE / "holdout.py"),
    }


def evaluate(archive, labels, head, check, np):
    groups = {split: [make_pair(worlds, LAYER, labels, head, False, np)
                      for (stage, arm, _), worlds in sorted(archive.items())
                      if (stage, arm) == (split, "focused")]
              for split in ("calibration", "development")}
    assert (len(groups["calibration"]), len(groups["development"])) == (8, 16)
    cal, dev = groups["calibration"], groups["development"]
    unique = {frozenset(row["tokens"]) for row in cal}
    assert len(unique) == 8 and len(set().union(*unique)) == 16
    assert all(frozenset(row["tokens"]) in unique for row in dev)
    folds = {}
    rows = []
    for pair in dev:
        check()
        key = frozenset(pair["tokens"])
        if key not in folds:
            eligible = [item for item in cal if key.isdisjoint(item["tokens"])]
            assert len(eligible) == 7
            assert all(key.isdisjoint(row["tokens"]) for row in eligible)
            folds[key] = cal_bases(np.stack([row["cal_write"] for row in eligible]),
                                   np.stack([row["cal_direction"] for row in eligible]), 4, np)
        rows.append(score_pair(pair, folds[key], {}, head, np))
    assert len(folds) == 8
    return summarize(rows), rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--primary", type=Path, required=True)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable Python assertions")
    manifest = registration(args.primary)
    if args.freeze:
        write_new(LOCK, manifest)
        return
    if not args.snapshot or not args.output or json.loads(LOCK.read_bytes()) != manifest:
        cli.error("Require frozen label holdout, snapshot, and output")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    base = json.loads((READPATH / "registration.json").read_bytes())
    assert args.snapshot.name == cfg["revision"] == base["revision"]
    with lease(cfg) as (check, samples):
        assert (version("numpy"), version("tokenizers")) == ("2.5.5", "0.20.3")
        import numpy as np
        from tokenizers import Tokenizer

        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        bare = {label: tok.encode(label, add_special_tokens=False).ids for label in "ABCDEFGHIJKLMNOP"}
        assert all(len(v) == 1 for v in bare.values()) and len({v[0] for v in bare.values()}) == 16
        labels = {k: v[0] for k,v in bare.items()}
        roster, _ = joined(base)
        args.output.mkdir(parents=True, exist_ok=False)
        aggregate = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for backend in base["backends"]:
                archive, error = read_archive(backend, cfg["layers"], base, roster, head, check, np)
                summary, rows = evaluate(archive, labels, head, check, np)
                write_new(args.output / f"{backend}_private.json", rows)
                aggregate[backend] = {"native_replay_max_abs_error": error, **summary}
                print(backend, summary["roots"], flush=True)
        assert not any(k == "torch" or k.startswith("torch.") for k in sys.modules)
    write_new(args.output / "aggregate.json", {"experiment": manifest["experiment"], "status": "completed",
        "supplemental_manifest_sha256": digest(LOCK), "formal_comparisons": 0,
        "new_pretrained_forwards": 0, "sealed_test_accessed": False, "per_backend": aggregate,
        "resources": {"max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                      "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)}})


if __name__ == "__main__":
    main()
