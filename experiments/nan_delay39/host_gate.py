"""Independent exact-label gate for a fresh DELAY39 native tranche."""
import argparse
import json
from pathlib import Path

from experiments.nan_comp14.parser import label
from experiments.nan_ranksite13.resources import below_normal
from experiments.nan_readout34.runtime_plan import write_new
from .source import ARMS, HERE, digest


def assess(rows, gold, cfg):
    outcomes = {}
    for row in rows:
        if row["id"] in outcomes or type(row["token_id"]) is not int or row["token_id"] < 0:
            raise ValueError("Duplicate or invalid native token record")
        if not isinstance(row["text"], str):
            raise ValueError("A decoded native token is required")
        outcomes[row["id"]] = row
    expected = {(item["root"], item["world"], item["arm"]): item for item in gold}
    roots = sorted({key[0] for key in expected})
    if (len(roots) != cfg["source_roots"] or len(expected) != 6 * len(roots)
            or set(outcomes) != {item["id"] for item in expected.values()}):
        raise ValueError("The native result roster does not close exactly once")
    both = dict.fromkeys(ARMS, 0)
    failed_erased_identity = 0
    for root in roots:
        for arm in ARMS:
            a, b = [expected[root, world, arm] for world in ("A", "B")]
            ta, tb = [outcomes[item["id"]] for item in (a, b)]
            both[arm] += (
                label(ta["text"]) == a["answer"] and label(tb["text"]) == b["answer"]
            )
            if arm == "focused_erased" and (ta["token_id"], ta["text"]) != (tb["token_id"], tb["text"]):
                failed_erased_identity += 1
    admitted = (both["focused"] >= cfg["minimum_clean_both_world_roots"]
                and both["copy"] >= cfg["minimum_literal_copy_both_world_roots"]
                and failed_erased_identity == 0 and both["focused_erased"] == 0)
    return {"passed": admitted, "registered_calibration_roots": len(roots),
            "native_correct_both_world_roots": both, "erased_identical_input_disagreements": failed_erased_identity,
            "records_checked": len(outcomes), "independent_host_gold_in_model_process": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--native-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    below_normal()
    cfg = json.loads((HERE / "registration.json").read_bytes())
    pins = json.loads((HERE / "SOURCE_CHECK.json").read_bytes())
    gold = json.loads((args.source_dir / "gold.json").read_bytes())
    if digest(gold) != pins["gold_digest"]:
        raise ValueError("Independent host gold digest failed")
    records = json.loads(args.native_records.read_bytes())
    result = assess(records, gold, cfg)
    result["native_record_digest"] = digest(records)
    write_new(args.output, result)


if __name__ == "__main__":
    main()
