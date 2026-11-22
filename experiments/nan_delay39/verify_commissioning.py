"""Synthetic output fixture for native checkpoint qualification before interventions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.nan_ranksite13.resources import below_normal
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from . import pilot
from .source import HERE


def main():
    if not __debug__:
        raise RuntimeError("Use assertions for the commissioning fixture")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    below_normal()
    cfg = json.loads((HERE / "registration.json").read_bytes())
    gold = json.loads((args.source_dir / "gold.json").read_bytes())
    raw = json.loads((args.source_dir / "tokens.json").read_bytes())
    label_ids = raw["bare_label_token_ids"]
    by_token = {v: k for k, v in label_ids.items()}
    data, targets = {}, {}
    for row in gold:
        key = row["root"], row["arm"], row["world"]
        ids = raw["row_bindings"][row["id"]]["ids"]
        data[key] = (row["id"], ids)
        desired = "A" if row["arm"] == "focused_erased" else row["answer"]
        targets[tuple(ids)] = label_ids[desired]
    assert len(data) == 48 and len(targets) == 40
    bad_ids = tuple(next(ids for (_root, arm, world), (_, ids) in data.items()
                         if arm == "copy" and world == "A"))

    class Logits:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, phase):
            assert phase == 1
            return self

        def argmax(self, dim):
            assert dim == -1
            return self

        def tolist(self):
            return self.values

    class Engine:
        def __init__(self, fail=False):
            self.fail = fail

        def capture(self, ids):
            return {"logits": Logits([
                0 if self.fail and tuple(row) == bad_ids else targets[tuple(row)] for row in ids
            ])}

    class Tokenizer:
        def decode(self, ids, **options):
            return by_token.get(ids[0], "X")

    worker_args = SimpleNamespace(backend="nf4", source_dir=args.source_dir.resolve())
    calls = []

    def measure(_engine, _ids, _donor, _V, _U, _labels, _cfg, tag, _np, _guard):
        calls.append(tag)
        return {"baseline": {"top_token": [label_ids["B"], label_ids["A"]]}, "cases": {}}

    args.output_dir.mkdir(exist_ok=False)
    passed = args.output_dir / "qualification_pass"
    failed = args.output_dir / "qualification_fail"
    passed.mkdir()
    failed.mkdir()
    with patch.object(pilot, "get_bank", return_value=(None, None)), patch.object(pilot, "measure_pair", side_effect=measure):
        a = pilot.run_groups(Engine(), data, list(label_ids.values()), Tokenizer(),
                             worker_args, passed, cfg, None, lambda: None)
        b = pilot.run_groups(Engine(True), data, list(label_ids.values()), Tokenizer(),
                             worker_args, failed, cfg, None, lambda: None)
    assert a["status"] == "closed" and a["closed_focused_worlds"] == 16
    assert b["status"] == "native_gate_failed" and b["closed_focused_worlds"] == 0
    assert len(calls) == len(set(calls)) == 16 and not (failed / "closed.jsonl").exists()
    artifact = {
        "status": "passed", "scope": "synthetic outputs with actual host qualification subprocess and journal closure",
        "native_roster_rows_per_attempt": 48, "closed_focused_fixture_rows": 16,
        "failed_gate_prevented_all_synthetic_edits": True,
        "source_sha256": {name: sha(HERE / name) for name in ("pilot.py", "host_gate.py", "verify_commissioning.py")},
    }
    write_new(args.output_dir / "verification.json", artifact)
    print(json.dumps(artifact))


if __name__ == "__main__":
    main()
