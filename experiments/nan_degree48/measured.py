"""Apply the exact degree null to a closed DELAY39 finite-measurement backend."""
import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.nan_delay39 import pilot
from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound
from experiments.nan_select47 import run as selectors

from .null import analyze
from .run import HERE, MANIFEST, registration


def measured(work, gold, tokens, scales, selection_cfg, check):
    custody = selectors.collect(work, gold, tokens, scales, selection_cfg, check)
    source = {row["id"]: row for row in gold if row["arm"] == "focused"}
    by_pair = {(row["root"], row["world"]): row for row in source.values()}
    old_cfg = json.loads((selectors.DELAY / "registration.json").read_bytes())
    profiles = defaultdict(dict)
    if custody["status"] != "native_gate_failed":
        journal = work / "closed.jsonl"
        raw = journal.read_bytes()
        if sha256(raw).hexdigest() != custody["native_journal_sha256"]:
            raise ValueError("Device journal changed after the independent selector check")
        terminals = work / "terminal.json"
        if sha(terminals) != custody["terminal_sha256"]:
            raise ValueError("Device terminal changed after the independent selector check")
        for i, entry in enumerate(json.loads(line) for line in raw.splitlines()):
            check()
            path = work / f"episode_{i:04d}.json"
            if path.name != entry["file"]:
                raise ValueError("Device journal sequence differs from the source seal")
            item = json.loads(bound(path, entry["sha256"]))
            row = source[item["source_id"]]
            root, world = row["root"], row["world"]
            twin = by_pair[root, "B" if world == "A" else "A"]
            graph, _ = selectors.extract(item, row, twin, scales, selection_cfg, old_cfg)
            if graph is None:
                continue
            profiles[root][world] = analyze(graph["source"], graph["middle"], graph["sink"],
                                            selection_cfg["calibration_cutoff"], check)
    pairs = [worlds for worlds in profiles.values() if set(worlds) == {"A", "B"}]
    if len(pairs) != custody["complete_CAL_roots"]:
        raise ValueError("Complete fixed-degree graph count differs from selector custody")
    root_delta = []
    for worlds in pairs:
        values = [sum(v["observed_backbone"]) - Fraction(int(v["null_total_backbone_numerator"]),
                  int(v["null_configurations"])) for v in worlds.values()]
        root_delta.append(sum(values) / 2)
    return dict(profiles), {
        "backend": custody["backend"], "native_terminal_sha256": custody["terminal_sha256"],
        "selector_custody_status": custody["status"], "registered_worlds": custody["registered_worlds"],
        "complete_worlds": sum(map(len, profiles.values())), "complete_both_world_roots": len(pairs),
        "complete_root_macro_observed_minus_exact_null_expected_size": fmean(float(x) for x in root_delta)
        if root_delta else None,
        "cutoff": selection_cfg["calibration_cutoff"], "inferential_tests": 0, "model_calls": 0,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--backend", choices=("nf4", "bf16"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    selectors.verify()
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("Fixed-degree solver differs from its audited freeze")
    selection_cfg = json.loads(selectors.CONFIG.read_bytes())
    pins = json.loads((selectors.DELAY / "SOURCE_CHECK.json").read_bytes())
    gold = json.loads((args.source_dir / "gold.json").read_bytes())
    tokens = json.loads((args.source_dir / "tokens.json").read_bytes())["row_bindings"]
    if digest(gold) != pins["gold_digest"] or digest(tokens) != pins["token_digest"]:
        raise ValueError("Frozen native source or token receipt differs")
    work = pilot.ARTIFACT / args.backend
    if not (work / "terminal.json").is_file():
        raise RuntimeError("Awaiting the closed DELAY39 device terminal")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    with lease(cfg) as (check, samples):
        import numpy as np

        _, writes = pilot.get_bank(args.backend, np)
        scales = np.linalg.norm(writes, axis=1).tolist()
        if any(not math.isfinite(x) or x <= 0 for x in scales):
            raise ValueError("The frozen write norms are invalid")
        private, summary = measured(work, gold, tokens, scales, selection_cfg, check)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = args.output / "private.json"
    write_new(receipt, private)
    write_new(args.output / "aggregate.json", summary | {
        "experiment": "NAN-DEGREE-48", "solver_manifest_sha256": sha(MANIFEST),
        "adapter_sha256": sha(__file__), "selector_manifest_sha256": sha(selectors.MANIFEST),
        "private_counts_sha256": sha(receipt), "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
    })


if __name__ == "__main__":
    main()
