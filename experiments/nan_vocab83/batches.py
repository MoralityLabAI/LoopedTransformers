"""Run the original B2 schedule and seal each root after observation."""

import json

from experiments.nan_delay39.source import digest
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_vocab82.capture import observed_baseline, observed_trial
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_writer77.pipeline import source_index

from .custody import EXPERIMENT


def record_baseline(work, capture, payload, cfg, check, reducer, classes):
    baseline, focus = observed_baseline(capture, payload, cfg["arms"], check, reducer, classes)
    if len(baseline) != len(payload["public"]) or len(focus) != cfg["source_roots"]:
        raise ValueError("Baseline capture missed a registered B2 pair")
    write_new(work / "baseline.json", baseline)
    compact = [{k: v for k, v in row.items() if k != "class_maxima"} for row in baseline]
    return focus, digest(compact), len(baseline) // 2


def gate_binding(work, baseline_digest, classes, roots):
    report = json.loads((work / "qualification.json").read_bytes())
    gate = report["gate"]
    if (report["experiment"] != EXPERIMENT or report["baseline_sha256"] != sha(work / "baseline.json")
            or report["partition_sha256"] != partition_sha(classes) or report["vocab"] != len(classes)
            or gate["baseline_digest"] != baseline_digest or gate["registered_roots"] != len(roots)
            or not set(gate["qualified_focused_roots"]) <= set(roots) or type(gate["passed"]) is not bool):
        raise ValueError("Oracle qualification did not bind the captured baseline")
    return gate


def record_roots(work, public, roots, gate, trial):
    index, _, _ = source_index(public)
    qualified = set(gate["qualified_focused_roots"])
    closed = 0
    with (work / "closed.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        for i, root in enumerate(sorted(roots)):
            if gate["passed"] and root in qualified:
                record = trial(root, roots[root])
            else:
                gate_status = "native_root_unqualified" if gate["passed"] else "native_gate_failed"
                record = {"status": gate_status}
            ids = [index[root, "focused", w]["id"] for w in ("A", "B")]
            path = work / f"root_{i:03d}.json"
            write_new(path, record | {"root": root, "source_ids": ids})
            journal.write(json.dumps({"file": path.name, "sha256": sha(path)}) + "\n")
            journal.flush()
            closed += 1
    return closed


def execute_pairs(work, capture, payload, cfg, fit, masks, np, classes, check, host_gate):
    from experiments.nan_vocab82.heads import Reducer

    reduce = Reducer(classes)
    focus, baseline_digest, batches = record_baseline(work, capture, payload, cfg, check, reduce, classes)
    host_gate()
    gate = gate_binding(work, baseline_digest, classes, focus)
    labels = payload["tokens"]["bare_label_token_ids"]

    def run(root, pair):
        return observed_trial(capture, root, *pair, labels, fit, masks, np, check, reduce, classes)

    count = record_roots(work, payload, focus, gate, run)
    return batches, count
