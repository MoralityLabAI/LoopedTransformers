"""Run the fixed B2 roster and retain exact admission and endpoint custody."""

import json

from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import write_new
from experiments.nan_vocab82.heads import Reducer, partition_sha
from experiments.nan_vocab83.batches import record_baseline, record_roots
from experiments.nan_writer77.pipeline import head_rows

from . import roster
from .core import run_root


class RootCapture:
    def __init__(self, capture, anchor, prompts, labels):
        self.capture_engine = capture
        self.anchor = anchor
        self.prompts = prompts
        self.labels = labels
        self.rows = []

    def predict(self, clean, after):
        return self.capture_engine.predict(clean, after)

    def capture(self, prompts, **options):
        from experiments.nan_premlp64.capture import same_bf16

        if self.anchor is not None:
            if prompts != self.prompts or options:
                raise ValueError("The first root capture must reuse the exact admitted B2 prompts")
            result, self.anchor = self.anchor, None
            return result
        result = self.capture_engine.capture(prompts, **options)
        edit = options.get("edit")
        if edit is not None and not same_bf16(*edit):
            self.rows.append({"rescued": options.get("rescue") is not None,
                              "phases": head_rows(result, self.labels)[1]})
        return result


def bind_qualification(work, digest, classes, roots, payload, admission):
    approved = json.loads((work/"qualification.json").read_bytes())
    gate = approved["gate"]
    pairs = approved["answer_token_pairs"]
    labels = set(payload["tokens"]["bare_label_token_ids"].values())
    admitted = set(gate["qualified_focused_roots"]) if gate["passed"] else set()
    if (approved["experiment"] != roster.EXPERIMENT
            or any(approved[k] != admission[k] for k in ("runtime_sha256", "backend", "mode"))
            or approved["baseline_sha256"] != sha(work/"baseline.json")
            or approved["partition_sha256"] != partition_sha(classes) or approved["vocab"] != len(classes)
            or gate["baseline_digest"] != digest or gate["registered_roots"] != len(roots)
            or type(gate["passed"]) is not bool or not set(gate["qualified_focused_roots"]) <= set(roots)
            or set(pairs) != admitted or any(len(v) != 2 or len(set(v)) != 2 or not set(v)<=labels
                                             for v in pairs.values())):
        raise ValueError("The host qualification does not bind the exact B2 evidence")
    return gate, pairs


def drive(work, capture, payload, admission, classes, planner, design, host_gate, check, np):
    from .host import CFG

    roots, digest, batches = record_baseline(work,capture,payload,CFG,check,Reducer(classes),classes)
    host_gate()
    gate, pairs = bind_qualification(work,digest,classes,roots,payload,admission)
    ordinals = {root:i for i,root in enumerate(sorted(roots))}
    labels = payload["tokens"]["bare_label_token_ids"]

    def trial(root, entry):
        anchor,prompts = entry
        path = work/f"plan_{ordinals[root]:03d}.json"
        observed = RootCapture(capture,anchor,prompts,labels)
        output = run_root(observed,root,prompts,pairs[root],planner,design,lambda plan:write_new(path,plan),check,np)
        if output["edited_pass2_runs"] != len(observed.rows):
            raise ValueError("Native kernel and staged observer counted different edited heads")
        return output | {"plan_sha256":sha(path)} | ({"head_observations":observed.rows} if observed.rows else {})

    count = record_roots(work,payload,roots,gate,trial)
    return batches,count
