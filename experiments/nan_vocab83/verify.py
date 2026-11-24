"""Round-trip a synthetic B2 attempt through the production file boundaries."""

import argparse
from math import exp, log
import json
from pathlib import Path
from shutil import copyfile, copytree
import sys

from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound
from experiments.nan_vocab82.heads import INVALID, LABELS, partition_sha
from experiments.nan_writer77.pipeline import MODES
from experiments.nan_writer77.source import load, worker_inputs

from .batches import gate_binding, record_roots
from .custody import HOST_BUDGET, OBSERVER, collect, config, vocabulary
from .lane import host, registration, terminal, verified


FIXTURE = "synthetic_fixture"


class SyntheticHead:
    def __init__(self, classes, bare):
        self.vocab = len(classes)
        self.classes = classes
        self.bare = bare
        self.first_ids = [classes.index(i) for i in range(INVALID + 1)]

    def read(self, target, foil, third, *, edited=False):
        changed = {self.bare[target]: 3., self.bare[foil]: .5 if edited else 1., self.bare[third]: 2.}
        if edited:
            changed[self.first_ids[-1]] = 5.
        maxima = [[token, 0.] for token in self.first_ids]
        for token, value in changed.items():
            maxima[self.classes[token]] = [token, value]
        winner, peak = min(changed.items(), key=lambda pair: (-pair[1], pair[0]))
        reading = {"winner": winner, "peak": peak, "second_logit": sorted(changed.values())[-2],
                   "log_normalizer": log(self.vocab + sum(exp(v)-1 for v in changed.values())),
                   "labels": {c: changed.get(i, 0.) for c, i in self.bare.items()}}
        return reading, maxima


def baseline_rows(source, roots, head, *, gate_passes):
    gold = {p["id"]:p["answer"] for p in source["gold"]}
    rows = []
    for p in source["public"]:
        key = p["id"]
        target = gold[key] if p["arm"] != "focused_erased" else "A"
        foil = next(c for c in LABELS if c != target)
        if p["arm"] == "focused":
            foil = gold[roots[p["root"]]["A" if p["world"] == "B" else "B"]]
        third = next(c for c in LABELS if c not in (target, foil))
        first = head.read(foil, target, third) if p["arm"] == "focused" else head.read(target, foil, third)
        after = (head.read(foil, target, third) if p["arm"] == "focused" and not gate_passes
                 else head.read(target, foil, third))
        rows.append({"id": key, "token_sha256": source["tokens"]["row_bindings"][key]["sha256"],
                     "phases": [first[0], after[0]], "class_maxima": [first[1], after[1]]})
    return rows


def eligible_trial(source, pair, head, fixture):
    gold = {p["id"]:p["answer"] for p in source["gold"]}
    worlds = []
    for i, letter in enumerate(("A","B")):
        target = gold[pair[letter]]
        foil = gold[pair["A" if letter == "B" else "B"]]
        third = next(c for c in LABELS if c not in (target, foil))
        first, clean, edited = (head.read(foil, target, third), head.read(target, foil, third),
                                head.read(target, foil, third, edited=True))
        second = {mode: edited if mode == "active" else clean for mode in MODES}
        worlds.append(fixture["worlds"][i] | {"arms": {mode:[first[0], second[mode][0]] for mode in MODES},
                      "class_maxima": {mode:[first[1], second[mode][1]] for mode in MODES}})
    return {"status":"closed", "plans":fixture["plans"], "worlds":worlds}


def round_trip(args, work, public, source, roots, classes, head, fixture, frozen, *, gate_passes):
    work.mkdir()
    commit = "f" * 40
    target = runtime.TARGETS["bf16"]
    write_new(work / "admission.json", {"published_commit":commit, "runtime_sha256":frozen, "mode":FIXTURE})
    write_new(work / "preflight.json", {"depths": {str(p): {"tested_max_position": registration()["max_position"]}
                                                  for p in (1,2)},
              "both_heads_exact":True, "partition_sha256":partition_sha(classes)})
    rows = baseline_rows(source, roots, head, gate_passes=gate_passes)
    write_new(work / "baseline.json", rows)
    host(args, work, qualify_only=True, fixture=True)
    compact = [{k:v for k,v in row.items() if k!="class_maxima"} for row in rows]
    gate = gate_binding(work, digest(compact), classes, roots)
    if gate["passed"] != gate_passes:
        raise AssertionError("The real host returned the wrong native competence decision")
    first_root = min(roots)

    def trial(root, pair):
        if not gate_passes:
            raise AssertionError("The gated edit ran without native competence")
        return eligible_trial(source, pair, head, fixture) if root == first_root else {"status":"geometry_unqualified"}

    count = record_roots(work, public, roots, gate, trial)
    terminal(work, frozen=frozen, commit=commit, classes=classes, batches=len(rows)//2, roots=count,
             before=target, after=target, quant="synthetic", quant_after="synthetic", peak=0, mode=FIXTURE)
    host(args, work, fixture=True)
    aggregate = json.loads((work / "aggregate.json").read_bytes())
    assert aggregate["mode"] == FIXTURE
    assert aggregate["private_outcomes_sha256"] == sha(work / "private_outcomes.json")
    return aggregate


def reject(work, source, classes, tok, frozen, message, *, mode=FIXTURE):
    try:
        collect(work, source, classes, tok, frozen, mode=mode)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Invalid runtime evidence was accepted")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    cli.add_argument("--observer-native", type=Path, required=True)
    args = cli.parse_args()
    frozen = verified()
    args.work.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(OBSERVER.read_bytes())["host_and_native"]["native_fixture"]
    fixture = json.loads(bound(args.observer_native, receipt["native_private_sha256"]))
    with lease(HOST_BUDGET) as (check, samples):
        worker_only = args.work / "worker_only"
        worker_only.mkdir()
        for name in ("public.json", "tokens.json"):
            copyfile(args.source_dir / name, worker_only / name)
        public = worker_inputs(worker_only, args.snapshot, check)
        assert set(public) == {"public","tokens"} and not (worker_only/"gold.json").exists()
        source, proof = load(args.source_dir, ("public", "gold", "tokens"))
        tok, classes = vocabulary(args.snapshot, check, proof["tokenizer_pins"])
        head = SyntheticHead(classes, source["tokens"]["bare_label_token_ids"])
        roots = {}
        for row in public["public"]:
            if row["arm"] == "focused":
                roots.setdefault(row["root"], {})[row["world"]] = row["id"]
        good = args.work / "closed_example"
        ok = round_trip(args, good, public, source, roots, classes, head, fixture, frozen, gate_passes=True)
        refused = round_trip(args, args.work / "refused_example", public, source, roots, classes,
                             head, fixture, frozen, gate_passes=False)
        primary = ok["contrasts"]["active_minus_matched_seeded"]["pass2"]
        assert primary["root_macro_delta"]["full_margin"] == -3
        assert primary["root_macro_delta"]["paired_margin"] == .5
        assert primary["regressions"] == 2 and primary["unknown_worlds"] == 22
        assert refused["legacy"]["root_status"] == {"native_gate_failed":config()["source_roots"]}
        assert refused["contrasts"]["active_minus_matched_seeded"]["pass2"]["unknown_worlds"] == 24
        reject(good, source, classes, tok, frozen, "does not certify", mode="pretrained_cuda")
        altered = args.work / "altered_journal"
        copytree(good, altered)
        with (altered / "closed.jsonl").open("ab") as out:
            out.write(b"\n")
        reject(altered, source, classes, tok, frozen, "Source hash mismatch: closed.jsonl")
        altered_file = args.work / "altered_root"
        copytree(good, altered_file)
        with (altered_file / "root_000.json").open("ab") as out:
            out.write(b" ")
        reject(altered_file, source, classes, tok, frozen, "Source hash mismatch: root_000.json")
        changed = args.work / "altered_baseline"
        copytree(good, changed)
        with (changed / "baseline.json").open("ab") as out:
            out.write(b" ")
        reject(changed, source, classes, tok, frozen, "Source hash mismatch: baseline.json")
        assert "torch" not in sys.modules
        check()
    assert verified() == frozen
    write_new(args.work / "verification.json", {"status":"passed", "runtime_sha256":frozen,
              "mode":FIXTURE, "host_positive_replay":True, "worker_input_without_gold":True,
              "journal_tamper_refused":True, "baseline_tamper_refused":True, "root_tamper_refused":True,
              "fixture_refused_as_pretrained":True, "roots_written":len(roots),
              "native_gate_refused_roots":refused["legacy"]["root_status"]["native_gate_failed"],
              "b2_baseline_batches":len(public["public"])//2, "primary_fixture":primary,
              "host_peak_private_bytes":max(ok["max_sampled_private_bytes"],refused["max_sampled_private_bytes"]),
              "host_peak_working_set_bytes":max(ok["max_sampled_working_set_bytes"],refused["max_sampled_working_set_bytes"]),
              "fixture_driver_max_private_bytes":max(s["private_bytes"] for s in samples),
              "minimum_available_RAM_bytes":min(ok["min_sampled_available_RAM_bytes"],
                                                refused["min_sampled_available_RAM_bytes"],
                                                *(s["available_physical_bytes"] for s in samples)),
              "formal_comparisons":0, "sealed_test_accessed":False, "pretrained_forwards":0})
    print("VOCAB83 synthetic custody and gate refusals passed", flush=True)


if __name__ == "__main__":
    main()
