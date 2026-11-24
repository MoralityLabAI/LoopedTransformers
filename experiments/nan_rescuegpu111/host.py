"""Audit grouped stage custody, then reuse the frozen scientific scorer."""

import argparse
import json
from math import isfinite
from pathlib import Path
import sys

from experiments.nan_boundrescue109 import host as serial_host, roster
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound, write_new
from experiments.nan_vocab83.custody import HOST_BUDGET, vocabulary

from .worker import OFFLOADED_LAYERS, VARIANT, validate_usage


def audited_group(work, frozen, *, mode=serial_host.MODE):
    terminal = work/"terminal.json"
    terminal_sha = sha(terminal)
    end = json.loads(bound(terminal,terminal_sha))
    admission = json.loads(bound(work/"admission.json",end["admission_sha256"]))
    preflight = json.loads(bound(work/"preflight.json",end["preflight_sha256"]))
    identity = (roster.EXPERIMENT,"bf16",mode,frozen,VARIANT)
    keys = ("experiment","backend","mode","runtime_sha256","execution_variant")
    if (tuple(end[k] for k in keys) != identity or tuple(admission[k] for k in keys) != identity
            or preflight["execution_variant"] != VARIANT):
        raise ValueError("The terminal lacks the admitted grouped BF16 execution identity")
    position = json.loads(roster.RECEIPT.read_bytes())["max_position"]
    probes = preflight["grouped_probes"]
    if len(probes)!=2 or {r["probe"] for r in probes}!={"A","B"}:
        raise ValueError("Grouped capacity verification requires both fixed duplicate probes")
    for probe in probes:
        if (probe["position"] != position or probe["full_heads_exact"] is not True
                or probe["native_vectors_exact"] is not True or probe["exact_K_restored"] is not True
                or probe["phase2_branches"] != 2 or probe["grouped_stage_copies"] != OFFLOADED_LAYERS
                or probe["prime_stage_copies"] != OFFLOADED_LAYERS or probe["serial_stage_copies"] != OFFLOADED_LAYERS*4
                or not isfinite(probe["serial_wall_seconds"]) or not isfinite(probe["grouped_wall_seconds"])
                or min(probe["serial_wall_seconds"],probe["grouped_wall_seconds"])<=0):
            raise ValueError("Grouped CUDA capacity or exact serial parity failed custody")
    journal = bound(work/"closed.jsonl",end["journal_sha256"]).splitlines()
    if len(journal)!=end["root_records"] or end["baseline_stage_copies"]!=2*OFFLOADED_LAYERS*end["baseline_batches"]:
        raise ValueError("The grouped roster or baseline stage count is incomplete")
    copies = end["baseline_stage_copies"]
    counted = 0
    for i,line in enumerate(journal):
        entry = json.loads(line)
        name = f"root_{i:03d}.json"
        if entry["file"] != name:
            raise ValueError("Grouped journal is out of canonical order")
        record = json.loads(bound(work/name,entry["sha256"]))
        if record["status"] in ("native_gate_failed","native_root_unqualified"):
            if "grouped_usage" in record:
                raise ValueError("A competence refusal emitted grouped work")
            continue
        validate_usage(record)
        copies += record["grouped_usage"]["stage_copies"]
        counted += record["grouped_usage"]["grouped_endpoint_pairs"]
    if copies != end["roster_stage_copies"]:
        raise ValueError("The staged-copy total does not reconcile with the closed root journal")
    return {"execution_variant":VARIANT,"roster_stage_copies":copies,"baseline_stage_copies":end["baseline_stage_copies"],
            "grouped_endpoint_pairs":counted,"grouped_capacity_probes":len(probes),"terminal_sha256":terminal_sha}


def collect(work, source, classes, tok, frozen, *, mode=serial_host.MODE):
    measured = audited_group(work,frozen,mode=mode)
    report, private = serial_host.collect(work,source,classes,tok,frozen,"bf16",mode=mode)
    if measured["terminal_sha256"] != report["terminal_sha256"]:
        raise ValueError("Scientific score and grouped work came from different terminals")
    return report | measured,private


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir","snapshot","work"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--qualification",action="store_true")
    args = cli.parse_args()
    from .lane import verified

    frozen = verified()
    if "torch" in sys.modules or args.snapshot.name != roster.CONFIG["checkpoint_revision"]:
        raise RuntimeError("The grouped host requires only the pinned tokenizer")
    with lease(HOST_BUDGET) as (check,samples):
        source,pins = serial_host.host_inputs(args.source_dir)
        tok,classes = vocabulary(args.snapshot,check,pins["tokenizer_pins"])
        if args.qualification:
            admission = json.loads((args.work/"admission.json").read_bytes())
            if (admission["runtime_sha256"],admission["mode"],admission["backend"],admission["execution_variant"]) != (
                    frozen,serial_host.MODE,"bf16",VARIANT):
                raise ValueError("Qualification did not bind the admitted grouped runtime")
            baseline = args.work/"baseline.json"
            pin = sha(baseline)
            write_new(args.work/"qualification.json",
                      serial_host.qualify(source,json.loads(bound(baseline,pin)),pin,classes,tok,admission))
            return
        report, private = collect(args.work,source,classes,tok,frozen)
    write_new(args.work/"private_outcomes.json",private)
    write_new(args.work/"aggregate.json",report | {
        "private_outcomes_sha256":sha(args.work/"private_outcomes.json"),
        "max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
