"""Verify the grouped capacity procedure and independent host before GPU admission."""

import argparse
import json
import os
from pathlib import Path
import sys
from importlib.metadata import version

from experiments.nan_boundrescue109 import host as serial_host, roster
from experiments.nan_boundrescue109.native import BoundaryCapture
from experiments.nan_boundrescue109.verify_runtime import FIXTURE, refused, round_trip
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound, write_new
from experiments.nan_rescuegroup110.group import PairGroup
from experiments.nan_stagegroup103.verify import prepare_fixture
from experiments.nan_vocab83.custody import HOST_BUDGET, vocabulary
from experiments.nan_vocab83.verify import SyntheticHead

from .device import capacity
from .host import collect
from .lane import REPO, verified
from .worker import OFFLOADED_LAYERS, QUALIFIED_SWEEPS, VARIANT


def staged_fixture(snapshot, work, check, np, torch):
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    model,loop,_ = prepare_fixture(snapshot,work,torch)
    head = StreamedVocabularyHead(model.lm_head.weight,loop.device,13)
    serial,group = BoundaryCapture(loop,head,chunk=3),PairGroup(loop,head,chunk=3)
    try:
        pair = [[1,4,32,5,29,6,42],[8,39,3,4,65,77,91]]
        receipts = capacity(serial,group,loop,pair,(7,19),9,check,lambda:None,np,torch)
        assert len(receipts)==2 and all(r["full_heads_exact"] and r["exact_K_restored"] for r in receipts)
        assert all(r["serial_stage_copies"]==12 and r["grouped_stage_copies"]==3 for r in receipts)
    finally:
        serial.close()
        group.close()
        head.close()
    assert not torch.cuda.is_initialized()
    return {"probes":2,"exact_grouped_branches":4,"cpu_fixture_position":9,
            "observed_offloaded_layers":3,"cuda_initialized":False,"pretrained_forwards":0,
            "native_BF16_head_and_projection_match":True}


def replace(path, value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def staged_host(args, work, frozen, check):
    source,pins = serial_host.host_inputs(args.source_dir)
    tok,classes = vocabulary(args.snapshot,check,pins["tokenizer_pins"])
    public = {"public":source["public"],"tokens":source["tokens"]}
    head = SyntheticHead(classes,source["tokens"]["bare_label_token_ids"])
    roots = {}
    for row in source["public"]:
        if row["arm"]=="focused":
            roots.setdefault(row["root"],{})[row["world"]] = row["id"]
    old,_ = round_trip(work,source,public,roots,head,classes,tok,frozen,passes=True)
    end_path = work/"terminal.json"
    end = json.loads(end_path.read_bytes())
    admission = json.loads((work/"admission.json").read_bytes())
    admission |= {"backend":"bf16","execution_variant":VARIANT}
    replace(work/"admission.json",admission)
    baseline = work/"baseline.json"
    gate = serial_host.qualify(source,json.loads(baseline.read_bytes()),sha(baseline),classes,tok,admission)
    replace(work/"qualification.json",gate)
    preflight_path = work/"preflight.json"
    preliminary = json.loads(preflight_path.read_bytes())
    position = json.loads(roster.RECEIPT.read_bytes())["max_position"]
    preliminary |= {"execution_variant":VARIANT,"grouped_probes":[{
        "probe":key,"position":position,"full_heads_exact":True,"native_vectors_exact":True,"exact_K_restored":True,
        "phase2_branches":2,"grouped_stage_copies":18,"prime_stage_copies":18,"serial_stage_copies":72,
        "grouped_wall_seconds":2.,"serial_wall_seconds":5.} for key in ("A","B")]}
    replace(preflight_path,preliminary)
    journal = work/"closed.jsonl"
    entries = [json.loads(line) for line in journal.read_bytes().splitlines()]
    edited = 0
    for i,entry in enumerate(entries):
        path = work/entry["file"]
        record = json.loads(path.read_bytes())
        if record["status"]=="native_root_unqualified":
            continue
        done = record["status"]=="closed"
        stage_copies = QUALIFIED_SWEEPS*OFFLOADED_LAYERS if done else 0
        record["grouped_usage"] = {"stage_copies":stage_copies,"grouped_endpoint_pairs":16 if done else 0,
                                   "primed_worlds":2 if done else 0,"wall_seconds":3.}
        replace(path,record)
        entry["sha256"] = sha(path)
        edited += stage_copies
    journal.write_text("".join(json.dumps(e)+"\n" for e in entries),encoding="utf-8")
    target = runtime.TARGETS["bf16"]
    baseline_copies = 2*OFFLOADED_LAYERS*end["baseline_batches"]
    end |= admission | {"target_before":target,"target_after":target,"baseline_stage_copies":baseline_copies,
                       "roster_stage_copies":baseline_copies+edited}
    for k,name in (("admission","admission.json"),("preflight","preflight.json"),("qualification","qualification.json"),
                   ("journal","closed.jsonl")):
        end[f"{k}_sha256"] = sha(work/name)
    replace(end_path,end)
    result,private = collect(work,source,classes,tok,frozen,mode=FIXTURE)
    assert result["closed_root_macro"]==old["closed_root_macro"] and result["closed_roots"]==len(private)==1
    assert result["grouped_endpoint_pairs"]==16 and result["roster_stage_copies"]==baseline_copies+540
    refused(lambda:collect(work,source,classes,tok,frozen),"execution identity")
    replace(end_path,end|{"roster_stage_copies":end["roster_stage_copies"]+18})
    refused(lambda:collect(work,source,classes,tok,frozen,mode=FIXTURE),"staged-copy total")
    replace(end_path,end)
    changed = dict(preliminary)
    changed["grouped_probes"] = [p|{"grouped_stage_copies":36} for p in preliminary["grouped_probes"]]
    replace(preflight_path,changed)
    replace(end_path,end|{"preflight_sha256":sha(preflight_path)})
    refused(lambda:collect(work,source,classes,tok,frozen,mode=FIXTURE),"capacity or exact serial parity")
    replace(preflight_path,preliminary)
    end["preflight_sha256"] = sha(preflight_path)
    replace(end_path,end)
    assert "torch" not in sys.modules
    return {"fixture_closed_roots":1,"synthetic_refused_as_pretrained":True,"tampered_grouped_count_refused":True,
            "tampered_capacity_refused":True,"fixture_signed_macro":result["closed_root_macro"],
            "fixture_grouped_endpoint_pairs":result["grouped_endpoint_pairs"],"pretrained_forwards":0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode",choices=("cpu","host"),required=True)
    for name in ("snapshot","source-dir","work"):
        cli.add_argument("--"+name,type=Path)
    args = cli.parse_args()
    frozen = verified()
    if not all((args.work,args.snapshot)) or args.snapshot.name!=roster.CONFIG["checkpoint_revision"]:
        cli.error("Require pinned snapshot and a fresh output path")
    if args.mode=="host" and not args.source_dir:
        cli.error("Host fixture requires the frozen input directory")
    args.work.mkdir(parents=True,exist_ok=False)
    with lease(BUDGET if args.mode=="cpu" else HOST_BUDGET) as (check,samples):
        if args.mode=="cpu":
            os.environ.update(USE_TORCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                HF_HOME=str((args.work/"hf").resolve()),HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
            os.environ.pop("TRANSFORMERS_CACHE",None)
            base = json.loads((REPO/"experiments/nan_readpath37/registration.json").read_bytes())
            for name in ("config.json","modeling_nanbeige.py","model.safetensors.index.json"):
                bound(args.snapshot/name,base["snapshot_pins"][name])
            bound(args.snapshot/"configuration_nanbeige.py",CONFIGURATION)
            assert {k:version(k) for k in PACKAGES}==PACKAGES
            import numpy as np
            import torch

            torch.set_num_threads(1)
            proof = staged_fixture(args.snapshot,args.work,check,np,torch)
        else:
            proof = staged_host(args,args.work/"host_fixture",frozen,check)
    write_new(args.work/"verification.json",proof|{"runtime_sha256":frozen,
        "status":"passed","max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)})


if __name__=="__main__":
    main()
