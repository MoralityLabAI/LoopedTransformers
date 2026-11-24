"""CPU exact-parity and stage-copy audit for grouped boundary rescue."""

import argparse
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_boundrescue109 import lane
from experiments.nan_boundrescue109.core import SIGNS, WORLDS, run_root
from experiments.nan_boundrescue109.native import BoundaryCapture, HeadPlan
from experiments.nan_boundrescue109.worker import RootCapture
from experiments.nan_centersham108.search import midpoint_round
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_radial76.native import WRITER
from experiments.nan_reader71.native import SELF
from experiments.nan_readpath37.run import bf16, bound, digest, write_new
from experiments.nan_stagegroup103.verify import prepare_fixture
from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore, logical_target_digest

from .group import ALL_SITES, PairGroup, RootBranches, grouped_root


HERE = Path(__file__).resolve().parent
MANIFEST = HERE/"SOURCE_MANIFEST.json"
EXPERIMENT = "NAN-RESCUEGROUP-110"


def registration():
    source = lane.verified()
    lane.commissioned(source)
    files = [HERE / name for name in (".gitattributes","__init__.py","group.py","verify.py","PROTOCOL.md")]
    files += [lane.MANIFEST,lane.COMM] + [lane.REPO/name for name in (
        "experiments/nan_stagegroup103/verify.py","experiments/nan_bf31/config.json",
        "experiments/nan_boundrescue109/native.py","experiments/nan_boundrescue109/core.py",
        "experiments/nan_boundrescue109/worker.py","experiments/nan_reader71/native.py",
        "mechinterp/nanbeige_bpd_gym/chunked_cache.py","mechinterp/nanbeige_bpd_gym/lazy_bf16_storage.py")]
    return {"experiment":EXPERIMENT,"runtime_sha256":source,"maximum_simultaneous_B2_pass2_branches":2,
            "budget":BUDGET,"files_sha256":{p.relative_to(lane.REPO).as_posix():digest(p) for p in files}}


def expect_refusal(call, expected):
    try:
        call()
    except (ValueError, RuntimeError) as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError("Contradictory grouped branch was accepted")


def packet_equal(observed, expected, torch):
    assert torch.equal(observed["logits"],expected["logits"])
    assert (observed["writer_edits"],observed["rescues"]) == (expected["writer_edits"],expected["rescues"])
    for site in ALL_SITES:
        assert observed[site].keys() == expected[site].keys()
        assert all(same_bf16(v,expected[site][k]) for k,v in observed[site].items())


def plans(anchor, np, *, reject=False):
    out = {}
    for i, world in enumerate(WORLDS):
        raw = anchor["writer_raw"][1,i].float().numpy().astype(np.float64)
        for sign in SIGNS:
            v,w = np.zeros_like(raw),np.zeros_like(raw)
            v[12],w[13] = sign*.004,sign*.004
            a,b = bf16(raw-v,np).astype(np.float64),bf16(raw+v,np).astype(np.float64)
            c,d = midpoint_round((a+b)/2,w[None,:],a,b,np)
            out[world,sign] = a,b,c[0],d[0]
    if reject:
        out.pop(("B",-1))
    return out,{"fixture_only":True}


def verify_pair(serial,group,loop,ids,check,torch):
    prompts = [ids,ids]
    clean = serial.capture(prompts)
    base = clean["writer_raw"][WRITER]
    after = base.clone()
    after[12] += .015625
    head = serial.predict(clean,after)
    pair = base,after
    edited = serial.capture(prompts,edit=pair,planned=head)
    restored = serial.capture(prompts,edit=pair,planned=head,rescue=(edited["k_raw"][SELF],clean["k_raw"][SELF]))
    p = group.prime(prompts,clean)
    snapshot = p.raw.clone()
    before = loop.stage_copies
    stale = HeadPlan(torch.zeros_like(head.normalized),head.logits)
    expect_refusal(lambda:group.run(p,prompts,pair,stale,clean,check),"boundary differs")
    bad_head = HeadPlan(head.normalized,head.logits + 1)
    expect_refusal(lambda:group.run(p,prompts,pair,bad_head,clean,check),"head differs")
    expect_refusal(lambda:group.run(p,[ids+[2]]*2,pair,head,clean,check),"bound duplicate")
    expect_refusal(lambda:group.run(p,prompts,(base+1.,after),head,clean,check),"stale")
    assert loop.stage_copies == before
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        check()
        if calls == 4:
            raise RuntimeError("intentional guard interrupt")

    phase = loop.phase,loop.active_layer
    expect_refusal(lambda:group.run(p,prompts,pair,head,clean,interrupt),"intentional guard")
    assert not group.cache.instrument.active and phase==(loop.phase,loop.active_layer)
    for actual,expected in zip(group.run(p,prompts,pair,head,clean,check),(edited,restored),strict=True):
        packet_equal(actual,expected,torch)
    assert torch.equal(snapshot,p.raw) and not group.cache.caches
    return len(ALL_SITES)*4*2


def exercise(snapshot, work, check, np, torch):
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    model,loop,store = prepare_fixture(snapshot,work,torch)
    before,hooks = logical_target_digest(model,store),hook_count(loop)
    prompts = [[1,4,32,5,29,6,42],[8,39,3,4,65,77,91]]
    labels,design = (7,13),np.arange(2,97,13)
    bare = dict(zip("ABCDEFGHIJKLMNOP",range(1,17),strict=True))
    with ExitStack() as stack:
        head = StreamedVocabularyHead(model.lm_head.weight,loop.device,13)
        stack.callback(head.close)
        serial = BoundaryCapture(loop,head,chunk=3)
        stack.callback(serial.close)
        group = PairGroup(loop,head,chunk=3)
        stack.callback(group.close)
        audited = sum(verify_pair(serial,group,loop,ids,check,torch) for ids in (prompts[0],prompts[1]+[6,9]))
        anchor = serial.capture(prompts)
        reference = RootCapture(serial,anchor,prompts,bare)
        accepted = []
        rejected = []
        grouped = []
        start = loop.stage_copies,store.bytes_copied

        def plan(packet,_root,_check):
            return plans(packet,np)

        expected = run_root(reference,"fixture",prompts,labels,plan,design,accepted.append,check,np)
        serial_cost = loop.stage_copies-start[0],store.bytes_copied-start[1]
        start = loop.stage_copies,store.bytes_copied
        actual, observations, usage = grouped_root(serial,group,"fixture",prompts,labels,plan,design,
                                                    grouped.append,check,np,anchor=anchor,bare=bare)
        grouped_cost = loop.stage_copies-start[0],store.bytes_copied-start[1]
        assert expected == actual and observations == reference.rows and grouped == accepted
        assert actual["status"]=="closed" and len(observations)==32 and usage=={
            "grouped_endpoint_pairs":16,"primed_worlds":2}
        assert serial_cost[0]==228 and grouped_cost[0]==90 and grouped_cost[1]*228==serial_cost[1]*90

        def missing(packet,_root,_check):
            return plans(packet,np,reject=True)

        before_refusal = loop.stage_copies
        state, observations, usage = grouped_root(serial,group,"blocked",prompts,labels,missing,design,
                                                   rejected.append,check,np,anchor=anchor,bare=bare)
        assert rejected[0]["status"]=="geometry_unqualified" and state["edited_pass2_runs"]==0
        assert not observations and not any(usage.values()) and loop.stage_copies==before_refusal
        adapter = RootBranches(serial,group,check)
        duplicate = [prompts[0]]*2
        clean = adapter.capture(duplicate)
        base = clean["writer_raw"][WRITER]
        other = base.clone()
        other[12] += .015625
        prediction = adapter.predict(clean,other)
        expect_refusal(lambda:adapter.capture(duplicate,edit=(base,other),planned=prediction),"qualification")
        adapter.approved = True
        edited = adapter.capture(duplicate,edit=(base,other),planned=prediction)
        before_wrong = loop.stage_copies
        expect_refusal(lambda:adapter.capture(duplicate,edit=(base,other),planned=prediction,
            rescue=(edited["k_raw"][SELF],clean["k_raw"][SELF]+1.)),"bound rescue branch")
        assert loop.stage_copies == before_wrong and adapter.pending is None
    assert before==logical_target_digest(model,store) and hook_count(loop)==hooks and not torch.cuda.is_initialized()
    return {"full_root_exact":True,"native_physical_vectors_exact":audited,"full_B2_edited_heads_exact":32,
            "independent_exact_k_rescues":16,"serial_stage_copies":serial_cost[0],"grouped_stage_copies":grouped_cost[0],
            "serial_requested_parameter_bytes":serial_cost[1],"grouped_requested_parameter_bytes":grouped_cost[1],
            "blocked_root_zero_edits":True,"stale_plan_before_staging":True,"partial_group_guard_recovery":True,
            "unexpected_rescue_refused":True,"cuda_initialized":False,"pretrained_forwards":0}


def project(snapshot):
    index = CheckpointStore(snapshot)
    totals = Counter()
    for name,span in index.spans.items():
        if name.startswith("model.layers."):
            totals[int(name.split(".")[2])] += span.length
    resident = json.loads((lane.REPO/"experiments/nan_bf31/config.json").read_bytes())["BF16_resident_decoder_layers"]
    if len(totals)!=22 or resident!=4:
        raise ValueError("The registered staged layout changed")
    sweep = sum(totals[i] for i in range(resident,22))
    return {"offloaded_layers":22-resident,"staged_parameter_bytes_per_sweep":sweep,
            "serial_stage_sweeps_per_qualified_root_after_baseline":76,
            "grouped_stage_sweeps_per_qualified_root_after_baseline":30,
            "requested_parameter_bytes_saved_per_qualified_root":46*sweep}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--work",type=Path)
    args = cli.parse_args()
    lock = registration()
    if args.freeze:
        write_new(MANIFEST,lock)
        return
    base = json.loads((lane.REPO/"experiments/nan_readpath37/registration.json").read_bytes())
    if not args.snapshot or not args.work or args.snapshot.name != base["revision"] or (
            lock != json.loads(MANIFEST.read_bytes())) or "torch" in sys.modules:
        cli.error("Use the frozen grouped fixture and pinned snapshot")
    with lease(BUDGET) as (check,samples):
        args.work.mkdir(parents=True,exist_ok=False)
        os.environ.update(USE_TORCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            HF_HOME=str((args.work/"hf").resolve()),HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE",None)
        for name in ("modeling_nanbeige.py","config.json","model.safetensors.index.json"):
            bound(args.snapshot/name,base["snapshot_pins"][name])
        bound(args.snapshot/"configuration_nanbeige.py",CONFIGURATION)
        assert {name:version(name) for name in PACKAGES} == PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        result = exercise(args.snapshot,args.work,check,np,torch)
        projection = project(args.snapshot)
    write_new(args.work/"verification.json",result | {"experiment":EXPERIMENT,"status":"passed",
        "source_manifest_sha256":digest(MANIFEST),"sealed_test_accessed":False,"formal_comparisons":0,
        "staged_BF16_projection":projection,"max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)})


if __name__ == "__main__":
    main()
