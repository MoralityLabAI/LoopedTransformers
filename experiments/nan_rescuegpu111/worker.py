"""Run grouped BOUNDRESCUE109 roots through the existing independent host contract."""

from time import monotonic

from experiments.nan_boundrescue109 import roster
from experiments.nan_boundrescue109.host import CFG
from experiments.nan_boundrescue109.worker import bind_qualification
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import write_new
from experiments.nan_rescuegroup110.group import grouped_root
from experiments.nan_vocab82.heads import Reducer
from experiments.nan_vocab83.batches import record_baseline, record_roots


VARIANT = "NAN-RESCUEGPU-111"
QUALIFIED_SWEEPS = 30
BASELINE_SWEEPS = 2
OFFLOADED_LAYERS = 18


def validate_usage(record):
    edited = record.get("edited_pass2_runs", 0)
    usage = record["grouped_usage"]
    count = usage["stage_copies"]
    seconds = usage["wall_seconds"]
    from math import isfinite

    if type(count) is not int or count < 0 or count % OFFLOADED_LAYERS or (
            type(seconds) not in (int,float) or not isfinite(seconds) or seconds < 0):
        raise ValueError("The root lacks valid grouped execution counters")
    if record["status"] == "closed":
        if (edited,usage["grouped_endpoint_pairs"],usage["primed_worlds"],count) != (
                32,16,2,QUALIFIED_SWEEPS*OFFLOADED_LAYERS):
            raise ValueError("The closed root stage or edited-branch count differs from the grouped schedule")
    elif edited or usage["grouped_endpoint_pairs"] or usage["primed_worlds"]:
        raise ValueError("A refused root executed grouped edits")


def drive(work, serial, grouped, payload, admission, classes, planner, design, host_gate, check, np):
    loop = grouped.loop
    start = loop.stage_copies
    roots, digest, batches = record_baseline(work,serial,payload,CFG,check,Reducer(classes),classes)
    baseline_stage_copies = loop.stage_copies-start
    if baseline_stage_copies != BASELINE_SWEEPS*batches*OFFLOADED_LAYERS:
        raise RuntimeError("The native baseline stage-copy count differs")
    host_gate()
    gate, pairs = bind_qualification(work,digest,classes,roots,payload,admission)
    order = {r:i for i,r in enumerate(sorted(roots))}
    bare = payload["tokens"]["bare_label_token_ids"]

    def trial(root, entry):
        anchor, prompts = entry
        plan = work/f"plan_{order[root]:03d}.json"
        before, began = loop.stage_copies,monotonic()
        result, heads, usage = grouped_root(serial,grouped,root,prompts,pairs[root],planner,design,
                                            lambda value:write_new(plan,value),check,np,anchor=anchor,bare=bare)
        usage |= {"stage_copies":loop.stage_copies-before,"wall_seconds":monotonic()-began}
        output = result | {"plan_sha256":sha(plan),"grouped_usage":usage}
        if heads:
            output["head_observations"] = heads
        validate_usage(output)
        return output

    count = record_roots(work,payload,roots,gate,trial)
    return {"baseline_batches":batches,"root_records":count,"baseline_stage_copies":baseline_stage_copies,
            "roster_stage_copies":loop.stage_copies-start}
