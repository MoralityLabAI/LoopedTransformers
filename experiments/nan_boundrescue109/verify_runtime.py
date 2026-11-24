"""Exercise host judgement, source isolation and immutable root closure without Torch."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile, copytree
import sys

from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import write_new
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab83.batches import record_roots
from experiments.nan_vocab83.custody import HOST_BUDGET, vocabulary
from experiments.nan_vocab83.verify import SyntheticHead, baseline_rows

from . import roster
from .core import ARMS, SIGNS, WORLDS
from .host import ENDPOINTS, MODE, check_plan, collect, host_inputs, margin, qualify
from .lane import verified
from .worker import bind_qualification


FIXTURE = "synthetic_fixture"


def refused(action, marker):
    try:
        action()
    except ValueError as e:
        assert marker in str(e), str(e)
    else:
        raise AssertionError("The intentionally invalid receipt was accepted")


def sample(root, ids, source, baseline, head):
    gold = {p["id"]:p["answer"] for p in source["gold"]}
    geometry, signed, worlds, observations = {}, {}, {}, []
    summaries = {"first_pass":0.,"second_pass":.125,"delayed":.125,
                 "delayed_with_clean_k":0.,"k_rescue_removed":.125}
    for world, other in (("A","B"),("B","A")):
        own, foil = gold[ids[world]],gold[ids[other]]
        third = next(k for k in head.bare if k not in (own,foil))
        clean = baseline[ids[world]]["phases"]
        changed, _ = head.read(own,foil,third,edited=True)
        delta = margin(changed,own,foil)-margin(clean[1],own,foil)
        assert delta == .5
        geometry[world] = {"parent":{"status":"measured"},"signed":{}}
        signed[world], worlds[world] = {}, {}
        for sign in SIGNS:
            key = str(sign)
            geometry[world]["signed"][key] = {"centered":{"differential":{"qualified":True}},
                "sham":{"status":"matched","reached_abs_chord_cosine":.01,
                        "chord_relative_error":.01,"max_endpoint_radius_relative_error":.02}}
            signed[world][key] = {"passed":True,"arm_gates":[{
                "passed":True,"greedy_same":True,"max_abs_logit_delta":.0625,"margin_delta_abs":0.} for _ in ARMS],
                "active_rms":[.01,.01],"sham_rms":[.01,.01],"absolute_rms_gaps":[0.,0.],"excluded_rms_gaps":[0.,0.],
                "paired_label_contrast":0.}
            arms = {}
            for arm in ARMS:
                record = {"zeroed_k_delta":True}
                for mode in ENDPOINTS:
                    changed_arm = mode == "edited" and sign == 1 and arm == "active_t2"
                    outcome = changed if changed_arm else clean[1]
                    observations.append({"rescued":mode=="k_rescue","phases":[clean[0],outcome]})
                    record[mode] = {"answer_margin_change":delta if changed_arm else 0.,
                        "greedy_token_changed":bool(changed_arm),"reached_greedy_correct":not changed_arm,
                        "reached_projection":{"k":{"l2_change":0. if mode=="k_rescue" else .125}}}
                arms[arm] = record
            expected = .25 if sign==1 else 0.
            worlds[world][key] = {"arms":arms,"contrast":{"first_pass":0.,"second_pass":expected,
                        "second_pass_with_clean_k":0.,"delayed":expected,"delayed_with_clean_k":0.,"k_rescue_removed":expected}}
    return {"root":root,"geometry":geometry,"signed":signed,"status":"closed","edited_pass2_runs":len(observations),
            "head_observations":observations,"worlds":worlds,"root_signed_contrast":summaries}


def round_trip(work, source, public, roots, head, classes, tok, frozen, *, passes):
    work.mkdir()
    admission = {"experiment":roster.EXPERIMENT,"mode":FIXTURE,"backend":"nf4",
                 "runtime_sha256":frozen,"published_commit":"f"*40}
    write_new(work/"admission.json",admission)
    position = json.loads(roster.RECEIPT.read_bytes())["max_position"]
    write_new(work/"preflight.json",{"native_heads_exact":True,"head_placement_exact":True,
              "depths":{str(k):{"tested_max_position":position} for k in (1,2)}})
    rows = baseline_rows(source,roots,head,gate_passes=passes)
    if passes:
        incomplete = next(r for r in rows if r["id"] == roots[max(roots)]["B"])
        incomplete["phases"][1] = incomplete["phases"][0]
        incomplete["class_maxima"][1] = incomplete["class_maxima"][0]
    write_new(work/"baseline.json",rows)
    baseline = {r["id"]:r for r in rows}
    pin = sha(work/"baseline.json")
    grant = qualify(source,rows,pin,classes,tok,admission)
    write_new(work/"qualification.json",grant)
    compact = [{k:v for k,v in r.items() if k!="class_maxima"} for r in rows]
    gate,pairs = bind_qualification(work,digest(compact),classes,roots,public,admission)
    assert gate["passed"] is passes
    assert len(pairs) == (15 if passes else 0) and max(roots) not in pairs
    first = min(roots)
    ordinal = {r:i for i,r in enumerate(sorted(roots))}

    def trial(root, ids):
        assert root in pairs
        record = sample(root,ids,source,baseline,head) if root==first else {
            "root":root,"geometry":{},"signed":{},"status":"geometry_unqualified","edited_pass2_runs":0}
        plan = {k:record[k] for k in ("root","geometry","signed","status")}
        if root==first:
            plan["status"] = "qualified"
        path = work/f"plan_{ordinal[root]:03d}.json"
        write_new(path,plan)
        return record | {"plan_sha256":sha(path)}

    closed = record_roots(work,public,roots,gate,trial)
    target = runtime.TARGETS["nf4"]
    pins = {f"{key}_sha256":sha(work/file) for key,file in (
        ("admission","admission.json"),("baseline","baseline.json"),("qualification","qualification.json"),
        ("preflight","preflight.json"),("journal","closed.jsonl"))}
    write_new(work/"terminal.json",admission | pins | {"status":"completed","target_before":target,
              "target_after":target,"quant_before":{},"quant_after":{},"partition_sha256":partition_sha(classes),
              "checkpoint_verified_after":True,"root_records":closed,"baseline_batches":len(rows)//2})
    return collect(work,source,classes,tok,frozen,"nf4",mode=FIXTURE)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","source-dir","work"):
        cli.add_argument("--"+name,type=Path,required=True)
    args = cli.parse_args()
    frozen = verified()
    args.work.mkdir(parents=True,exist_ok=False)
    with lease(HOST_BUDGET) as (check,samples):
        copied = args.work/"public"
        copied.mkdir()
        for n in ("public.json","tokens.json"):
            copyfile(args.source_dir/n,copied/n)
        public = roster.worker_source(copied,args.snapshot,check)
        assert set(public)=={"public","tokens"} and not (copied/"gold.json").exists()
        source,proof = host_inputs(args.source_dir)
        tok,classes = vocabulary(args.snapshot,check,proof["tokenizer_pins"])
        head = SyntheticHead(classes,source["tokens"]["bare_label_token_ids"])
        roots = {}
        for row in public["public"]:
            if row["arm"] == "focused":
                roots.setdefault(row["root"],{})[row["world"]] = row["id"]
        success, private = round_trip(args.work/"success",source,public,roots,head,classes,tok,frozen,passes=True)
        empty,_ = round_trip(args.work/"refusal",source,public,roots,head,classes,tok,frozen,passes=False)
        assert success["status_counts"] == {"closed":1,"geometry_unqualified":14,"native_root_unqualified":1}
        assert success["closed_root_macro"] == {"first_pass":0.,"second_pass":.125,"delayed":.125,
                                              "delayed_with_clean_k":0.,"k_rescue_removed":.125}
        assert success["answer_events_by_signed_world"]["active_t2_regressions_vs_t1"] == 2
        assert success["answer_events_by_signed_world"]["active_t2_rescue_repairs"] == 2
        assert len(private)==1 and empty["status_counts"]=={"native_gate_failed":16} and empty["closed_roots"]==0
        refused(lambda:collect(args.work/"success",source,classes,tok,frozen,"nf4",mode=MODE), "execution mode")
        bad = args.work/"tampered"
        copytree(args.work/"success",bad)
        with (bad/"root_000.json").open("ab") as stream:
            stream.write(b" ")
        refused(lambda:collect(bad,source,classes,tok,frozen,"nf4",mode=FIXTURE),"Source hash mismatch")
        plan = json.loads((args.work/"success"/"plan_000.json").read_bytes())
        episode = json.loads((args.work/"success"/"root_000.json").read_bytes())
        plan["signed"]["A"]["1"]["excluded_rms_gaps"][0] = .5
        altered = deepcopy(episode)
        altered["signed"] = plan["signed"]
        refused(lambda:check_plan(plan,altered),"two RMS gates")
        assert "torch" not in sys.modules
    write_new(args.work/"verification.json",{"status":"passed","runtime_sha256":frozen,"baseline_batches":48,
              "gate_failed_zero_edits":True,"native_root_unqualified_zero_edits":True,
              "oracle_bound_without_worker_gold":True,"archived_head_audited":True,"tamper_refused":True,
              "synthetic_refused_as_pretrained":True,"pretrained_forwards":0,
              "fixture_root_macro":success["closed_root_macro"],"formal_comparisons":0,"sealed_test_accessed":False,
              "max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
              "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)})


if __name__=="__main__":
    main()
