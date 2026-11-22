"""NEC-03W: immutable, read-only sham-control coverage anatomy.

This is a retrospective source-training archive audit of the already
completed NEC-03V experiment (original Actions run 44968927511).
No Torch, model/SAE checkpoint load, new model forward, new candidate
search, changed sham, or new correctness endpoint. Only the original
saved strong-sham choice is examined under diagnostic tolerance grids.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1, sha256
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median

SCHEMA="NEC-03W/read-only-original-sham-coverage-anatomy/v1"
LOCK_NAME="NEC03W_ARCHIVE_LOCK.json"
ROOTS=tuple(range(8))
QUALIFIED=(0,1,2,4,6,7)
ORIGINAL_ZEROS=(3,5)
ARMS=("composed","additive")
BUDGETS={"four":4,"six":6}
PANEL=tuple(range(32,40))
ANGLE_GRID=(.25,.35,.50)
COHESION_GRID=(.50,.75,1.0)
PRIORITY=("no_source_site","insufficient_topk","angle","cohesion","dose","mismatch")
OLD_MODES=(
    "clean","hubs","native_nonhub","full_affine",
    "contrib16_only","coherent4_source","coherent4_sham",
    "coherent6_source","coherent6_sham","hash4_source","hash4_sham",
)


def sha(path:Path):
    return sha256(path.read_bytes()).hexdigest()


def read(path:Path):
    return json.loads(path.read_text())


def finite(value,what):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise AssertionError("nonfinite or malformed archived "+what)
    return float(value)


def verify_lock():
    path=Path(__file__).with_name(LOCK_NAME)
    cfg=read(path)
    exact={
        "schema":"NEC-03W/frozen-retrospective-archive-coverage-method/v1",
        "old_nec03v_actions_run":44968927511,
        "original_qualified_worlds":list(QUALIFIED),
        "original_source_ITT_zeros":list(ORIGINAL_ZEROS),
        "original_root_indices":list(ROOTS),
        "original_training_source_groups":list(PANEL),
        "original_histories_per_teacher":256,
        "original_model_arms":list(ARMS),
        "original_modes":list(OLD_MODES),
        "original_group_budgets":BUDGETS,
        "original_angle_gate":.25,
        "original_cohesion_gate":.50,
        "dose_floor":1e-6,
        "dose_relative_match":.01,
        "angle_grid":list(ANGLE_GRID),
        "cohesion_grid":list(COHESION_GRID),
        "first_failure_priority":list(PRIORITY),
        "original_primary_accuracy_effect":25/2048,
        "original_source_effect_p_exploratory":.5,
        "no_new_sham_candidates":True,
        "posthoc_grid_has_no_counterfactual_predictions":True,
        "new_model_forwards":0,
        "model_or_SAE_training_updates":0,
        "new_development_eval_scored":0,
        "new_reserved_scored":0,
        "new_recruited_worlds_scored":0,
        "new_embargoed_five_input_scored":0,
        "new_four_input_confirmation_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k)!=v for k,v in exact.items()):
        raise AssertionError("frozen original source/threshold-grid method changed")
    if set(cfg["original_root_pins"])!=set(map(str,ROOTS)):
        raise AssertionError("original eight root hash manifest incomplete")
    if set(cfg["original_artifact_ids"])!=set(map(str,ROOTS))|{"aggregate"}:
        raise AssertionError("original artifact identity changed")
    if len(cfg["original_aggregate_sha256"])!=64:
        raise AssertionError("original eight-root ITT SHA missing")
    for i in ROOTS:
        pin=cfg["original_root_pins"][str(i)]
        if set(pin)!={"world_sha256","root_report","original_audit",
                       "composed_raw","additive_raw","original_source_lock"}:
            raise AssertionError("original root source evidence entry incomplete")
        if any(len(x)!=64 for x in pin.values()):
            raise AssertionError("original root source evidence hash malformed")
    root=Path(__file__).resolve().parents[2]
    for filename,pinned in cfg["git_blob_sha1"].items():
        raw=(root/filename).read_bytes()
        actual=sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        if actual!=pinned:
            raise AssertionError("read-only NEC03W source code differs: "+filename)
    return cfg,sha(path)


def summary_stats(values):
    if not values:
        return {"n":0,"median":None,"mean":None,"p10":None,"p90":None,
                "min":None,"max":None}
    a=sorted(map(lambda v:finite(v,"physical measurement"),values))
    return {"n":len(a),"median":median(a),"mean":fmean(a),
            "p10":a[int(.1*(len(a)-1))],"p90":a[int(.9*(len(a)-1))],
            "min":a[0],"max":a[-1]}


def raw_site_flags(group,mask):
    if group["k"] not in (4,6):
        raise AssertionError("unregistered group budget")
    active=group["active_ids"][mask]
    exact=group["exactly_16_positive_active_codes"][mask]
    if (not isinstance(exact,bool) or len(active)!=len(set(active))
        or any(type(v)!=int or not 0<=v<128 for v in active)
        or exact!=(len(active)==16)):
        raise AssertionError("original TopK16 support or source feature IDs changed")
    k=group["k"]
    source=group["source_ids"][mask]
    sham=group["sham_ids"][mask]
    count=group["exhaustive_control_count"][mask]
    angle=group["angular_gap"][mask]
    cohesion=group["cohesion_gap"][mask]
    dose=group["angular_physical_dose"]
    if exact:
        if (len(source)!=k or len(sham)!=k or
            len(set(source))!=k or len(set(sham))!=k or
            set(source)&set(sham) or not set(source+sham)<=set(active) or
            count!=math.comb(16-k,k)):
            raise AssertionError("original source/sham disjoint active roster changed")
        angle=finite(angle,"original angular gap")
        cohesion=finite(cohesion,"original cohesion gap")
        near_joint=finite(group["sham_joint_score"][mask],"original sham joint score")
        if abs(near_joint-(angle+.5*cohesion))>1e-4:
            raise AssertionError("frozen original sham score differs from its saved geometry")
        if angle<0 or cohesion<0:
            raise AssertionError("negative original angular/cohesion gap")
        finite(group["source_physical_cohesion"][mask],"physical cohesion")
        finite(group["sham_physical_cohesion"][mask],"physical cohesion")
    elif (source or sham or count or angle is not None or cohesion is not None
          or group["sham_joint_score"][mask] is not None):
        raise AssertionError("unsupported SAE TopK sample had invented control")
    raw0=finite(dose["raw_source_32d_dose"][mask],"raw source dose")
    raw1=finite(dose["raw_sham_32d_dose"][mask],"raw sham dose")
    shared=finite(dose["shared_output_32d_dose"][mask],"shared source/sham dose")
    gain0=finite(dose["source_gain"][mask],"source gain")
    gain1=finite(dose["sham_gain"][mask],"sham gain")
    if (min(raw0,raw1)<0 or shared<0 or min(gain0,gain1)<0
        or max(gain0,gain1)>1.000001
        or abs(shared-min(raw0,raw1))>max(1e-5,1e-5*abs(shared))):
        raise AssertionError("unmatched or amplified original physical dose")
    nonzero=dose["nonzero_dose"][mask]
    matched=dose["matched_1pct"][mask]
    if type(nonzero)!=bool or type(matched)!=bool or nonzero!=(shared>1e-6):
        raise AssertionError("original positive shared dose policy changed")
    if nonzero and (
        abs(gain0*raw0-shared)/max(shared,1e-6)>.0101 or
        abs(gain1*raw1-shared)/max(shared,1e-6)>.0101):
        raise AssertionError("previously matched physical output doses changed")
    flags=set()
    if not exact:flags.add("insufficient_topk")
    if angle is not None and angle>.25:flags.add("angle")
    if cohesion is not None and cohesion>.50:flags.add("cohesion")
    if not nonzero:flags.add("dose")
    if not matched:flags.add("mismatch")
    return flags,{"angle":angle,"cohesion":cohesion,"raw_source":raw0,
                  "raw_sham":raw1,"shared":shared,
                  "source_gain":gain0,"sham_gain":gain1,
                  "support":exact,"nonzero":nonzero,"matched":matched}


def admissible(site_values, *, angle, cohesion):
    return bool(site_values) and all(
        s["support"] and s["angle"] is not None and s["angle"]<=angle
        and s["cohesion"] is not None and s["cohesion"]<=cohesion
        and s["nonzero"] and s["matched"]
        for s in site_values)


def original_correctness(record, counts):
    if (set(record["logits"])!=set(OLD_MODES)
        or set(record["source_predictions"])!=set(OLD_MODES)):
        raise AssertionError("original 11 intervention outcomes were replaced or omitted")
    truth=record["original_source_true_labels"]
    if len(truth)!=32 or any(type(y)!=int or y not in (0,1) for y in truth):
        raise AssertionError("original source label incomplete")
    histories=record["source_input_histories"]
    if (len(histories)!=32 or len({tuple(row) for row in histories})!=32
        or any(len(row)!=12 or any(bit not in (0,1) for bit in row)
               for row in histories)):
        raise AssertionError("original complete source context/mask set changed")
    for mode in OLD_MODES:
        logits=record["logits"][mode]
        pred=record["source_predictions"][mode]
        if len(logits)!=32 or len(pred)!=32:
            raise AssertionError("source case predictions or logits omitted")
        for i in range(32):
            if not isinstance(logits[i],list) or len(logits[i])!=2:
                raise AssertionError("not a binary logit source prediction")
            a,b=(finite(v,"old model logit") for v in logits[i])
            forecast=int(b>a)
            if forecast!=pred[i]:
                raise AssertionError("previous original scored model prediction altered")
            counts[mode]+=int(forecast==truth[i])


def validate_original_parent(agg,pins):
    if (agg["schema"]!="NEC-03V/eight-world-coherent-group-source-only-ITT/v1"
        or agg["original_world_root_ITT_count"]!=8
        or agg["independently_audited_root_count"]!=8
        or not agg["all_eight_independent_root_neural_audits_complete"]
        or agg["original_paired_source_eligible_roots"]!=list(QUALIFIED)
        or agg["original_source_ineligible_zero_roots"]!=list(ORIGINAL_ZEROS)
        or agg["original_source_groups"]!=list(PANEL)
        or agg["source_training_histories_per_arm"]!=256
        or agg["original_native_or_SAE_optimizer_updates"]!=0
        or agg["source_only_mechanism_resolution_gate_passed"] is not False
        or agg["primary_eight_root_macro"]!=25/2048
        or agg["exploratory_two_sided_exact_root_signflip_p"]!=.5
        or agg["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or len(agg["roots"])!=8
        or any(agg[k]!=0 for k in (
            "new_development_eval_groups_scored","reserved_groups_scored",
            "new_recruited_worlds_scored","five_input_embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"))):
        raise AssertionError("original NEC03V independently audited ITT modified")
    for index,row in enumerate(agg["roots"]):
        pin=pins[str(index)]
        if (row["world_index"]!=index or row["world_sha256"]!=pin["world_sha256"]
            or not row["independent_audit_complete"]
            or row["root_report_sha256"]!=pin["root_report"]
            or row["independent_audit_sha256"]!=pin["original_audit"]
            or row["original_paired_source_ITT_failure"]!=(index in ORIGINAL_ZEROS)):
            raise AssertionError("original NEC03V world root or receipt changed")


def process_root(index,folder,original,cfg):
    pin=cfg["original_root_pins"][str(index)]
    rp=folder/"nec03v_source_group_root_report.json"
    ap=folder/"independent_nec03v_source_group_audit.json"
    if sha(rp)!=pin["root_report"] or sha(ap)!=pin["original_audit"]:
        raise AssertionError("original source world report or independent audit mutated")
    report=read(rp)
    audit=read(ap)
    if (report["schema"]!="NEC-03V/source-only-coherent-group-necessity/v1"
        or audit["schema"]!="NEC-03V/independent-coherent-group-physical-null-replay/v1"
        or report["world_index"]!=index or audit["world_index"]!=index
        or report["world_sha256"]!=pin["world_sha256"]
        or audit["world_sha256"]!=pin["world_sha256"]
        or report["source_lock_sha256"]!=pin["original_source_lock"]
        or audit["source_lock_sha256"]!=pin["original_source_lock"]
        or audit["original_root_report_sha256"]!=sha(rp)
        or report["source_groups_scored"]!=list(PANEL)
        or report["source_cases_per_model_arm"]!=256
        or audit["source_training_cases_per_arm"]!=256
        or report["paired_qualified_in_original_source_roster"]!=(index in QUALIFIED)
        or report["original_ITT_source_failure"]!=(index in ORIGINAL_ZEROS)
        or any(report[k]!=0 for k in (
            "new_native_or_SAE_training_updates","new_development_eval_groups_scored",
            "reserved_groups_scored","new_recruited_worlds_scored",
            "five_input_embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"))
        or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or audit["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("original source-only world provenance changed")
    if not all(audit[k] for k in (
        "all_original_source_inputs_and_Boolean_labels_independently_replayed",
        "frozen_model_and_SAE_sha256_verified",
        "all_coherent_group_source_ids_and_disjoint_shams_reconstructed",
        "all_original_sham_corners_and_physical_32d_output_doses_recomputed",
        "all_eleven_full_recurrent_neural_outcomes_independently_replayed",
        "all_source_case_dropout_and_hash_controls_verified",
        "both_original_teacher_arms_replayed")):
        raise AssertionError("missing original independent complete neural audit")
    arms={}
    cases=[]
    historical={}
    for arm in ARMS:
        receipt=report["arms"][arm]
        verified=audit["arms"][arm]
        original_arm=original["arms"][arm]
        raw_path=folder/f"{arm}.nec03v_source_coherent_groups.jsonl"
        if (receipt["raw_filename"]!=raw_path.name
            or receipt["raw_sha256"]!=sha(raw_path)
            or sha(raw_path)!=pin[f"{arm}_raw"]
            or verified["raw_sha256"]!=sha(raw_path)
            or receipt["correct_counts"]!=verified["correct_counts"]
            or receipt["correct_counts"]!=original_arm["correct_counts"]
            or receipt["matched_group_valid_cases"]!=original_arm["valid_physical_null_cases"]
            or receipt["matched_group_valid_cases"]!=verified["valid_physical_null_cases"]
            or receipt["source_model_sha256"]!=verified["model_sha256"]
            or receipt["source_sae_checkpoint_sha256"]!=verified["SAE_sha256"]
            or receipt["source_sae_rank_sha256"]!=verified["source_rank_sha256"]
            or receipt["source_groups_scored"]!=list(PANEL)
            or receipt["cases_per_arm"]!=256):
            raise AssertionError("original audited raw root/arm receipt changed")
        rows=[json.loads(line) for line in raw_path.read_text().splitlines()]
        if len(rows)!=8:
            raise AssertionError("original source raw root missing complete 32-mask group")
        modes=Counter()
        per_budget={}
        input_histories=[]
        for budget in BUDGETS:
            per_budget[budget]={
                "valid":0,"hash_valid":0,"case_priority":Counter(),
                "case_overlap":Counter(),"site_priority":Counter(),
                "site_overlap":Counter(),"site_count":0,
                "without_angle":0,"without_cohesion":0,
                "hypothetical_grid":{f"{a:.2f}|{c:.2f}":0
                    for a in ANGLE_GRID for c in COHESION_GRID},
                "physical_arrays":{key:[] for key in (
                    "angle_gap","cohesion_gap","raw_source","raw_sham",
                    "shared","source_gain","sham_gain")},
            }
        for expected_group,row in zip(PANEL,rows):
            if (row["schema"]!="NEC-03V/source-only-coherent-group-necessity/v1"
                or row["source_group"]!=expected_group
                or row["arm"]!=arm or row["focal_masks"]!=list(range(32))
                or row["teacher_labels_used_for_group_selection"] is not False
                or set(row["valid_group_source_sham_on_all_visits"])!=set(BUDGETS)
                or len(row["valid_hash_on_all_visits"])!=32):
                raise AssertionError("original source group, treatment or order changed")
            original_correctness(row,modes)
            input_histories.extend(row["source_input_histories"])
            sites=row["site_geometry"]
            if not isinstance(sites,dict):
                raise AssertionError("source physical site geometry missing")
            if any(len(row["valid_group_source_sham_on_all_visits"][b])!=32
                   for b in BUDGETS):
                raise AssertionError("source control availability mask missing")
            for mask in range(32):
                for budget in BUDGETS:
                    k=BUDGETS[budget]
                    values=[]
                    flags=set()
                    if not sites:
                        flags.add("no_source_site")
                    for visit,site in sites.items():
                        if (set(site["groups"])!=set(BUDGETS)
                            or site["groups"][budget]["k"]!=k):
                            raise AssertionError("saved physical source group width changed")
                        site_flags,detail=raw_site_flags(site["groups"][budget],mask)
                        values.append(detail)
                        per_budget[budget]["site_count"]+=1
                        for key,val in detail.items():
                            if key in per_budget[budget]["physical_arrays"] and val is not None:
                                per_budget[budget]["physical_arrays"][key].append(val)
                        flags.update(site_flags)
                        for reason in site_flags:
                            per_budget[budget]["site_overlap"][reason]+=1
                        for reason in PRIORITY:
                            if reason in site_flags:
                                per_budget[budget]["site_priority"][reason]+=1
                                break
                    archived=row["valid_group_source_sham_on_all_visits"][budget][mask]
                    strict=not flags
                    if type(archived)!=bool or strict!=archived:
                        raise AssertionError("original strict physical gate fails reconstruction")
                    per_budget[budget]["valid"]+=int(strict)
                    for flag in flags:per_budget[budget]["case_overlap"][flag]+=1
                    for reason in PRIORITY:
                        if reason in flags:
                            per_budget[budget]["case_priority"][reason]+=1
                            break
                    no_angle=bool(values) and all(
                        v["support"] and v["cohesion"] is not None
                        and v["cohesion"]<=.5 and v["nonzero"] and v["matched"] for v in values)
                    no_cohesion=bool(values) and all(
                        v["support"] and v["angle"] is not None
                        and v["angle"]<=.25 and v["nonzero"] and v["matched"] for v in values)
                    per_budget[budget]["without_angle"]+=int(no_angle)
                    per_budget[budget]["without_cohesion"]+=int(no_cohesion)
                    membership={}
                    for angle in ANGLE_GRID:
                        for cohesion in COHESION_GRID:
                            key=f"{angle:.2f}|{cohesion:.2f}"
                            result=admissible(values,angle=angle,cohesion=cohesion)
                            membership[key]=result
                            per_budget[budget]["hypothetical_grid"][key]+=int(result)
                    cases.append({
                        "root_index":index,"world_id":report["world_id"],
                        "arm":arm,"source_group":expected_group,
                        "focal_mask":mask,"budget":budget,
                        "selected_recurrent_sites":len(sites),
                        "original_angular_sham_valid":strict,
                        "original_hash_sham_valid":row["valid_hash_on_all_visits"][mask]
                            if budget=="four" else None,
                        "original_failed_constraints":sorted(flags),
                        "first_failure_priority":next((f for f in PRIORITY if f in flags),None),
                        "same_original_sham_without_angle_gate":no_angle,
                        "same_original_sham_without_cohesion_gate":no_cohesion,
                        "same_original_sham_tolerance_grid":membership,
                    })
                if type(row["valid_hash_on_all_visits"][mask])!=bool:
                    raise AssertionError("original hash sham control validity not Boolean")
                per_budget["four"]["hash_valid"]+=int(row["valid_hash_on_all_visits"][mask])
        if len(input_histories)!=256 or any(modes[mode]!=receipt["correct_counts"][mode]
                                             for mode in OLD_MODES):
            raise AssertionError("original correctness or source training group total altered")
        if any(per_budget[b]["valid"]!=receipt["matched_group_valid_cases"][b]
               for b in BUDGETS):
            raise AssertionError("original strong sham gate mismatches audited source receipt")
        if per_budget["four"]["hash_valid"]!=receipt["hash_valid_cases"]:
            raise AssertionError("original hash sham control availability changed")
        historical[arm]={
            "input_histories":input_histories,
            "correct_counts":dict(modes),
            "physical_sha256":receipt["raw_sha256"],
        }
        arms[arm]={
            "source_groups":list(PANEL),
            "source_cases":256,
            "original_source_candidate_present":receipt["original_source_candidate_present"],
            "original_arm_source_eligible":receipt["original_arm_source_eligible"],
            "original_correct_counts":dict(modes),
            "physical_controls":{},
        }
        for name,x in per_budget.items():
            arms[arm]["physical_controls"][name]={
                "original_valid":x["valid"],
                "original_hash_valid":x["hash_valid"] if name=="four" else None,
                "unavailable_cases":256-x["valid"],
                "site_count":x["site_count"],
                "case_failure_priority":{f:x["case_priority"][f] for f in PRIORITY},
                "case_failure_overlap":{f:x["case_overlap"][f] for f in PRIORITY},
                "site_failure_priority":{f:x["site_priority"][f] for f in PRIORITY},
                "site_failure_overlap":{f:x["site_overlap"][f] for f in PRIORITY},
                "original_selected_sham_without_angle_gate":x["without_angle"],
                "original_selected_sham_without_cohesion_gate":x["without_cohesion"],
                "posthoc_existing_sham_coverage_grid":x["hypothetical_grid"],
                "original_visit_geometry":{k:summary_stats(v)
                                          for k,v in x["physical_arrays"].items()},
            }
    if historical["composed"]["input_histories"]!=historical["additive"]["input_histories"]:
        raise AssertionError("original source world composed/additive histories unpaired")
    raw_did=(
        arms["composed"]["original_correct_counts"]["coherent4_sham"]-
        arms["composed"]["original_correct_counts"]["coherent4_source"]-
        arms["additive"]["original_correct_counts"]["coherent4_sham"]+
        arms["additive"]["original_correct_counts"]["coherent4_source"])/256
    if (abs(raw_did-report["raw_root_primary_coherent4_necessity_DID"])>1e-12
        or abs((raw_did if index in QUALIFIED else 0)-original["primary_root_effect"])>1e-12):
        raise AssertionError("original root correctness ITT changed")
    return {
        "world_index":index,"world_id":report["world_id"],
        "world_sha256":report["world_sha256"],
        "original_report_sha256":sha(rp),"original_independent_audit_sha256":sha(ap),
        "original_primary_root_effect":original["primary_root_effect"],
        "original_ITT_zero":index in ORIGINAL_ZEROS,
        "arms":arms,
    },cases


def aggregate(input_root,original_aggregate,out):
    cfg,locksha=verify_lock()
    if sha(original_aggregate)!=cfg["original_aggregate_sha256"]:
        raise AssertionError("original NEC03V aggregate SHA256 changed")
    parent=read(original_aggregate)
    validate_original_parent(parent,cfg["original_root_pins"])
    expected={f"nec03v-source-root-{i}" for i in ROOTS}
    found={p.name for p in input_root.iterdir() if p.is_dir()}
    if found!=expected:
        raise AssertionError("missing or unregistered root source archive directory")
    roots=[]
    cases=[]
    for i in ROOTS:
        r,rows=process_root(
            i,input_root/f"nec03v-source-root-{i}",parent["roots"][i],cfg)
        roots.append(r)
        cases.extend(rows)
    if len(cases)!=8*2*256*2:
        raise AssertionError("incomplete original per-budget source control observations")
    totals={}
    for budget in BUDGETS:
        totals[budget]={}
        for arm in ARMS:
            relevant=[r["arms"][arm]["physical_controls"][budget]
                      for r in roots if r["world_index"] in QUALIFIED]
            totals[budget][arm]={
                "original_qualified_root_count":len(QUALIFIED),
                "original_source_cases":256*len(QUALIFIED),
                "original_strict_valid":sum(v["original_valid"] for v in relevant),
                "original_unconditioned_hash_valid":(
                    sum(v["original_hash_valid"] for v in relevant)
                    if budget=="four" else None),
                "case_failure_priority":{f:sum(v["case_failure_priority"][f] for v in relevant)
                                         for f in PRIORITY},
                "case_failure_overlap":{f:sum(v["case_failure_overlap"][f] for v in relevant)
                                        for f in PRIORITY},
                "site_failure_priority":{f:sum(v["site_failure_priority"][f] for v in relevant)
                                         for f in PRIORITY},
                "site_failure_overlap":{f:sum(v["site_failure_overlap"][f] for v in relevant)
                                        for f in PRIORITY},
                "source_site_input_observations":sum(v["site_count"] for v in relevant),
                "posthoc_same_selected_sham_without_angle_gate":
                    sum(v["original_selected_sham_without_angle_gate"] for v in relevant),
                "posthoc_same_selected_sham_without_cohesion_gate":
                    sum(v["original_selected_sham_without_cohesion_gate"] for v in relevant),
                "posthoc_same_selected_sham_tolerance_grid":{
                    f"{a:.2f}|{c:.2f}":
                        sum(v["posthoc_existing_sham_coverage_grid"][f"{a:.2f}|{c:.2f}"]
                            for v in relevant)
                    for a in ANGLE_GRID for c in COHESION_GRID
                },
            }
            baseline=totals[budget][arm]["original_strict_valid"]
            if baseline!=totals[budget][arm]["posthoc_same_selected_sham_tolerance_grid"]["0.25|0.50"]:
                raise AssertionError("original physical sham gate differs from diagnostic grid")
            if sum(totals[budget][arm]["case_failure_priority"].values())+baseline!=256*len(QUALIFIED):
                raise AssertionError("case priority misses original unavailable cases")
    composed=totals["four"]["composed"]["original_strict_valid"]
    additive=totals["four"]["additive"]["original_strict_valid"]
    if (composed!=517 or additive!=931 or
        parent["primary_eight_root_macro"]!=25/2048 or
        parent["eligible_root_positive_count"]!=2):
        raise AssertionError("original NEC03V failed physical sham coverage or effect changed")
    result={
        "schema":SCHEMA,
        "frozen_new_archive_lock_sha256":locksha,
        "old_actions_run":44968927511,
        "old_aggregate_sha256":sha(original_aggregate),
        "old_eight_original_independent_neural_audits_verified":True,
        "root_count":len(roots),"teacher_arms_per_root":2,
        "source_training_groups":list(PANEL),
        "source_cases_per_teacher":256,
        "original_source_qualified_roots":list(QUALIFIED),
        "original_source_zero_ITT_roots":list(ORIGINAL_ZEROS),
        "old_registered_accuracy_ITT_effect":25/2048,
        "old_registered_accuracy_signflip_p":.5,
        "old_registered_meaningful_floor":.02,
        "old_mechanism_resolution_gate_passed":False,
        "new_model_forwards":0,
        "new_SAE_or_model_training_updates":0,
        "new_source_interventions":0,
        "new_sham_candidates_searched":0,
        "new_model_correctness_outcomes_scored":0,
        "all_original_root_and_arm_report_AUDIT_RAW_SHA256_verified":True,
        "all_original_neural_predictions_recomputed_from_saved_logits":True,
        "posthoc_sham_eligibility_has_no_scored_counterfactual_outcomes":True,
        "new_development_eval_groups_scored":0,
        "new_reserved_context_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "new_five_input_embargoed_worlds_scored":0,
        "new_old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "qualified_source_coverage":totals,
        "original_roots":roots,
    }
    out.mkdir(parents=True,exist_ok=True)
    case_bytes=b"".join(
        (json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        for row in cases)
    (out/"nec03w_casewise_control_failure_receipts.jsonl").write_bytes(case_bytes)
    result["case_receipt_sha256"]=sha256(case_bytes).hexdigest()
    (out/"nec03w_original_sham_coverage_anatomy.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    with (out/"nec03w_original_selected_sham_grid.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["group","arm","original_eligible_source_roots","source_cases",
                         "angle_gate","cohesion_gate","original_selected_sham_eligible_cases"])
        for budget in BUDGETS:
            for arm in ARMS:
                row=totals[budget][arm]
                for a in ANGLE_GRID:
                    for c in COHESION_GRID:
                        writer.writerow([budget,arm,len(QUALIFIED),1536,a,c,
                            row["posthoc_same_selected_sham_tolerance_grid"][f"{a:.2f}|{c:.2f}"]])
    print("NEC03W_IMMUTABLE_ARCHIVAL_SHAM_COVERAGE_PASS",
          composed,additive,"original_ITT",25/2048,"cases",len(cases))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    aggregate(a.root,a.original_aggregate,a.out)
