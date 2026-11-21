"""Independent NEC-03K-B stdlib raw causal, source and angular-null auditor.

Re-enumerate every disjoint currently active TopK16 candidate from the
reported physical 32-D decoder contributions. Check selection was the
nearest angle without observing labels, and reconstruct every dose gain,
linear margin displacement, exact nonlinear margin difference, source
versus control correctness and twelve recurrent-visit memory distances.

All inputs are old, already exposed NEC-03H development worlds. An
unrecognized root or original confirmation/reserved group fails closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

from mechinterp.neuron_communities.nec03g_worlds import (
    worlds,registry,context_catalog,
)
from mechinterp.neuron_communities.nec03k_source_audit import (
    dot,cosine,norm,vector_sum,close,file_hash,
)

SCHEMA="NEC-03K-B/source-angle-matched-control-recurrent-causal-dev/v1"
INDICES=tuple(range(8,16))
ANGLES=.25


def read(p):
    return json.loads(p.read_text())


def prediction(l):
    if len(l)!=2 or not all(math.isfinite(float(x)) for x in l):
        raise AssertionError("invalid binary output logits")
    return int(l[1]>l[0])


def margin(l,orient):
    if orient not in (0,1):
        raise AssertionError("invalid clean-argmax margin")
    return float(l[orient])-float(l[1-orient])


def verify_root(root:Path,old_panel:Path,source_cal:Path,
                measured:Path,dev_lock:Path,source_lock:Path,v2_lock:Path):
    world=read(root/"world_spec.json")
    index=world["world_index"]
    if index not in INDICES or world!=worlds()[index]:
        raise AssertionError("source validation attempted unregistered/confirmation world")
    _,original_contexts=context_catalog(world)
    if read(root/"context_source_manifest.json")!=original_contexts:
        raise AssertionError("input context split differs from frozen source")
    report=read(measured/"nec03k_development_report.json")
    old=read(root/"root_summary.json")
    source_manifest=read(root/"composed.source_freeze.json")
    source_reference=read(source_cal/"nec03k_source_calibration.json")
    if (report["schema"]!=SCHEMA
        or report["world_sha256"]!=world["world_sha256"]
        or report["causal_dev_protocol_sha256"]!=file_hash(dev_lock)
        or report["direction_matching_source_lock_sha256"]!=file_hash(source_lock)
        or report["original_NEC03H_v2_lock_sha256"]!=file_hash(v2_lock)
        or report["source_calibration_sha256"]!=file_hash(
            source_cal/"nec03k_source_calibration.json")
        or not source_reference["source_null_feasible"]
        or report["native_model_sha256"]!=source_manifest["model_sha256"]
        or report["source_sae_sha256"]!=source_manifest["source_sae"]["source_sae_sha256"]
        or report["source_rank_sha256"]!=file_hash(root/"composed.source_freeze.json")
        or report["world_index"]!=index
        or report["reserved_contexts_scored"]!=0
        or report["confirmation_worlds_scored"]!=0
        or report["trained_native_or_SAE_weights_modified"] is not False
        or report["source_root_specific_result_is_posthoc"] is not True):
        raise AssertionError("source lock/SAE/native model or confirmation contract changed")
    panel=read(old_panel/"source_blind_control_panel.json")
    if (report["old_hash_control_panel_sha256"]!=file_hash(
            old_panel/"source_blind_control_panel.json")
        or panel["world_sha256"]!=world["world_sha256"]
        or panel["native_model_sha256"]!=source_manifest["model_sha256"]):
        raise AssertionError("historical baseline drifted")
    first=panel["comparisons"][0]
    path=old_panel/first["raw_filename"]
    if file_hash(path)!=first["raw_context_sha256"]:
        raise AssertionError("original baseline byte hash changed")
    old_rows=[json.loads(s) for s in path.read_text().splitlines()]
    if len(old_rows)!=48:
        raise AssertionError("original baseline has incomplete source context roster")
    original_records={x["context_index"]:x for x in old_rows}
    out=measured/report["raw_context_file"]
    if file_hash(out)!=report["raw_context_sha256"]:
        raise AssertionError("causal runner raw outcomes changed")
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    if len(rows)!=48 or {r["context_index"] for r in rows}!=set(range(48)):
        raise AssertionError("new angular results do not have 48 complete source contexts")
    truth=[(world["truth_table_16"]>>k)&1 for k in range(16)]
    left_correct=right_correct=joint=positive=matched=total=0
    retained=[]
    context_signed=[]
    abs_lin=[]
    abs_nonlin=[]
    nonlinear_abs_remainder=[]
    for row in rows:
        ctx=row["context_index"]
        oldrecord=original_records[ctx]
        if (row["schema"]!=SCHEMA or row["world_id"]!=world["world_id"]
            or row["labels"]!=truth
            or row["source_context_bits"]!=[
                int(x) for x in original_contexts["contexts_ordered"][160+ctx]]
            or row["focal_mask_order"]!=list(range(16))
            or row["true_labels_used_for_control_selection"] is not False):
            raise AssertionError("causal evaluation changed teacher or context provenance")
        if set(row["site_direction_and_dose"])!=set(
                str(n.split(":")[0]) for n in source_manifest["source_graph_candidate"]["nodes"]):
            raise AssertionError("source selected GELU recurrent sites were changed")
        for key,old_key in (("clean_logits","clean_logits"),
                            ("hubs_logits","hubs_logits")):
            if len(row[key])!=16 or len(oldrecord[old_key])!=16:
                raise AssertionError("missing original baseline logits")
            for cur,oldlogits in zip(row[key],oldrecord[old_key]):
                if any(abs(float(a)-float(b))>3e-5+1e-5*abs(float(b))
                       for a,b in zip(cur,oldlogits)):
                    raise AssertionError("original source-frozen clean/hub outputs modified")
        linear=[0.]*16
        for visit,site in row["site_direction_and_dose"].items():
            v=int(visit)
            rank=source_manifest["source_ranking_per_visit"][v]
            order={code:i for i,code in enumerate(rank)}
            if len(site["routes"])!=16:
                raise AssertionError("missing per-focal control route")
            d=site["dose"]
            for i,route in enumerate(site["routes"]):
                active=route["all_active"]
                src=route["source_top_active"]
                ctrl=route["selected_disjoint_control"]
                n=len(src)
                contributions=route["active_output_contributions32"]
                C={f["id"]:f["vector"] for f in contributions}
                if (len(active)!=16 or len(C)!=16 or set(C)!=set(active)
                    or not 1<=n<=8 or len(ctrl)!=n or set(src)&set(ctrl)
                    or src!=sorted(active,key=lambda f:order[f])[:n]):
                    raise AssertionError("not source-highest active or disjoint donor")
                pool=sorted(set(active)-set(src))
                choices=list(combinations(pool,n))
                if len(choices)!=route["candidate_count"]:
                    raise AssertionError("exhaustive active subset roster changed")
                grad=route["hub_gradient32"]
                source32=vector_sum(src,C)
                control32=vector_sum(ctrl,C)
                for original,predicted in zip(route["source_projection32"],source32):
                    close(float(original),predicted,atol=4e-5)
                cos_src=cosine(source32,grad)
                cos_ctrl=cosine(control32,grad)
                gap=abs(cos_src-cos_ctrl)
                min_gap=min(abs(cosine(vector_sum(p,C),grad)-cos_src)
                            for p in choices)
                if gap>min_gap+5e-5:
                    raise AssertionError("control was selected with evaluation outcome information")
                for key,value in (("source_cosine",cos_src),
                                  ("control_cosine",cos_ctrl),
                                  ("absolute_cosine_gap",gap)):
                    close(float(route[key]),value,atol=5e-5)
                a=norm(source32)
                b=norm(control32)
                shared=min(a,b)
                valid_angle=(gap<=ANGLES and norm(grad)>1e-7
                             and a>1e-6 and b>1e-6)
                if route["within_angle_gate"]!=valid_angle:
                    raise AssertionError("angular eligibility gate manipulated")
                for key,value in (("full_left_dose",a),
                                  ("full_right_dose",b),
                                  ("shared_achievable_dose",shared),
                                  ("realized_left_dose",shared),
                                  ("realized_right_dose",shared)):
                    close(float(d[key][i]),value)
                expected_a=shared/a if a>1e-10 else 0.
                expected_b=shared/b if b>1e-10 else 0.
                close(float(d["left_gain"][i]),expected_a)
                close(float(d["right_gain"][i]),expected_b)
                if d["left_gain"][i]>1.00001 or d["right_gain"][i]>1.00001:
                    raise AssertionError("donor amplification under false physical match")
                close(float(d["retained_fraction_of_full_left_dose"][i]),expected_a)
                close(float(site["actual_matched_source32_norm"][i]),shared)
                close(float(site["actual_matched_control32_norm"][i]),shared)
                if d["matched_within_1pct"][i] is not True:
                    raise AssertionError("unequal physical dose")
                active_dose=shared>1e-6
                if d["shared_nonzero"][i]!=active_dose:
                    raise AssertionError("false positive shared physical dose")
                delta32=[(sa*expected_a-ct*expected_b)
                         for sa,ct in zip(source32,control32)]
                pred=dot(delta32,grad)
                close(float(site["hubs_first_order_margin_delta"][i]),pred,atol=3e-5)
                linear[i]+=pred
                total+=1
                matched+=1
                positive+=int(active_dose)
                joint+=int(valid_angle and active_dose)
                retained.append(expected_a)
        signedctx=0
        if (len(row["source_minus_control_correctness"])!=16
            or len(row["hubs_first_order_margin_delta"])!=16
            or len(row["exact_nonlinear_margin_delta"])!=16
            or len(row["nonlinear_remainder"])!=16
            or len(row["source_control_memory_distance_12visits"])!=16):
            raise AssertionError("incomplete 16-state margin and recurrence trace")
        for i in range(16):
            clean=row["clean_logits"][i]
            chosen=int(float(clean[1])>float(clean[0]))
            if chosen!=row["clean_chosen_class"][i]:
                raise AssertionError("gradient was oriented using evaluation label")
            src=row["source_logits"][i]
            ctrl=row["angle_control_logits"][i]
            if len(src)!=2 or len(ctrl)!=2:
                raise AssertionError("incomplete binary output")
            la=int(int(src[1]>src[0])==truth[i])
            lb=int(int(ctrl[1]>ctrl[0])==truth[i])
            delta=la-lb
            if row["source_minus_control_correctness"][i]!=delta:
                raise AssertionError("incorrect signed classification effect")
            left_correct+=la
            right_correct+=lb
            signedctx+=delta
            margin_source=float(src[chosen])-float(src[1-chosen])
            margin_control=float(ctrl[chosen])-float(ctrl[1-chosen])
            nonlin=margin_source-margin_control
            pred=linear[i]
            close(float(row["hubs_first_order_margin_delta"][i]),pred,atol=5e-5)
            close(float(row["exact_nonlinear_margin_delta"][i]),nonlin,atol=5e-5)
            close(float(row["nonlinear_remainder"][i]),nonlin-pred,atol=5e-5)
            distances=row["source_control_memory_distance_12visits"][i]
            if len(distances)!=12 or any(
                    not math.isfinite(float(x)) or float(x)<0 for x in distances):
                raise AssertionError("invalid 12-visit memory transport")
            abs_lin.append(abs(pred))
            abs_nonlin.append(abs(nonlin))
            nonlinear_abs_remainder.append(abs(nonlin-pred))
        context_signed.append(signedctx)
    observed=left_correct-right_correct
    if (report["source_correct_of_768"]!=left_correct
        or report["angle_control_correct_of_768"]!=right_correct
        or report["observed_correctness_gain_source_minus_angle_control"]!=observed):
        raise AssertionError("reported correctness contradicted raw logits")
    from statistics import median,mean
    physical=(matched/total>=.99 and positive/total>=.8 and median(retained)>=.25)
    directional=(joint/total>=.8 and physical)
    if (abs(report["angle_matched_positive_shared_fraction"]-joint/total)>1e-7
        or report["new_direction_control_feasible"]!=directional
        or abs(report["dose"]["median_fraction_full_left_dose_preserved"]-median(retained))>1e-7):
        raise AssertionError("new control physical/angle admission mismatched")
    original_clean=old["native_clean_scores"]["composed"]["accuracy"]>=.90
    expected_itt=observed if directional and original_clean else 0
    if (report["clean_composed_head_competent"]!=original_clean
        or report["development_ITT_gain_source_minus_angle_control"]!=expected_itt):
        raise AssertionError("new development ITT failure policy moved after outcomes")
    expected_context={
        "positive":sum(x>0 for x in context_signed),
        "negative":sum(x<0 for x in context_signed),
        "zero":sum(x==0 for x in context_signed)}
    if report["context_signs"]!=expected_context:
        raise AssertionError("context sign clustering differs")
    for field,pred in (("mean_absolute_hubs_tangent_displacement",mean(abs_lin)),
                       ("mean_absolute_exact_margin_displacement",mean(abs_nonlin)),
                       ("mean_absolute_nonlinear_remainder",mean(nonlinear_abs_remainder))):
        close(float(report[field]),pred,atol=1e-5)
    return {
        "world_id":world["world_id"],
        "old_primary_ITT_effect":report["root_original_NEC03H_v2_ITT"],
        "old_site_kind":old["source_site_kinds"]["composed"],
        "clean_model_eligible":original_clean,
        "angle_and_direct_dose_coverage":joint/total,
        "physical_eligible":physical,
        "new_angle_control_eligible":directional,
        "median_source_dose_retained":median(retained),
        "observed_source_minus_angle_control_correct":observed,
        "new_posthoc_ITT_correct":expected_itt,
        "context_signs":expected_context,
        "mean_linear_abs_margin_shift":mean(abs_lin),
        "mean_nonlinear_abs_margin_shift":mean(abs_nonlin),
        "mean_nonlinear_abs_remainder":mean(nonlinear_abs_remainder),
        "all_48_context_groups_reaudited":True,
        "all_possible_active_control_directions_reenumerated":True,
        "original_model_and_source_ranks_unchanged":True,
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0,
    }


def aggregate(roots:Path,panels:Path,source_calibrations:Path,measured:Path,out:Path,
              old_lock:Path,source_lock:Path,dev_lock:Path):
    results=[]
    for idx in INDICES:
        results.append(verify_root(
            roots/f"nec03h-v2-development-root-{idx}",
            panels/f"nec03ib-v2-root-{idx}",
            source_calibrations/f"nec03k-source-root-{idx}",
            measured/f"nec03k-development-root-{idx}",
            dev_lock,source_lock,old_lock))
    if len(results)!=8 or len({r["world_id"] for r in results})!=8:
        raise AssertionError("missing development world")
    final={
        "schema":"NEC-03K-B/independent-direction-matched-causal-worlds/v1",
        "eight_old_worlds":8,
        "causal_contexts_scored":8*48,
        "correlated_focal_cases_scored":8*48*16,
        "worlds_directional_control_feasible":sum(r["new_angle_control_eligible"] for r in results),
        "worlds_primary_posthoc_ITT_eligible":sum(
            r["new_angle_control_eligible"] and r["clean_model_eligible"] for r in results),
        "root_mean_posthoc_ITT_effect":sum(
            r["new_posthoc_ITT_correct"] for r in results)/(8*768),
        "context_samples_are_not_independent_roots":True,
        "all_historical_model_and_original_ITT_unchanged":True,
        "confirmation_worlds_scored":0,
        "reserved_contexts_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "worlds":results,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03k_direction_matched_dev_audit.json").write_text(
        json.dumps(final,indent=2,sort_keys=True)+"\n")
    print("NEC03K_EIGHT_ROOT_DIRECTION_MATCHED_DEV_AUDIT_COMPLETE")
    return final


def main(argv=None):
    p=argparse.ArgumentParser()
    for k in ("roots","panels","source-calibrations","measured","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--old-lock",type=Path,default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    p.add_argument("--source-lock",type=Path,default=Path(__file__).with_name("NEC03K_SOURCE_LOCK.json"))
    p.add_argument("--dev-lock",type=Path,default=Path(__file__).with_name("NEC03K_DEV_LOCK.json"))
    a=p.parse_args(argv)
    aggregate(a.roots,a.panels,a.source_calibrations,a.measured,a.out,
              a.old_lock,a.source_lock,a.dev_lock)


if __name__=="__main__":
    main()
