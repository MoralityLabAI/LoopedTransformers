"""NEC-03J independent stdlib audit: 64D/32D causal tangent vs 12-visit drift.

The original heldout source, model, SAE and hash-ranked donor outputs are
verified before trusting any new gradient or transport field. Full 16-case
quartets are nested inside 48 contexts and eight distinct teacher worlds.
No optimizer, PyTorch import, confirmation or reserved context access.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from mechinterp.neuron_communities.nec03g_worlds import (
    context_catalog, registry, worlds,
)

SCHEMA="NEC-03J/GELU-Jacobian-and-tied-recurrent-transport/v1"
VARIANTS=tuple(range(4))
INDICES=tuple(range(8,16))
L=12


def read(path):
    value=json.loads(path.read_text())
    if not isinstance(value,dict):
        raise AssertionError("expected JSON source manifest")
    return value


def digest_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def expected_class(logits):
    if len(logits)!=2 or not all(math.isfinite(float(x)) for x in logits):
        raise AssertionError("nonfinite binary margin")
    return int(logits[1]>logits[0])


def exact_margin(logits,chosen):
    if chosen not in (0,1) or len(logits)!=2:
        raise AssertionError("wrong predicted-class gradient target")
    return float(logits[chosen])-float(logits[1-chosen])


def within(a,b,tolerance=2e-5):
    return abs(float(a)-float(b)) <= tolerance+1e-5*abs(float(b))


def finite_vector(value,n):
    if len(value)!=n or any(not math.isfinite(float(x)) for x in value):
        raise AssertionError("nonfinite or wrong-length directional measurement")


def check_record(rec,old,world,variant,source):
    """Verify ALL 16 case-level nonlinear and gradient-angle claims."""
    ctx=rec["context_index"]
    if (rec["schema"]!=SCHEMA or rec["world_id"]!=world["world_id"]
        or rec["variant"]!=variant
        or ctx not in range(48)
        or rec["focal_mask_order"]!=list(range(16))
        or rec["source_context_bits"]!=[
            int(x) for x in source["contexts_ordered"][160+ctx]]):
        raise AssertionError("wrong world/context/variant source identity")
    labels=[(world["truth_table_16"]>>j)&1 for j in range(16)]
    if rec["labels"]!=labels or old["labels"]!=labels:
        raise AssertionError("registered teacher labels changed")
    for new_key,old_key in (("clean_logits","clean_logits"),
                            ("hubs_logits","hubs_logits"),
                            ("source_logits","source_logits"),
                            ("control_logits","control_logits")):
        rows=rec[new_key]
        baseline=old[old_key]
        if len(rows)!=16 or len(baseline)!=16:
            raise AssertionError("incorrect focal history count")
        for got,original in zip(rows,baseline):
            if any(not within(a,b) for a,b in zip(got,original)):
                raise AssertionError("original native/SAE/control nonlinear outputs modified")
    tangent=rec["direction_and_transport"]
    names=("clean_linear_margin_delta","hubs_linear_margin_delta",
           "nonlinear_exact_margin_delta","nonlinear_minus_hubs_linear_error",
           "gradient_direction_sign_agreement")
    for name in names:
        if len(tangent[name])!=16:
            raise AssertionError("incomplete signed margin trace")
    for name in names[:-1]:
        finite_vector(tangent[name],16)
    for k in ("source_vs_control_memory_drift_by_visit",
              "source_vs_hubs_memory_drift_by_visit",
              "control_vs_hubs_memory_drift_by_visit"):
        rows=tangent[k]
        if len(rows)!=16:
            raise AssertionError("incomplete memory-state transport")
        for history in rows:
            finite_vector(history,L)
            if any(x<0 for x in history):
                raise AssertionError("negative physical memory distance")
    sites=tangent["local_jacobian"]
    if not sites:
        raise AssertionError("missing selected source memory-token recurrent sites")
    for visit,values in sites.items():
        if int(visit) not in range(12):
            raise AssertionError("outside recurrence horizon")
        for key in ("clean_directional_margin_derivative",
                    "hubs_directional_margin_derivative",
                    "direct_32d_delta_norm",
                    "hubs_margin_gradient_32d_norm",
                    "hubs_direction_cosine",
                    "native_dose","control_dose","achieved_common_dose",
                    "left_gain","right_gain"):
            finite_vector(values[key],16)
        if values["max_local_64d_32d_jacobian_identity_error"]>0.001001:
            raise AssertionError("64D GELU and 32D MLP-down Jacobian identity failed")
        for i in range(16):
            if (abs(float(values["hubs_direction_cosine"][i]))>1.00001
                or values["direct_32d_delta_norm"][i]<0
                or values["hubs_margin_gradient_32d_norm"][i]<0
                or values["native_dose"][i]<0 or values["control_dose"][i]<0
                or values["achieved_common_dose"][i]<0
                or not 0<=values["left_gain"][i]<=1.000001
                or not 0<=values["right_gain"][i]<=1.000001):
                raise AssertionError("physically impossible recurrent directional geometry")
        old_site=old["local_direct_doses"].get(visit)
        if old_site is None:
            raise AssertionError("NEC03J intervention site differs from original hash panel")
        for field,old_field in (("native_dose","full_left_dose"),
                                ("control_dose","full_right_dose"),
                                ("achieved_common_dose","shared_achievable_dose"),
                                ("left_gain","left_gain"),
                                ("right_gain","right_gain")):
            for a,b in zip(values[field],old_site[old_field]):
                if not within(a,b,3e-6):
                    raise AssertionError("historical pairwise dose receipts changed")
    if set(sites)!=set(old["local_direct_doses"]):
        raise AssertionError("source intervention visit family modified")
    max_err=tangent["max_native_gelu_to_mlpdown_jacobian_discrepancy"]
    if max_err<0 or max_err>0.001001:
        raise AssertionError("GELU/post-MLP direction equivalence failed")
    result={"source_correct":0,"control_correct":0,
            "source_only":0,"control_only":0,
            "nonzero_actual":0,"clean_linear_same_sign":0,
            "hubs_linear_same_sign":0,"hubs_linear_nonzero":0,
            "linear_absolute_error":0.,"memory_final_drift":0.,
            "memory_peak_drift":0.,"cosine_sum":0.,
            "cosine_count":0}
    for i in range(16):
        clean=rec["clean_logits"][i]
        chosen=rec["clean_chosen_class"][i]
        if chosen!=expected_class(clean):
            raise AssertionError("gradient orientation was selected from teacher label")
        actual=exact_margin(rec["source_logits"][i],chosen)-exact_margin(
            rec["control_logits"][i],chosen)
        recorded=tangent["nonlinear_exact_margin_delta"][i]
        if not within(actual,recorded,2e-5):
            raise AssertionError("nonlinear margin direction not recomputed from original outputs")
        pred_hubs=sum(values["hubs_directional_margin_derivative"][i]
                      for values in sites.values())
        pred_clean=sum(values["clean_directional_margin_derivative"][i]
                       for values in sites.values())
        if (not within(pred_hubs,tangent["hubs_linear_margin_delta"][i],2e-5)
            or not within(pred_clean,tangent["clean_linear_margin_delta"][i],2e-5)):
            raise AssertionError("source-local Jacobian contributions not equal registered sum")
        remainder=actual-pred_hubs
        if not within(remainder,tangent["nonlinear_minus_hubs_linear_error"][i],2e-5):
            raise AssertionError("nonlinear remainder differs from local tangent")
        expected_sign=(int(actual*pred_hubs>0)
                       if abs(actual)>1e-7 and abs(pred_hubs)>1e-7 else None)
        if tangent["gradient_direction_sign_agreement"][i]!=expected_sign:
            raise AssertionError("mismatched Jacobian sign descriptor")
        y=labels[i]
        correct_src=int(expected_class(rec["source_logits"][i])==y)
        correct_alt=int(expected_class(rec["control_logits"][i])==y)
        if rec["source_vs_control_correctness_delta"][i]!=correct_src-correct_alt:
            raise AssertionError("incorrect scored causal donor correctness")
        result["source_correct"]+=correct_src
        result["control_correct"]+=correct_alt
        result["source_only"]+=int(correct_src>correct_alt)
        result["control_only"]+=int(correct_src<correct_alt)
        result["linear_absolute_error"]+=abs(remainder)
        result["memory_final_drift"]+=tangent["source_vs_control_memory_drift_by_visit"][i][-1]
        result["memory_peak_drift"]+=max(tangent["source_vs_control_memory_drift_by_visit"][i])
        if abs(actual)>1e-7:
            result["nonzero_actual"]+=1
            result["clean_linear_same_sign"]+=int(actual*pred_clean>0)
            if abs(pred_hubs)>1e-7:
                result["hubs_linear_nonzero"]+=1
                result["hubs_linear_same_sign"]+=int(actual*pred_hubs>0)
        for s in sites.values():
            result["cosine_sum"]+=s["hubs_direction_cosine"][i]
            result["cosine_count"]+=1
    return result


def audit_one(original_root: Path, original_panel: Path, generated: Path,lock: Path):
    world=read(original_root/"world_spec.json")
    i=world["world_index"]
    if i not in INDICES or world!=worlds()[i] or world["cohort"]!="development":
        raise AssertionError("fresh/confirmation world was substituted")
    original_report=read(original_root/"root_summary.json")
    report=read(generated/"direction_transport_report.json")
    panels=read(original_panel/"source_blind_control_panel.json")
    source=read(original_root/"context_source_manifest.json")
    _,expected=context_catalog(world)
    if source!=expected:
        raise AssertionError("development vs reserved source membership altered")
    direction_lock=Path(__file__).with_name("NEC03J_LOCK.json")
    if report.get("NEC03J_direction_lock_sha256")!=digest_file(direction_lock):
        raise AssertionError("original frozen NEC03J derivative protocol changed")
    if (report["schema"]!=SCHEMA or report["world_sha256"]!=world["world_sha256"]
        or report["world_id"]!=world["world_id"]
        or report["original_v2_lock_sha256"]!=digest_file(lock)
        or report["source_model_weights_unchanged"] is not True
        or report["confirmation_worlds_scored"]!=0
        or report["reserved_context_groups_scored"]!=0
        or report["original_nec03h_run_id"]!=44878411694
        or report["original_nec03ib_run_id"]!=44885003105
        or report["original_source_control_panel_sha256"]!=digest_file(
             original_panel/"source_blind_control_panel.json")
        or report["original_ITT_correctness"]!=
             original_report["primary_effect"]["root_primary_effect"]):
        raise AssertionError("new trajectory evidence source/archival claims invalid")
    source_freeze=read(original_root/"composed.source_freeze.json")
    sae_sha=source_freeze["source_sae"]["source_sae_sha256"]
    if (report["native_model_sha256"]!=panels["native_model_sha256"]
        or sae_sha!=panels["source_sae_checkpoint_sha256"]):
        raise AssertionError("model/SAE source matches a different checkpoint")
    if report["source_steering_candidate_id"]!=read(original_root/"composed.source_freeze.json")["source_graph_candidate"]["id"]:
        raise AssertionError("local source candidate changed from original")
    finite=report["source_training_only_fd_preflight"]
    if (not finite["original_16_source_histories_only"]
        or finite["source_split"]!="train[:16] only"
        or finite["max_absolute_fd_error"]>0.020001
        or finite["epsilon"]!=0.003):
        raise AssertionError("source-only finite-difference gradient verifier failed")
    if (report["raw_variant_context_rows"]!=4*48
        or report["raw_focal_predictions"]!=4*48*16
        or report["evidence_class"]!="posthoc development, no prospective causal confirmation"
        or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("incorrect experimental class or completeness")
    rawpath=generated/report["raw_file"]
    if digest_file(rawpath)!=report["raw_sha256"]:
        raise AssertionError("direction/memory raw evidence tampered")
    rows=[json.loads(l) for l in rawpath.read_text().splitlines()]
    if len(rows)!=4*48:
        raise AssertionError("incomplete original repeated context family")
    histories={}
    for v in VARIANTS:
        historic=panels["comparisons"][v]
        p=original_panel/historic["raw_filename"]
        if digest_file(p)!=historic["raw_context_sha256"]:
            raise AssertionError("historical hash-control raw evidence changed")
        lines=[json.loads(line) for line in p.read_text().splitlines()]
        if len(lines)!=48 or {x["context_index"] for x in lines}!=set(range(48)):
            raise AssertionError("original 48 repeated contexts missing")
        histories[v]={x["context_index"]:x for x in lines}
    seen=set()
    metrics=defaultdict(lambda:Counter())
    totals=defaultdict(lambda:Counter())
    for item in rows:
        v=item["variant"]
        c=item["context_index"]
        if v not in VARIANTS or c not in range(48) or (v,c) in seen:
            raise AssertionError("duplicated or unregistered variant/context")
        seen.add((v,c))
        obs=check_record(item,histories[v][c],world,v,source)
        for k,n in obs.items():
            totals[v][k]+=n
    if seen!={(v,c) for v in VARIANTS for c in range(48)}:
        raise AssertionError("source world/variant pair count incorrect")
    out=[]
    for v in VARIANTS:
        data=totals[v]
        old=panels["comparisons"][v]
        delta=data["source_correct"]-data["control_correct"]
        if delta!=old["net_source_ranked_correct_gain"]:
            raise AssertionError("new Jacobian analysis changes original signed donor outcome")
        count=48*16
        out.append({
            "variant":v,
            "signed_correct_gain_original":delta,
            "source_only_correct":data["source_only"],
            "control_only_correct":data["control_only"],
            "final_memory_drift_mean":data["memory_final_drift"]/count,
            "maximum_visit_memory_drift_mean":data["memory_peak_drift"]/count,
            "gradient_nonzero_actual_count":data["nonzero_actual"],
            "hubs_jacobian_sign_agreement_count":data["hubs_linear_same_sign"],
            "hubs_jacobian_nonzero_count":data["hubs_linear_nonzero"],
            "clean_jacobian_sign_agreement_count":data["clean_linear_same_sign"],
            "mean_absolute_nonlinear_linearization_remainder":
                data["linear_absolute_error"]/count,
            "mean_direct_32d_hubs_jacobian_direction_cosine":
                data["cosine_sum"]/data["cosine_count"],
        })
    return {
        "world_id":world["world_id"],"world_index":i,
        "source_site_kind":original_report["source_site_kinds"]["composed"],
        "world_sha256":world["world_sha256"],
        "raw_direction_transport_sha256":report["raw_sha256"],
        "original_v2_root_summary_sha256":digest_file(original_root/"root_summary.json"),
        "original_active_sham_panel_sha256":report["original_source_control_panel_sha256"],
        "source_fd_verification":finite,
        "exact_nonlin_logit_and_physical_dose_replay":True,
        "within_world_correlated_768_histories":True,
        "variants":out,
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0,
    }


def aggregate(roots:Path,panels:Path,traces:Path,out:Path,lock:Path):
    rows=[]
    for i in INDICES:
        r=roots/f"nec03h-v2-development-root-{i}"
        p=panels/f"nec03ib-v2-root-{i}"
        t=traces/f"nec03j-transport-root-{i}"
        rows.append(audit_one(r,p,t,lock))
    if len(rows)!=8 or {x["world_index"] for x in rows}!=set(INDICES):
        raise AssertionError("original eight distinct development worlds missing")
    result={
        "schema":"NEC-03J/independent-eight-world-jacobian-transport-audit/v1",
        "independent_world_count":8,
        "reused_original_native_and_sae_checkpoints":True,
        "original_nec03h_ITT_and_matched_null_gates_unchanged":True,
        "reused_world_cohort":list(INDICES),
        "total_variant_contexts":8*4*48,
        "total_focal_case_records":8*4*48*16,
        "statistical_unit":"one independently seeded Boolean teacher/focal quartet world",
        "worlds":rows,
        "development_only":True,
        "no_independent_pvalues":True,
        "no_new_training_steps":True,
        "sham_tail_95_95_calibrated":False,
        "confirmation_worlds_scored":0,
        "reserved_contexts_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY"}
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03j_independent_jacobian_transport.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NEC03J_INDEPENDENT_FULL_SOURCE_AUDIT_WRITTEN")
    return result


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--roots",type=Path,required=True)
    ap.add_argument("--panels",type=Path,required=True)
    ap.add_argument("--traces",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--lock",type=Path,default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    a=ap.parse_args(argv)
    aggregate(a.roots,a.panels,a.traces,a.out,a.lock)


if __name__=="__main__":
    main()
