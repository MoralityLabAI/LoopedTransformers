"""NEC-03K-B: posthoc causal replay with SOURCE-CALIBRATED angle shams.

The exhaustive 32-D angle selector and all feasibility thresholds were
frozen and source-calibrated in NEC-03K before this code scores any of
the ALREADY EXPOSED dev-008..015 development-evaluation contexts.

This program reuses exactly the old native/SAE checkpoints and visits.
It scores source TopK versus disjoint equal-cardinality control matched
both in direct output norm and hubs-only gradient cosine. No training,
unused dev-world enrollment, reserved context or confirmation readout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03k_source_geometry import (
    exhaustive_direction_match,verify_k_lock,ANGLE_LIMIT,canonical,
)
from mechinterp.neuron_communities.nec03ib_active_shams import (
    load_frozen_source,check_logits,
)
from mechinterp.neuron_communities.nec03j_direction_transport import (
    traced_forward,oriented_margin,hash_file,panel_source_records,
)
from mechinterp.neuron_communities.nec03g_worlds import target_bits,registry
from mechinterp.neuron_communities.nec03e_pairs import (
    attenuate_pair,pair_dose_stats,
)
from mechinterp.neuron_communities.matched_dose import by_visit
from mechinterp.neuron_communities.source_sae import clean_activations
from research.pef07.pef07 import as_tensor,model_digest

SCHEMA="NEC-03K-B/source-angle-matched-control-recurrent-causal-dev/v1"
DEV_CONTEXTS=48
MIN_CLEAN=.90


def verify_dev_lock():
    path=Path(__file__).with_name("NEC03K_DEV_LOCK.json")
    obj=json.loads(path.read_text())
    source_lock=Path(__file__).with_name("NEC03K_SOURCE_LOCK.json")
    contents=source_lock.read_bytes()
    original_blob=__import__("hashlib").sha1(
        b"blob "+str(len(contents)).encode()+b"\x00"+contents).hexdigest()
    if (obj.get("schema")!="NEC-03K-B/dev-causal-locked-v1"
        or obj.get("source_lock_git_blob_sha1")!=original_blob
        or obj.get("source_calibration_run")!=44892904616
        or obj.get("calibrated_source_world_indices")!=list(range(8,16))
        or obj.get("heldout_development_contexts_per_world")!=DEV_CONTEXTS
        or obj.get("original_confirmation_worlds_scored")!=0
        or obj.get("reserved_context_groups_scored")!=0
        or obj.get("clean_head_admission_floor")!=MIN_CLEAN
        or obj.get("original_registry_sha256")!=registry()["registry_sha256"]
        or obj.get("confirmation_spec_commitment")!=
            registry()["confirmation_full_spec_sha256"]):
        raise AssertionError("NEC03K causal protocol or original model registry not frozen")
    for name,expected in obj["git_blob_sha1"].items():
        content=(Path(__file__).parent/name).read_bytes()
        blob=__import__("hashlib").sha1(
            b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()
        if blob!=expected:
            raise AssertionError("NEC03K causal program changed after lock: "+name)
    obj["lock_sha256"]=hash_file(path)
    return obj


def run(original_root:Path, old_controls:Path, source_calibration:Path,
        out:Path, old_v2_lock:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dev_lock=verify_dev_lock()
    source_lock=verify_k_lock()
    old=load_frozen_source(original_root,old_v2_lock)
    world=old["world"]
    w=world["world_id"]
    source_cal=json.loads((source_calibration/"nec03k_source_calibration.json").read_text())
    if (source_cal["world_sha256"]!=world["world_sha256"]
        or source_cal["original_model_sha256"]!=old["model_digest"]
        or source_cal["NEC03K_source_lock_sha256"]!=source_lock["lock_sha256"]
        or not source_cal["source_null_feasible"]
        or source_cal["heldout_development_contexts_scored"]!=0
        or source_cal["confirmation_worlds_scored"]!=0):
        raise AssertionError("causal runner lacks a pre-existing qualified source-only calibration")
    panel,original_rows=panel_source_records(old_controls)
    if (panel["world_sha256"]!=world["world_sha256"]
        or panel["native_model_sha256"]!=old["model_digest"]):
        raise AssertionError("historical old active-control source mismatch")
    m=old["model"]
    catalog=old["catalog"]
    candidate=old["candidate"]
    source_sites=by_visit(candidate["nodes"])
    hubs=by_visit(old["frozen_source"]["graph_hubs"])
    ranks=old["frozen_source"]["source_ranking_per_visit"]
    sae=old["sae"]
    mean,scale=old["mean"],old["scale"]
    down=m.block.mlp_down.weight.detach()
    out.mkdir(parents=True,exist_ok=True)
    raw=[]
    all_doses=[]
    matched_angle=0
    case_visits=0
    exact_left=exact_right=0
    context_signed=[]
    mean_linear=[]
    mean_nonlin=[]
    for ctx in range(DEV_CONTEXTS):
        group=catalog["development"][ctx*16:(ctx+1)*16]
        x=as_tensor(group)
        y=target_bits(x,world)
        clean=traced_forward(m,x,mode="clean")
        chosen=clean["logits"].argmax(1)
        acts=clean_activations(m,x)
        base_donors={}
        gates={}
        for visit in source_sites:
            base_donors[visit]={"hubs":torch.zeros(16,64)}
            mask=torch.zeros(64)
            if visit in hubs:
                mask[list(sorted(hubs[visit]))]=1.
            gates[visit]=mask
        hubs_trace=traced_forward(m,x,mode="hubs",
                                  orientation=chosen,
                                  donors=base_donors,gates=gates,derivative=True)
        old_rec=original_rows[0][ctx]
        check_logits(clean["logits"],old_rec["clean_logits"])
        check_logits(hubs_trace["logits"],old_rec["hubs_logits"])
        donors={}
        dose_context={}
        site_data={}
        lin=torch.zeros(16)
        for visit,units in sorted(source_sites.items()):
            source,control,proof=exhaustive_direction_match(
                sae,mean,scale,acts[:,visit,:],ranks[visit],
                hubs_trace["grad_down"][visit],down,count=len(units))
            source,control,dose=attenuate_pair(source,control,down)
            dose_context[visit]=dose
            matched_angle+=sum(
                bool(info["within_angle_gate"] and dose["matched_within_1pct"][i]
                     and dose["shared_nonzero"][i])
                for i,info in enumerate(proof))
            case_visits+=len(proof)
            donors[visit]={"hubs":torch.zeros_like(source),
                           "native":source,"sham":control}
            delta32=(source-control)@down.T
            predicted=(hubs_trace["grad_down"][visit]*delta32).sum(1)
            lin+=predicted
            site_data[str(visit)]={
                "routes":proof,"dose":dose,
                "hubs_first_order_margin_delta":predicted.double().tolist(),
                "actual_matched_source32_norm":(source@down.T).norm(dim=-1).double().tolist(),
                "actual_matched_control32_norm":(control@down.T).norm(dim=-1).double().tolist()
            }
        all_doses.append(dose_context)
        with torch.no_grad():
            trace_source=traced_forward(
                m,x,mode="native",orientation=chosen,
                donors=donors,gates=gates)
            trace_ctrl=traced_forward(
                m,x,mode="sham",orientation=chosen,
                donors=donors,gates=gates)
        correct_a=(trace_source["logits"].argmax(1)==y)
        correct_b=(trace_ctrl["logits"].argmax(1)==y)
        exact_left+=int(correct_a.sum())
        exact_right+=int(correct_b.sum())
        context_signed.append(int(correct_a.sum()-correct_b.sum()))
        nonlinear=(trace_source["margin"]-trace_ctrl["margin"])
        remainder=nonlinear-lin
        mean_linear.extend(lin.tolist())
        mean_nonlin.extend(nonlinear.tolist())
        memory_drift=(trace_source["memory"]-trace_ctrl["memory"]).norm(dim=-1)
        raw.append({
            "schema":SCHEMA,"world_id":w,
            "context_index":ctx,
            "source_context_bits":[int(v) for v in
                old["source"]["contexts_ordered"][160+ctx]],
            "focal_mask_order":list(range(16)),
            "labels":y.tolist(),
            "clean_chosen_class":chosen.tolist(),
            "clean_logits":clean["logits"].double().tolist(),
            "hubs_logits":hubs_trace["logits"].double().tolist(),
            "source_logits":trace_source["logits"].double().tolist(),
            "angle_control_logits":trace_ctrl["logits"].double().tolist(),
            "source_minus_control_correctness":(
                correct_a.int()-correct_b.int()).tolist(),
            "hubs_first_order_margin_delta":lin.double().tolist(),
            "exact_nonlinear_margin_delta":nonlinear.double().tolist(),
            "nonlinear_remainder":remainder.double().tolist(),
            "source_control_memory_distance_12visits":
                memory_drift.double().tolist(),
            "site_direction_and_dose":site_data,
            "true_labels_used_for_control_selection":False,
        })
    physical=pair_dose_stats(all_doses)
    coverage=matched_angle/case_visits
    eligible=(coverage>=source_lock["min_angle_matched_fraction"]
              and physical["candidate_physically_comparable_and_signal_sufficient"])
    composed_clean=(old["original_root"]["native_clean_scores"]["composed"]["accuracy"]>=MIN_CLEAN)
    observed=exact_left-exact_right
    itt=observed if eligible and composed_clean else 0
    failures=[]
    if not composed_clean:
        failures.append("composed_clean_head_under_90pct")
    if not eligible:
        failures.append("angle_or_dose_matched_null_source_infeasible")
    rawblob=b"".join(canonical(item) for item in raw)
    (out/"nec03k_angle_matched_causal.jsonl").write_bytes(rawblob)
    report={
        "schema":SCHEMA,"world_id":w,"world_index":world["world_index"],
        "world_sha256":world["world_sha256"],
        "native_model_sha256":old["model_digest"],
        "source_sae_sha256":old["frozen_source"]["source_sae"]["source_sae_sha256"],
        "source_rank_sha256":hash_file(original_root/"composed.source_freeze.json"),
        "source_calibration_sha256":hash_file(
            source_calibration/"nec03k_source_calibration.json"),
        "source_calibration_original_run":44892904616,
        "old_hash_active_control_run":44885003105,
        "old_hash_control_panel_sha256":hash_file(
            old_controls/"source_blind_control_panel.json"),
        "original_NEC03H_v2_lock_sha256":hash_file(old_v2_lock),
        "direction_matching_source_lock_sha256":source_lock["lock_sha256"],
        "causal_dev_protocol_sha256":dev_lock["lock_sha256"],
        "trained_native_or_SAE_weights_modified":False,
        "retrospective_development_contexts":DEV_CONTEXTS,
        "source_training_16_context_calibration_passed":True,
        "true_labels_used_for_control_selection":False,
        "angle_matched_positive_shared_fraction":coverage,
        "dose":physical,
        "clean_composed_head_competent":composed_clean,
        "new_direction_control_feasible":eligible,
        "root_original_NEC03H_v2_ITT":
            old["original_root"]["primary_effect"]["root_primary_effect"],
        "source_correct_of_768":exact_left,
        "angle_control_correct_of_768":exact_right,
        "observed_correctness_gain_source_minus_angle_control":observed,
        "development_ITT_gain_source_minus_angle_control":itt,
        "ITT_admission_failure_reasons":failures,
        "context_signs":{
            "positive":sum(v>0 for v in context_signed),
            "negative":sum(v<0 for v in context_signed),
            "zero":sum(v==0 for v in context_signed)},
        "mean_absolute_hubs_tangent_displacement":sum(abs(v) for v in mean_linear)/len(mean_linear),
        "mean_absolute_exact_margin_displacement":sum(abs(v) for v in mean_nonlin)/len(mean_nonlin),
        "mean_absolute_nonlinear_remainder":sum(abs(a-b) for a,b in zip(mean_linear,mean_nonlin))/len(mean_linear),
        "independent_statistical_unit":"one original Boolean teacher/world root; contexts and masks are repeated",
        "raw_context_file":"nec03k_angle_matched_causal.jsonl",
        "raw_context_sha256":sha256(rawblob).hexdigest(),
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0,
        "source_root_specific_result_is_posthoc":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec03k_development_report.json").write_bytes(canonical(report))
    if model_digest(m)!=old["model_digest"]:
        raise AssertionError("old native frozen model mutated by angle-matched intervention")
    print("NEC03K_DEVELOPMENT_DIRECTION_CAUSAL_REPLAY_COMPLETE")
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--old-controls",type=Path,required=True)
    p.add_argument("--source-calibration",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--old-lock",type=Path,
                   default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    args=p.parse_args(argv)
    run(args.root,args.old_controls,args.source_calibration,args.out,args.old_lock)


if __name__=="__main__":
    main()
