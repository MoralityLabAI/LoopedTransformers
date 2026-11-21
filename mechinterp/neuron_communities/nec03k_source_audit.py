"""NEC-03K independent pure-stdlib source-only geometry audit.

Reconstruct every candidate's 32D MLP-down output direction from logged
active-feature projections. Enumerate all C(16-n,n) disjoint controls, and
prove that the selected active sham minimizes source-angle gap without
using teacher labels or outcome-based screening. Recalculate physical
norms, dose matching, site/sample availability and original file hashes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

from mechinterp.neuron_communities.nec03g_worlds import (
    registry, worlds, context_catalog,
)

SCHEMA="NEC-03K/source-only-direction-matched-active-sham/v1"
ANGLE_LIMIT=.25
ROOTS=tuple(range(8,16))


def file_hash(p:Path):
    h=sha256()
    with p.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def read(path:Path):
    return json.loads(path.read_text())


def dot(a,b):
    if len(a)!=32 or len(b)!=32:
        raise AssertionError("direction projected in wrong physical output coordinates")
    return sum(float(x)*float(y) for x,y in zip(a,b))


def norm(a):
    return math.sqrt(max(dot(a,a),0.))


def cosine(a,b):
    return dot(a,b)/max(norm(a)*norm(b),1e-10)


def vector_sum(ids,contrib):
    out=[0.]*32
    for key in ids:
        for j,value in enumerate(contrib[key]):
            out[j]+=value
    return out


def close(a,b,atol=1e-4):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>atol+abs(b)*1e-5:
        raise AssertionError("frozen geometric measurement changed")


def audit_world(source_root:Path,generated:Path,original_v2_lock:Path,cal_lock:Path):
    w=read(source_root/"world_spec.json")
    index=w["world_index"]
    if index not in ROOTS or w!=worlds()[index]:
        raise AssertionError("unseen/confirmation world submitted to source calibration")
    catalog,source=context_catalog(w)
    if read(source_root/"context_source_manifest.json")!=source:
        raise AssertionError("world source assignment differs")
    report=read(generated/"nec03k_source_calibration.json")
    original=read(source_root/"composed.source_freeze.json")
    oldroot=read(source_root/"root_summary.json")
    if (report["schema"]!=SCHEMA
        or report["world_sha256"]!=w["world_sha256"]
        or report["original_v2_lock_sha256"]!=file_hash(original_v2_lock)
        or report["NEC03K_source_lock_sha256"]!=file_hash(cal_lock)
        or report["original_model_sha256"]!=oldroot["world_sha256"] and
           report["original_model_sha256"]!=original["model_sha256"]):
        raise AssertionError("new geometric pilot uses a different frozen source")
    if (report["original_native_checkpoint_sha256"]!=
            read(source_root/"composed.report.json")["train"]["checkpoint_sha256"]
        or report["original_topk16_checkpoint_sha256"]!=
            original["source_sae"]["source_sae_sha256"]
        or report["original_source_rank_freeze_sha256"]!=
            file_hash(source_root/"composed.source_freeze.json")
        or report["heldout_development_contexts_scored"]!=0
        or report["reserved_contexts_scored"]!=0
        or report["confirmation_worlds_scored"]!=0):
        raise AssertionError("source identity, model hash or cohort leak")
    path=generated/report["raw_context_file"]
    if file_hash(path)!=report["raw_context_sha256"]:
        raise AssertionError("source geometric evidence raw hash changed")
    items=[json.loads(line) for line in path.read_text().splitlines()]
    if len(items)!=16 or {x["source_context_index"] for x in items}!=set(range(16)):
        raise AssertionError("not exactly sixteen source calibration context quartets")
    total=matched=positive=angle_matched=joint=0
    retained=[]
    sites=set()
    choose_counts=[]
    for item in items:
        ctx=item["source_context_index"]
        if (item["schema"]!=SCHEMA or item["world_id"]!=w["world_id"]
            or item["source_context_bits"]!=[int(x) for x in source["contexts_ordered"][ctx]]
            or item["input_focal_masks"]!=list(range(16))
            or item["source_training_context_only"] is not True
            or item["output_labels_consulted_for_selection"] is not False):
            raise AssertionError("source geometry read unregistered context or task outcomes")
        if len(item["clean_chosen_class"])!=16 or len(item["clean_logits"])!=16:
            raise AssertionError("wrong source quartet clean logits")
        for i,logits in enumerate(item["clean_logits"]):
            if item["clean_chosen_class"][i] != int(float(logits[1])>float(logits[0])):
                raise AssertionError("source gradient conditioned on label instead of clean prediction")
        for visit,site in item["sites"].items():
            sites.add(int(visit))
            visit=int(visit)
            if visit not in range(12) or len(site["routes"])!=16:
                raise AssertionError("invalid recurrent visit/site row")
            weights=original["source_ranking_per_visit"][visit]
            order={f:i for i,f in enumerate(weights)}
            dose=site["dose"]
            for i,route in enumerate(site["routes"]):
                contrib={}
                entries=route["active_output_contributions32"]
                for record in entries:
                    f=record["id"]
                    if f in contrib or len(record["vector"])!=32:
                        raise AssertionError("duplicate/invalid active output feature projection")
                    contrib[f]=record["vector"]
                active=route["all_active"]
                source_ids=route["source_top_active"]
                ctrl_ids=route["selected_disjoint_control"]
                n=len(source_ids)
                if (len(active)!=16 or len(contrib)!=16 or set(active)!=set(contrib)
                    or not 1<=n<=8 or len(ctrl_ids)!=n
                    or set(source_ids)&set(ctrl_ids)):
                    raise AssertionError("disjoint active feature capacity mismatch")
                if source_ids!=sorted(active,key=lambda f:order[f])[:n]:
                    raise AssertionError("source donor was not source-ranked active TopK")
                if not set(ctrl_ids).issubset(set(active)):
                    raise AssertionError("matched control contains inactive codes")
                pool=sorted(set(active)-set(source_ids))
                choices=list(combinations(pool,n))
                if len(choices)!=route["candidate_count"]:
                    raise AssertionError("hidden candidate family or sample-dependent selection")
                grad=route["hub_gradient32"]
                source_vec=vector_sum(source_ids,contrib)
                ctrl_vec=vector_sum(ctrl_ids,contrib)
                for got,replayed in zip(route["source_projection32"],source_vec):
                    close(float(got),replayed)
                source_cos=cosine(source_vec,grad)
                control_cos=cosine(ctrl_vec,grad)
                gap=abs(source_cos-control_cos)
                for got,predicted in ((route["source_cosine"],source_cos),
                                      (route["control_cosine"],control_cos),
                                      (route["absolute_cosine_gap"],gap),
                                      (route["gradient_norm"],norm(grad)),
                                      (route["physical_source_norm"],norm(source_vec)),
                                      (route["physical_control_norm"],norm(ctrl_vec))):
                    close(float(got),predicted,atol=3e-5)
                best=min(abs(cosine(vector_sum(choice,contrib),grad)-source_cos)
                         for choice in choices)
                if gap>best+5e-5:
                    raise AssertionError("direction control did not minimize frozen angle gap")
                expected_angle=(gap<=ANGLE_LIMIT and norm(grad)>1e-7
                                and norm(source_vec)>1e-6 and norm(ctrl_vec)>1e-6)
                if route["within_angle_gate"]!=expected_angle:
                    raise AssertionError("angle match feasibility was relabeled")
                la=norm(source_vec)
                rb=norm(ctrl_vec)
                target=min(la,rb)
                for field,value in (("full_left_dose",la),
                                    ("full_right_dose",rb),
                                    ("shared_achievable_dose",target),
                                    ("realized_left_dose",target),
                                    ("realized_right_dose",target)):
                    close(float(dose[field][i]),value)
                if not 0<=dose["left_gain"][i]<=1.00001 or not 0<=dose["right_gain"][i]<=1.00001:
                    raise AssertionError("amplifying donor instead of pure attenuation")
                if (dose["matched_within_1pct"][i] is not True
                    or dose["relative_dose_mismatch"][i]>0.01001):
                    raise AssertionError("direct donor norms not matched")
                retain=target/la if la>1e-6 else 0.
                close(float(dose["retained_fraction_of_full_left_dose"][i]),retain)
                close(float(site["source_norm_after_matching"][i]),target)
                close(float(site["control_norm_after_matching"][i]),target)
                live=target>1e-6
                if dose["shared_nonzero"][i]!=live:
                    raise AssertionError("positive dose was misrepresented")
                if site["angle_passed_and_positive_with_matched_dose"][i] != (
                        expected_angle and live):
                    raise AssertionError("combined physical-angle eligibility misrepresented")
                total+=1
                angle_matched+=int(expected_angle)
                positive+=int(live)
                matched+=1
                joint+=int(expected_angle and live)
                retained.append(retain)
                choose_counts.append(len(choices))
    if not sites or total!=16*16*len(sites):
        raise AssertionError("wrong number of 16-state source visit cases")
    mean_ratio=median(retained)
    if (abs(report["angle_matched_fraction"]-angle_matched/total)>1e-7
        or abs(report["angle_and_physical_matched_fraction"]-joint/total)>1e-7
        or abs(report["dose"]["fraction_with_positive_shared_dose"]-positive/total)>1e-7
        or abs(report["dose"]["median_fraction_full_left_dose_preserved"]-mean_ratio)>1e-7):
        raise AssertionError("calibration summary differs from independently scored evidence")
    physical=(matched/total>=.99 and positive/total>=.8 and mean_ratio>=.25)
    feasible=joint/total>=.8 and physical
    if bool(report["source_null_feasible"])!=feasible:
        raise AssertionError("calibration gate differs from registered combination of geometric floors")
    return {
        "world_id":w["world_id"],
        "world_index":index,
        "source_site_kind":report["candidate_source_site_kind"],
        "source_contexts":16,
        "sampled_visit_cases":total,
        "source_angle_matched_fraction":angle_matched/total,
        "angle_and_physical_matched_fraction":joint/total,
        "positive_shared_dose_fraction":positive/total,
        "median_left_dose_retained":mean_ratio,
        "minimum_candidate_subset_count":min(choose_counts),
        "maximum_candidate_subset_count":max(choose_counts),
        "physical_signal_feasible":physical,
        "source_null_feasible":feasible,
        "all_active_candidate_directions_exhaustively_audited":True,
        "evaluation_contexts_scored":0,
        "confirmation_worlds_scored":0,
        "raw_sha256":report["raw_context_sha256"],
        "source_frozen_checkpoint_sha256":report["original_native_checkpoint_sha256"],
    }


def aggregate(roots:Path,calibrations:Path,old_lock:Path,cal_lock:Path,output:Path):
    results=[]
    for index in ROOTS:
        folder=roots/f"nec03h-v2-development-root-{index}"
        candidate=calibrations/f"nec03k-source-root-{index}"
        results.append(audit_world(folder,candidate,old_lock,cal_lock))
    if {x["world_index"] for x in results}!=set(ROOTS):
        raise AssertionError("duplicate/wrong root enrollment")
    output.mkdir(parents=True,exist_ok=True)
    obj={
        "schema":"NEC-03K/independent-source-angle-control-calibration-audit/v1",
        "original_model_and_SAE_training_steps_added":0,
        "original_v2_root_indices":list(ROOTS),
        "original_v2_ITT_results_unchanged":True,
        "old_confirmation_registry_sha256":registry()["confirmation_full_spec_sha256"],
        "eight_opened_worlds":8,
        "worlds_with_physical_signal":sum(x["physical_signal_feasible"] for x in results),
        "worlds_with_source_direction_sham_feasibility":sum(x["source_null_feasible"] for x in results),
        "mean_source_angle_coverage":sum(x["source_angle_matched_fraction"] for x in results)/8,
        "mean_angle_and_dose_coverage":sum(x["angle_and_physical_matched_fraction"] for x in results)/8,
        "source_only_16_context_groups_per_root":True,
        "development_evaluation_contexts_scored":0,
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0,
        "worlds":results,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "independent_world_power_unresolved":True,
    }
    (output/"nec03k_independent_source_feasibility.json").write_text(
        json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("NEC03K_EIGHT_ROOT_SOURCE_FEASIBILITY_AUDIT_COMPLETE")
    return obj


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--roots",type=Path,required=True)
    p.add_argument("--calibrations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--old-lock",type=Path,default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    p.add_argument("--calibration-lock",type=Path,default=Path(__file__).with_name("NEC03K_SOURCE_LOCK.json"))
    x=p.parse_args(argv)
    aggregate(x.roots,x.calibrations,x.old_lock,x.calibration_lock,x.out)


if __name__=="__main__":
    main()
