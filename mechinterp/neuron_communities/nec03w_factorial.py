"""NEC-03W: CPU-only source-custodied factorial-complementarity gym.

Status: CANDIDATE DESIGN AND SYNTHETIC GYM ONLY. No new native forward,
task outcome, trained SAE, score read, recruited root or confirmation access.

Old NEC03V selected one four-code group using geometric coactivation and failed
its source-only necessity gate. NEC03W registers a DISTINCT hypothesis:
two 2-code components might be *jointly necessary*, even when either can be
removed alone. Factorial excess loss is compared to a physically matched
disjoint four-code sham. Delayed 32D memory rescue is a separately scored
causal intervention; same-site full add-back is an identity/implementation
control and never counts as empirical evidence of rescue.

Provenance: original source SAE was fitted on groups 0..39, so these source
panels are developmental even when source behavioral readouts are disjoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack, original_lock_digest

SCHEMA="NEC-03W/source-only-factorial-candidate-roster/v1"
FACTORIAL_SCHEMA="NEC-03W/paired-2x2-complementarity-and-late-rescue/v1"
ORIGINAL_ROOTS=tuple(range(8))
SOURCE_FIT_DISCOVERY=tuple(range(0,16))
SOURCE_FUNCTION_CALIBRATION=tuple(range(16,24))
PREVIOUSLY_SCORED_SOURCE=tuple(range(24,32))
SOURCE_QUERY=tuple(range(32,40))
PANELS=(SOURCE_FIT_DISCOVERY,SOURCE_FUNCTION_CALIBRATION,
        PREVIOUSLY_SCORED_SOURCE,SOURCE_QUERY)
ARMS=("composed","additive")
FIT_ROWS=1280
FEATURES=128
MAX_CODES=16
GROUP_SIZE=4
PER_VISIT_CANDIDATES=12
FUNCTION_CALIBRATION_CASES=8*32
SOURCE_QUERY_CASES=8*32
PAIR_MATCH_SCHEMA="NEC-03W/matched-factorial-physical-sham/v1"
SALT="NEC03W-source-geometry-factorial-20261122-v1"


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False)


def digest(value):
    return sha256(encode(value).encode("utf-8")).hexdigest()


def _source_ids(rows):
    if not isinstance(rows,list) or len(rows)!=FIT_ROWS:
        raise ValueError("NEC03W requires exact 1280 original label-free source-fit IDs")
    for row in rows:
        if (not isinstance(row,list) or len(row)>MAX_CODES or
                any(type(f) is not int or not 0<=f<FEATURES for f in row) or
                row!=sorted(set(row))):
            raise ValueError("invalid original positive SAE support")
    return rows


def quartet_partitions(group):
    if len(group)!=GROUP_SIZE or tuple(sorted(set(group)))!=tuple(group):
        raise ValueError("NEC03W quartet must be four sorted unique feature IDs")
    a,b,c,d=group
    return [
        [[a,b],[c,d]],
        [[a,c],[b,d]],
        [[a,d],[b,c]],
    ]


def discover_candidates(rows, budget=PER_VISIT_CANDIDATES):
    """Top quartets based on original source discovery 0..15 ID incidence only.

    The *feature groups* are candidate menu items, not functional selections.
    Their function must be tested in separately registered calibration, then
    on original source query 32..39 (already seen by the historical SAE fit).
    """
    _source_ids(rows)
    if not 1<=budget<=32:
        raise ValueError("candidate budget must be 1..32")
    seen=Counter()
    for row in rows[:16*32]:
        seen.update(combinations(row,GROUP_SIZE))
    ranked=sorted(seen.items(),key=lambda entry:(-entry[1],entry[0]))
    chosen=[]
    for group,count in ranked:
        if any(len(set(group).intersection(other))>2 for other,_ in chosen):
            continue
        chosen.append((group,count))
        if len(chosen)==budget:
            break
    result=[]
    for group,count in chosen:
        # These later cohorts are inspected only for label-blind reporting of
        # activation feasibility, AFTER candidate selection is frozen.
        support=set(group)
        calibration=sum(support.issubset(ids) for ids in rows[512:768])
        query=sum(support.issubset(ids) for ids in rows[1024:1280])
        result.append({
            "quartet":list(group),"source_discovery_count":count,
            "source_discovery_fraction":count/512,
            "source_calibration_ID_support_only":calibration,
            "source_query_ID_support_only":query,
            "pair_partitions":quartet_partitions(group),
        })
    return result


def build_roster(source_pack:Path,budget=PER_VISIT_CANDIDATES):
    manifest,shards,manifest_sha=read_pack(source_pack)
    if manifest["evidence"]!="registered_source_only_activation_extraction_unscored":
        raise ValueError("candidate discovery requires authenticated original source code archive")
    root=manifest["world_index"]
    if type(root) is not int or root not in ORIGINAL_ROOTS:
        raise PermissionError("NEC03W only original dev5-000..007 source roots")
    arms={}
    for arm in ARMS:
        items={}
        for visit,spec in sorted(shards[arm]["visits"].items(),key=lambda kv:int(kv[0])):
            candidates=discover_candidates(spec["source_fit_active"],budget)
            items[visit]={
                "source_fit_rows_used_for_rank":list(SOURCE_FIT_DISCOVERY),
                "candidate_count":len(candidates),
                "selected_by_functional_effect":False,
                "candidates":candidates,
            }
        arms[arm]={"visits":items,
                   "original_no_source_visits":not bool(items)}
    roster={
        "schema":SCHEMA,
        "original_world_id":manifest["world_id"],
        "world_index":root,
        "source_pack_manifest_sha256":manifest_sha,
        "original_NEC03T_lock_sha256":original_lock_digest(),
        "original_source_evidence":manifest["evidence"],
        "frozen_candidate_budget":budget,
        "split":{
            "geometry_discovery":list(SOURCE_FIT_DISCOVERY),
            "functional_calibration":list(SOURCE_FUNCTION_CALIBRATION),
            "previously_behavior_scored_excluded":list(PREVIOUSLY_SCORED_SOURCE),
            "developmental_query":list(SOURCE_QUERY),
            "source_query_is_SAE_fit_training_data":True,
            "original_source_rank_40_to_79_unchanged":True,
        },
        "arms":arms,
        "allowed_new_native_forwards":0,
        "task_labels_read":0,"model_logits_read":0,
        "optimizer_steps":0,
        "protected_cohort_reads":0,
        "interventions_scored":0,
        "experimental_stage":"source-only candidate feasibility, functional selection deferred",
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    roster["roster_sha256"]=digest(roster)
    return roster


def projected(ids,coefficients,decoder32):
    if any(i not in coefficients for i in ids):
        raise ValueError("invalid inactive source feature used for 32D projector")
    if len(decoder32)!=128 or any(len(v)!=32 for v in decoder32):
        raise ValueError("frozen original SAE physical decoder must be 128x32")
    return tuple(math.fsum(coefficients[i]*decoder32[i][j] for i in ids)
                 for j in range(32))


def norm(v):
    return math.sqrt(math.fsum(x*x for x in v))


def cosine(u,v):
    a,b=norm(u),norm(v)
    if min(a,b)<=1e-10:
        raise ValueError("nonzero physical vectors required")
    return math.fsum(x*y for x,y in zip(u,v))/(a*b)


def matched_pair_geometry(active_codes,cols,quartet,partition, *,
                          root,arm,group,mask,visit,trials=96):
    """Match both 2-code subgroups, joint 4-code dose AND pair-angle geometry.

    Output depends only on original positive code coefficients, frozen 32D
    decoder columns and preregistered ID hashing. No model outcomes.
    """
    if root not in ORIGINAL_ROOTS or arm not in ARMS or not 0<=group<40 or not 0<=mask<32:
        raise PermissionError("protected cohort or unregistered input identity")
    if not 0<=int(visit)<12 or not 8<=trials<=512:
        raise ValueError("invalid original recurrence visit or sham search budget")
    coeffs={int(p[0]):float(p[1]) for p in active_codes}
    if (len(coeffs)!=len(active_codes) or len(coeffs)>MAX_CODES or
            any(not math.isfinite(v) or v<=1e-7 for v in coeffs.values())
            or any(i not in range(FEATURES) for i in coeffs)):
        raise ValueError("positive source TopK feature coefficients invalid")
    frozen=tuple(sorted(quartet))
    if tuple(quartet)!=frozen or len(frozen)!=4:
        raise ValueError("invalid frozen feature quartet")
    if len(coeffs)!=16 or not set(frozen).issubset(coeffs):
        return {"schema":PAIR_MATCH_SCHEMA,"admissible":False,
                "reason":"incomplete_positive_topk_or_quartet_support"}
    if (len(partition)!=2 or any(len(x)!=2 for x in partition)
            or set(partition[0])|set(partition[1])!=set(frozen)
            or set(partition[0])&set(partition[1])):
        raise ValueError("each original feature belongs to exactly one of two pairs")
    pA,pB=tuple(partition[0]),tuple(partition[1])
    a,b=projected(pA,coeffs,cols),projected(pB,coeffs,cols)
    g=tuple(x+y for x,y in zip(a,b))
    ng=norm(g)
    if ng<0.1 or min(norm(a),norm(b))<0.05:
        return {"schema":PAIR_MATCH_SCHEMA,"admissible":False,"reason":"insufficient_joint_physical_dose"}
    # Deterministic, SHA256-stable proposals. This is an outcome-blind physical
    # sham match, with two disjoint pairs selected from 12 other active codes.
    other=tuple(sorted(set(coeffs)-set(frozen)))
    # Reuse each of the 66 disjoint 2-code physical directions. This cache
    # is purely source geometry and does not use any observed task response.
    pair_vectors={pair:projected(pair,coeffs,cols)
                  for pair in combinations(other,2)}
    attempts=[]
    for trial in range(trials):
        key=f"{SALT}|{root}|{arm}|{group}|{mask}|{visit}|{frozen}|{trial}"
        order=sorted(other,key=lambda f:sha256(f"{key}|{f}".encode()).digest())
        chosen=tuple(sorted(order[:4]))
        for pieces in quartet_partitions(chosen):
            for first,second in (pieces,list(reversed(pieces))):
                c,d=pair_vectors[tuple(first)],pair_vectors[tuple(second)]
                h=tuple(x+y for x,y in zip(c,d))
                nh=norm(h)
                if min(norm(c),norm(d),nh)<0.05:
                    continue
                ratioA,ratioB=norm(a)/ng,norm(b)/ng
                ratioC,ratioD=norm(c)/nh,norm(d)/nh
                norm_gap=max(abs(ratioA-ratioC),abs(ratioB-ratioD))
                angle_gap=abs(cosine(a,b)-cosine(c,d))
                cross=cosine(g,h)
                dmin=min(ng,nh)
                score=norm_gap+angle_gap+0.1*abs(math.log(ng/nh))
                if norm_gap>0.35 or angle_gap>0.40 or cross>0.95 or dmin<0.1:
                    continue
                attempts.append((score,tuple(first),tuple(second),chosen,
                                 norm_gap,angle_gap,cross,dmin,nh,c,d,h))
    if not attempts:
        return {"schema":PAIR_MATCH_SCHEMA,"admissible":False,
                "reason":"no_pairwise_physical_sham_with_original_limits",
                "sham_trials":trials}
    winner=min(attempts,key=lambda v:(round(v[0],12),v[1],v[2]))
    _,c_ids,d_ids,h_ids,gap,angle_gap,cross,dmin,nh,c,d,h=winner
    sg,sh=dmin/ng,dmin/nh
    if min(sg,sh)<0 or max(sg,sh)>1+1e-12:
        raise AssertionError("source or sham physical contribution was amplified")
    scale=lambda v,k:[round(k*x,8) for x in v]
    out={
        "schema":PAIR_MATCH_SCHEMA,"admissible":True,
        "root":root,"arm":arm,"group":group,"mask":mask,"visit":int(visit),
        "pair_A":list(pA),"pair_B":list(pB),
        "matched_control_C":list(c_ids),"matched_control_D":list(d_ids),
        "group_quartet":list(frozen),"control_quartet":list(h_ids),
        "source_gain":round(sg,8),"sham_gain":round(sh,8),
        "shared_dose32":round(dmin,8),
        "max_pair_subgroup_norm_ratio_gap":round(gap,8),
        "pair_internal_cosine_gap":round(angle_gap,8),
        "source_sham_combined_direction_cosine":round(cross,8),
        "projected32":{
            "A":scale(a,sg),"B":scale(b,sg),"AB":scale(g,sg),
            "C":scale(c,sh),"D":scale(d,sh),"CD":scale(h,sh),
        },
        "original_source_only":True,"label_free":True,
        "delayed_rescue_visit":int(visit)+1 if int(visit)<11 else None,
        "immediate_same_site_addback_is_identity_only":True,
    }
    # Numeric projection equality is part of the receipt, not assumed.
    V=out["projected32"]
    if (max(abs((x+y)-z) for x,y,z in zip(V["A"],V["B"],V["AB"]))>5e-8
            or max(abs((x+y)-z) for x,y,z in zip(V["C"],V["D"],V["CD"]))>5e-8
            or abs(norm(V["AB"])-norm(V["CD"]))/dmin>0.01):
        raise AssertionError("matched source/control joint 32D geometry changed")
    out["geometry_sha256"]=digest(out)
    return out


def _binary_loss(value):
    if type(value) is not int or value not in (0,1):
        raise ValueError("factorial outcome must be 0/1 error, not score/logit")
    return value


def factorial_metrics(records, *, denominator=256):
    """Compute fixed-denominator joint necessity and delayed rescue controls.

    A positive excess error interaction (AB - A - B + baseline) measures a
    *redundancy-type* joint necessity under this chosen dose. The comparison
    subtracts physically matched C/D sham interaction. This remains
    developmental: matching marginal pair norms and angles cannot guarantee
    behavioral null resolution.
    """
    if len(records)!=denominator or denominator<1:
        raise ValueError("original source evaluation needs full fixed case denominator")
    names=("baseline","remove_A","remove_B","remove_AB",
           "sham_C","sham_D","sham_CD",
           "late_target_rescue","late_matched_sham_rescue",
           "same_visit_exact_addback")
    sums={key:0 for key in names}
    used=0
    combined_excess=sham_excess=rescue=0
    for case in records:
        if set(case)!={"eligible","delayed_rescue_eligible","errors"}:
            raise ValueError("factorial case fields include unexpected outcome/custody data")
        observations=case["errors"]
        if set(observations)!=set(names):
            raise ValueError("complete original source 2x2 controls are mandatory")
        for name in names:
            sums[name]+=_binary_loss(observations[name])
        base=observations["baseline"]
        if observations["same_visit_exact_addback"]!=base:
            raise ValueError("exact same-visit feature restoration must replay baseline")
        if type(case["eligible"]) is not bool or type(case["delayed_rescue_eligible"]) is not bool:
            raise ValueError("factorial eligibility must be explicit boolean")
        if not case["eligible"]:
            if any(v!=base for v in observations.values()):
                raise ValueError("ineligible original source case must have identical no-op controls")
            continue
        used+=1
        if not case["delayed_rescue_eligible"] and (
                observations["late_target_rescue"]!=observations["remove_AB"] or
                observations["late_matched_sham_rescue"]!=observations["remove_AB"]):
            raise ValueError("missing next visit must preserve paired zero-rescue controls")
        combined_excess+=observations["remove_AB"]-observations["remove_A"]-observations["remove_B"]+base
        sham_excess+=observations["sham_CD"]-observations["sham_C"]-observations["sham_D"]+base
        rescue+=observations["late_matched_sham_rescue"]-observations["late_target_rescue"]
    return {
        "schema":FACTORIAL_SCHEMA,"denominator":denominator,
        "admissible_cases":used,
        "group_joint_error_excess":combined_excess/denominator,
        "matched_sham_joint_error_excess":sham_excess/denominator,
        "factorial_group_minus_sham_interaction":(combined_excess-sham_excess)/denominator,
        "delayed_target_minus_sham_rescue_benefit":rescue/denominator,
        "observed_binary_error_counts":sums,
        "same_site_identity_control_passed":True,
        "p_value_or_population_claim":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def select_from_calibration(candidate_calibration:dict, *, minimum_admissible=64):
    """Calibrate only from explicitly supplied SOURCE 16..23 outcome records.

    The caller must custody these records; the withheld source query panel
    32..39 is never a parameter or accessible binding to this method.
    """
    if not candidate_calibration:
        raise ValueError("functional calibration needs at least one preregistered candidate")
    eligible=[]
    for identifier,records in candidate_calibration.items():
        if not isinstance(identifier,str) or not identifier:
            raise ValueError("candidate identity must be a frozen nonempty string")
        m=factorial_metrics(records,denominator=FUNCTION_CALIBRATION_CASES)
        if m["admissible_cases"]>=minimum_admissible:
            eligible.append((m["factorial_group_minus_sham_interaction"],identifier,m))
    if not eligible:
        return {"selected":None,"reason":"insufficient_original_source_calibration_coverage"}
    best=min(eligible,key=lambda row:(-row[0],row[1]))
    return {"selected":best[1],"source_calibration_effect":best[0],
            "source_calibration_sha256":digest({
                "candidate":best[1],"metrics":best[2]}),
            "source_calibration_cases":FUNCTION_CALIBRATION_CASES,
            "source_query_32_to_39_read":False,
            "selection_conditioning":"max over preregistered candidate pool",
            "scientific_gate":"REDESIGN_OR_PILOT_ONLY"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--candidate-budget",type=int,default=PER_VISIT_CANDIDATES)
    args=parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("candidate roster must have a fresh output file")
    result=build_roster(args.source_pack,args.candidate_budget)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(encode(result)+"\n",encoding="utf-8")
    print(encode({
        "original_world_id":result["original_world_id"],
        "candidate_roster_sha256":result["roster_sha256"],
        "source_only":True,"model_forwards":0,
        "candidate_counts":{
            arm:{v:entry["candidate_count"] for v,entry in a["visits"].items()}
            for arm,a in result["arms"].items()}}))
    return result


if __name__=="__main__":
    main()
