"""Independent NEC-03V source-geometry audit, with a separate matching implementation.

No import from nec03v_geometry. No Torch, checkpoint load, model forward,
task label, outcome logit, recruited world or confirmation-set access.

Recomputes frozen source-fit coactivation and projected SAE decoder geometry
and independently checks the chosen four-code source and disjoint sham
interventions. This audits *internal integrity*, not the external provenance
of the supplied label-blind geometry archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

ROOTS=tuple(range(8))
ARMS=("composed","additive")
GROUPS=tuple(range(24,32))
MASKS=tuple(range(32))
FIT_ROWS=1280
FEAT=128
WIDTH=32
K=4


def canon(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,
                      separators=(",",":"),allow_nan=False)


def sha(path):
    h=sha256()
    with path.open("rb") as stream:
        for b in iter(lambda:stream.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def vecdot(a,b):
    return math.fsum(x*y for x,y in zip(a,b))


def size(v):
    return math.sqrt(vecdot(v,v))


def independent_coactivation(fit,columns):
    occurrences=Counter()
    joint=Counter()
    for row in fit:
        if len(row)>16 or len(set(row))!=len(row) or list(row)!=sorted(row):
            raise AssertionError("invalid positive source-fit TopK16 row")
        occurrences.update(row)
        joint.update(combinations(sorted(row),2))
    lengths=[size(x) for x in columns]
    weights={}
    for x in range(FEAT):
        for y in range(x+1,FEAT):
            numerator=joint[x,y]
            union=occurrences[x]+occurrences[y]-numerator
            jac=numerator/union if union else 0.0
            direction=(vecdot(columns[x],columns[y])/(lengths[x]*lengths[y])
                       if lengths[x]>1e-12 and lengths[y]>1e-12 else 0.0)
            weights[x,y]=.5*jac+.5*max(0.,direction)
    return weights


def cohesive(ids, weights):
    pairs=list(combinations(ids,2))
    return math.fsum(weights[p] for p in pairs)/len(pairs)


def linked(group, weights):
    active={group[0]}
    while True:
        more=set(active)
        for member in active:
            for other in group:
                if member==other:
                    continue
                if weights[min(member,other),max(member,other)]>.01:
                    more.add(other)
        if more==active:
            return len(active)==len(group)
        active=more


def aggregate(ids,acts,columns):
    return tuple(math.fsum(acts[x]*columns[x][j] for x in ids)
                 for j in range(WIDTH))


def validate_number(a,b,field):
    if type(b) not in (int,float) or not math.isfinite(b):
        raise AssertionError(f"nonfinite receipt {field}")
    if not math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=2e-7):
        raise AssertionError(f"incorrect physical geometry {field}: {a} != {b}")


def evaluate(fit,columns, weights, active, cache=None):
    coefficients={int(item[0]):float(item[1]) for item in active}
    if (len(coefficients)!=len(active) or len(coefficients)>16
            or any(x<=1e-7 or not math.isfinite(x) for x in coefficients.values())):
        raise AssertionError("invalid positive active features")
    active_ids=tuple(sorted(coefficients))
    if len(active_ids)!=16:
        return {"ok":False,"reason":"fewer_than_16_positive_topk_codes",
                "candidate_count":0,"positive_codes":len(active_ids)}
    # Independent from the scorer: preserve the entire candidate enumeration
    # and source-rank optimization, cache by graph-fixed active IDs only.
    if cache is None:
        cache={}
    if active_ids not in cache:
        candidates=[]
        for combination in combinations(active_ids, K):
            if linked(combination,weights):
                weight=cohesive(combination,weights)
                if weight>=.10:
                    candidates.append((combination,weight))
        if not candidates:
            cache[active_ids]=None
        else:
            best=max(candidates,key=lambda v:(v[1],tuple(-x for x in v[0])))
            choices=[(ids,score) for ids,score in candidates
                     if not set(ids)&set(best[0])]
            cache[active_ids]=(best,choices)
    frozen=cache[active_ids]
    if frozen is None:
        return {"ok":False,"reason":"missing_coherent_source_group",
                "candidate_count":0}
    (chosen,weight), alternatives=frozen
    source=aggregate(chosen,coefficients,columns)
    ns=size(source)
    if not alternatives:
        return {"ok":False,"reason":"missing_disjoint_structural_sham",
                "candidate_count":0, "source":list(chosen)}
    def rank(item):
        ids,score=item
        v=aggregate(ids,coefficients,columns)
        return (round(abs(weight-score),12),
                round(abs(ns-size(v))/max(ns,1e-4),12),
                tuple(ids))
    other,other_score=min(alternatives,key=rank)
    sh=aggregate(other,coefficients,columns)
    nh=size(sh)
    gap=abs(weight-other_score)
    dose=min(ns,nh)
    record={
        "source":list(chosen),"sham":list(other),
        "candidate_count":len(alternatives),
        "source_coherence":weight,
        "sham_coherence":other_score,
        "coherence_gap":gap,
        "source_norm":ns,"sham_norm":nh,
    }
    if gap>.35:
        return {"ok":False,"reason":"sham_coherence_gap",**record}
    if dose<1e-4:
        return {"ok":False,"reason":"weak_shared_32d_dose",**record}
    gs=dose/ns
    hs=dose/nh
    gvec=[gs*v for v in source]
    hvec=[hs*v for v in sh]
    equal=abs(size(gvec)-size(hvec))/max(dose,1e-4)
    if equal>.01 or gs>1+1e-12 or hs>1+1e-12:
        raise AssertionError("independent 32D matching/nonamplification failure")
    return {"ok":True,**record,
            "shared_dose":dose,"source_gain":gs,"sham_gain":hs,
            "actual_dose_source":size(gvec),
            "actual_dose_sham":size(hvec),
            "relative_match_error":equal,
            "scaled_source32":gvec, "scaled_sham32":hvec}


def audit(pack:Path, report:Path):
    src=json.loads((pack/"manifest.json").read_text(encoding="utf-8"))
    out=json.loads(report.read_text(encoding="utf-8"))
    if (out.get("schema")!="NEC-03V/geometry-preflight-unscored/v1"
            or src.get("schema")!="NEC-03V/label-blind-source-geometry/v1"
            or out.get("world_index")!=src.get("world_index")
            or out.get("world_id")!=src.get("world_id")
            or src.get("world_index") not in ROOTS
            or src.get("source_panel")!=list(GROUPS)
            or out.get("origin_pack_manifest_sha256")!=sha(pack/"manifest.json")):
        raise AssertionError("source geometry or output manifest identity changed")
    lock=Path(__file__).with_name("NEC03T_SOURCE_LOCK.json")
    if (src.get("original_lock_sha256")!=sha(lock)
            or out.get("original_nec03t_lock_sha256")!=sha(lock)):
        raise AssertionError("historical NEC03T lock changed")
    if (src.get("evidence") not in ("synthetic_geometry_fixture",
                                     "cached_source_fit_rank_geometry_unscored",
                                     "registered_source_only_activation_extraction_unscored")
            or out.get("evidence_kind")!=src.get("evidence")
            or set(src.get("arm_shards",{}))!=set(ARMS)
            or set(src.get("source_checkpoint_receipts",{}))!=set(ARMS)):
        raise AssertionError("source evidence or arm scope changed")
    if src["evidence"] == "registered_source_only_activation_extraction_unscored":
        receipt=pack/"extraction_receipt.json"
        if (sha(receipt)!=src.get("extraction_receipt_sha256")
                or json.loads(receipt.read_text()).get("task_labels_accessed")!=0):
            raise AssertionError("source activation receipt failed")
    for k in ("model_forwards","labels_accessed","logits_accessed",
              "new_neural_or_SAE_training_updates","new_recruited_root_reads",
              "embargoed_or_confirmation_reads"):
        if out.get(k)!=0:
            raise AssertionError(f"NEC03V outcome or protected-source access prohibited: {k}")
    if (out.get("registered_primary_scoring_enabled") is not False
            or out.get("source_only_pilot_scored") is not False
            or out.get("behavioral_effect") is not None
            or out.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("preflight claimed behavioral scoring")
    records=out.get("case_receipts")
    if not isinstance(records,list) or len(records)!=512:
        raise AssertionError("all 256 x 2 registered original source cases required")
    if out.get("method_source_sha256") != sha(Path(__file__).with_name("nec03v_geometry.py")):
        raise AssertionError("frozen source-group preflight method changed")
    frozen_roster = [
        {"arm": case["arm"], "group": case["group"], "mask": case["mask"],
         "eligible": case["eligible_all_visits"],
         "visits": {v: {"source": row.get("source", []),
                         "sham": row.get("sham", []),
                         "effective_zero_pair": row["effective_zero_pair"]}
                    for v, row in sorted(case["visits"].items(), key=lambda pair:int(pair[0]))}}
        for case in records
    ]
    if out.get("frozen_group_selection_sha256") != sha256(canon(frozen_roster).encode()).hexdigest():
        raise AssertionError("frozen coherent-group and sham roster hash changed")
    lookup={(r["arm"],r["group"],r["mask"]):r for r in records}
    if len(lookup)!=512:
        raise AssertionError("duplicate geometry case receipt")
    checked=0
    arm_counts={}
    for arm in ARMS:
        item=src["arm_shards"][arm]
        if (set(item)!={"file","sha256"} or item["file"]!=f"{arm}.json"
                or sha(pack/item["file"])!=item["sha256"]):
            raise AssertionError("geometry shard bytes or path changed")
        shard=json.loads((pack/item["file"]).read_text(encoding="utf-8"))
        if (set(shard)!={"schema","arm","world_index","visits","cases"}
                or shard["schema"]!="NEC-03V/source-geometry-arm/v1"
                or shard["arm"]!=arm or shard["world_index"]!=src["world_index"]
                or len(shard["cases"])!=256):
            raise AssertionError("registered original source arm changed")
        if (not shard["visits"] and src["world_index"] not in (3,5)
                or (src["source_checkpoint_receipts"][arm]["sae_sha256"] is None)
                   != (not shard["visits"])):
            raise AssertionError("missing eligible source site prohibited")
        weights={}
        for v,spec in shard["visits"].items():
            if set(spec)!={"decoder32","source_fit_active"}:
                raise AssertionError("source visit includes unauthorized fields")
            if (len(spec["decoder32"])!=FEAT or
                    any(len(col)!=WIDTH for col in spec["decoder32"]) or
                    len(spec["source_fit_active"])!=FIT_ROWS):
                raise AssertionError("source fit or 32D decoder size changed")
            weights[v]=independent_coactivation(spec["source_fit_active"],spec["decoder32"])
        per_visit_cache={visit:{} for visit in weights}
        eligible=0
        for i,case in enumerate(shard["cases"]):
            group,mask=GROUPS[i//32],MASKS[i%32]
            if (set(case)!={"group","mask","visits"} or
                    case["group"]!=group or case["mask"]!=mask or
                    set(case["visits"])!=set(weights)):
                raise AssertionError("invalid source cohort group/mask/visits")
            row=lookup[arm,group,mask]
            if set(row["visits"])!=set(weights):
                raise AssertionError("missing registered visit in output")
            expected={}
            for visit in sorted(weights,key=int):
                obs=row["visits"][visit]
                exp=evaluate(shard["visits"][visit]["source_fit_active"],
                             shard["visits"][visit]["decoder32"],
                             weights[visit],case["visits"][visit],
                             per_visit_cache[visit])
                if obs["ok"]!=exp["ok"]:
                    raise AssertionError("source/sham feasibility changed")
                if obs.get("reason") not in (None,exp.get("reason"),"valid",
                                              "other_visit_invalid_case_zero"):
                    raise AssertionError("unknown feature-group intervention failure")
                for field in ("source","sham","candidate_count"):
                    if field in exp and obs.get(field)!=exp[field]:
                        raise AssertionError(f"chosen feature group or sham changed: {field}")
                for field in ("source_coherence","sham_coherence","coherence_gap",
                              "source_norm","sham_norm","shared_dose","source_gain","sham_gain",
                              "actual_dose_source","actual_dose_sham","relative_match_error"):
                    if field in exp:
                        validate_number(exp[field],obs[field],field)
                for field in ("scaled_source32","scaled_sham32"):
                    if field in exp:
                        if len(obs[field])!=WIDTH:
                            raise AssertionError("wrong output projected dimension")
                        for x,y in zip(exp[field],obs[field]):
                            validate_number(x,y,field)
                expected[visit]=exp
            valid=bool(expected) and all(v["ok"] for v in expected.values())
            if row.get("eligible_all_visits") is not valid:
                raise AssertionError("all-visit source-case validity changed")
            eligible+=int(valid)
            for visit,exp in expected.items():
                obs=row["visits"][visit]
                zero=[0.0]*WIDTH
                if obs.get("effective_zero_pair") is not (not valid):
                    raise AssertionError("invalid case did not zero both paired donors")
                for field,op in (("effective_source32","scaled_source32"),
                                 ("effective_sham32","scaled_sham32")):
                    data=exp[op] if valid else zero
                    for x,y in zip(data,obs[field]):
                        validate_number(x,y,field)
            checked+=1
        arm_out=out["arms"][arm]
        if (arm_out["cases"]!=256 or arm_out["eligible_cases"]!=eligible
                or arm_out["invalid_as_intention_to_treat_zero_pair"]!=256-eligible):
            raise AssertionError("preflight source-case denominator or coverage changed")
        validate_number(eligible/256,arm_out["coverage"],"coverage")
        arm_counts[arm]={"cases":256,"eligible":eligible}
    return {"schema":"NEC-03V/independent-geometry-audit/v1",
            "world_id":src["world_id"],"case_arm_records":checked,
            "arms":arm_counts,"geometry_audit_passed":True,
            "external_source_provenance_authenticated":False,
            "new_model_forwards":0,"protected_reads":0,
            "original_nec03t_lock_sha256":sha(lock),
            "preflight_sha256":sha(report),"source_pack_sha256":sha(pack/"manifest.json"),
            "frozen_group_selection_sha256":out["frozen_group_selection_sha256"],
            "next_gate":"authenticate frozen source-fit and SAE checkpoint geometry externally",
            "scientific_gate":"REDESIGN_OR_PILOT_ONLY"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack",type=Path,required=True)
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise ValueError("audit output must be a fresh file")
    receipt=audit(args.pack,args.preflight)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(canon(receipt)+"\n",encoding="utf-8")
    print(canon(receipt))
    return receipt


if __name__=="__main__":
    main()
