"""NEC-03V-B original source-only causal group-removal pilot.

One registered coherent four-code group and a disjoint, physically matched
sham per original source case and visit, drawn ONLY from previously audited
source-fit groups. Frozen native teachers, original 12 recurrent passes,
no new training or recruited/confirmation access.

Source-only behavioral results are developmental, never population claims.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack
from .nec03v_source_attest import attest, canonical, sha256_file

ROOTS=tuple(range(8))
ELIGIBLE=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
GROUPS=tuple(range(24,32))
ARMS=("composed","additive")
MODES=("full_affine","group_025","sham_025","group_05","sham_05","group_1","sham_1")
RESULT="NEC-03V/source-only-group-removal-behavior/v1"


def load_locks():
    base=Path(__file__).parent
    panel=json.loads((base/"NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json").read_text("utf-8"))
    method=json.loads((base/"NEC03V_BEHAVIOR_METHOD_LOCK.json").read_text("utf-8"))
    if (panel.get("schema")!="NEC-03V/frozen-original-eight-source-physical-roster/v1"
            or sorted(map(int,panel["roots"]))!=list(ROOTS)
            or panel["source_eligibility_roots"]!=list(ELIGIBLE)
            or panel["fixed_ITT_zero_roots"]!=list(ITT_ZERO)
            or panel["source_groups"]!=list(GROUPS)
            or panel["source_input_per_arm"]!=256
            or panel["physical_outcome_blind_gate"]["minimum_meaningful_behavioral_root_macro_effect"]!=.02
            or method.get("schema")!="NEC-03V/frozen-original-source-behavior-method/v1"
            or method.get("allowed_original_roots")!=list(ROOTS)
            or method.get("source_groups")!=list(GROUPS)
            or method.get("doses")!=[0,.25,.5,1]
            or method.get("source_only_new_scoring") is not True
            or method.get("new_training_updates")!=0
            or method.get("recruited_or_confirmation_worlds_allowed") is not False
            or method.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("NEC03V source-only behavioral panel or protocol changed")
    methods=method.get("git_blob_sha1",{})
    expected={
        "mechinterp/neuron_communities/nec03v_behavior_source.py",
        "mechinterp/neuron_communities/nec03v_behavior_independent_audit.py",
        "mechinterp/neuron_communities/NEC03V_BEHAVIOR_SOURCE_PROTOCOL.md",
    }
    if set(methods)!=expected:
        raise ValueError("behavior method independent-audit SHA registration missing")
    root=base.parents[1]
    for name,pin in {**methods,**method["dependency_git_blob_sha1"]}.items():
        data=(root/name).read_bytes()
        sha=sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if sha!=pin:
            raise ValueError("frozen source behavioral code changed: "+name)
    return panel,method


def physical_null_gate(records, *, root, floor):
    if root not in ROOTS or len(records)!=512:
        raise ValueError("full two-arm 256-case source-only roster required")
    counts={arm:0 for arm in ARMS}
    angles,doses,errors=[],[],[]
    for record in records:
        if record["arm"] not in counts or record["group"] not in GROUPS:
            raise ValueError("unexpected source-only case identity")
        if not record["eligible_all_visits"]:
            continue
        counts[record["arm"]]+=1
        for item in record["visits"].values():
            a,b=item["effective_source32"],item["effective_sham32"]
            if len(a)!=32 or len(b)!=32:
                raise ValueError("source and sham are physically 32D")
            da=math.sqrt(math.fsum(x*x for x in a))
            db=math.sqrt(math.fsum(x*x for x in b))
            if min(da,db)<=0:
                raise ValueError("admitted source case lacks nonzero physical perturbation")
            angles.append(math.fsum(x*y for x,y in zip(a,b))/(da*db))
            doses.append(min(da,db))
            errors.append(abs(da-db)/min(da,db))
    fraction=sum(v<=.95 for v in angles)/len(angles) if angles else 0.0
    pass_gate=(root in ITT_ZERO or (
        min(counts.values())/256>=floor["min_eligible_case_fraction_each_original_paired_source_arm"]
        and min(doses)>=floor["min_shared_dose_32D"]
        and fraction>=floor["min_fraction_selected_visits_source_sham_cosine_lte_0_95"]
        and max(errors)<=floor["max_physical_matched_relative_L2_error"]))
    if not pass_gate:
        raise ValueError("predeclared original-source physical null gate failed")
    return {"root":root,"eligible_by_arm":counts,"selected_site_doses":len(doses),
            "minimum_shared_dose":min(doses) if doses else 0.,
            "fraction_source_sham_direction_cosine_le_095":fraction,
            "largest_physical_dose_discrepancy":max(errors) if errors else 0.,
            "gate_passed":True,"behavioral_labels_used":False}


def inputs(root:int, source:Path,t:Path,reset:Path|None,pack:Path,
           physical:Path,independent_physical:Path,independent_activation:Path):
    if type(root) is not int or root not in ROOTS:
        raise PermissionError("NEC03V excludes recruited, reserved and confirmation worlds")
    panel,method=load_locks()
    old=attest(root,source,t,reset)
    meta,shards,manifest_sha=read_pack(pack)
    phys=json.loads(physical.read_text("utf-8"))
    verify=json.loads(independent_physical.read_text("utf-8"))
    activation=json.loads(independent_activation.read_text("utf-8"))
    pin=panel["roots"][str(root)]
    if (pin["source_pack_manifest_sha256"]!=manifest_sha
            or pin["physical_preflight_sha256"]!=sha256_file(physical)
            or pin["physical_independent_audit_sha256"]!=sha256_file(independent_physical)
            or pin["frozen_group_selection_sha256"]!=phys["frozen_group_selection_sha256"]
            or meta["world_index"]!=root
            or meta["source_checkpoint_receipts"]!=old["source_checkpoint_receipts"]
            or phys["world_id"]!=old["world_id"]
            or phys["origin_pack_manifest_sha256"]!=manifest_sha
            or phys["model_forwards"]!=0 or phys["labels_accessed"]!=0
            or phys["registered_primary_scoring_enabled"] is not False
            or verify.get("geometry_audit_passed") is not True
            or verify.get("case_arm_records")!=512
            or verify.get("source_pack_sha256")!=manifest_sha
            or verify.get("preflight_sha256")!=sha256_file(physical)
            or activation.get("independent_activation_audit_passed") is not True
            or activation.get("source_geometry_manifest_sha256")!=manifest_sha
            or activation.get("original_archival_lock_verified") is not True
            or activation.get("source_only_native_forward_calls")!=
               activation.get("independent_source_only_native_forward_calls")
            or activation.get("protected_cohort_queries")!=0
            or activation.get("task_labels_read")!=0):
        raise ValueError("original-source physical/activation provenance or cohort identity changed")
    if {arm:phys["arms"][arm]["eligible_cases"] for arm in ARMS}!=pin["eligible_case_count_by_arm"]:
        raise ValueError("physical cohort pre-locked coverage mismatch")
    physical_check=physical_null_gate(
        phys["case_receipts"],root=root,floor=panel["physical_outcome_blind_gate"])
    return old,shards,phys,physical_check


def original_rows(t:Path,arm:str):
    report=json.loads((t/"nec03t_source_necessity_report.json").read_text("utf-8"))
    meta=report["arms"][arm]
    path=t/meta["raw_file"]
    if sha256_file(path)!=meta["raw_sha256"]:
        raise ValueError("old NEC03T full-affine source-case logit SHA changed")
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(rows)!=8 or [v["source_group"] for v in rows]!=list(GROUPS):
        raise ValueError("old NEC03T source group ordering changed")
    return rows


def donor(frozen,cases,phys_cases,alpha:float,kind:str):
    import torch
    down=frozen["model"].block.mlp_down.weight.detach()
    sae=frozen["sae"]
    gates,donors={},{}
    for visit in sorted(frozen["source_sites"]):
        name=str(visit)
        gate=torch.zeros(64,dtype=down.dtype)
        if visit in frozen["source_hubs"]:
            gate[sorted(frozen["source_hubs"][visit])]=1
        nonhub=1-gate
        z=torch.zeros((32,128),dtype=down.dtype)
        for case_id,case in enumerate(cases):
            for f,value in case["visits"][name]:
                z[case_id,int(f)]=float(value)
        full=(sae.decoder(z)*frozen["scale"]+frozen["mean"])*nonhub
        delta=torch.zeros_like(full)
        for i,row in enumerate(phys_cases):
            chosen=row["visits"][name]
            if chosen["effective_zero_pair"]:
                continue
            ids=chosen["source" if kind=="group" else "sham"]
            gain=chosen["source_gain" if kind=="group" else "sham_gain"]
            direction=(z[i,ids]@sae.decoder.weight[:,ids].T)*frozen["scale"]*nonhub*gain
            expected=torch.tensor(
                chosen["effective_source32" if kind=="group" else "effective_sham32"],
                dtype=down.dtype)
            if not torch.allclose(direction@down.T,expected,atol=2e-4,rtol=2e-5):
                raise ValueError("original 64D removals differ from independent physical 32D audit")
            if not 0 <= gain <= 1.0000001:
                raise ValueError("nonamplifying four-code source/sham gain violated")
            delta[i]=direction
        gates[visit]=gate
        donors[visit]={"native":full-alpha*delta}
    return donors,gates


def run(root:int,source:Path,t:Path,reset:Path|None,pack:Path,physical:Path,
        physical_audit:Path,activation_audit:Path,out:Path,
        *,execute_registered_source_scoring=False):
    old,shards,ph,gate=inputs(root,source,t,reset,pack,physical,physical_audit,activation_audit)
    if not execute_registered_source_scoring:
        return {"schema":RESULT,"root":root,"dry_run":True,
                "physical_gate":gate,"model_forwards":0,
                "labels_read":0,"protected_world_reads":0}
    if out.exists():
        raise ValueError("source-only behavioral pilot output must be a fresh directory")
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("NEC03V behavioral pilot requires CPU-only PyTorch")
    from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
    from mechinterp.neuron_communities.nec03l_five_input_worlds import labels
    from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
    from research.pef07.pef07 import as_tensor,model_digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    indexed={(v["arm"],v["group"],v["mask"]):v
             for v in ph["case_receipts"]}
    if len(indexed)!=512:
        raise ValueError("duplicate physical case/arm IDs")
    rows=[]
    totals={}
    calls={}
    for arm in ARMS:
        frozen=load_source_arm(root,arm,source,reset=reset)
        frozen_model_hash=model_digest(frozen["model"])
        if (frozen_model_hash!=old["source_checkpoint_receipts"][arm]["native_sha256"]
                or frozen["sae_sha256"]!=old["source_checkpoint_receipts"][arm]["sae_sha256"]):
            raise ValueError("original source-native or SAE checkpoint changed")
        packed=shards[arm]["cases"]
        legacy=original_rows(t,arm)
        totals[arm]={name:0 for name in MODES}
        calls[arm]=0
        model=frozen["model"].eval()
        with torch.no_grad():
            for j,g in enumerate(GROUPS):
                sample=packed[j*32:(j+1)*32]
                physical_cases=[indexed[arm,g,i] for i in range(32)]
                if any(sample[i]["group"]!=g or sample[i]["mask"]!=i for i in range(32)):
                    raise ValueError("original source panel group/mask is reordered")
                bits=frozen["catalog"]["train"][g*32:(g+1)*32]
                original=legacy[j]
                if ([tuple(row) for row in original["input_histories"]]!=bits
                        or original["focal_mask_order"]!=list(range(32))):
                    raise ValueError("original source input cases disagree with NEC03T")
                sample_bits=as_tensor(bits)
                truth=labels(sample_bits,frozen["world"],control=(arm=="additive")).tolist()
                if truth!=original["true_teacher_labels"]:
                    raise ValueError("original NEC03T source teacher labels changed")
                logits={}
                for mode in MODES:
                    kind="sham" if mode.startswith("sham") else "group"
                    alpha=0. if mode=="full_affine" else (
                        .25 if mode.endswith("025") else (.5 if mode.endswith("05") else 1.))
                    donors,gates=donor(frozen,sample,physical_cases,alpha,kind)
                    trace=traced_forward(model,sample_bits,mode="native",donors=donors,gates=gates)
                    output=trace["logits"].detach()
                    if tuple(output.shape)!=(32,2):
                        raise ValueError("original 12-pass source logits shape changed")
                    calls[arm]+=1
                    logits[mode]=output.tolist()
                    predicted=output.argmax(dim=1).tolist()
                    totals[arm][mode]+=sum(int(a==b) for a,b in zip(predicted,truth))
                    if mode=="full_affine":
                        legacy_logits=torch.tensor(original["logits"]["full_affine"],dtype=output.dtype)
                        if (not torch.allclose(output,legacy_logits,atol=1e-3,rtol=2e-5)
                                or predicted!=original["class_predictions"]["full_affine"]):
                            raise ValueError("zero-dose source full-affine output differs from NEC03T")
                for i in range(32):
                    if not physical_cases[i]["eligible_all_visits"]:
                        full=logits["full_affine"][i]
                        for name in MODES:
                            if max(abs(x-y) for x,y in zip(full,logits[name][i]))>1e-6:
                                raise ValueError("invalid matched case unexpectedly changed native logits")
                rows.append({
                    "schema":RESULT,"world_index":root,"arm":arm,"source_group":g,
                    "focal_mask_order":list(range(32)),
                    "source_histories":original["input_histories"],
                    "labels":truth,
                    "registered_physical_case_hashes":[sha256(canonical(r).encode()).hexdigest()
                                                       for r in physical_cases],
                    "eligible":[v["eligible_all_visits"] for v in physical_cases],
                    "logits":logits,"group_selected_without_labels":True})
        if model_digest(model)!=frozen_model_hash:
            raise ValueError("source teacher weights mutated during behavioral pilot")
    raw_effect=((totals["composed"]["sham_1"]-totals["composed"]["group_1"])
                -(totals["additive"]["sham_1"]-totals["additive"]["group_1"]))/256
    out.mkdir(parents=True,exist_ok=False)
    raw=out/"source_behavior_raw.jsonl"
    raw.write_text("".join(canonical(item)+"\n" for item in rows),encoding="utf-8")
    report={
        "schema":RESULT,"world_index":root,"world_id":old["world_id"],
        "original_source_groups":list(GROUPS),"source_cases_per_arm":256,
        "registered_source_panel_lock_sha256":sha256_file(
            Path(__file__).with_name("NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json")),
        "registered_behavior_method_lock_sha256":sha256_file(
            Path(__file__).with_name("NEC03V_BEHAVIOR_METHOD_LOCK.json")),
        "original_NEC03T_report_sha256":old["original_nec03t_report_sha256"],
        "source_pack_manifest_sha256":sha256_file(pack/"manifest.json"),
        "original_physical_preflight_sha256":sha256_file(physical),
        "frozen_source_group_roster_sha256":ph["frozen_group_selection_sha256"],
        "physical_gate":gate,"correctness_totals":totals,
        "raw_behavior_file":raw.name,"raw_behavior_sha256":sha256_file(raw),
        "raw_root_composed_minus_additive_correctness_DID":raw_effect,
        "registered_ITT_root_effect":0. if root in ITT_ZERO else raw_effect,
        "original_fixed_ITT_zero":root in ITT_ZERO,
        "source_only_native_forward_calls":sum(calls.values()),
        "source_native_forward_calls_by_arm":calls,
        "recurrent_visits_per_forward":12,
        "new_optimizer_updates":0,"feature_group_selection_uses_labels":False,
        "recruited_or_embargoed_world_queries":0,
        "development_evaluation_or_reserved_queries":0,
        "old_confirmation_queries":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "confirmatory_population_claim":False}
    (out/"source_behavior_report.json").write_text(canonical(report)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-index",type=int,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--nec03t-root",type=Path,required=True)
    parser.add_argument("--reset-root",type=Path)
    parser.add_argument("--geometry-pack",type=Path,required=True)
    parser.add_argument("--physical-preflight",type=Path,required=True)
    parser.add_argument("--independent-physical-audit",type=Path,required=True)
    parser.add_argument("--independent-activation-audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--execute-registered-source-scoring",action="store_true")
    a=parser.parse_args(argv)
    result=run(a.world_index,a.source_root,a.nec03t_root,a.reset_root,
               a.geometry_pack,a.physical_preflight,a.independent_physical_audit,
               a.independent_activation_audit,a.out,
               execute_registered_source_scoring=a.execute_registered_source_scoring)
    print(canonical(result))
    return result


if __name__=="__main__":
    main()
