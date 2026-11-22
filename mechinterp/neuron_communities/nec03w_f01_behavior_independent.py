"""NEC-03W/F01 independent original-source 12-pass factorial behavioral replay.

For each original source calibration group16..23, reconstruct all ten
interventions using the original frozen SAE tensors and a NEW native
MLP-down prehook plus recurrent block posthook. This module does NOT import
the F01 source donor planner or its neural-forward function.

All positive code IDs, candidate/sham group IDs, original training-source
bit histories, root/arm identities, 32D physical vectors and prior audit
SHA-256s are checked before recomputing every original task label and logit.
The independent arithmetic reproduces matched FP32 GEMV for native transport;
the independent physical source-vector audit remains FP64.

Historical source-training developmental result, NOT population confirmation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03v_source_attest import attest,canonical
from .nec03w_f01_behavior_registry import registered_behavior
from .nec03w_f01_physical_audit import verify as verify_original_physical

SCHEMA="NEC-03W/F01/independent-original-calibration-factorial-behavior-audit/v1"
MODES=("baseline","remove_A","remove_B","remove_AB","sham_C","sham_D",
       "sham_CD","late_target_rescue","late_matched_sham_rescue",
       "same_visit_exact_addback")


def digest(obj):
    return sha256(canonical(obj).encode("utf-8")).hexdigest()


def original_trace(model,bits,original_donors,original_gates,*,late_visit=None,late_32=None):
    """Independent original twelve-pass memory-token intervention."""
    import torch
    pre=[0]
    post=[0]
    def pre_down(_module,args):
        v=pre[0]
        pre[0]+=1
        if v not in original_donors:
            return None
        src=args[0]
        if tuple(src.shape)!=(len(bits),2,64):
            raise ValueError("source memory GELU original physical dimension changed")
        out=src.clone()
        out[:,1,:]=src[:,1,:]*original_gates[v][None,:]+original_donors[v]
        return (out,)
    def on_step(_module,_args,value):
        v=post[0]
        post[0]+=1
        if tuple(value.shape)!=(len(bits),2,32):
            raise ValueError("source 12-pass original recurrent block output changed")
        if late_visit is None or v!=late_visit:
            return None
        altered=value.clone()
        altered[:,1,:]+=late_32
        return altered
    hooks=[model.block.mlp_down.register_forward_pre_hook(pre_down),
           model.block.register_forward_hook(on_step)]
    try:
        with torch.no_grad():
            logits=model.logits(bits)
    finally:
        for h in hooks:
            h.remove()
    if pre!=[12] or post!=[12] or tuple(logits.shape)!=(len(bits),2):
        raise ValueError("source teacher did not perform exact twelve recurrent visits")
    return logits.detach().clone()


def separate_donors(frozen,code_arrays,geometry,site,mode):
    """Independently derive one site's two-pair lesions and 32D delayed rescue.

    Use scorer-equivalent FP32 GEMV for native recurrence, separately
    double-precision-checked recorded pair directions from the original
    physical audit. Formula and identities are checked per case.
    """
    import torch
    sae=frozen["sae"]
    down=frozen["model"].block.mlp_down.weight.detach()
    if site is None:
        if any(row["admissible"] for row in geometry):
            raise ValueError("original source-ineligible arm has nonzero physical intervention")
        return {},{},None,None
    if site not in frozen["source_sites"]:
        raise ValueError("registered original source candidate points to nonexistent recurrence site")
    donors={}
    gates={}
    later=None
    late_direction=None
    for v in sorted(frozen["source_sites"]):
        z=code_arrays[v]
        if tuple(z.shape)!=(32,128):
            raise ValueError("original source TopK coefficient batch shape changed")
        gate=torch.zeros(64,dtype=down.dtype)
        for index in frozen["source_hubs"].get(v,()):
            gate[int(index)]=1.
        nonhub=1-gate
        background=(torch.nn.functional.linear(z,sae.decoder.weight,sae.decoder.bias)
                    *frozen["scale"]+frozen["mean"])*nonhub
        gates[v]=gate
        if v!=site:
            donors[v]=background
            continue
        removals={name:torch.zeros_like(background) for name in ("A","B","C","D")}
        target=torch.zeros((32,32),dtype=down.dtype)
        sham=torch.zeros_like(target)
        for j,receipt in enumerate(geometry):
            if not receipt["admissible"]:
                continue
            if (receipt["root"]!=frozen["world"]["world_index"]
                    or receipt["visit"]!=v):
                raise ValueError("registered source F01 physical control belongs to wrong world/site")
            parts={
                "A":(receipt["pair_A"],receipt["source_gain"]),
                "B":(receipt["pair_B"],receipt["source_gain"]),
                "C":(receipt["matched_control_C"],receipt["sham_gain"]),
                "D":(receipt["matched_control_D"],receipt["sham_gain"]),
            }
            for name,(feature_ids,gain) in parts.items():
                raw=z[j,feature_ids]@sae.decoder.weight[:,feature_ids].T
                native=raw*frozen["scale"]*nonhub*float(gain)
                expected=torch.tensor(receipt["projected32"][name],dtype=down.dtype)
                if not torch.allclose(native@down.T,expected,rtol=2e-5,atol=2e-4):
                    raise ValueError("independent source F01 physical output differs from frozen 32D receipt")
                removals[name][j]=native
            target[j]=torch.tensor(receipt["projected32"]["AB"],dtype=down.dtype)
            sham[j]=torch.tensor(receipt["projected32"]["CD"],dtype=down.dtype)
        modes={
            "baseline":background,
            "remove_A":background-removals["A"],
            "remove_B":background-removals["B"],
            "remove_AB":background-removals["A"]-removals["B"],
            "sham_C":background-removals["C"],
            "sham_D":background-removals["D"],
            "sham_CD":background-removals["C"]-removals["D"],
            "same_visit_exact_addback":(
                background-removals["A"]-removals["B"]+
                removals["A"]+removals["B"]),
        }
        modes["late_target_rescue"]=modes["remove_AB"]
        modes["late_matched_sham_rescue"]=modes["remove_AB"]
        donors[v]=modes[mode]
        if mode.startswith("late_") and v<11:
            later=v+1
            late_direction=target if mode=="late_target_rescue" else sham
    return donors,gates,later,late_direction


def independent_metrics(records):
    """Recompute registered full-denominator interaction from binary losses."""
    if len(records)!=256:
        raise ValueError("independent original case denominator must be 256")
    counts={mode:0 for mode in MODES}
    admitted=0
    joint=sham=rescue=0
    for row in records:
        if set(row)!={"eligible","delayed","errors"} or set(row["errors"])!=set(MODES):
            raise ValueError("independent registered factorial modes omitted")
        e=row["errors"]
        if any(type(v) is not int or v not in (0,1) for v in e.values()):
            raise ValueError("source original correctness must be binary")
        if e["same_visit_exact_addback"]!=e["baseline"]:
            raise ValueError("same-site exact addback must preserve original baseline class")
        for mode in MODES:
            counts[mode]+=e[mode]
        if row["eligible"]:
            admitted+=1
            if not row["delayed"] and (
                e["late_target_rescue"]!=e["remove_AB"]
                or e["late_matched_sham_rescue"]!=e["remove_AB"]):
                raise ValueError("last recurrence visit cannot deliver delayed rescue")
            joint+=e["remove_AB"]-e["remove_A"]-e["remove_B"]+e["baseline"]
            sham+=e["sham_CD"]-e["sham_C"]-e["sham_D"]+e["baseline"]
            rescue+=e["late_matched_sham_rescue"]-e["late_target_rescue"]
        elif any(v!=e["baseline"] for v in e.values()):
            raise ValueError("original invalid physical case must be all-mode no-op")
    return {
        "admissible_cases":admitted,
        "group_joint_error_excess":joint/256,
        "matched_sham_joint_error_excess":sham/256,
        "factorial_group_minus_sham_interaction":(joint-sham)/256,
        "delayed_target_minus_sham_rescue_benefit":rescue/256,
        "observed_binary_error_counts":counts,
    }


def verify(root:int,source:Path,t:Path,reset:Path|None,
           pack:Path,codes:Path,code_audit:Path,
           shortlist:Path,shortlist_audit:Path,
           physical:Path,physical_audit:Path,
           behavior:Path):
    from .nec03w_f01_behavior_registry import registered_behavior
    lock=registered_behavior()
    if type(root) is not int or root not in range(8):
        raise PermissionError("only historical source roots0..7 may be audited")
    orig=attest(root,source,t,reset)
    phy=verify_original_physical(
        pack,codes,code_audit,shortlist,shortlist_audit,physical)
    saved_phy=json.loads(physical.read_text("utf-8"))
    source_codes=json.loads(codes.read_text("utf-8"))
    meta,_,packsha=read_pack(pack)
    receipt_path=behavior/"factorial_source_calibration_report.json"
    raw=behavior/"factorial_source_calibration_raw.jsonl"
    report=json.loads(receipt_path.read_text("utf-8"))
    pin=lock["original_source_input_roots"][str(root)]
    if (report.get("schema")!="NEC-03W/F01/original-source-functional-calibration-factorial-behavior/v1"
            or report.get("world_index")!=root
            or report.get("world_id")!=orig["world_id"]
            or report.get("source_cases_per_arm")!=256
            or report.get("original_calibration_groups")!=list(range(16,24))
            or report.get("original_source_pack_manifest_sha256")!=packsha
            or report.get("source_code_calibration_sha256")!=file_hash(codes)
            or report.get("physical_roster_sha256")!=pin["physical_roster_sha256"]
            or report.get("physical_roster_file_sha256")!=file_hash(physical)
            or report.get("physical_independent_audit_file_sha256")!=file_hash(physical_audit)
            or report.get("source_F01_behavior_method_lock_sha256")!=file_hash(
                Path(__file__).with_name("NEC03W_F01_BEHAVIOR_LOCK.json"))
            or report.get("original_source_behavior_raw_sha256")!=file_hash(raw)
            or report.get("native_tied_recurrent_visits")!=12
            or report.get("new_original_source_behavior_native_forward_calls")!=160
            or report.get("new_optimizer_updates")!=0
            or report.get("source_quartet_selected_by_task_labels") is not False
            or report.get("source_query32_to39_outcomes_read") is not False
            or report.get("old_NEC03V_source24_to31_outcomes_read") is not False
            or report.get("recruited_reserved_embargoed_confirmation_reads")!=0
            or report.get("population_or_confirmation_claim") is not False
            or report.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or phy.get("independent_original_source_physical_audit_passed") is not True
            or set(report["factorial_outcome_by_arm"])!={"composed","additive"}):
        raise ValueError("original-source F01 behavior report, proof or cohort SHA changed")
    data=[json.loads(s) for s in raw.read_text("utf-8").splitlines() if s.strip()]
    if len(data)!=16 or len({(row["arm"],row["source_calibration_group"]) for row in data})!=16:
        raise ValueError("source F01 original two-arm eight calibration group archive incomplete")
    lookup={(row["arm"],row["source_calibration_group"]):row for row in data}
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("independent F01 original behavior audit requires CPU-only Torch")
    from .nec03r_source_recoverability import load_source_arm
    from .nec03l_five_input_worlds import labels
    from research.pef07.pef07 import as_tensor,model_digest
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    by_arm={}
    forwards={}
    maximum_abs_logit_difference=0.
    for arm in ("composed","additive"):
        frozen=load_source_arm(root,arm,source,reset=reset)
        model=frozen["model"].eval()
        orig_hash=model_digest(model)
        if (orig_hash!=orig["source_checkpoint_receipts"][arm]["native_sha256"]
                or frozen["sae_sha256"]!=orig["source_checkpoint_receipts"][arm]["sae_sha256"]):
            raise ValueError("independent F01 teacher or SAE original checkpoint changed")
        cohort=frozen["catalog"]["train"][512:768]
        expected_sha=digest([list(s) for s in cohort])
        if expected_sha!=source_codes["arms"][arm]["original_original_source_input_sha256"]:
            raise ValueError("independent original calibration bit histories changed")
        source_site=saved_phy["arms"][arm]["original_source_site"]
        selected_visit=source_site["visit"] if source_site else None
        code_rows=source_codes["arms"][arm]["positive_codes_by_visit"]
        original_rows=[]
        forwards[arm]=0
        for group_index,group in enumerate(range(16,24)):
            case=lookup[arm,group]
            group_bits=cohort[group_index*32:(group_index+1)*32]
            bits=as_tensor(group_bits)
            truth=labels(bits,frozen["world"],control=(arm=="additive")).tolist()
            if (case["source_input_histories"]!=[list(x) for x in group_bits]
                    or case["teacher_truth"]!=truth
                    or case["focal_mask_order"]!=list(range(32))
                    or case["world_index"]!=root):
                raise ValueError("original source F01 calibration input identity or truth changed")
            phys=saved_phy["arms"][arm]["cases"][group_index*32:(group_index+1)*32]
            if (case["original_physical_case_sha256"]!=[digest(row) for row in phys]
                    or case["eligible"]!=[r["geometry"]["admissible"] for r in phys]):
                raise ValueError("calibration source F01 committed physical case SHA or no-op mask changed")
            geom=[r["geometry"] for r in phys]
            tensor_codes={}
            for visit,rows in code_rows.items():
                z=torch.zeros((32,128),dtype=model.block.mlp_down.weight.dtype)
                for i,features in enumerate(rows[group_index*32:(group_index+1)*32]):
                    for feature,val in features:
                        z[i,int(feature)]=float(val)
                tensor_codes[int(visit)]=z
            if set(tensor_codes)!=set(frozen["source_sites"]):
                raise ValueError("independent original source SAE selected visits changed")
            preds={}
            errors={}
            for mode in MODES:
                donors,gates,late,addition=separate_donors(
                    frozen,tensor_codes,geom,selected_visit,mode)
                logits=original_trace(model,bits,donors,gates,late_visit=late,late_32=addition)
                expected=torch.tensor(case["mode_logits"][mode],dtype=logits.dtype)
                if tuple(expected.shape)!=(32,2):
                    raise ValueError("recorded original source F01 logits wrong shape")
                diff=float((logits-expected).abs().max())
                maximum_abs_logit_difference=max(maximum_abs_logit_difference,diff)
                if not torch.allclose(logits,expected,atol=1e-4,rtol=2e-5):
                    raise ValueError(
                        f"independently replayed original F01 source logits mismatch "
                        f"world={root} arm={arm} group={group} mode={mode} "
                        f"max_abs={diff:.9g} (fixed abs threshold 1e-4)")
                preds[mode]=logits.argmax(1).tolist()
                errors[mode]=[int(a!=b) for a,b in zip(preds[mode],truth)]
                forwards[arm]+=1
            for mask in range(32):
                original_rows.append({
                    "eligible":bool(geom[mask]["admissible"]),
                    "delayed":bool(geom[mask]["admissible"]
                                   and selected_visit is not None and selected_visit<11),
                    "errors":{mode:errors[mode][mask] for mode in MODES},
                })
        independent=independent_metrics(original_rows)
        scored=report["factorial_outcome_by_arm"][arm]
        original_factorial=scored["factorial"]
        for key in ("admissible_cases","group_joint_error_excess",
                    "matched_sham_joint_error_excess",
                    "factorial_group_minus_sham_interaction",
                    "delayed_target_minus_sham_rescue_benefit"):
            if not math.isclose(independent[key],original_factorial[key],
                                abs_tol=1e-9,rel_tol=1e-9):
                raise ValueError("independent F01 full-denominator factorial effect changed")
        if (scored["physical_admitted"]!=independent["admissible_cases"]
                or independent["observed_binary_error_counts"]!=original_factorial["observed_binary_error_counts"]
                or scored["correct_by_mode"]!={mode:256-n for mode,n in
                                              independent["observed_binary_error_counts"].items()}):
            raise ValueError("independently replayed original correct count ledger differs")
        if model_digest(model)!=orig_hash:
            raise ValueError("source F01 independent auditor mutated frozen native teacher")
        by_arm[arm]=independent
    if sum(forwards.values())!=160:
        raise ValueError("independent 2x2 original source forward-call count differs")
    effect=(by_arm["composed"]["factorial_group_minus_sham_interaction"]
            -by_arm["additive"]["factorial_group_minus_sham_interaction"])
    rescue=(by_arm["composed"]["delayed_target_minus_sham_rescue_benefit"]
            -by_arm["additive"]["delayed_target_minus_sham_rescue_benefit"])
    if (not math.isclose(effect,report["raw_root_composed_minus_additive_source_interaction"],abs_tol=1e-9)
            or not math.isclose(rescue,report["raw_root_composed_minus_additive_delayed_rescue"],abs_tol=1e-9)
            or report["registered_root_ITT_source_interaction"]!=(0. if root in (3,5) else effect)
            or report["root_fixed_original_ITT_zero"]!=(root in (3,5))):
        raise ValueError("original F01 primary and delayed control root effects differ")
    return {
        "schema":SCHEMA,"world_index":root,"world_id":orig["world_id"],
        "independent_original_source_factorial_replay_passed":True,
        "native_recurrent_passes":12,
        "source_case_arm_records_verified":512,
        "independent_source_native_forwards":sum(forwards.values()),
        "maximum_absolute_native_logit_replay_error":maximum_abs_logit_difference,
        "independent_root_composed_minus_additive_interaction":effect,
        "independent_root_composed_minus_additive_delayed_rescue":rescue,
        "independent_binary_error_metrics":by_arm,
        "behavior_original_raw_sha256":file_hash(raw),
        "behavior_original_report_sha256":file_hash(receipt_path),
        "physical_roster_file_sha256":file_hash(physical),
        "new_optimizer_updates":0,
        "new_source_query_group32_to39_reads":0,
        "recruited_embargoed_reserved_confirmation_reads":0,
        "population_or_confirmation_claim":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--independent-calibration-audit",type=Path,required=True)
    p.add_argument("--label-blind-shortlist",type=Path,required=True)
    p.add_argument("--independent-shortlist-audit",type=Path,required=True)
    p.add_argument("--physical",type=Path,required=True)
    p.add_argument("--independent-physical",type=Path,required=True)
    p.add_argument("--behavior",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("independent source F01 behavior audit output must be fresh")
    result=verify(a.world_index,a.source_root,a.nec03t_root,a.reset_root,
                  a.original_pack,a.calibration_codes,a.independent_calibration_audit,
                  a.label_blind_shortlist,a.independent_shortlist_audit,
                  a.physical,a.independent_physical,a.behavior)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(canonical(result)+"\n",encoding="utf-8")
    print(canonical(result))
    return result


if __name__=="__main__":
    main()
