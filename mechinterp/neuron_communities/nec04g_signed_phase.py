"""NEC-04G: read-only signed fifth-order XOR Fourier-phase audit.

Uses only previously accepted NEC-04F six-original-calibration-world archive
(roots 12..17), source/owned source-label ledger and independently replayed
native final logits. No model imports, new optimizer/inference, SAE training,
recruited/heldout/confirmation-world data or causal sham access.

Retrospective *diagnostic* of whether the unsigned 5-bit Walsh power in
NEC-04F represents correct teacher context phase. A teacher-informed
"phase oracle" is a posthoc analysis, never a feasible learned correction.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, isfinite, sqrt
from pathlib import Path

SCHEMA="NEC-04G/read-only-six-calibration-root-signed-XOR-phase/v1"
ROOTS=tuple(range(12,18))
ARMS=("aux_shuffled","aux_prefix","vanilla")
STEPS=(240,900)
SOURCE=2560
VALIDATION=768
FIFTH=31
NEAR_ZERO=1e-9
LOW_POWER=1e-20
HIGH_POWER_SHARE=.10


def file_sha(path:Path):
    return sha256(path.read_bytes()).hexdigest()


def require(value,why):
    if not value:
        raise AssertionError(why)


def sign5(mask):
    return -1 if mask.bit_count()%2 else 1


def walsh(margins):
    if len(margins)!=32 or any(type(v) not in (int,float) or not isfinite(v) for v in margins):
        raise ValueError("32 finite original class1-minus-class0 native logit margins required")
    return [fsum(margins[mask]*(-1 if (mask & subset).bit_count()%2 else 1)
                 for mask in range(32))/32 for subset in range(32)]


def analyze_context(source_rows,source_logits):
    """One ORIGINAL parity-context group with all 32 original focal masks."""
    if len(source_rows)!=32 or len(source_logits)!=32:
        raise ValueError("original parity context requires complete 32-mask array")
    mux={row["mux_selector"] for row in source_rows}
    require(mux=={1},"parity-only source context analysis cannot inspect alternate MUX")
    require([row["focal_mask"] for row in source_rows]==list(range(32)),
            "original source-parity 32-mask ordering altered")
    labels=[row["composed_label"] for row in source_rows]
    require(all(type(y) is int and y in (0,1) for y in labels),
            "original binary composed teacher required")
    phase=labels[0]
    require(all(label==((mask.bit_count()%2)^phase)
                for mask,label in enumerate(labels)),
            "original composed teacher context is not raw XOR with fixed phase")
    require(all("xor_prefix" not in row or row["xor_prefix"]["5"]==mask.bit_count()%2
                for mask,row in enumerate(source_rows)),
            "original auxiliary true 5-focal prefix disagrees with focal-mask XOR")
    margins=[]
    for raw in source_logits:
        if (not isinstance(raw,list) or len(raw)!=2
                or any(type(v) not in (int,float) or not isfinite(v) for v in raw)):
            raise ValueError("original model final native logits must be finite 2-class arrays")
        margins.append(raw[1]-raw[0])
    coefficient=walsh(margins)
    energy=fsum(c*c for c in coefficient[1:])
    amplitude=coefficient[FIFTH]
    # Expected sign of class-1 minus class-0 fifth Fourier coefficient:
    # For y(mask)=XOR(mask) xor phase, margin should be -chi_5 when phase0,
    # and +chi_5 when phase1. sign_target = 2*phase-1.
    sign_target=2*phase-1
    oriented=sign_target*amplitude
    if oriented>NEAR_ZERO:status="teacher_aligned"
    elif oriented< -NEAR_ZERO:status="teacher_inverted"
    else:status="near_zero"
    original=sum(int((value>0)==(truth==1))
                 for value,truth in zip(margins,labels))
    corrected_amp=sign_target*abs(amplitude)
    repaired=[value+(corrected_amp-amplitude)*sign5(mask)
              for mask,value in enumerate(margins)]
    corrected=sum(int((value>0)==(truth==1))
                  for value,truth in zip(repaired,labels))
    return {
        "context_phase_from_focal_mask_zero":phase,
        "original_true_parity_labels_balanced":sum(labels)==16,
        "fifth_order_unsigned_walsh_coefficient":amplitude,
        "teacher_signed_fifth_order_walsh_coefficient":oriented,
        "nonconstant_walsh_energy":energy,
        "fifth_order_unsigned_power_share":(
            amplitude*amplitude/energy if energy>LOW_POWER else None),
        "teacher_signed_normalized_fifth_coefficient":(
            oriented/sqrt(energy) if energy>LOW_POWER else None),
        "teacher_phase_status":status,
        "high_fifth_order_power_share":(
            energy>LOW_POWER and amplitude*amplitude/energy>=HIGH_POWER_SHARE),
        "original_correct_out_of_32":original,
        "posthoc_teacher_phase_oracle_correct_out_of_32":corrected,
        "original_context_sample_size":32,
    }


def summarize(contexts):
    require(contexts,"at least one frozen original parity context required")
    original=sum(x["original_correct_out_of_32"] for x in contexts)
    oracle=sum(x["posthoc_teacher_phase_oracle_correct_out_of_32"] for x in contexts)
    with_energy=[x for x in contexts if x["fifth_order_unsigned_power_share"] is not None]
    high=[x for x in contexts if x["high_fifth_order_power_share"]]
    return {
        "original_parity_context_count":len(contexts),
        "original_parity_case_denominator":32*len(contexts),
        "original_parity_correct":original,
        "original_parity_accuracy":original/(len(contexts)*32),
        "posthoc_teacher_phase_oracle_correct":oracle,
        "posthoc_teacher_phase_oracle_accuracy":oracle/(len(contexts)*32),
        "posthoc_oracle_minus_native_accuracy":(oracle-original)/(len(contexts)*32),
        "mean_unsigned_fifth_order_power_share":(
            fsum(x["fifth_order_unsigned_power_share"] for x in with_energy)/len(with_energy)
            if with_energy else None),
        "mean_teacher_signed_fifth_walsh_coefficient":(
            fsum(x["teacher_signed_fifth_order_walsh_coefficient"] for x in contexts)/len(contexts)),
        "mean_teacher_signed_normalized_fifth_coefficient":(
            fsum(x["teacher_signed_normalized_fifth_coefficient"] for x in with_energy)/len(with_energy)
            if with_energy else None),
        "phase_aligned_contexts":sum(x["teacher_phase_status"]=="teacher_aligned" for x in contexts),
        "phase_inverted_contexts":sum(x["teacher_phase_status"]=="teacher_inverted" for x in contexts),
        "phase_near_zero_contexts":sum(x["teacher_phase_status"]=="near_zero" for x in contexts),
        "high_power_contexts":len(high),
        "high_power_phase_inverted_contexts":sum(
            x["teacher_phase_status"]=="teacher_inverted" for x in high),
        "high_power_phase_aligned_contexts":sum(
            x["teacher_phase_status"]=="teacher_aligned" for x in high),
        "contexts":contexts,
    }


def analyze_validation(rows,logits):
    require(len(rows)==VALIDATION and len(logits)==SOURCE+VALIDATION,
            "all original source and owned calibration-world validation scores required")
    contexts=[]
    for context in range(VALIDATION//32):
        batch=rows[32*context:32*(context+1)]
        selectors={row["mux_selector"] for row in batch}
        require(len(selectors)==1,"original context MUX branch changed within 32 focal masks")
        if selectors=={1}:
            out=analyze_context(batch,logits[SOURCE+32*context:SOURCE+32*(context+1)])
            out["original_owned_validation_context_index"]=context
            contexts.append(out)
    return summarize(contexts)


def read_root(root:int,ledger:dict,pretraining:Path,folder:Path,ledger_sha:str):
    require(root in ROOTS,"source archive supplied a non-calibration root")
    ref=ledger["roots"][root-ROOTS[0]]
    require(ref["index"]==root,"original 6-root order changed")
    rows_file=pretraining/ref["rows_file"]
    require(file_sha(rows_file)==ref["rows_sha256"],"original source/owned label rows SHA changed")
    rows=[json.loads(line) for line in rows_file.read_text().splitlines() if line.strip()]
    require(len(rows)==SOURCE+VALIDATION,"original 80 source +24 owned validation context panel changed")
    original_path=folder/"nec04f_native_root_result.json"
    audit_path=folder/"nec04f_independent_native_audit.json"
    original=json.loads(original_path.read_text())
    audit=json.loads(audit_path.read_text())
    require(
        original.get("schema")=="NEC-04F/three-arm-recurrent-auxiliary-XOR-native-900steps/v1"
        and audit.get("schema")=="NEC-04F/independent-three-arm-native-optimizer-aux-logit-replay/v1"
        and original.get("root_index")==audit.get("root_index")==root
        and original.get("world_id")==audit.get("world_id")==ref["world_id"]
        and original.get("world_sha256")==audit.get("world_sha256")==ref["world_sha256"]
        and original.get("original_program_sha256")==ref["program_sha256"]
        and original.get("original_source_validation_sha256")==ref["rows_sha256"]
        and audit.get("original_source_and_validation_sha256")==ref["rows_sha256"]
        and original.get("pretraining_oracle_ledger_sha256")==ledger_sha
        and audit.get("original_pretraining_ledger_sha256")==ledger_sha
        and audit.get("original_root_scored_receipt_sha256")==file_sha(original_path)
        and original.get("source_cases")==SOURCE
        and original.get("owned_calibration_validation_cases")==VALIDATION
        and original.get("optimizer_updates_per_arm")==900
        and original.get("native_initial_weights_shared_all_arms") is True
        and original.get("source_batch_order_matched_all_arms") is True
        and original.get("model_checkpoints_sealed_before_validation") is True
        and audit.get("all_six_native_model_optimizer_checkpoints_byte_sha256_verified") is True
        and audit.get("all_six_native_full_and_four_auxiliary_logits_independently_replayed") is True
        and audit.get("all_six_continuous_optimizer_state_counters_verified") is True
        and audit.get("native_checkpoints_replayed")==6
        and original.get("new_source_SAE_training_updates")==0
        and audit.get("native_SAE_updates")==0
        and original.get("old_nec03p_roots_scored")==0
        and original.get("development_world_roots_materialized")==0
        and original.get("confirmation_world_roots_materialized")==0
        and original.get("original_calibration_reserved_contexts_scored")==0
        and original.get("FSI_01_gate")==audit.get("FSI_01_gate")=="REDESIGN_OR_PILOT_ONLY"
        and set(original["arms"])==set(audit["arms"])==set(ARMS),
        "original NEC04F native checkpoint/root/split/independent replay custody changed",
    )
    data={}
    for arm in ARMS:
        obs=original["arms"][arm]["observations"]
        reviewed=audit["arms"][arm]
        require(len(obs)==len(reviewed)==2,"original 240/900 native checkpoint set changed")
        for i,step in enumerate(STEPS):
            item=obs[i]
            prior=reviewed[i]
            require(item["step"]==prior["step"]==step
                    and item["original_checkpoint_sha256"]==prior["original_checkpoint_sha256"]
                    and item["original_model_sha256"]==prior["model_weight_sha256"]
                    and prior["all_source_owned_native_12step_and_aux_logits_replayed"] is True,
                    "original final native + four auxiliary logits not independently replayed")
            result=analyze_validation(rows[SOURCE:],item["all_final_native_logits"])
            expected=item["source_and_owned_validation_scores"]["owned_calibration_validation"][
                "native_final_by_mux_branch"]["parity"]
            require(result["original_parity_case_denominator"]==expected["cases"]
                    and result["original_parity_correct"]==expected["correct"],
                    "signed source spectrum disagrees with previously accepted native accuracy")
            data.setdefault(arm,{})[str(step)]=result
    return {
        "root_index":root,"world_id":ref["world_id"],"owned_rows_sha256":ref["rows_sha256"],
        "native_root_result_sha256":file_sha(original_path),
        "independent_native_replay_sha256":file_sha(audit_path),
        "models_and_optimizers_retrained":False,
        "model_final_logits_independently_replayed_in_NEC04F":True,
        "analysis":data,
    }


def aggregate(pretraining:Path,roots:Path):
    ledger_file=pretraining/"nec04f_pretraining_oracles.json"
    ledger=json.loads(ledger_file.read_text())
    require(
        ledger.get("schema")=="NEC-04F/original-six-calibration-root-auxiliary-teacher-ledger/v1"
        and ledger.get("root_indices")==list(ROOTS)
        and ledger.get("training_arms")==list(ARMS)
        and ledger.get("checkpoint_steps")==list(STEPS)
        and ledger.get("original_source_rows_per_root")==SOURCE
        and ledger.get("original_owned_validation_rows_per_root")==VALIDATION
        and ledger.get("optimizer_updates_before_preflight")==0
        and ledger.get("FSI_01_gate")=="REDESIGN_OR_PILOT_ONLY"
        and len(ledger.get("roots",[]))==6,
        "original source/owned pretraining ledger or sealed population changed")
    require({p.name for p in roots.iterdir() if p.is_dir()}==
            {f"nec04f-native-root-{i}" for i in ROOTS},
            "exact six original independently audited calibration-world output folders required")
    ledger_sha=file_sha(ledger_file)
    found=[read_root(i,ledger,pretraining,roots/f"nec04f-native-root-{i}",ledger_sha)
           for i in ROOTS]
    comparison={}
    for step in STEPS:
        snapshots={}
        for arm in ARMS:
            individual=[r["analysis"][arm][str(step)] for r in found]
            keys=("original_parity_accuracy","posthoc_teacher_phase_oracle_accuracy",
                  "posthoc_oracle_minus_native_accuracy",
                  "mean_unsigned_fifth_order_power_share",
                  "mean_teacher_signed_fifth_walsh_coefficient",
                  "mean_teacher_signed_normalized_fifth_coefficient")
            snapshots[arm]={
                key:fsum(v[key] for v in individual)/6
                if all(v[key] is not None for v in individual) else None
                for key in keys
            }
            snapshots[arm]["phase_inverted_contexts_total"]=sum(
                v["phase_inverted_contexts"] for v in individual)
            snapshots[arm]["high_power_phase_inverted_contexts_total"]=sum(
                v["high_power_phase_inverted_contexts"] for v in individual)
            snapshots[arm]["high_power_contexts_total"]=sum(
                v["high_power_contexts"] for v in individual)
            snapshots[arm]["parity_contexts_total"]=sum(
                v["original_parity_context_count"] for v in individual)
            snapshots[arm]["by_original_root"]=[
                {"root_index":r["root_index"],
                 "unsigned_power":v["mean_unsigned_fifth_order_power_share"],
                 "signed_normalized":v["mean_teacher_signed_normalized_fifth_coefficient"],
                 "inverted_contexts":v["phase_inverted_contexts"],
                 "total_contexts":v["original_parity_context_count"],
                 "native_parity_accuracy":v["original_parity_accuracy"],
                 "oracle_parity_accuracy":v["posthoc_teacher_phase_oracle_accuracy"],
                 "high_power_inverted":v["high_power_phase_inverted_contexts"]}
                for r,v in zip(found,individual)]
        comparison[str(step)]=snapshots
    return {
        "schema":SCHEMA,"original_calibration_root_indices":list(ROOTS),
        "model_arms":list(ARMS),"continuous_checkpoint_steps":list(STEPS),
        "original_pretraining_ledger_sha256":ledger_sha,
        "original_source_cases_per_root":SOURCE,
        "owned_calibration_validation_cases_per_root":VALIDATION,
        "fifth_order_subset_index":FIFTH,
        "teacher_sign_rule":"focal-mask-zero label y0 gives target sign 2*y0-1",
        "high_power_share_descriptor_threshold":HIGH_POWER_SHARE,
        "phase_near_zero_absolute_coeff_threshold":NEAR_ZERO,
        "comparison":comparison,
        "roots":found,
        "original_NEC04F_primary_negative_preserved":True,
        "phase_oracle_uses_validation_truth_posthoc":True,
        "phase_oracle_is_not_trained_or_deployable":True,
        "original_source_checkpoint_states_only":True,
        "new_model_forwards":0,"new_optimizer_updates":0,
        "new_SAE_fitting":False,"new_causal_sham_interventions":0,
        "new_development_worlds":0,"new_confirmation_worlds":0,
        "new_reserved_contexts":0,"old_NEC03P_recruited_root_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "population_p_value_or_feature_necessity_claim":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretraining",type=Path,required=True)
    p.add_argument("--roots",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("NEC04G analysis must create a new immutable output path")
    report=aggregate(a.pretraining,a.roots)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    last=report["comparison"]["900"]
    print("NEC04G_SIX_ORIGINAL_CALIBRATION_SIGNED_PHASE_READONLY",
          json.dumps({
            "signed_aux_prefix":last["aux_prefix"]["mean_teacher_signed_normalized_fifth_coefficient"],
            "signed_aux_shuffled":last["aux_shuffled"]["mean_teacher_signed_normalized_fifth_coefficient"],
            "phase_inverted_aux_prefix":last["aux_prefix"]["phase_inverted_contexts_total"],
            "root_phase":last["aux_prefix"]["by_original_root"],
            "new_model_forwards":0,
            "new_optimizer_updates":0,
            "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
          },sort_keys=True))
    return report


if __name__=="__main__":
    main()
