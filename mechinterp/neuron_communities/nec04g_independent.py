"""NEC-04G second implementation: FWHT and teacher-sign arithmetic replay.

Reads the same frozen NEC04F independently checkpoint-audited original
source/owned rows and native logits. Does not import NEC04G primary scorer.
Computes the entire 32-vector spectrum via an in-place Walsh-Hadamard
butterfly and independently checks every source-parity context outcome,
teacher sign and posthoc phase-oracle counterfactual.

This is read-only CPU/std-lib; no trained model or new teacher generation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, sqrt, isclose
from pathlib import Path

ROOTS=tuple(range(12,18))
ARMS=("aux_shuffled","aux_prefix","vanilla")
STEPS=(240,900)
NATIVE_SOURCE=2560
OWNED=768


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def require(ok,why):
    if not ok:
        raise AssertionError(why)


def hadamard(values):
    if len(values)!=32:
        raise ValueError("independent Walsh is defined on all five original focal bits")
    values=[float(x) for x in values]
    stride=1
    while stride<32:
        for start in range(0,32,2*stride):
            for j in range(stride):
                i=start+j
                a,b=values[i],values[i+stride]
                values[i]=a+b
                values[i+stride]=a-b
        stride*=2
    return [x/32 for x in values]


def context(source,logits):
    require(len(source)==len(logits)==32,"independent original full 32-mask parity group required")
    require([x["focal_mask"] for x in source]==list(range(32))
            and all(x["mux_selector"]==1 for x in source),
            "independent context mask order or source MUX branch differs")
    truth=[int(x["composed_label"]) for x in source]
    require(all(x in (0,1) for x in truth),"binary original context teacher required")
    phase=truth[0]
    require(all(truth[mask]==((mask.bit_count()&1)^phase) for mask in range(32)),
            "original source context teacher phase does not match actual focal XOR")
    m=[float(x[1])-float(x[0]) for x in logits]
    spectrum=hadamard(m)
    power=fsum(x*x for x in spectrum[1:])
    raw_correct=sum((m[i]>0)==(truth[i]==1) for i in range(32))
    raw=spectrum[31]
    teacher=2*phase-1
    corrected=teacher*abs(raw)
    oracle=sum((m[i]+(corrected-raw)*(-1 if i.bit_count()&1 else 1)>0)==(truth[i]==1)
               for i in range(32))
    return {
        "phase":phase,
        "walsh_31":raw,
        "signed":raw*teacher,
        "power":power,
        "share":raw*raw/power if power>1e-20 else None,
        "signed_norm":raw*teacher/sqrt(power) if power>1e-20 else None,
        "original_correct":raw_correct,
        "oracle_correct":oracle,
        "status":("teacher_aligned" if raw*teacher>1e-9
                  else "teacher_inverted" if raw*teacher< -1e-9
                  else "near_zero"),
    }


def audit(pretraining:Path,roots:Path,report_path:Path):
    ledger_path=pretraining/"nec04f_pretraining_oracles.json"
    ledger=json.loads(ledger_path.read_text())
    report=json.loads(report_path.read_text())
    require(report.get("schema")=="NEC-04G/read-only-six-calibration-root-signed-XOR-phase/v1"
            and report.get("original_calibration_root_indices")==list(ROOTS)
            and report.get("model_arms")==list(ARMS)
            and report.get("continuous_checkpoint_steps")==list(STEPS)
            and report.get("original_pretraining_ledger_sha256")==sha(ledger_path)
            and report.get("phase_oracle_uses_validation_truth_posthoc") is True
            and report.get("phase_oracle_is_not_trained_or_deployable") is True
            and report.get("new_model_forwards")==report.get("new_optimizer_updates")==0
            and report.get("new_development_worlds")==report.get("new_confirmation_worlds")==0
            and report.get("new_reserved_contexts")==report.get("old_NEC03P_recruited_root_reads")==0
            and report.get("population_p_value_or_feature_necessity_claim") is False
            and report.get("scientific_gate")=="REDESIGN_OR_PILOT_ONLY",
            "registered original six-world posthoc report or protected-cohort declaration changed")
    require(set(str(x["root_index"]) for x in report["roots"])==set(str(i) for i in ROOTS)
            and len(report["roots"])==6,
            "all six original root receipts required")
    by_root={x["root_index"]:x for x in report["roots"]}
    audited_contexts=0
    original_predictions=0
    for root in ROOTS:
        ref=ledger["roots"][root-12]
        file=pretraining/ref["rows_file"]
        require(sha(file)==ref["rows_sha256"],"original source/owned label row file SHA drift")
        labels=[json.loads(x) for x in file.read_text().splitlines() if x.strip()]
        require(len(labels)==NATIVE_SOURCE+OWNED,"original source/owned panel incomplete")
        folder=roots/f"nec04f-native-root-{root}"
        original_file=folder/"nec04f_native_root_result.json"
        audit_file=folder/"nec04f_independent_native_audit.json"
        payload=json.loads(original_file.read_text())
        prior=json.loads(audit_file.read_text())
        result=by_root[root]
        require(result["native_root_result_sha256"]==sha(original_file)
                and result["independent_native_replay_sha256"]==sha(audit_file)
                and prior.get("original_root_scored_receipt_sha256")==sha(original_file)
                and prior.get("all_six_native_full_and_four_auxiliary_logits_independently_replayed") is True
                and prior.get("all_six_native_model_optimizer_checkpoints_byte_sha256_verified") is True
                and payload.get("old_nec03p_roots_scored")==0
                and payload.get("original_calibration_reserved_contexts_scored")==0,
                "original archived independent native result identity changed")
        for arm in ARMS:
            obs=payload["arms"][arm]["observations"]
            for step_idx,step in enumerate(STEPS):
                saved=result["analysis"][arm][str(step)]
                raw=obs[step_idx]["all_final_native_logits"]
                require(len(raw)==NATIVE_SOURCE+OWNED,"original signed Fourier has incomplete native logits")
                existing=saved["contexts"]
                computed=[]
                for j in range(24):
                    source=labels[NATIVE_SOURCE+32*j:NATIVE_SOURCE+32*(j+1)]
                    if source[0]["mux_selector"]!=1:
                        continue
                    derived=context(source,raw[NATIVE_SOURCE+32*j:NATIVE_SOURCE+32*(j+1)])
                    found=existing[len(computed)]
                    computed.append(derived)
                    require(found["original_owned_validation_context_index"]==j
                            and found["context_phase_from_focal_mask_zero"]==derived["phase"]
                            and found["teacher_phase_status"]==derived["status"]
                            and found["original_correct_out_of_32"]==derived["original_correct"]
                            and found["posthoc_teacher_phase_oracle_correct_out_of_32"]==derived["oracle_correct"],
                            "independent source teacher context, corrected oracle or prediction differs")
                    for original_key,new_key in (
                        ("fifth_order_unsigned_walsh_coefficient","walsh_31"),
                        ("teacher_signed_fifth_order_walsh_coefficient","signed"),
                        ("nonconstant_walsh_energy","power"),
                        ("fifth_order_unsigned_power_share","share"),
                        ("teacher_signed_normalized_fifth_coefficient","signed_norm"),
                    ):
                        reported=found[original_key]
                        fresh=derived[new_key]
                        if fresh is None:
                            require(reported is None,"zero-variance fifth-order phase cannot be scored")
                        else:
                            require(reported is not None
                                    and isclose(float(reported),fresh,rel_tol=1e-9,abs_tol=1e-9),
                                    "independent FWHT disagrees with original fifth-order phase or power")
                    require(found["high_fifth_order_power_share"]==
                            (derived["share"] is not None and derived["share"]>=.10),
                            "predeclared posthoc 10-percent high-power diagnostic changed")
                    audited_contexts+=1
                    original_predictions+=32
                require(len(computed)==saved["original_parity_context_count"],
                        "original parity-context denominator changed")
                require(saved["original_parity_correct"]==
                        sum(x["original_correct"] for x in computed),
                        "source original native accuracy changed")
                require(saved["phase_inverted_contexts"]==
                        sum(x["status"]=="teacher_inverted" for x in computed),
                        "source teacher-phase inversion counting changed")
                require(saved["posthoc_teacher_phase_oracle_correct"]==
                        sum(x["oracle_correct"] for x in computed),
                        "posthoc teacher-informed phase oracle score changed")
    return {
        "schema":"NEC-04G/independent-scalar-FWHT-six-root-audit/v1",
        "original_calibration_roots":list(ROOTS),
        "native_arms":list(ARMS),
        "original_checkpoints":list(STEPS),
        "original_owned_parity_context_arm_checkpoint_records":audited_contexts,
        "original_owned_parity_logit_pairs_rechecked":original_predictions,
        "read_only_fifth_Walsh_FHWT_and_teacher_phase_replay_passed":True,
        "original_signed_phase_report_sha256":sha(report_path),
        "no_training_or_new_native_inference":True,
        "new_model_forwards":0,"new_optimizer_steps":0,
        "reserved_development_confirmation_or_recruited_roots_scored":0,
        "root_level_population_inference":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretraining",type=Path,required=True)
    p.add_argument("--roots",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("NEC04G independent FWH transform audit must use a fresh output")
    result=audit(args.pretraining,args.roots,args.report)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NEC04G_SIX_ROOT_INDEPENDENT_FHWT_TEACHER_SIGN_AUDIT_PASS",
          json.dumps(result,sort_keys=True))
    return result


if __name__=="__main__":
    main()
