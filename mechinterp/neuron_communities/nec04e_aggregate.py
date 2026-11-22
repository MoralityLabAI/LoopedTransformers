"""NEC-04E pure-stdlib six-world hidden parity recoverability aggregate.

Requires complete preregistered 6×2×2×12 source-only readout and independently
replayed original frozen task-model checkpoints. One calibration WORLD is
the statistical unit, not its 32-mask context observations. Descriptive
only; never promote the NEC04B/04C headroom, SAE or FSI mechanism gates.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path

ROOTS=tuple(range(6,12))
ARMS=("vanilla","parity_curriculum")
BUDGETS=(240,900)
REPRESENTATIONS=(
    "trained_hidden", "random_initial_hidden", "visible_raw_bits",
    "source_shuffled_trained_hidden",
)
SCHEMA="NEC-04E/six-world-12visit-parity-memory-readout-aggregate/v1"


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def check(ok,msg):
    if not ok:
        raise AssertionError(msg)


def summary(numbers):
    if not numbers:
        raise AssertionError("empty original independent-root diagnostic")
    avg=sum(numbers)/len(numbers)
    var=(sum((x-avg)**2 for x in numbers)/(len(numbers)-1)
         if len(numbers)>1 else 0.)
    return {
        "root_count":len(numbers),
        "mean":avg,
        "root_sd_descriptive":sqrt(var),
        "root_standard_error_descriptive":sqrt(var/len(numbers)),
        "original_values_in_root_order":numbers,
    }


def pick(model,visit,task,kind="trained_hidden"):
    result=model["visits"][visit-1]
    check(result["visit"]==visit,"original recurrent visit omitted")
    return result["tasks"][task][kind]["owned_validation_metrics"]["accuracy"]


def root_snapshot(rec,arm,step):
    idx=BUDGETS.index(step)
    model=rec["arms"][arm][idx]
    check(model["step"]==step and len(model["visits"])==12,
          "original two-budget or 12-step readout absent")
    focal=rec["original_focal_positions"]
    all_seen=max(focal)+1
    score={kind:pick(model,all_seen,"prefix_current",kind)
           for kind in REPRESENTATIONS}
    final={kind:pick(model,12,"full_parity_teacher",kind)
           for kind in REPRESENTATIONS}
    first=sorted(focal)[0]+1
    second=sorted(focal)[1]+1
    future=max(focal)
    one={kind:pick(model,first,"prefix1",kind)
         for kind in REPRESENTATIONS}
    two={kind:pick(model,second,"prefix2",kind)
         for kind in REPRESENTATIONS}
    negative={kind:pick(model,future,"future_unseen",kind)
              for kind in REPRESENTATIONS}
    contrast=score["trained_hidden"]-max(
        score["random_initial_hidden"],score["visible_raw_bits"]
    )
    passroot=score["trained_hidden"]>=.70 and contrast>=.10
    return {
        "root_index":rec["root_index"],
        "world_id":rec["world_id"],
        "first_focal_seen_visit":first,
        "second_focal_seen_visit":second,
        "all_five_focal_seen_visit":all_seen,
        "prefix1_first_seen_validation":one,
        "prefix2_first_complete_validation":two,
        "prefix5_first_complete_validation":score,
        "future_last_unseen_bit_validation":negative,
        "final_original_parity_target_visit12_validation":final,
        "prefix5_trained_minus_best_random_raw":contrast,
        "individual_five_bit_signal_gate":passroot,
        "native_logit_replay_max_abs_error":
            model["native_final_head_replayed_max_absolute_logit_error"],
    }


def aggregate(roots:Path,preflight:Path,out:Path):
    prefile=preflight/"nec04e_preinference_lock.json"
    pre=json.loads(prefile.read_text())
    check(pre["schema"]==
          "NEC-04E/preinference-original-six-root-native-checkpoint-receipts/v1"
          and pre["root_indices"]==list(ROOTS)
          and pre["native_arms"]==list(ARMS)
          and pre["original_native_checkpoint_budgets"]==list(BUDGETS)
          and pre["all_24_historical_native_checkpoint_byte_shas_verified"]
          and pre["six_prior_native_full_logit_audits_reverified"]
          and pre["new_hidden_inference_executed_at_preflight"] is False
          and pre["new_task_model_training_updates"]==0,
          "original frozen code/model preinference source invalid")
    check({p.name for p in roots.iterdir() if p.is_dir()}==
          {f"nec04e-probe-root-{i}" for i in ROOTS},
          "all six fixed original calibration world readouts and audits required")
    received=[]
    total_readouts=0
    for index in ROOTS:
        folder=roots/f"nec04e-probe-root-{index}"
        recorded_file=folder/"nec04e_loopwise_hidden_probe_root.json"
        audit_file=folder/"nec04e_independent_loopwise_probe_audit.json"
        record=json.loads(recorded_file.read_text())
        independently=json.loads(audit_file.read_text())
        ref=pre["roots"][str(index)]
        check(
            record["schema"]==
            "NEC-04E/source-only-loopwise-32D-parity-linear-probe/v1"
            and independently["schema"]==
            "NEC-04E/independent-all-visits-native-ridge-readout-replay/v1"
            and record["root_index"]==independently["root_index"]==index
            and record["world_id"]==independently["world_id"]==ref["world_id"]
            and record["world_sha256"]==independently["original_calibration_world_sha256"]
            ==ref["world_sha256"]
            and record["original_source_validation_sha256"]
            ==independently["original_source_validation_file_sha256"]
            ==ref["source_validation_sha256"]
            and record["original_native_score_sha256"]==ref["original_score_receipt_sha256"]
            and record["original_native_replay_sha256"]
            ==ref["original_independent_replay_sha256"]
            and record["same_seed_random_initial_weight_sha256"]
            ==ref["initial_weights_sha256"]
            and record["preinference_receipt_sha256"]
            ==independently["preinference_lock_receipt_sha256"]==sha(prefile)
            and independently["original_score_receipt_sha256"]==sha(recorded_file)
            and independently["native_checkpoints_replayed"]==4
            and independently["all_12_original_recurrent_states_independently_replayed"]
            and independently["all_source_fit_probe_labels_and_predictions_replayed"]
            and independently["source_only_linear_readout_fits_recomputed"]>0
            and record["all_original_checkpoints_replayed_before_probe_scoring"]
            and record["source_only_ridge_fit"]
            and record["new_native_optimizer_updates"]==0
            and independently["trained_task_model_optimizer_updates"]==0
            and record["source_SAE_training_updates"]==
            independently["source_SAE_training_updates"]==0
            and all(record[k]==0 for k in (
                "reserved_contexts_scored",
                "development_world_root_specs_materialized",
                "confirmation_world_root_specs_materialized",
                "old_NEC03P_recruited_worlds_scored",
            ))
            and all(independently[k]==0 for k in (
                "reserved_contexts_scored",
                "development_WORLD_roots_materialized",
                "confirmation_WORLD_roots_materialized",
                "old_NEC03P_causal_sham_worlds_scored",
            ))
            and record["FSI_01_gate"]==independently["FSI_01_gate"]
            =="REDESIGN_OR_PILOT_ONLY"
            and set(record["arms"])==set(ARMS)
            and set(independently["arms"])==set(ARMS),
            "source-only root readout or independent native checkpoint audit incomplete"
        )
        for arm in ARMS:
            check(len(record["arms"][arm])==len(BUDGETS)
                  and len(independently["arms"][arm])==len(BUDGETS),
                  "root model checkpoint roster changed")
            for position,step in enumerate(BUDGETS):
                model=record["arms"][arm][position]
                audit=independently["arms"][arm][position]
                required=ref["original_model_checkpoints"][arm][str(step)]
                check(
                    model["step"]==audit["step"]==step
                    and model["original_checkpoint_byte_sha256"]
                    ==audit["original_checkpoint_sha256"]
                    ==required["checkpoint_sha256"]
                    and model["model_weights_sha256"]==required["model_sha256"]
                    and audit["all_12_original_states_byte_sha256_reproduced"]
                    and audit["independently_replayed_source_only_ridge_probes"]>0
                    and len(model["visits"])==12,
                    "original trained native model, visits or controls omitted"
                )
                total_readouts+=audit["independently_replayed_source_only_ridge_probes"]
        received.append({
            "root_index":index,
            "world_id":record["world_id"],
            "root_result_sha256":sha(recorded_file),
            "independent_native_replay_sha256":sha(audit_file),
            "original_source_validation_sha256":ref["source_validation_sha256"],
            "original_focal_positions":record["original_focal_positions"],
            "arms":record["arms"],
        })
    macro={}
    for arm in ARMS:
        snapshots=[]
        for step in BUDGETS:
            root_rows=[root_snapshot(r,arm,step) for r in received]
            values={
                "prefix5_trained_hidden":[r["prefix5_first_complete_validation"]["trained_hidden"]
                                          for r in root_rows],
                "prefix5_random_initial_hidden":[r["prefix5_first_complete_validation"]["random_initial_hidden"]
                                                   for r in root_rows],
                "prefix5_visible_raw_bits":[r["prefix5_first_complete_validation"]["visible_raw_bits"]
                                             for r in root_rows],
                "prefix5_shuffled_source_labels":[r["prefix5_first_complete_validation"]["source_shuffled_trained_hidden"]
                                                   for r in root_rows],
                "prefix5_trained_minus_best_control":[r["prefix5_trained_minus_best_random_raw"]
                                                       for r in root_rows],
                "final_teacher_trained_hidden":[r["final_original_parity_target_visit12_validation"]["trained_hidden"]
                                                for r in root_rows],
                "future_last_unseen_trained_hidden":[r["future_last_unseen_bit_validation"]["trained_hidden"]
                                                    for r in root_rows],
                "low_order_prefix1_trained_hidden":[r["prefix1_first_seen_validation"]["trained_hidden"]
                                                    for r in root_rows],
                "low_order_prefix2_trained_hidden":[r["prefix2_first_complete_validation"]["trained_hidden"]
                                                    for r in root_rows],
            }
            stats={k:summary(v) for k,v in values.items()}
            positive=sum(r["individual_five_bit_signal_gate"] for r in root_rows)
            qualified=(
                step==900
                and positive>=4
                and stats["prefix5_trained_hidden"]["mean"]>=.70
                and stats["prefix5_trained_minus_best_control"]["mean"]>=.10
            )
            snapshots.append({
                "step":step,
                "root_count":6,
                "qualified_readout_roots":positive,
                "registered_minimum_qualified_roots":4,
                "registered_minimum_validation_accuracy":.70,
                "registered_minimum_gain_vs_controls":.10,
                "meets_source_fitted_parity_recoverability_gate":qualified,
                "root_macro_descriptive":stats,
                "per_root":root_rows,
            })
        macro[arm]=snapshots
    # Predeclared primary diagnostic: 900-update curriculum arm, complete
    # five-focal prefix parity at the first original recurrent visit with
    # all five focal bits observed. The vanilla arm remains a secondary.
    gate=macro["parity_curriculum"][-1]["meets_source_fitted_parity_recoverability_gate"]
    verdict=(
        "CALIBRATION_HIDDEN_PREFIX5_LINEAR_READOUT_SIGNAL_REQUIRES_FRESH_PROTOCOL"
        if gate else "NO_QUALIFIED_HIDDEN_PREFIX5_LINEAR_READOUT_SIGNAL"
    )
    result={
        "schema":SCHEMA,
        "preinference_lock_receipt_sha256":sha(prefile),
        "root_indices":list(ROOTS),
        "native_arms":list(ARMS),
        "existing_native_optimizer_budgets":list(BUDGETS),
        "statistical_unit":"one previously opened NEC04A calibration world root",
        "independent_world_roots":6,
        "original_native_model_checkpoints_verified":24,
        "model_checkpoints_independently_replayed_12_visits":24,
        "source_only_linear_readout_fits_independently_replayed":total_readouts,
        "source_SAE_models_trained":0,
        "new_task_model_optimizer_updates":0,
        "new_world_root_outcomes":0,
        "reserved_contexts_scored":0,
        "development_WORLD_roots_materialized":0,
        "confirmation_WORLD_roots_materialized":0,
        "old_NEC03P_causal_sham_roots_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "per_root_original_evidence":received,
        "root_macro_diagnostics":macro,
        "decision":verdict,
        "source_SAE_or_causal_campaign_promoted":False,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec04e_six_root_loopwise_probe_aggregate.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    last=macro["parity_curriculum"][-1]
    print("NEC04E_INDEPENDENT_SIX_WORLD_12VISIT_READOUT_RESULT",
          "trained_prefix5",last["root_macro_descriptive"]["prefix5_trained_hidden"]["mean"],
          "random",last["root_macro_descriptive"]["prefix5_random_initial_hidden"]["mean"],
          "raw",last["root_macro_descriptive"]["prefix5_visible_raw_bits"]["mean"],
          "qualifying",last["qualified_readout_roots"],
          "decision",verdict,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--roots",type=Path,required=True)
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    aggregate(args.roots,args.preflight,args.out)


if __name__=="__main__":
    main()
