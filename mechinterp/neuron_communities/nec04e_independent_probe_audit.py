"""NEC-04E independent native recurrent replay and source-ridge score audit.

Reconstruct each original trained and seeded-untrained 12-visit Looper state,
refit linear readouts from old source-only parity-context labels and verify
every root/arm/budget/visit target/control saved prediction and accuracy.
No original model training, no new root, no reserved-context read.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION, seed_for_root,
)
from mechinterp.neuron_communities.nec04d_parity_walsh_audit import checked_original_rows
from mechinterp.neuron_communities.nec04e_hidden_probe import (
    SCHEMA as SCORE_SCHEMA, VISITS, TARGETS, REPRESENTATIONS,
)
from mechinterp.neuron_communities.nec04e_preflight import (
    lock, code_identity, sha, require, SCHEMA as PREFLIGHT_SCHEMA,
)
from research.pef07.pef07 import Looper, model_digest, as_tensor

AUDIT_SCHEMA="NEC-04E/independent-all-visits-native-ridge-readout-replay/v1"
CHUNK=128
CUTOFF=.5
PRED_TIE_TOL=1e-8
REG=1.
STD_FLOOR=1e-4


def tensor_sha(t: torch.Tensor) -> str:
    return sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_shuffle(index: int, visit: int, target: str, n: int):
    seed = int.from_bytes(
        sha256(
            b"NEC04E/source-only-label-shuffle/v1|"
            + str(index).encode() + b"|" + str(visit).encode()
            + b"|" + target.encode()
        ).digest()[:8], "big"
    ) % (2**63-1)
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def own_targets(rows: list[dict], focal: list[int], visit: int):
    ordered = sorted(focal)
    seen = [b for b in ordered if b < visit]
    future = [b for b in ordered if b >= visit]
    tasks = {"full_parity_teacher": [
        row["composed_label"] for row in rows
    ]}
    if seen:
        tasks["prefix1"] = [
            row["history_bits12"][seen[0]] for row in rows
        ]
    if len(seen)>=2:
        tasks["prefix2"] = [
            (row["history_bits12"][seen[0]] ^
             row["history_bits12"][seen[1]]) for row in rows
        ]
        tasks["prefix_current"] = [
            sum(row["history_bits12"][bit] for bit in seen) % 2 for row in rows
        ]
    if future:
        tasks["future_unseen"] = [
            sum(row["history_bits12"][bit] for bit in future) % 2 for row in rows
        ]
    return tasks


@torch.no_grad()
def independent_hidden_states(model: Looper, bits: torch.Tensor):
    # Re-executes the original network's Looper.step in chronological order.
    model.eval()
    all_steps = [[] for _ in range(12)]
    ends=[]
    check=0.
    for start in range(0,len(bits),CHUNK):
        chunk=bits[start:start+CHUNK]
        memory=model.memory_init.reshape(1,-1).expand(len(chunk),-1)
        for bit_idx in range(12):
            e=model.bit_embedding(chunk[:,bit_idx])
            memory=model.step(memory,e)
            all_steps[bit_idx].append(memory.detach().cpu())
        ends.append(model.head(memory).detach().cpu())
        check=max(check,float((memory-model.memory(chunk)).abs().max()))
    return [torch.cat(xs) for xs in all_steps], torch.cat(ends), check


def independent_score(pred: torch.Tensor, truth: torch.Tensor):
    pred=pred.long().cpu()
    truth=truth.long().cpu()
    n=len(truth)
    positives=int((truth==1).sum())
    require(n==len(pred) and 0<positives<n,"audited probe target not binary/balanced")
    tp=int(((pred==1)&(truth==1)).sum())
    tn=int(((pred==0)&(truth==0)).sum())
    return {
        "cases":n,
        "correct":tp+tn,
        "accuracy":(tp+tn)/n,
        "balanced_accuracy":.5*(tp/positives+tn/(n-positives)),
        "true_positive_cases":positives,
        "predicted_positive_cases":int(pred.sum()),
        "majority_class_baseline_accuracy":max(positives,n-positives)/n,
    }


def compare_score(actual: dict, expected: dict):
    for key in (
        "cases", "correct", "true_positive_cases",
        "predicted_positive_cases"
    ):
        require(actual[key]==expected[key],"independent probe class-count mismatch "+key)
    for key in ("accuracy","balanced_accuracy","majority_class_baseline_accuracy"):
        require(abs(actual[key]-expected[key]) <= 1e-9,
                "independent probe score mismatch "+key)


def independent_fit(
    x_source: torch.Tensor, y_source: torch.Tensor,
    x_valid: torch.Tensor, y_valid: torch.Tensor,
    y_source_true: torch.Tensor,
):
    # Independent normal equations: affine ridge where bias has zero penalty;
    # transformations depend on original source-only examples, never validation.
    src=x_source.to(torch.float64)
    validation=x_valid.to(torch.float64)
    avg=src.sum(0)/len(src)
    centered=src-avg
    sigma=(centered.square().sum(0)/len(src)).sqrt().clamp_min(STD_FLOOR)
    design=torch.cat((
        centered/sigma, torch.ones((len(src),1),dtype=torch.float64)
    ),dim=1)
    held=torch.cat((
        (validation-avg)/sigma,
        torch.ones((len(validation),1),dtype=torch.float64)
    ),dim=1)
    gram=design.mT@design
    diagonal=torch.arange(gram.shape[0]-1)
    gram[diagonal,diagonal]+=REG
    beta=torch.linalg.solve(gram,design.mT@y_source.to(torch.float64))
    pred_train=(design@beta >= CUTOFF).long()
    pred_valid=(held@beta >= CUTOFF).long()
    return {
        "source_fit_label_metrics":independent_score(pred_train,y_source),
        "source_true_label_metrics":independent_score(pred_train,y_source_true),
        "owned_validation_metrics":independent_score(pred_valid,y_valid),
        "source_only_feature_mean_sha256":tensor_sha(avg),
        "source_only_feature_std_sha256":tensor_sha(sigma),
        "source_only_ridge_coefficient_sha256":tensor_sha(beta),
        "source_fit_label_sha256":tensor_sha(y_source.to(torch.uint8)),
        "source_true_label_sha256":tensor_sha(y_source_true.to(torch.uint8)),
        "owned_validation_target_sha256":tensor_sha(y_valid.to(torch.uint8)),
        "owned_validation_predictions_sha256":tensor_sha(pred_valid.to(torch.uint8)),
    }


def audit(index: int, old: Path, pretraining: Path,
          preflight: Path, output: Path):
    require(index in ROOTS, "root outside six old NEC04C calibration worlds")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec, lock_sha = lock()
    preflight_file = preflight/"nec04e_preinference_lock.json"
    pre = json.loads(preflight_file.read_text())
    ledger = json.loads((pretraining/"nec04c_pretraining_oracles.json").read_text())
    oldref = ledger["roots"][index-ROOTS[0]]
    rootref = pre["roots"][str(index)]
    oldinput = pretraining / oldref["training_file"]
    native_file=old/"nec04c_native_root_result.json"
    prev_replay=old/"nec04c_independent_native_audit.json"
    score_file=output/"nec04e_loopwise_hidden_probe_root.json"
    score=json.loads(score_file.read_text())
    require(
        pre["schema"]==PREFLIGHT_SCHEMA
        and pre["preinference_lock_sha256"]==lock_sha
        and pre["new_probe_code_sha256_before_new_hidden_inference"]==code_identity()
        and pre["root_indices"]==list(ROOTS)
        and pre["original_native_checkpoint_budgets"]==list(CHECKPOINTS)
        and score["schema"]==SCORE_SCHEMA
        and score["root_index"]==index
        and score["preinference_receipt_sha256"]==sha(preflight_file)
        and score["original_source_validation_sha256"]==sha(oldinput)
        ==rootref["source_validation_sha256"]
        and score["original_native_score_sha256"]==sha(native_file)
        ==rootref["original_score_receipt_sha256"]
        and score["original_native_replay_sha256"]==sha(prev_replay)
        ==rootref["original_independent_replay_sha256"]
        and score["original_source_train_rows"]==SOURCE
        and score["original_owned_validation_rows"]==VALIDATION
        and score["same_seed_random_initial_weight_sha256"]
        ==rootref["initial_weights_sha256"]
        and score["all_original_checkpoints_replayed_before_probe_scoring"] is True
        and score["source_only_ridge_fit"] is True
        and score["new_native_optimizer_updates"]==0
        and score["source_SAE_training_updates"]==0
        and score["reserved_contexts_scored"]==0
        and score["development_world_root_specs_materialized"]==0
        and score["confirmation_world_root_specs_materialized"]==0
        and score["old_NEC03P_recruited_worlds_scored"]==0
        and score["FSI_01_gate"]=="REDESIGN_OR_PILOT_ONLY"
        and set(score["arms"])==set(ARMS),
        "original model/root/probe registration, scope or previous replay changed",
    )
    rows=checked_original_rows(index,oldref,pretraining)
    world,_=gym.calibration_world(index)
    focal=world["focal_positions"]
    require(score["world_sha256"]==world["world_sha256"]
            and score["original_focal_positions"]==focal
            and score["five_focal_first_complete_visit"]==max(focal)+1,
            "original recurrent visit/focal indexing changed")
    src_indices=torch.tensor([
        i for i,r in enumerate(rows[:SOURCE]) if r["mux_selector"]==1
    ],dtype=torch.long)
    val_indices=torch.tensor([
        SOURCE+i for i,r in enumerate(rows[SOURCE:])
        if r["mux_selector"]==1
    ],dtype=torch.long)
    require(len(src_indices)==score["source_parity_selector_rows"]
            and len(val_indices)==score["owned_validation_parity_selector_rows"]
            and len(src_indices)%32==len(val_indices)%32==0,
            "original whole source/owned parity context panel differs")
    bits=as_tensor([row["history_bits12"] for row in rows])
    native=json.loads(native_file.read_text())
    torch.manual_seed(seed_for_root(index))
    random_model=Looper().eval()
    require(model_digest(random_model)==rootref["initial_weights_sha256"],
            "matched initial random Looper model SHA differs")
    random_states,_,diff=independent_hidden_states(random_model,bits)
    require(diff<=.0003,"random initialized Looper original memory replay failed")
    require([tensor_sha(x) for x in random_states]
            ==score["random_initial_hidden_state_sha256_by_visit"],
            "prior random-initialized native hidden states changed")
    outcomes={}
    verified=0
    fit_count=0
    for arm in ARMS:
        reported=score["arms"][arm]
        require(len(reported)==len(CHECKPOINTS),"old optimizer checkpoint roster changed")
        audits=[]
        for k,step in enumerate(CHECKPOINTS):
            saved=reported[k]
            past=rootref["original_model_checkpoints"][arm][str(step)]
            chk_file=old/past["checkpoint_file"]
            require(saved["step"]==step
                    and saved["original_checkpoint_byte_sha256"]
                    ==sha(chk_file)==past["checkpoint_sha256"]
                    and saved["model_weights_sha256"]==past["model_sha256"],
                    "original native checkpoint byte seal altered")
            blob=torch.load(chk_file,map_location="cpu",weights_only=True)
            model=Looper().eval()
            model.load_state_dict(blob["model_state"],strict=True)
            require(model_digest(model)==past["model_sha256"],
                    "native checkpoint weights changed before independent memory audit")
            states,logits,memorydiff=independent_hidden_states(model,bits)
            require(memorydiff<=.0003,
                    "independent native hidden-state replay differs from Looper.memory")
            archived_logits=(
                native["arms"][arm]["scored_checkpoints"][k]
                ["source_context_training_metrics"]["native_logits"]+
                native["arms"][arm]["scored_checkpoints"][k]
                ["owned_calibration_world_validation_metrics"]["native_logits"]
            )
            max_error=float((logits-torch.tensor(archived_logits,dtype=torch.float32)).abs().max())
            require(max_error<=.0003
                    and max_error<=.0003
                    and saved["native_final_head_replayed_max_absolute_logit_error"]<=.0003,
                    "independent final recurrent logits mismatch original scored checkpoint")
            require([tensor_sha(x) for x in states]
                    ==saved["trained_hidden_state_sha256_by_visit"],
                    "independently recomputed all twelve hidden states changed")
            visited=0
            for visit_block in saved["visits"]:
                visit=visit_block["visit"]
                require(visit==visited+1 and visit in VISITS,
                        "source-only readout omitted a recurrent visit")
                visited+=1
                own_tasks=own_targets(rows,focal,visit)
                require(set(visit_block["tasks"])==set(own_tasks)
                        and visit_block["observed_focal_count"]
                        ==sum(pos<visit for pos in focal)
                        and visit_block["unseen_focal_count"]
                        ==sum(pos>=visit for pos in focal),
                        "target definition or true observed/future bit order changed")
                for task,labels in own_tasks.items():
                    y=torch.tensor(labels,dtype=torch.long)
                    original_y_source=y[src_indices]
                    true_y_val=y[val_indices]
                    require(int(original_y_source.sum())*2==len(original_y_source)
                            and int(true_y_val.sum())*2==len(true_y_val),
                            "original parity-context balanced target changed")
                    for control, saved_readout in visit_block["tasks"][task].items():
                        require(control in REPRESENTATIONS,
                                "unknown original source-only readout control")
                        if control=="visible_raw_bits":
                            features=bits[:,:visit].float()
                        elif control=="random_initial_hidden":
                            features=random_states[visit-1]
                        else:
                            features=states[visit-1]
                        fit_y=original_y_source
                        if control=="source_shuffled_trained_hidden":
                            fit_y=fit_y[source_shuffle(
                                index,visit,task,len(fit_y)
                            )]
                        independently=independent_fit(
                            features[src_indices],fit_y,
                            features[val_indices],true_y_val,original_y_source
                        )
                        for section in (
                            "source_fit_label_metrics",
                            "source_true_label_metrics",
                            "owned_validation_metrics",
                        ):
                            compare_score(independently[section],saved_readout[section])
                        for name in (
                            "source_fit_label_sha256","source_true_label_sha256",
                            "owned_validation_target_sha256",
                            "owned_validation_predictions_sha256",
                        ):
                            require(independently[name]==saved_readout[name],
                                    "independently sourced class target/prediction SHA mismatch")
                        require(saved_readout["ridge_penalty"]==REG
                                and saved_readout["source_feature_std_clamp"]==STD_FLOOR
                                and saved_readout["source_rows"]==len(src_indices)
                                and saved_readout["validation_rows"]==len(val_indices),
                                "original source-only ridge hyperparameter or split changed")
                        fit_count+=1
            require(visited==12,"one of 12 recurrent states omitted")
            audits.append({
                "step":step,
                "original_checkpoint_sha256":past["checkpoint_sha256"],
                "all_12_original_states_byte_sha256_reproduced":True,
                "model_final_logit_max_abs_error":max_error,
                "independently_replayed_source_only_ridge_probes":sum(
                    len(tasks) * len(REPRESENTATIONS) for tasks in
                    (v["tasks"] for v in saved["visits"])
                ),
            })
            verified+=1
            print("NEC04E_INDEPENDENT_12VISIT_SOURCE_ONLY_RIDGE_REPLAY_PASS",
                  index,arm,step,flush=True)
        outcomes[arm]=audits
    report={
        "schema":AUDIT_SCHEMA,
        "root_index":index,
        "world_id":world["world_id"],
        "original_score_receipt_sha256":sha(score_file),
        "preinference_lock_receipt_sha256":sha(preflight_file),
        "original_calibration_world_sha256":world["world_sha256"],
        "original_source_validation_file_sha256":sha(oldinput),
        "native_checkpoints_replayed":verified,
        "all_12_original_recurrent_states_independently_replayed":True,
        "all_source_fit_probe_labels_and_predictions_replayed":True,
        "source_only_linear_readout_fits_recomputed":fit_count,
        "trained_task_model_optimizer_updates":0,
        "source_SAE_training_updates":0,
        "reserved_contexts_scored":0,
        "development_WORLD_roots_materialized":0,
        "confirmation_WORLD_roots_materialized":0,
        "old_NEC03P_causal_sham_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":outcomes,
    }
    (output/"nec04e_independent_loopwise_probe_audit.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    )
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-index",required=True,type=int)
    p.add_argument("--old-root",required=True,type=Path)
    p.add_argument("--pretraining",required=True,type=Path)
    p.add_argument("--preflight",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    audit(a.root_index,a.old_root,a.pretraining,a.preflight,a.out)


if __name__=="__main__":
    main()
