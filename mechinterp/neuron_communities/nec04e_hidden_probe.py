"""NEC-04E: 12-visit Looper hidden parity readout on old native checkpoints.

Only existing NEC04C source and owned calibration-world validation histories.
Frozen Looper weights, no task-model optimizer updates. Source-only ridge fits
for trained memory, matched initial random memory and visible-bit controls.
Independent original model/logit and preinference SHA checks precede readouts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION, seed_for_root,
)
from mechinterp.neuron_communities.nec04e_preflight import (
    SCHEMA as PREFLIGHT_SCHEMA, lock, code_identity, sha, require,
)
from research.pef07.pef07 import Looper, model_digest, as_tensor

SCHEMA = "NEC-04E/source-only-loopwise-32D-parity-linear-probe/v1"
VISITS = tuple(range(1, 13))
TARGETS = (
    "prefix1", "prefix2", "prefix_current", "future_unseen",
    "full_parity_teacher",
)
REPRESENTATIONS = (
    "trained_hidden", "random_initial_hidden", "visible_raw_bits",
    "source_shuffled_trained_hidden",
)
ALPHA = 1.
STDMIN = 1e-4
TOL = 3e-4
PRED_TIE_TOL = 1e-8
CHUNK = 128


def hash_tensor(t: torch.Tensor):
    return sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def shuffle_seed(index: int, visit: int, task: str) -> int:
    raw = sha256(
        b"NEC04E/source-only-label-shuffle/v1|"
        + str(index).encode() + b"|" + str(visit).encode()
        + b"|" + task.encode()
    ).digest()
    return int.from_bytes(raw[:8], "big") % (2**63 - 1)


def eligible_targets(visit: int, focal: list[int]) -> tuple[str, ...]:
    if not 1 <= visit <= 12 or len(focal) != 5 or len(set(focal)) != 5:
        raise ValueError("requires one of twelve original visits and five focal positions")
    seen = sum(position < visit for position in focal)
    output = []
    if seen >= 1:
        output.append("prefix1")
    if seen >= 2:
        output.extend(("prefix2", "prefix_current"))
    if seen < 5:
        output.append("future_unseen")
    output.append("full_parity_teacher")
    return tuple(output)


def target_labels(bits: torch.Tensor, original: torch.Tensor,
                  focal: list[int], visit: int, task: str) -> torch.Tensor:
    if bits.ndim != 2 or bits.shape[1] != 12 or len(original) != len(bits):
        raise AssertionError("original full 12-bit Boolean input required")
    positions = sorted(focal)
    seen = [pos for pos in positions if pos < visit]
    unseen = [pos for pos in positions if pos >= visit]
    if task not in eligible_targets(visit, focal):
        raise ValueError("unregistered target for this recurrent visit")
    if task == "full_parity_teacher":
        return original.long().clone()
    positions = {
        "prefix1": seen[:1],
        "prefix2": seen[:2],
        "prefix_current": seen,
        "future_unseen": unseen,
    }[task]
    return (bits[:, positions].sum(1) % 2).long()


def class_score(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    if len(pred) != len(truth) or len(pred) == 0:
        raise AssertionError("empty/mismatched source-or-validation probe predictions")
    pred = pred.detach().cpu().long()
    truth = truth.detach().cpu().long()
    pos = int(truth.sum())
    cases = len(truth)
    if pos <= 0 or pos >= cases:
        raise AssertionError("source/owned-validation target class is degenerate")
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    return {
        "cases": cases,
        "correct": tp + tn,
        "accuracy": (tp + tn) / cases,
        "balanced_accuracy": .5*(tp/pos + tn/(cases-pos)),
        "true_positive_cases": pos,
        "predicted_positive_cases": int(pred.sum()),
        "majority_class_baseline_accuracy": max(pos, cases-pos)/cases,
    }


def fit_source_only_ridge(
    train_x: torch.Tensor, train_y: torch.Tensor,
    validation_x: torch.Tensor, validation_y: torch.Tensor,
    true_train_y: torch.Tensor,
) -> dict:
    if (
        train_x.ndim != 2 or validation_x.ndim != 2
        or train_x.shape[1] != validation_x.shape[1]
        or len(train_x) != len(train_y) or len(validation_x) != len(validation_y)
    ):
        raise AssertionError("source-only probe feature or input dimensions invalid")
    x = train_x.double()
    v = validation_x.double()
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(STDMIN)
    ones_train = torch.ones((len(x), 1), dtype=torch.float64)
    ones_valid = torch.ones((len(v), 1), dtype=torch.float64)
    design = torch.cat(((x-mean)/std, ones_train), dim=1)
    valid_design = torch.cat(((v-mean)/std, ones_valid), dim=1)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * ALPHA
    regularizer[-1, -1] = 0.
    coeff = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ train_y.double(),
    )
    source_logits = design @ coeff
    validation_logits = valid_design @ coeff
    # An exact half-score carries no positive evidence. Treat values within
    # numerical tolerance as ties to make independent BLAS ridge replay stable.
    pred_source = (source_logits > .5 + PRED_TIE_TOL).long()
    pred_validation = (validation_logits > .5 + PRED_TIE_TOL).long()
    return {
        "source_fit_label_metrics": class_score(pred_source, train_y),
        "source_true_label_metrics": class_score(pred_source, true_train_y),
        "owned_validation_metrics": class_score(pred_validation, validation_y),
        "source_only_feature_mean_sha256": hash_tensor(mean),
        "source_only_feature_std_sha256": hash_tensor(std),
        "source_only_ridge_coefficient_sha256": hash_tensor(coeff),
        "source_fit_label_sha256": hash_tensor(train_y.to(torch.uint8)),
        "source_true_label_sha256": hash_tensor(true_train_y.to(torch.uint8)),
        "owned_validation_target_sha256": hash_tensor(validation_y.to(torch.uint8)),
        "owned_validation_predictions_sha256": hash_tensor(
            pred_validation.to(torch.uint8)
        ),
        "ridge_penalty": ALPHA,
        "source_feature_std_clamp": STDMIN,
        "source_rows": len(train_x),
        "validation_rows": len(validation_x),
    }


@torch.no_grad()
def recurrent_states(model: Looper, histories: torch.Tensor):
    """Compute h1..h12 by original chronological step calls, no intervention."""
    model.eval()
    collector = [[] for _ in VISITS]
    endings = []
    direct_max_diff = 0.
    for start in range(0, len(histories), CHUNK):
        bits = histories[start:start + CHUNK]
        h = model.memory_init[None,:].expand(len(bits),-1)
        for step in range(12):
            h = model.step(h, model.bit_embedding(bits[:, step]))
            collector[step].append(h.detach().cpu().contiguous())
        # Separate original model.memory replay cross-check for h12.
        reference_h = model.memory(bits)
        direct_max_diff = max(
            direct_max_diff, float((reference_h-h).abs().max())
        )
        endings.append(model.head(h).detach().cpu())
    states = [torch.cat(parts, 0) for parts in collector]
    final_logits = torch.cat(endings, 0)
    require(len(states) == 12
            and all(x.shape == (len(histories), 32) for x in states)
            and final_logits.shape == (len(histories),2)
            and direct_max_diff <= TOL,
            "recurrent-state step-by-step replay differs from original Looper.memory")
    return states, final_logits, direct_max_diff


def _check_source(index, old: Path, pretraining: Path, preflight: Path):
    if index not in ROOTS:
        raise PermissionError("NEC04E only original calibration roots6..11")
    _, lock_sha = lock()
    receipt_file = preflight / "nec04e_preinference_lock.json"
    receipt = json.loads(receipt_file.read_text())
    ledger_file = pretraining / "nec04c_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    if (
        receipt["schema"] != PREFLIGHT_SCHEMA
        or receipt["preinference_lock_sha256"] != lock_sha
        or receipt["new_probe_code_sha256_before_new_hidden_inference"]
        != code_identity()
        or receipt["original_nec04c_pretraining_sha256"] != sha(ledger_file)
        or receipt["root_indices"] != list(ROOTS)
        or receipt["native_arms"] != list(ARMS)
        or receipt["original_native_checkpoint_budgets"] != list(CHECKPOINTS)
        or receipt["all_24_historical_native_checkpoint_byte_shas_verified"] is not True
        or receipt["six_prior_native_full_logit_audits_reverified"] is not True
        or receipt["new_hidden_inference_executed_at_preflight"] is not False
        or receipt["new_task_model_training_updates"] != 0
        or receipt["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
        or (os.environ.get("GITHUB_SHA")
            and receipt["preinference_source_commit_sha"] != os.environ["GITHUB_SHA"])
    ):
        raise AssertionError("pre-inference original model/code/checkpoint lock changed")
    ref = ledger["roots"][index-ROOTS[0]]
    old_record = receipt["roots"][str(index)]
    source_path = pretraining / ref["training_file"]
    score_path = old / "nec04c_native_root_result.json"
    replay_path = old / "nec04c_independent_native_audit.json"
    require(ref["world_id"] == old_record["world_id"]
            and ref["world_sha256"] == old_record["world_sha256"]
            and sha(source_path) == ref["training_file_sha256"]
            == old_record["source_validation_sha256"]
            and sha(score_path) == old_record["original_score_receipt_sha256"]
            and sha(replay_path) == old_record["original_independent_replay_sha256"]
            and ref["matched_native_seed"] == old_record["matched_model_seed"]
            == seed_for_root(index),
            "old source/oracle/model/audit identity differs from pre-inference commit")
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    require(len(rows) == SOURCE + VALIDATION
            and all(r["root_index"] == index for r in rows)
            and len({tuple(r["history_bits12"]) for r in rows}) == SOURCE+VALIDATION,
            "old root split altered")
    world, _ = gym.calibration_world(index)
    require(world["world_sha256"] == ref["world_sha256"]
            and world["canonical_program"][:2] ==
            ["MUX", ["VAR", world["context_positions"][0]]],
            "old root program or original context-selector label changed")
    original_score = json.loads(score_path.read_text())
    old_replay = json.loads(replay_path.read_text())
    require(old_replay["four_original_checkpoints_byte_SHA256_verified"]
            and old_replay["all_four_native_source_validation_logits_independently_replayed"]
            and original_score["native_initial_weights_sha256"]
            == old_record["initial_weights_sha256"],
            "old independent native output audit invalid")
    return receipt, old_record, ref, rows, world, original_score


def run_root(index: int, old: Path, pretraining: Path,
             preflight: Path, out: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt, old_record, ref, rows, world, previous = _check_source(
        index, old, pretraining, preflight
    )
    bits = as_tensor([row["history_bits12"] for row in rows])
    raw_targets = torch.tensor(
        [row["composed_label"] for row in rows],dtype=torch.long
    )
    selected_source = torch.tensor(
        [i for i, row in enumerate(rows[:SOURCE]) if row["mux_selector"] == 1],
        dtype=torch.long
    )
    selected_val = torch.tensor(
        [SOURCE+i for i, row in enumerate(rows[SOURCE:])
         if row["mux_selector"] == 1],dtype=torch.long
    )
    require(
        0 < len(selected_source) < SOURCE and 0 < len(selected_val) < VALIDATION
        and len(selected_source)%32 == 0 and len(selected_val)%32 == 0
        and int(raw_targets[selected_source].sum())*2 == len(selected_source)
        and int(raw_targets[selected_val].sum())*2 == len(selected_val),
        "original full-mask parity context closure or class balance broken",
    )
    focal = world["focal_positions"]
    torch.manual_seed(seed_for_root(index))
    initial_model = Looper().eval()
    require(
        model_digest(initial_model) == old_record["initial_weights_sha256"],
        "same-seed random Looper initial weights changed",
    )
    random_states,_,random_replay_diff = recurrent_states(initial_model,bits)
    arms = {}
    for arm in ARMS:
        models = []
        oldcheck = previous["arms"][arm]["scored_checkpoints"]
        for ordinal, step in enumerate(CHECKPOINTS):
            source_checkpoint = old_record["original_model_checkpoints"][arm][str(step)]
            chk_file = old / source_checkpoint["checkpoint_file"]
            require(
                source_checkpoint["checkpoint_file"] ==
                    f"nec04c.root{index}.{arm}.step{step}.pt"
                and sha(chk_file) == source_checkpoint["checkpoint_sha256"]
                and source_checkpoint["checkpoint_sha256"] ==
                    oldcheck[ordinal]["checkpoint_sha256"],
                "old native model optimizer checkpoint SHA changed",
            )
            payload = torch.load(chk_file,map_location="cpu",weights_only=True)
            require(payload["step"] == payload["optimizer_updates"] == step
                    and payload["training_arm"] == arm
                    and payload["root_index"] == index
                    and payload["native_seed"] == seed_for_root(index),
                    "old native optimizer training receipt inconsistent")
            steps = [
                int(item["step"].item())
                for item in payload["optimizer_state"]["state"].values()
                if "step" in item
            ]
            require(bool(steps) and all(count == step for count in steps),
                    "old optimizer state step counters disagree with frozen budget")
            model = Looper().eval()
            model.load_state_dict(payload["model_state"],strict=True)
            require(model_digest(model) == source_checkpoint["model_sha256"],
                    "native Looper weights mismatch previous independent replay")
            states, logits, recurrence_diff = recurrent_states(model,bits)
            archived = (
                oldcheck[ordinal]["source_context_training_metrics"]["native_logits"]
                + oldcheck[ordinal]["owned_calibration_world_validation_metrics"]["native_logits"]
            )
            require(len(archived) == SOURCE + VALIDATION,
                    "incomplete old 12-step Looper logits")
            old_logits = torch.tensor(archived,dtype=torch.float32)
            max_difference = float((logits - old_logits).abs().max())
            require(max_difference <= TOL,
                    "reconstructed 12-step Looper logits differ from sealed NEC04C output")
            byvisit = []
            for visit in VISITS:
                targets = {}
                for task in eligible_targets(visit, focal):
                    y = target_labels(bits,raw_targets,focal,visit,task)
                    truth_source = y[selected_source]
                    truth_validation = y[selected_val]
                    require(
                        int(truth_source.sum())*2 == len(truth_source)
                        and int(truth_validation.sum())*2 == len(truth_validation),
                        "original per-mask parity/prefix/future target must be balanced",
                    )
                    spec = {}
                    source_y = truth_source
                    validation_y = truth_validation
                    for kind in REPRESENTATIONS:
                        if kind in ("trained_hidden","source_shuffled_trained_hidden"):
                            features = states[visit-1]
                        elif kind == "random_initial_hidden":
                            features = random_states[visit-1]
                        else:
                            features = bits[:,:visit].float()
                        src = features[selected_source]
                        val = features[selected_val]
                        fit_y = source_y
                        if kind == "source_shuffled_trained_hidden":
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(shuffle_seed(index,visit,task))
                            order = torch.randperm(len(fit_y),generator=generator)
                            fit_y = fit_y[order]
                        spec[kind] = fit_source_only_ridge(
                            src,fit_y,val,validation_y,source_y
                        )
                    targets[task] = spec
                byvisit.append({
                    "visit":visit,
                    "observed_focal_count":sum(p<visit for p in focal),
                    "unseen_focal_count":sum(p>=visit for p in focal),
                    "source_parity_context_groups":len(selected_source)//32,
                    "owned_validation_parity_context_groups":len(selected_val)//32,
                    "tasks":targets,
                })
            models.append({
                "step":step,
                "original_checkpoint_byte_sha256":source_checkpoint["checkpoint_sha256"],
                "model_weights_sha256":source_checkpoint["model_sha256"],
                "native_final_head_replayed_max_absolute_logit_error":max_difference,
                "native_original_memory_replay_max_absolute_error":recurrence_diff,
                "trained_hidden_state_sha256_by_visit":[hash_tensor(s) for s in states],
                "visits":byvisit,
            })
            print("NEC04E_SIX_ROOT_NATIVE_RECURRENT_MEMORY_PROBES_SCORED",
                  index,arm,step,round(max_difference,8),flush=True)
        arms[arm] = models
    report = {
        "schema":SCHEMA,
        "root_index":index,
        "world_id":ref["world_id"],
        "world_sha256":ref["world_sha256"],
        "preinference_source_commit_sha":receipt["preinference_source_commit_sha"],
        "original_source_validation_sha256":ref["training_file_sha256"],
        "preinference_receipt_sha256":sha(preflight/"nec04e_preinference_lock.json"),
        "original_native_score_sha256":old_record["original_score_receipt_sha256"],
        "original_native_replay_sha256":old_record["original_independent_replay_sha256"],
        "same_seed_random_initial_weight_sha256":old_record["initial_weights_sha256"],
        "same_seed_initial_memory_replay_max_absolute_error":random_replay_diff,
        "random_initial_hidden_state_sha256_by_visit":[hash_tensor(s) for s in random_states],
        "source_parity_selector_rows":len(selected_source),
        "owned_validation_parity_selector_rows":len(selected_val),
        "original_source_train_rows":SOURCE,
        "original_owned_validation_rows":VALIDATION,
        "original_focal_positions":focal,
        "five_focal_first_complete_visit":max(focal)+1,
        "all_original_checkpoints_replayed_before_probe_scoring":True,
        "new_native_optimizer_updates":0,
        "source_only_ridge_fit":True,
        "source_SAE_training_updates":0,
        "reserved_contexts_scored":0,
        "development_world_root_specs_materialized":0,
        "confirmation_world_root_specs_materialized":0,
        "old_NEC03P_recruited_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":arms,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec04e_loopwise_hidden_probe_root.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    print("NEC04E_SOURCE_ONLY_LOOPWISE_PROBE_ROOT_COMPLETE",index,flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-index",required=True,type=int)
    p.add_argument("--old-root",required=True,type=Path)
    p.add_argument("--pretraining",required=True,type=Path)
    p.add_argument("--preflight",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run_root(args.root_index,args.old_root,args.pretraining,args.preflight,args.out)


if __name__=="__main__":
    main()
