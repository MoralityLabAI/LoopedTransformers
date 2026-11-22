"""NEC-04C fixed six-root Looper parity curriculum versus matched vanilla baseline.

Two native composed-teacher arms per untouched NEC04A calibration world6..11.
Both arms start with identical 32D Looper weights and AdamW settings,
perform 900 continuous source-only updates, save checkpoints at 240/900,
and only then score originally owned calibration-world validation contexts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION, SCHEMA as ORACLE_SCHEMA,
    digest, verify_lock, code_identity, seed_for_root,
)
from mechinterp.neuron_communities.nec04b_native_pilot import (
    score_logits, score_predictions,
)
from research.pef07.pef07 import Looper, model_digest, as_tensor

SCHEMA = "NEC-04C/native-matched-parity-curriculum-pilot/v1"
BATCH = 64
STEPS = 900
STAGE1 = 240
LR = 0.003
CLIP = 1.0
WEIGHT_DECAY = 0.0
SAMPLER_XOR = 0x5DEECE66D


def checked_inputs(index: int, folder: Path):
    if index not in ROOTS:
        raise PermissionError("NEC04C permits only previously untouched calibration roots6..11")
    _, lock_sha = verify_lock()
    ledger_file = folder / "nec04c_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    if (
        ledger["schema"] != ORACLE_SCHEMA
        or ledger["lock_sha256"] != lock_sha
        or ledger["root_indices"] != list(ROOTS)
        or ledger["training_arms"] != list(ARMS)
        or ledger["checkpoint_steps"] != list(CHECKPOINTS)
        or ledger["source_cases_per_root"] != SOURCE
        or ledger["owned_validation_cases_per_root"] != VALIDATION
        or ledger["code_sha256_before_training"] != code_identity()
        or ledger["optimizer_steps_before_preflight"] != 0
    ):
        raise AssertionError("frozen source, model, context ownership or registered code changed")
    if os.environ.get("GITHUB_SHA") and ledger["preflight_source_commit_sha"] != os.environ["GITHUB_SHA"]:
        raise AssertionError("training checkout differs from pretraining commit SHA")
    ref = ledger["roots"][index - ROOTS[0]]
    f = folder / ref["training_file"]
    if (
        ref["root_index"] != index
        or ref["matched_native_seed"] != seed_for_root(index)
        or digest(f) != ref["training_file_sha256"]
    ):
        raise AssertionError("original calibration-world source and validation oracle changed")
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    if (
        len(rows) != SOURCE + VALIDATION
        or any(row["split"] != "source_train" for row in rows[:SOURCE])
        or any(row["split"] != "calibration_world_validation" for row in rows[SOURCE:])
        or any(row["root_index"] != index for row in rows)
        or len({tuple(row["history_bits12"]) for row in rows}) != SOURCE + VALIDATION
    ):
        raise AssertionError("original 80 source + 24 owned validation contexts changed")
    return ledger, ref, rows, digest(ledger_file)


def sample_indices(
    arm: str,
    step: int,
    rng: torch.Generator,
    parity_indices: torch.Tensor,
    alternate_indices: torch.Tensor,
) -> torch.Tensor:
    if arm == "vanilla":
        return torch.randint(SOURCE, (BATCH,), generator=rng)
    if arm != "parity_curriculum" or not 1 <= step <= STEPS:
        raise ValueError("unregistered NEC04C native training arm or step")
    if step <= STAGE1:
        chosen = torch.randint(len(parity_indices), (BATCH,), generator=rng)
        return parity_indices[chosen]
    parity = parity_indices[torch.randint(len(parity_indices), (BATCH // 2,), generator=rng)]
    alternate = alternate_indices[
        torch.randint(len(alternate_indices), (BATCH // 2,), generator=rng)
    ]
    take = torch.cat((parity, alternate))
    return take[torch.randperm(BATCH, generator=rng)]


def train_arm(index: int, arm: str, ref: dict, data: list, folder: Path, lock_sha: str):
    seed = seed_for_root(index)
    torch.manual_seed(seed)
    model = Looper().train()
    initial = model_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    rng = torch.Generator(device="cpu").manual_seed(seed ^ SAMPLER_XOR)
    train_x = as_tensor([row["history_bits12"] for row in data[:SOURCE]])
    train_y = torch.tensor(
        [int(row["composed_label"]) for row in data[:SOURCE]],
        dtype=torch.long,
    )
    parity_indices = torch.tensor(
        [i for i, row in enumerate(data[:SOURCE]) if row["mux_selector"] == 1],
        dtype=torch.long,
    )
    alternate_indices = torch.tensor(
        [i for i, row in enumerate(data[:SOURCE]) if row["mux_selector"] == 0],
        dtype=torch.long,
    )
    if (
        train_x.shape != (SOURCE, 12)
        or not 0 < int(train_y.sum()) < SOURCE
        or not 0 < len(parity_indices) < SOURCE
        or not 0 < len(alternate_indices) < SOURCE
        or len(parity_indices) + len(alternate_indices) != SOURCE
    ):
        raise AssertionError("original composed teacher or MUX source contexts malformed")
    frozen = []
    first_loss = None
    for step in range(1, STEPS + 1):
        take = sample_indices(arm, step, rng, parity_indices, alternate_indices)
        bx, by = train_x[take], train_y[take]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.logits(bx), by)
        if not bool(torch.isfinite(loss)):
            raise AssertionError("nonfinite original source composed loss")
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if not bool(torch.isfinite(grad)):
            raise AssertionError("nonfinite original source Looper gradients")
        optimizer.step()
        if step in CHECKPOINTS:
            model_sha = model_digest(model)
            record = {
                "schema": SCHEMA,
                "root_index": index,
                "world_id": ref["world_id"],
                "world_sha256": ref["world_sha256"],
                "source_lock_sha256": lock_sha,
                "source_validation_sha256": ref["training_file_sha256"],
                "training_arm": arm,
                "native_seed": seed,
                "native_initial_model_sha256": initial,
                "native_model_sha256": model_sha,
                "optimizer": "AdamW",
                "optimizer_updates": step,
                "step": step,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": CLIP,
                "batch_size": BATCH,
                "sampler_stage1_updates": STAGE1 if arm == "parity_curriculum" else 0,
                "sampler_stage2_updates": STEPS - STAGE1 if arm == "parity_curriculum" else 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_path = folder / f"nec04c.root{index}.{arm}.step{step}.pt"
            torch.save(record, save_path)
            frozen.append({
                "step": step,
                "file": save_path.name,
                "checkpoint_sha256": digest(save_path),
                "native_model_sha256": model_sha,
                "native_initial_model_sha256": initial,
                "source_loss_first_update": first_loss,
                "source_loss_last_update": float(loss.detach()),
            })
            print(
                "NEC04C_MATCHED_NATIVE_CHECKPOINT_SEALED",
                ref["world_id"], arm, step, frozen[-1]["checkpoint_sha256"],
                flush=True,
            )
    if len(frozen) != len(CHECKPOINTS):
        raise AssertionError("fixed 240/900 matched checkpoint schedule incomplete")
    manifest = {
        "schema": SCHEMA,
        "root_index": index,
        "world_id": ref["world_id"],
        "arm": arm,
        "lock_sha256": lock_sha,
        "source_validation_sha256": ref["training_file_sha256"],
        "original_composed_program_sha256": ref["program_sha256"],
        "native_initial_model_sha256": initial,
        "continuous_AdamW_optimizer": True,
        "optimizer_resets": 0,
        "total_updates": STEPS,
        "checkpoint_files_frozen_before_any_owned_validation_scoring": True,
        "checkpoints": frozen,
    }
    (folder / f"nec04c.root{index}.{arm}.checkpoints.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def run(index: int, pretraining: Path, output: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock, lock_sha = verify_lock()
    ledger, ref, data, ledger_sha = checked_inputs(index, pretraining)
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for arm in ARMS:
        manifests[arm] = train_arm(index, arm, ref, data, output, lock_sha)
    initials = {manifests[arm]["native_initial_model_sha256"] for arm in ARMS}
    if len(initials) != 1:
        raise AssertionError("treatment arms did not begin from identical native weights")
    # Both arms have completed all 900 updates and all four original checkpoint
    # byte SHA256s were sealed before we read any owned validation target.
    x = as_tensor([row["history_bits12"] for row in data])
    source_y = [int(row["composed_label"]) for row in data[:SOURCE]]
    validation_y = [int(row["composed_label"]) for row in data[SOURCE:]]
    if not 0 < sum(validation_y) < VALIDATION:
        raise AssertionError("owned validation target class degenerate")
    outcomes = {}
    for arm in ARMS:
        manifest = manifests[arm]
        checkpoints = []
        for frozen in manifest["checkpoints"]:
            path = output / frozen["file"]
            if digest(path) != frozen["checkpoint_sha256"]:
                raise AssertionError("native source checkpoint modified before validation")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            replay = Looper().eval()
            replay.load_state_dict(payload["model_state"], strict=True)
            if model_digest(replay) != frozen["native_model_sha256"]:
                raise AssertionError("native checkpoint weights changed before validation")
            src = score_predictions(score_logits(replay, x[:SOURCE]), source_y)
            val = score_predictions(score_logits(replay, x[SOURCE:]), validation_y)
            checkpoints.append({
                "step": frozen["step"],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "native_model_sha256": frozen["native_model_sha256"],
                "source_context_training_metrics": src,
                "owned_calibration_world_validation_metrics": val,
            })
        outcomes[arm] = {
            "sealed_checkpoint_manifest_sha256": digest(
                output / f"nec04c.root{index}.{arm}.checkpoints.json"
            ),
            "checkpoint_manifest": manifest,
            "scored_checkpoints": checkpoints,
        }
    result = {
        "schema": SCHEMA,
        "root_index": index,
        "world_id": ref["world_id"],
        "world_sha256": ref["world_sha256"],
        "structural_difficulty_tier": ref["tier"],
        "original_composed_program_sha256": ref["program_sha256"],
        "source_lock_sha256": lock_sha,
        "pretraining_ledger_sha256": ledger_sha,
        "original_source_validation_file_sha256": ref["training_file_sha256"],
        "preflight_source_commit_sha": ledger["preflight_source_commit_sha"],
        "native_initial_weights_shared_between_arms": True,
        "native_initial_weights_sha256": next(iter(initials)),
        "source_train_rows": SOURCE,
        "owned_validation_rows": VALIDATION,
        "both_arms_all_four_checkpoints_sealed_before_validation": True,
        "source_only_training_updates_per_arm": STEPS,
        "torch_version": str(torch.__version__),
        "processor": platform.processor(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "native_SAE_training_updates": 0,
        "original_reserved_contexts_scored": 0,
        "development_world_roots_materialized": 0,
        "confirmation_world_roots_materialized": 0,
        "old_NEC03P_worlds_scored": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "arms": outcomes,
    }
    (output / "nec04c_native_root_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("NEC04C_NATIVE_NEW_CALIBRATION_WORLD_ROOT_SAVED", ref["world_id"], flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-index", type=int, required=True)
    p.add_argument("--pretraining", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.root_index, args.pretraining, args.out)


if __name__ == "__main__":
    main()
