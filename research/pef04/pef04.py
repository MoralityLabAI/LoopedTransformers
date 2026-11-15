"""PEF-04: prospective, pair-closed, six-seed natural history transfer.

Read research/pef04/PRE_REGISTRATION.md before making any experimental
interpretation. A pretrained task head never receives oldest-bit
supervision; after all six trunks are frozen, a later oldest-bit linear
head is fitted using the independent probe split.

Natural 8+2 visit states only. No off-manifold activation patch appears
in the scored test. All original/counterfactual pairs stay together
across train/probe/heldout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from research.pef03.pef03 import (
    BATCH, L, TRAIN_STEPS, BitMemoryLooper, fit_ridge, hash_weights,
    labels, load_frozen, predict_ridge, tensor
)

SEEDS = (101, 109, 127, 131, 149, 157)
PAIR_COUNTS = {"train": 80, "probe": 24, "heldout": 24}
CONTINUATIONS = {"h8": (), "h10_00": (0, 0), "h10_11": (1, 1)}
PRIMARY_ARM = "h10_00"
REGISTERED_TRAINING_STEPS = 250
MODEL_INIT_CONTROL_OFFSET = 50000


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest_tensor_sequence(tensors: Iterable[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for value in tensors:
        item = value.detach().cpu().contiguous()
        h.update(str(tuple(item.shape)).encode())
        h.update(str(item.dtype).encode())
        h.update(bytes(item.view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def paired_split() -> tuple[dict[str, tuple[tuple[int, ...], ...]], dict]:
    suffixes = list(product((0, 1), repeat=L-1))
    suffixes.sort(key=lambda suffix: (
        hashlib.sha256(b"pef04-v1-pair:" + bytes(suffix)).digest(), bytes(suffix)
    ))
    catalog = {name: [] for name in PAIR_COUNTS}
    assignment = {}
    offset = 0
    for name, count in PAIR_COUNTS.items():
        for suffix in suffixes[offset:offset+count]:
            assignment["".join(map(str,suffix))] = name
            for bit in (0, 1):
                catalog[name].append((bit, *suffix))
        offset += count
    if offset != 128:
        raise AssertionError("expected all 128 paired suffixes")
    assert all(len(catalog[name]) == 2 * PAIR_COUNTS[name] for name in PAIR_COUNTS)
    for name, rows in catalog.items():
        for i in range(0, len(rows), 2):
            if rows[i][0] != 0 or rows[i+1][0] != 1 or rows[i][1:] != rows[i+1][1:]:
                raise AssertionError(f"split {name} is not oldest-bit counterfactual closed")
    sets = [set(catalog[name]) for name in PAIR_COUNTS]
    if len(set.union(*sets)) != 256 or sum(len(s) for s in sets) != 256:
        raise AssertionError("split histories overlap or are missing")
    manifest = {
        "campaign": "PEF-04",
        "split_algorithm": "sort SHA256(pef04-v1-pair: + seven chronological suffix bytes)",
        "suffixes_sorted_by_sha256": ["".join(map(str, x)) for x in suffixes],
        "allocation_pairs": PAIR_COUNTS,
        "allocation_histories": {key: len(value) for key, value in catalog.items()},
        "pair_assignment": assignment,
        "heldout_counterfactuals_are_never_trunk_or_probe_training": True,
        "oldest_bit_conditionally_fair_given_seven_bit_suffix": True,
    }
    manifest["split_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    return {name: tuple(rows) for name, rows in catalog.items()}, manifest


def train_seed(seed: int, train_bits: torch.Tensor, checkpoint: Path,
               steps: int = REGISTERED_TRAINING_STEPS) -> dict:
    if seed not in SEEDS or steps <= 0:
        raise ValueError("unsupported PEF-04 seed or steps")
    torch.manual_seed(seed)
    model = BitMemoryLooper().train()
    initial_hash = hash_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed * 7919 + 59)
    loss_first = loss_last = None
    for i in range(steps):
        samples = train_bits[torch.randint(len(train_bits), (BATCH,), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.logits(samples), labels(samples))
        if not torch.isfinite(loss):
            raise AssertionError("training loss nonfinite")
        if i == 0:
            loss_first = float(loss.detach())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise AssertionError("training gradient nonfinite")
        optimizer.step()
        loss_last = float(loss.detach())
    model.eval()
    frozen_hash = hash_weights(model)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "seed": seed, "steps": steps,
        "init_hash": initial_hash, "final_hash": frozen_hash,
    }, checkpoint)
    return {
        "seed": seed,
        "steps": steps,
        "initial_loss": loss_first,
        "final_loss": loss_last,
        "initial_weights_sha256": initial_hash,
        "frozen_weights_sha256": frozen_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_file": checkpoint.name,
        "trained_on_only_recent_two_bit_task": True,
        "oldest_bit_has_no_supervised_loss": True,
    }


@torch.no_grad()
def natural_memory(model: BitMemoryLooper, histories: torch.Tensor,
                   continuation: tuple[int, ...]) -> torch.Tensor:
    if any(type(bit) is not int or bit not in (0, 1) for bit in continuation):
        raise ValueError("continuation must be a tuple of binary bits")
    state = model.memory(histories)
    for bit in continuation:
        b = torch.full((len(histories),), bit, dtype=torch.long)
        state = model.step(state, model.bit_embedding(b))
    if not torch.isfinite(state).all():
        raise AssertionError("nonfinite natural continuation state")
    return state


def _pair_scoring(prediction: torch.Tensor, truth: torch.Tensor, hidden: torch.Tensor) -> dict:
    if prediction.ndim != 1 or truth.ndim != 1 or len(prediction) != len(truth):
        raise ValueError("invalid readout scoring tensors")
    if len(prediction) % 2 or hidden.shape[0] != len(prediction):
        raise ValueError("pair-closed evaluation requires even correctly shaped input")
    if not (torch.all(truth[0::2] == 0) and torch.all(truth[1::2] == 1)):
        raise AssertionError("paired oldest-bit labels are not 0 then 1")
    count = int((prediction == truth).sum())
    both = int(((prediction[0::2] == 0) & (prediction[1::2] == 1)).sum())
    flips = int((prediction[0::2] != prediction[1::2]).sum())
    distances = (hidden[0::2] - hidden[1::2]).norm(dim=1)
    return {
        "correct_histories": count,
        "total_histories": len(prediction),
        "both_members_correct_pairs": both,
        "total_pairs": len(prediction) // 2,
        "prediction_changes_across_oldest_bit_flip_pairs": flips,
        "mean_natural_pair_hidden_l2": float(distances.mean()),
        "min_natural_pair_hidden_l2": float(distances.min()),
    }


@torch.no_grad()
def evaluate_seed(seed: int, catalog: dict, sealed: dict,
                  root: Path) -> dict:
    checkpoint = root / sealed["checkpoint_file"]
    pinned = sealed["checkpoint_sha256"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != pinned:
        raise AssertionError("frozen checkpoint bytes changed before readout fitting")
    model = load_frozen(checkpoint, seed)
    frozen_hash = hash_weights(model)
    if frozen_hash != sealed["frozen_weights_sha256"]:
        raise AssertionError("loaded source model weights do not match sealed manifest")
    torch.manual_seed(seed + MODEL_INIT_CONTROL_OFFSET)
    random_model = BitMemoryLooper().eval()
    random_init_hash = hash_weights(random_model)
    probe_bits, heldout_bits = (tensor(catalog[split]) for split in ("probe", "heldout"))
    label8 = labels(heldout_bits)
    source_task_pred = model.logits(heldout_bits).argmax(1)
    task_correct = int((source_task_pred == label8).sum())
    random_task_correct = int((random_model.logits(heldout_bits).argmax(1) == label8).sum())

    arms = {}
    for name, continuation in CONTINUATIONS.items():
        source_probe = natural_memory(model, probe_bits, continuation)
        source_test = natural_memory(model, heldout_bits, continuation)
        control_probe = natural_memory(random_model, probe_bits, continuation)
        control_test = natural_memory(random_model, heldout_bits, continuation)
        fitted = fit_ridge(source_probe, probe_bits[:, 0], penalty=1.0)
        control_fitted = fit_ridge(control_probe, probe_bits[:, 0], penalty=1.0)
        pred = predict_ridge(source_test, fitted)
        control_pred = predict_ridge(control_test, control_fitted)
        arms[name] = {
            "natural_continuation_bits": list(continuation),
            "source_frozen_trunk": {
                **_pair_scoring(pred, heldout_bits[:, 0], source_test),
                "probe_head_tensors_sha256": _digest_tensor_sequence(fitted),
            },
            "random_initialization_control": {
                **_pair_scoring(control_pred, heldout_bits[:, 0], control_test),
                "random_trunk_init_hash": random_init_hash,
                "probe_head_tensors_sha256": _digest_tensor_sequence(control_fitted),
            },
            "suffix_only_theoretical_control": {
                "correct_histories": 24,
                "total_histories": 48,
                "both_members_correct_pairs": 0,
                "total_pairs": 24,
                "reason": "each suffix has equal oldest-bit 0/1 members",
            },
            "trained_new_head_only_on_probe_split": True,
            "trunk_weights_unchanged": True,
        }
    if hash_weights(model) != frozen_hash:
        raise AssertionError("downstream evaluation modified trained trunk weights")
    if hash_weights(random_model) != random_init_hash:
        raise AssertionError("random-init baseline was modified")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != pinned:
        raise AssertionError("downstream evaluation modified sealed checkpoint")
    row = {
        "seed": seed,
        "checkpoint_sha256": pinned,
        "trained_checkpoint_weights_sha256": frozen_hash,
        "frozen_checkpoint_integrity_passed": True,
        "head_task_correct_heldout": task_correct,
        "head_task_total_heldout": len(heldout_bits),
        "random_init_head_task_correct": random_task_correct,
        "random_init_head_task_total": len(heldout_bits),
        "training_steps": sealed["steps"],
        "probe_training_histories": len(probe_bits),
        "heldout_histories": len(heldout_bits),
        "heldout_complete_counterfactual_pairs": len(heldout_bits) // 2,
        "downstream_arms": arms,
    }
    (root / f"seed{seed}.evaluation.json").write_bytes(_canonical_bytes(row))
    return row


def run(root: Path, *, seeds: tuple[int, ...] = SEEDS,
        steps: int = REGISTERED_TRAINING_STEPS) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root.mkdir(parents=True, exist_ok=True)
    catalog, split = paired_split()
    (root / "split_manifest.json").write_bytes(_canonical_bytes(split))
    source = tensor(catalog["train"])
    receipts = {}
    for seed in seeds:
        if seed in receipts:
            raise ValueError("duplicated model seed")
        receipt = train_seed(seed, source, root / f"seed{seed}.frozen.pt", steps)
        receipt["split_manifest_sha256"] = split["split_sha256"]
        receipts[seed] = receipt
        print(f"PEF04_TRAINING_FROZEN seed={seed} sha256={receipt['checkpoint_sha256']}",flush=True)
    ledger = {
        "campaign": "PEF-04",
        "stage": "ALL_TRAINED_CHECKPOINTS_FROZEN_BEFORE_EVALUATION",
        "pre_registration": "research/pef04/PRE_REGISTRATION.md",
        "split_manifest_sha256": split["split_sha256"],
        "training": [receipts[seed] for seed in seeds],
        "all_trunks_frozen_before_first_probe_or_test": True,
        "no_heldout_score_used_to_select_checkpoint": True,
    }
    (root / "sealed_training_ledger.json").write_bytes(_canonical_bytes(ledger))
    scored = [evaluate_seed(seed, catalog, receipts[seed], root) for seed in seeds]
    successes = [x["downstream_arms"][PRIMARY_ARM]["source_frozen_trunk"] for x in scored]
    transfer_passes = sum(
        x["correct_histories"] >= 36 and x["both_members_correct_pairs"] >= 12
        for x in successes
    )
    task_passes = sum(x["head_task_correct_heldout"] >= 41 for x in scored)
    preregistered = tuple(seeds) == SEEDS and steps == REGISTERED_TRAINING_STEPS
    result = {
        "campaign": "PEF-04",
        "protocol": "prospective-pair-closed-six-seed-natural-two-step-continuation",
        "registered_scoring_run": preregistered,
        "source_repos": {
            "model": "research.pef03.pef03.BitMemoryLooper",
            "theoretical": "MoralityLabAI/Compactification-Theory PR 142",
        },
        "torch_version": torch.__version__,
        "seed_order": list(seeds),
        "training_steps": steps,
        "split_manifest_sha256": split["split_sha256"],
        "split_history_counts": split["allocation_histories"],
        "primary_readout": PRIMARY_ARM,
        "primary_transferred_oldest_bit_successful_seeds": transfer_passes,
        "primary_supervised_task_successful_seeds": task_passes,
        "registered_transfer_decision": (
            "criterion_met" if transfer_passes >= 4 and task_passes >= 4
            else "criterion_not_met"
        ) if preregistered else "debug_or_nonregistered_run",
        "register_threshold": {
            "minimum_seeds": 4,
            "heldout_oldest_bit_correct_at_least": 36,
            "both_members_correct_pairs_at_least": 12,
            "original_four_way_task_correct_at_least": 41,
        },
        "source_suffix_only_baseline_correct": 24,
        "source_suffix_only_baseline_both_pairs": 0,
        "random_init_baselines_recorded": True,
        "all_historic_counterfactuals_unseen_by_trunk_and_probe_training": True,
        "result_per_seed": scored,
    }
    (root / "summary.json").write_bytes(_canonical_bytes(result))
    print(json.dumps({
        "campaign": result["campaign"],
        "registered_scoring_run": preregistered,
        "registered_transfer_decision": result["registered_transfer_decision"],
        "primary_transferred_oldest_bit_successful_seeds": transfer_passes,
        "primary_supervised_task_successful_seeds": task_passes,
        "seeds": [{
            "seed": x["seed"],
            "task_correct": x["head_task_correct_heldout"],
            "oldest_after_00_correct": x["downstream_arms"][PRIMARY_ARM]["source_frozen_trunk"]["correct_histories"],
            "oldest_after_00_both_pairs": x["downstream_arms"][PRIMARY_ARM]["source_frozen_trunk"]["both_members_correct_pairs"],
            "random_oldest_after_00_correct": x["downstream_arms"][PRIMARY_ARM]["random_initialization_control"]["correct_histories"],
        } for x in scored],
    }, sort_keys=True, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None,
                        help="one seed only for explicit NONREGISTERED debug smoke")
    parser.add_argument("--steps", type=int, default=REGISTERED_TRAINING_STEPS,
                        help="changing steps makes output nonregistered")
    args = parser.parse_args(argv)
    run(args.out, seeds=SEEDS if args.seed is None else (args.seed,),steps=args.steps)


if __name__ == "__main__":
    main()
