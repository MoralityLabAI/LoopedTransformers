"""PEF-03F: posthoc future reactivation of final-head dark-memory directions.

Frozen, SHA-pinned PEF-03 checkpoints. At intermediate visits, swap only
the 12D nullspace component of a memory state with its counterpart from
an oldest-bit-flipped history, then resume identical later inputs.
The patch is a hybrid state and may be off the model's normal manifold.
The same replacement at the *final* visit is task-logit invariant.

Separately differentiates the future-logit map through the remaining
recurrent visits. Nonzero J_future @ N shows a locally head-invisible
direction becoming future-head-visible through subsequent processing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.pef03.pef03 import (
    L, SEEDS, hash_weights, load_frozen, split_catalog, tensor,
)


def step_states(model, bits):
    """Collect all consecutive post-visit memory states without changing weights."""
    state = model.memory_init.unsqueeze(0).expand(len(bits), -1)
    out = []
    for visit in range(L):
        state = model.step(state, model.bit_embedding(bits[:, visit]))
        out.append(state)
    return out


def future_reactivation_audit(seed: int, root: Path) -> dict:
    receipt = json.loads((root / f"seed{seed}.json").read_text(encoding="utf-8"))
    ckpt = root / f"seed{seed}.frozen.pt"
    actual = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if receipt["training"]["checkpoint_sha256"] != actual:
        raise AssertionError("frozen source checkpoint integrity failure")
    model = load_frozen(ckpt, seed).eval()
    frozen_checksum = hash_weights(model)
    observed = tensor(split_catalog()["heldout"])
    changed = observed.clone()
    changed[:, 0] = 1 - changed[:, 0]
    assert torch.equal(observed[:, 1:], changed[:, 1:])
    with torch.no_grad():
        w = model.head.weight.detach().double()
        _, s, vh = torch.linalg.svd(w, full_matrices=True)
        rank = int((s > 1e-8).sum())
        if rank != 4 or tuple(w.shape) != (4, 16):
            raise AssertionError("head rank changed")
        row, null = vh[:4].T, vh[4:].T
        PR = row @ row.T
        PN = null @ null.T
        actual_states = step_states(model, observed)
        donor_states = step_states(model, changed)
        original_logits = model.head(actual_states[-1])
        standard_logits = model.logits(observed)
        if float((original_logits-standard_logits).abs().max()) > 1e-4:
            raise AssertionError("post-visit states are not original recurrence states")
        rows = []
        for visit in range(L):
            current = actual_states[visit].double()
            donor = donor_states[visit].double()
            patched_state = (current @ PR + donor @ PN).float()
            if float(((patched_state.double() @ null) -
                      (donor @ null)).abs().max()) > 2e-5:
                raise AssertionError("donor nullspace state transfer failed")
            state = patched_state
            for future in range(visit+1, L):
                state = model.step(state, model.bit_embedding(observed[:, future]))
            output = model.head(state)
            diff = output - original_logits
            rows.append({
                "patched_after_visit_zero_based": visit,
                "remaining_recurrent_visits": L-visit-1,
                "nullspace_transplant_only": True,
                "later_input_bits_unchanged": True,
                "hybrid_off_manifold_possible": True,
                "mean_final_task_logit_l2": float(diff.norm(dim=1).mean()),
                "max_abs_final_logit_shift": float(diff.abs().max()),
                "changed_predicted_task_classes": int(
                    (output.argmax(1) != original_logits.argmax(1)).sum()
                ),
                "paired_test_count": len(observed),
            })
        if rows[-1]["max_abs_final_logit_shift"] > 1e-4:
            raise AssertionError("terminal head-nullspace transplant changed head output")

    # Differential test at the *first* heldout history; the Jacobian
    # is with respect to the intermediate memory, along its actual
    # trajectory and with the later input bits frozen.
    reference = tensor([split_catalog()["heldout"][0]])
    with torch.no_grad():
        ref_states = step_states(model, reference)
    derivatives = []
    for visit in (0, 3, 6, 7):
        ref_state = ref_states[visit][0].detach()
        def suffix_head(value):
            state = value.unsqueeze(0)
            for future in range(visit+1, L):
                state = model.step(state, model.bit_embedding(reference[:,future]))
            return model.head(state)[0]
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(
                suffix_head, ref_state, vectorize=False
            )
        null_to_logits = jacobian.double() @ null
        norm = float(torch.linalg.matrix_norm(null_to_logits, ord="fro"))
        derivatives.append({
            "after_visit": visit,
            "remaining_visits": L-1-visit,
            "local_frobenius_norm_future_logit_jacobian_on_head_nullspace": norm,
        })
    if derivatives[-1]["local_frobenius_norm_future_logit_jacobian_on_head_nullspace"] > 1e-8:
        raise AssertionError("terminal nullspace acquired a spurious Jacobian")
    if hash_weights(model) != frozen_checksum:
        raise AssertionError("patch diagnostic modified trained weights")
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != actual:
        raise AssertionError("patch diagnostic modified checkpoint")
    result = {
        "campaign": "PEF-03F",
        "posthoc_after_frozen_mechanism_selection": True,
        "seed": seed,
        "source_checkpoint_sha256": actual,
        "final_task_head_rank": 4,
        "head_invisible_nullspace_dimension": 12,
        "paired_oldest_bit_source_interventions": len(observed),
        "counterfactual_later_input_bits_fixed": True,
        "posthoc_nullspace_transplants_are_hybrid_states": True,
        "frozen_checkpoint_unchanged": True,
        "per_visit_nullspace_transplant": rows,
        "reference_local_jacobian_reactivation": derivatives,
    }
    (root / f"seed{seed}.future.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    results = [future_reactivation_audit(seed, args.dir) for seed in SEEDS]
    output = {
        "campaign": "PEF-03F",
        "posthoc": True,
        "source_frozen_seeds": list(SEEDS),
        "results": [{
            "seed": x["seed"],
            "checkpoint_sha256": x["source_checkpoint_sha256"],
            "visit0_null_patch_task_class_changes":
                x["per_visit_nullspace_transplant"][0]["changed_predicted_task_classes"],
            "visit0_mean_final_logit_l2":
                x["per_visit_nullspace_transplant"][0]["mean_final_task_logit_l2"],
            "visit7_null_patch_task_class_changes":
                x["per_visit_nullspace_transplant"][-1]["changed_predicted_task_classes"],
            "visit0_local_future_head_on_nullspace_frobenius":
                x["reference_local_jacobian_reactivation"][0][
                    "local_frobenius_norm_future_logit_jacobian_on_head_nullspace"],
            "visit7_local_future_head_on_nullspace_frobenius":
                x["reference_local_jacobian_reactivation"][-1][
                    "local_frobenius_norm_future_logit_jacobian_on_head_nullspace"],
        } for x in results],
    }
    (args.dir / "future_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
