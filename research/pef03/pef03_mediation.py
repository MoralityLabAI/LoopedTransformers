"""PEF-03G: posthoc natural-bit-flip vs head-subspace hybrid mediation.

This audit is explicitly posthoc and runs only on the exact SHA256-sealed
original PEF-03 checkpoints. Original and counterfactual bit histories
are both on the model's natural reachable trajectory. Rowspace-only and
nullspace-only transplants are hybrid counterfactuals, possibly off-manifold.

The exact decomposition at each visit is:
    Delta_natural = Delta_row + Delta_null + Interaction,
where Interaction = Y(donor)-Y(row)-Y(null)+Y(original).
This identity is algebraic and does not imply linear causal mediation;
the nonlinear interaction can be large. The final head is affine, giving
Delta_null=Interaction=0 at visit 7 within float32 tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.pef03.pef03 import L, SEEDS, hash_weights, load_frozen, split_catalog, tensor
from research.pef03.pef03_future import step_states


def _resume_logits(model, state: torch.Tensor, bits: torch.Tensor, visit: int):
    for j in range(visit + 1, L):
        state = model.step(state, model.bit_embedding(bits[:, j]))
    return model.head(state)


def audit_seed(seed: int, directory: Path) -> dict:
    torch.set_num_threads(2)
    checkpoint = directory / f"seed{seed}.frozen.pt"
    receipt = json.loads((directory / f"seed{seed}.json").read_text(encoding="utf8"))
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if receipt["training"]["checkpoint_sha256"] != fingerprint:
        raise AssertionError("original checkpoint SHA256 mismatch")
    model = load_frozen(checkpoint, seed).eval()
    initial_hash = hash_weights(model)
    originals = tensor(split_catalog()["heldout"])
    donors = originals.clone()
    donors[:, 0] = 1 - donors[:, 0]
    if not torch.equal(donors[:, 1:], originals[:, 1:]):
        raise AssertionError("oldest-bit intervention changed newer history")
    if len(originals) != 52:
        raise AssertionError("unexpected frozen heldout cohort")

    with torch.no_grad():
        w = model.head.weight.detach().double()
        _, sv, vh = torch.linalg.svd(w, full_matrices=True)
        if w.shape != (4, 16) or int((sv > 1e-8).sum()) != 4:
            raise AssertionError("frozen head dimension/rank mismatch")
        row = vh[:4].T.contiguous()
        null = vh[4:].T.contiguous()
        row_projector, null_projector = row @ row.T, null @ null.T
        if float((w @ null).abs().max()) > 1e-10:
            raise AssertionError("head nullspace construction failed")
        if float((row_projector+null_projector-torch.eye(16, dtype=torch.float64)).abs().max()) > 1e-10:
            raise AssertionError("incomplete head subspace decomposition")

        original_states = step_states(model, originals)
        donor_states = step_states(model, donors)
        original_logits = model.head(original_states[-1])
        donor_logits = model.head(donor_states[-1])
        natural_delta = donor_logits - original_logits
        task_labels = (2 * originals[:, -2] + originals[:, -1])
        donor_task_labels = (2 * donors[:, -2] + donors[:, -1])
        if not torch.equal(task_labels, donor_task_labels):
            raise AssertionError("oldest-bit flip changed supervised target")
        baseline_pred = original_logits.argmax(1)
        natural_changes = int((donor_logits.argmax(1) != baseline_pred).sum())

        records = []
        for visit in range(L):
            orig = original_states[visit].double()
            donor = donor_states[visit].double()
            row_hybrid = (donor @ row_projector + orig @ null_projector).float()
            null_hybrid = (orig @ row_projector + donor @ null_projector).float()

            # Every arm uses the same later original inputs (identical to
            # donor later bits), keeping post-patch input interventions fixed.
            standard = _resume_logits(model, orig.float(), originals, visit)
            natural = _resume_logits(model, donor.float(), originals, visit)
            row_output = _resume_logits(model, row_hybrid, originals, visit)
            null_output = _resume_logits(model, null_hybrid, originals, visit)

            if float((standard-original_logits).abs().max()) > 1e-4:
                raise AssertionError("restarting original trajectory changed output")
            if float((natural-donor_logits).abs().max()) > 1e-4:
                raise AssertionError("restarting natural donor trajectory changed output")

            d_natural = natural - standard
            d_row = row_output - standard
            d_null = null_output - standard
            interaction = natural - row_output - null_output + standard
            reconstruction = d_natural - (d_row + d_null + interaction)
            if float(reconstruction.abs().max()) > 1e-5:
                raise AssertionError("paired decomposition did not reconstruct output")
            records.append({
                "patched_after_visit": visit,
                "remaining_visits": L - visit - 1,
                "full_natural_history_flip_is_on_manifold": True,
                "row_and_null_hybrids_may_be_off_manifold": True,
                "paired_count": len(originals),
                "natural_full_history_flip_task_class_changes": natural_changes,
                "rowspace_only_task_class_changes": int((row_output.argmax(1) != baseline_pred).sum()),
                "nullspace_only_task_class_changes": int((null_output.argmax(1) != baseline_pred).sum()),
                "mean_natural_final_logit_l2": float(d_natural.norm(dim=1).mean()),
                "mean_rowspace_patch_final_logit_l2": float(d_row.norm(dim=1).mean()),
                "mean_nullspace_patch_final_logit_l2": float(d_null.norm(dim=1).mean()),
                "mean_interaction_final_logit_l2": float(interaction.norm(dim=1).mean()),
                "max_algebraic_reconstruction_error": float(reconstruction.abs().max()),
            })

        if records[-1]["mean_nullspace_patch_final_logit_l2"] > 1e-4:
            raise AssertionError("nullspace patch changed terminal affine head")
        if records[-1]["mean_interaction_final_logit_l2"] > 1e-4:
            raise AssertionError("terminal affine head acquired false nonlinear interaction")

        # Local derivative in the *observed old-bit difference direction*,
        # rather than arbitrary unit vectors in the entire head nullspace.
        reference_original = originals[:1]
        reference_donor = donors[:1]
        ref_states = [s[:1] for s in original_states]
        donor_ref_states = [s[:1] for s in donor_states]

    jvp_rows = []
    for visit in (0, 3, 6, 7):
        base = ref_states[visit][0].detach().clone()
        contrast = donor_ref_states[visit][0].detach().double() - base.double()
        along_null = (contrast @ null_projector).float()

        def future_head(z):
            return _resume_logits(model, z.unsqueeze(0), reference_original, visit)[0]

        with torch.enable_grad():
            _, tangent = torch.autograd.functional.jvp(
                future_head, (base,), (along_null,), create_graph=False
            )
        jvp_rows.append({
            "visit": visit,
            "remaining_visits": L-1-visit,
            "natural_paired_nullspace_tangent_norm": float(along_null.norm()),
            "local_future_logits_jvp_along_natural_nullspace_delta_l2": float(tangent.norm()),
            "not_equivalent_to_discrete_intervention": True,
        })
    if jvp_rows[-1]["local_future_logits_jvp_along_natural_nullspace_delta_l2"] > 1e-5:
        raise AssertionError("terminal head saw a nullspace directional tangent")
    if hash_weights(model) != initial_hash:
        raise AssertionError("posthoc audit modified frozen weights")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fingerprint:
        raise AssertionError("posthoc audit modified source checkpoint")
    data = {
        "campaign": "PEF-03G",
        "posthoc": True,
        "checkpoint_sha256": fingerprint,
        "seed": seed,
        "n_heldout_original_histories": len(originals),
        "all_counterfactuals_oldest_bit_only": True,
        "supervised_targets_identical_in_pairs": True,
        "frozen_checkpoint_unchanged": True,
        "head_rank": 4,
        "head_nullspace_dimension": 12,
        "natural_oldest_bit_flip_task_class_changes": natural_changes,
        "mixed_subspace_mediation": records,
        "paired_direction_local_jvp": jvp_rows,
        "no_annealing_or_retraining": True,
        "hybrid_state_interpretation_cautioned": True,
    }
    (directory / f"seed{seed}.mediation.json").write_text(
        json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args(argv)
    results = [audit_seed(seed, args.dir) for seed in SEEDS]
    summary = {
        "campaign": "PEF-03G",
        "posthoc": True,
        "results": [
            {
                "seed": r["seed"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "natural_class_changes": r["natural_oldest_bit_flip_task_class_changes"],
                "visit0_row_patch_class_changes": r["mixed_subspace_mediation"][0]["rowspace_only_task_class_changes"],
                "visit0_null_patch_class_changes": r["mixed_subspace_mediation"][0]["nullspace_only_task_class_changes"],
                "visit0_interaction_mean_l2": r["mixed_subspace_mediation"][0]["mean_interaction_final_logit_l2"],
                "visit0_natural_mean_l2": r["mixed_subspace_mediation"][0]["mean_natural_final_logit_l2"],
            } for r in results
        ],
    }
    (args.dir / "mediation_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
