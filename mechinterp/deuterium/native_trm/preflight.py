"""DEUTERIUM-NATIVE-01 source-only provenance and genuine four-pass parity gate.

Runs on an original published LGA-02 checkpoint and the ALREADY EXPOSED
historical seed 31043. It never generates 731055 or 731057; scientific
steering and protected test scoring belong to separate later stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path

import torch

from mechinterp.loopscope04.gradient_flow import (
    TinyRecursiveModel, fixed_data, tensor_int64_sha
)
from mechinterp.loopscope05.fit_and_freeze import load_frozen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHEMA = "DEUTERIUM-NATIVE-01/native-source-only-preflight-v1"
TEACHERS = (103, 211, 307)
HISTORICAL_SEED = 31043
HISTORICAL_N = 256
RESERVED_CALIBRATION_SEED = 731055
RESERVED_HELDOUT_SEED = 731057
MODEL_SHA = "d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
META_SHA = "7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
SOURCE_BLOBS = {
    "mechinterp/loopscope04/gradient_flow.py": "ec1c601c321230b14f974bb76ba3610da6e9aa8f",
    "mechinterp/loopscope05/fit_and_freeze.py": "d933ce68cce6bc08670b8f5221d646ee11896629",
    "mechinterp/loopscope05/terminal_sae_test.py": "e9a9e77b306d1939e106267e8fd3cd1d41f8582d",
}
TARGET_NAME = "block.chan_mix.2.weight"
FORWARD_TOLERANCE = 1e-6
POSED_CHANGE_FLOOR = 1e-8
DOSE_RELATIVE_TOLERANCE = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\\x00" + raw).hexdigest()


def check_source_bytes():
    actual = {path: git_blob_sha(REPO / path) for path in SOURCE_BLOBS}
    if actual != SOURCE_BLOBS:
        raise RuntimeError("original LOOPSCOPE model/loader source blob changed")
    return actual


def four_pass_manual(model: TinyRecursiveModel, inputs: torch.Tensor):
    """Manual execution of the genuine frozen recurrent block four times."""
    h = model.initial_state(inputs)
    for _ in range(4):
        h = model.block(h)
    return model.head(h)


def closure_gate(model: TinyRecursiveModel, x: torch.Tensor, seed: int):
    """Tests real hidden activation and recurrent WEIGHT interventions."""
    model.eval().requires_grad_(False)
    if model.n_steps != 4 or sum(p.numel() for p in model.parameters()) != 44001:
        raise RuntimeError("four-pass model shape/parameter-count changed")
    with torch.no_grad():
        clean = model(x)
        native = four_pass_manual(model, x)
        forward_error = float((clean - native).abs().max())
        states = model.reason_states(x[:1])
        if len(states) != 5:
            raise RuntimeError("native model does not have five physical states h0..h4")
        identity = model.head(model.block(model.block(states[2])))
        zero_activation_error = float((clean[:1] - identity).abs().max())
        if max(forward_error, zero_activation_error) > FORWARD_TOLERANCE:
            raise RuntimeError("native forward/zero-hidden-edit replay failed")

        token = 10
        delta_native = states[3][0, token] - states[2][0, token]
        activation_dose = 0.25 * float(delta_native.norm())
        if activation_dose <= 0:
            raise RuntimeError("native transition has degenerate source-only dose")
        gen = torch.Generator(device="cpu").manual_seed(79000 + seed)
        direction = torch.randn(48, generator=gen)
        edit = direction / direction.norm().clamp_min(1e-12) * activation_dose
        h2 = states[2].clone()
        h2[0, token] += edit
        reached = model.head(model.block(model.block(h2)))
        activation_effect = float((reached - clean[:1]).abs().max())
        if activation_effect <= POSED_CHANGE_FLOOR:
            raise RuntimeError("native hidden edit did not propagate through remaining loops")

        weight = model.block.chan_mix[2].weight
        original = weight.detach().clone()
        if tuple(original.shape) != (48, 128):
            raise RuntimeError("recurrent target parameter changed shape")
        g = torch.Generator(device="cpu").manual_seed(88000 + seed)
        u = torch.randn(48, generator=g)
        v = torch.randn(128, generator=g)
        intended = torch.outer(u, v)
        dose = float(original.norm()) * 0.01
        intended = intended * (dose / intended.norm())
        if not bool(torch.isfinite(intended).all()):
            raise RuntimeError("nonfinite rank-one recurrent weight direction")
        applied = None
        patched = None
        patched_manual = None
        try:
            weight.copy_(original + intended)
            applied = weight.detach().clone() - original
            patched = model(x)
            patched_manual = four_pass_manual(model, x)
        finally:
            weight.copy_(original)
        restored = model(x)
        actual_dose = float(applied.norm())
        applied_relative_error = abs(actual_dose - dose) / dose
        native_weight_edit_error = float((patched - patched_manual).abs().max())
        restoration_error = float((restored - clean).abs().max())
        weight_restoration_error = float((weight - original).abs().max())
        weight_effect = float((patched - clean).abs().max())
        if (max(native_weight_edit_error, restoration_error, weight_restoration_error) >
                FORWARD_TOLERANCE):
            raise RuntimeError("real recurrent weight intervention/restore parity failed")
        if applied_relative_error > DOSE_RELATIVE_TOLERANCE:
            raise RuntimeError("actual FP32 recurrent weight dose differs from intended")
        if weight_effect <= POSED_CHANGE_FLOOR:
            raise RuntimeError("recurrent weight intervention had no native effect")
        return {
            "full_manual_replay_max_logit_error": forward_error,
            "zero_hidden_edit_max_logit_error": zero_activation_error,
            "hidden_edit_native_max_logit_change": activation_effect,
            "hidden_edit_intended_l2": activation_dose,
            "hidden_edit_actual_l2": float(edit.norm()),
            "weight_target": TARGET_NAME,
            "weight_shape": list(original.shape),
            "weight_edit_conceptual_rank": 1,
            "weight_edit_intended_frobenius": dose,
            "weight_edit_actual_frobenius": actual_dose,
            "weight_edit_actual_relative_dose_error": applied_relative_error,
            "weight_edit_max_matrix_delta_error": float((applied - intended).abs().max()),
            "weight_edit_native_max_logit_change": weight_effect,
            "weight_edit_native_manual_max_logit_error": native_weight_edit_error,
            "weight_restore_max_logit_error": restoration_error,
            "weight_restore_max_parameter_error": weight_restoration_error,
            "all_four_recurrent_weight_uses_edited": True,
        }


def run(teacher_bundle: Path, teacher_meta: Path, out: Path) -> dict:
    torch.set_num_threads(2)
    if sha256(teacher_bundle) != MODEL_SHA or sha256(teacher_meta) != META_SHA:
        raise RuntimeError("original LGA-02 teacher artifact SHA256 mismatch")
    source_blobs = check_source_bytes()
    meta, bundle = load_frozen(teacher_bundle, teacher_meta)
    if meta.get("teacher_init_seeds") != list(TEACHERS) or meta.get("mode") != "full":
        raise RuntimeError("original teacher roster/full training provenance invalid")

    # Historical old LGA-01/02 evaluation set: already generated, already scored.
    x, y = fixed_data(HISTORICAL_N, HISTORICAL_SEED)
    prior = meta["prior_test_used_only_for_reproduction"]
    if (prior.get("seed") != HISTORICAL_SEED or prior.get("n") != HISTORICAL_N or
        tensor_int64_sha(x) != prior.get("x_sha256") or
        tensor_int64_sha(y) != prior.get("y_sha256")):
        raise RuntimeError("historical source/target byte lineage mismatch")

    receipt = {
        "schema": SCHEMA,
        "status": "PREFLIGHT_RUNNING",
        "evidence_class": "HISTORICAL_SOURCE_ENGINEERING_ONLY",
        "original_teacher_bundle_sha256": MODEL_SHA,
        "original_teacher_meta_sha256": META_SHA,
        "original_source_git_blobs": source_blobs,
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "teacher_seeds": list(TEACHERS),
        "historical_source_seed": HISTORICAL_SEED,
        "historical_source_n": HISTORICAL_N,
        "historical_x_sha256": tensor_int64_sha(x),
        "historical_y_sha256": tensor_int64_sha(y),
        "new_calibration_seed_reserved": RESERVED_CALIBRATION_SEED,
        "new_heldout_seed_reserved": RESERVED_HELDOUT_SEED,
        "new_calibration_generated": False,
        "new_heldout_generated": False,
        "new_heldout_labels_read": False,
        "original_canonical_tensor_check": "PASS",
        "independent_original_reference_logits": "UNAVAILABLE",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "models": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\\n")
    for seed in TEACHERS:
        model = TinyRecursiveModel(48, 4)
        model.load_state_dict(bundle["models"][str(seed)], strict=True)
        model.eval().requires_grad_(False)
        correct = 0
        total = 0
        max_forward_error = 0.0
        with torch.no_grad():
            for left in range(0, len(x), 32):
                xx, yy = x[left:left + 32], y[left:left + 32]
                logits = model(xx)
                reference = four_pass_manual(model, xx)
                max_forward_error = max(
                    max_forward_error, float((logits - reference).abs().max()))
                correct += int((logits.argmax(-1) == yy).sum())
                total += int(yy.numel())
            accuracy = correct / total
            historical = float(meta["teacher_receipts"][str(seed)]["reproduced_accuracy"])
            if abs(accuracy - historical) > (1 / total + 1e-8):
                raise RuntimeError(f"historical clean accuracy drift for {seed}")
            if max_forward_error > FORWARD_TOLERANCE or accuracy < 0.95:
                raise RuntimeError(f"native clean source/replay gate failed for {seed}")
            controls = closure_gate(model, x[:4], seed)
        receipt["models"][str(seed)] = {
            "clean_old_source_accuracy": accuracy,
            "original_train_meta_old_source_accuracy": historical,
            "old_source_tokens": total,
            "full_manual_forward_max_logit_error": max_forward_error,
            **controls,
        }
        out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\\n")
    receipt["status"] = "PREFLIGHT_PASS"
    receipt["new_score_comparison_completed"] = False
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\\n")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-bundle", type=Path, required=True)
    p.add_argument("--teacher-meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        r = run(args.teacher_bundle, args.teacher_meta, args.out)
    except Exception as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "schema": SCHEMA, "status": "BLOCKED_SOURCE_PREFLIGHT",
            "new_calibration_generated": False, "new_heldout_generated": False,
            "new_heldout_labels_read": False,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        }, indent=2, sort_keys=True) + "\\n")
        raise
    print("DEUTERIUM_NATIVE_SOURCE_PREFLIGHT", r["status"], flush=True)


if __name__ == "__main__":
    main()
