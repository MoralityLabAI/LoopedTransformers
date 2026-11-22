"""INK-ROUTE-01 remote native checkpoint experiment.

Use --plan without NDIF access. The --run flag requires an NDIF_API_KEY;
no remote model data or native intervention result is produced by --plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .core import (
    ARMS, HF_READER_LAYER, TARGET_COORD, WIDTH, patch_vector,
    reconstruct_reader, sham_coordinate, vector_metrics,
)

MODEL_ID = "thinkingmachines/inkling-2-mini"
TRANSFORMERS_REVISION = "e3a65e2c21abb9c25a03704f2f25852f9ad03a18"
SCHEMA = "ink-route-01.native-dev.v1"
PROMPTS = {
    "capital": "The capital of France is",
    "geometry": "A triangle has exactly three",
    "story": "The sailor returned to the harbor after",
    "language": "The opposite of earlier is",
    "arithmetic": "Seven plus six equals",
    "agreement": "The musicians tuned their instruments before they",
    "instructions": "To unlock the door, she inserted the key into the",
    "geography": "The longest river in Africa is the",
}


def reader_call(pass_number: int, layer: int = HF_READER_LAYER) -> int:
    """0-based message_input call index, with 12 calls per later pass."""
    if pass_number not in (2, 3) or not 6 <= layer < 18:
        raise ValueError("reader must be a core layer in pass 2 or pass 3")
    return (pass_number - 2) * 12 + (layer - 6)


def plan(pass_number: int, limit: int) -> dict:
    if not 1 <= limit <= len(PROMPTS):
        raise ValueError("limit must be 1..8")
    config = {
        "schema_version": SCHEMA,
        "model_id": MODEL_ID,
        "transformers_revision": TRANSFORMERS_REVISION,
        "reader_layer_hf_zero_based": HF_READER_LAYER,
        "reader_layer_goodfire_one_based": HF_READER_LAYER + 1,
        "coordinate_zero_based": TARGET_COORD,
        "previous_loop_block3_bank_input": 1,
        "reader_pass": pass_number,
        "reader_call": reader_call(pass_number),
        "restore_call": reader_call(pass_number) + 1,
        "root_prompts": dict(list(PROMPTS.items())[:limit]),
        "arms": ["clean", *ARMS],
        "status": "development_only",
        "primary_readout": "beta3 change at layer-13 Q/K/V/MoE reader",
        "secondary_readout": "final reference-token log probability and top1",
        "scientific_effect_floor": "pending independent sham pilot; 0.10 beta is exploratory",
        "independent_unit": "distinct prompt root, with all arms within-root paired",
        "sealed_readout": False,
        "claims": "requires real NDIF execution; no mechanism confirmation from tests",
    }
    config["manifest_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config


def _np(proxy):
    return np.asarray(proxy.value.detach().float().cpu().numpy(), dtype=np.float64)


def apply_native_patch(bank3, arm: str, sham: int, target: int = TARGET_COORD):
    """Patch one final-token bank vector, returning a tensor to restore next call.

    These are torch/Nnsight proxy operations dispatched to the native model.
    Each arm has a matching numpy definition in core.patch_vector.
    """
    import torch

    current = bank3[:, -1, :]
    original = current.clone()
    patched = current.clone()
    if arm in ("zero_2027", "reader_rescue", "norm_preserving_zero"):
        patched[..., target] = 0
        if arm == "norm_preserving_zero":
            source_norm = torch.linalg.vector_norm(original.float(), dim=-1, keepdim=True)
            reduced_norm = torch.linalg.vector_norm(patched.float(), dim=-1, keepdim=True)
            patched = (patched.float() * (source_norm / reduced_norm.clamp_min(1e-12))).to(current.dtype)
    elif arm == "l2_sham":
        patched[..., sham] = original[..., sham] - original[..., target]
    elif arm == "swap":
        patched[..., target] = original[..., sham]
        patched[..., sham] = original[..., target]
    else:
        raise ValueError(f"unexpected arm {arm}")
    bank3[:, -1, :] = patched
    return original


def baseline(model, prompt, call):
    """Observe raw block-3 bank input, Q/K/V/MoE message, and all three ends."""
    llm = model.model.language_model
    with model.trace(prompt, remote=True) as tracer:
        visit17 = [llm.layers[17].output[0, -1, TARGET_COORD].clone().save() for _ in tracer.iter[:3]]
        for _ in tracer.iter[call]:
            bank2 = llm.message_input.inputs[0][0][0, -1, :].clone().save()
            bank3 = llm.message_input.inputs[0][1][0, -1, :].clone().save()
            message = llm.message_input.output[0, -1, :, :].clone().save()
        logits = model.output.logits[0, -1]
        top2 = logits.topk(2).indices.save()
        top2_logprobs = logits.log_softmax(-1).gather(0, top2).save()
    return {
        "bank2": _np(bank2), "bank3": _np(bank3), "message": _np(message),
        "layer17_coordinate_by_pass": [float(_np(x)) for x in visit17],
        "top2": [int(x) for x in top2.value.detach().cpu().tolist()],
        "top2_logprobs": [float(x) for x in _np(top2_logprobs)],
    }


def intervention(model, prompt, call, arm, sham, reference_id, clean_message):
    """Intervene at one reader call; restore source bank at the next call.

    Rescue restores the clean reader OUTPUT after the bank intervention.
    This isolates the reader's downstream pathway. The final state of the
    previous-pass bank is audited after the restoration hook.
    """
    llm = model.model.language_model
    with model.trace(prompt, remote=True) as tracer:
        for _ in tracer.iter[call]:
            bank3 = llm.message_input.inputs[0][1]
            original = apply_native_patch(bank3, arm, sham)
            after = bank3[0, -1, :].clone().save()
            before_rescue = llm.message_input.output[0, -1, :, :].clone().save()
            if arm == "reader_rescue":
                llm.message_input.output[:, -1, :, :] = clean_message.unsqueeze(0).to(
                    llm.message_input.output.dtype
                )
        for _ in tracer.iter[call + 1]:
            llm.message_input.inputs[0][1][:, -1, :] = original
            restored = llm.message_input.inputs[0][1][0, -1, :].clone().save()
        logits = model.output.logits[0, -1]
        top1 = logits.argmax(-1).save()
        logp = logits.log_softmax(-1)[reference_id].save()
    return {
        "bank3_after": _np(after),
        "message_before_rescue": _np(before_rescue),
        "restored_bank3": _np(restored),
        "top1": int(top1.value.item()),
        "reference_logprob": float(_np(logp)),
    }


def execute(out: Path, config: dict) -> None:
    if not os.getenv("NDIF_API_KEY"):
        raise SystemExit("NDIF_API_KEY is required for --run (request it from login.ndif.us)")
    import torch
    from nnsight import CONFIG, TransformersModel

    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
    model = TransformersModel(MODEL_ID, task="image-text-to-text")
    call = config["reader_call"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    records = []
    for root_id, prompt in config["root_prompts"].items():
        clean = baseline(model, prompt, call)
        if clean["bank3"].shape != (WIDTH,) or clean["message"].shape != (4, WIDTH):
            raise RuntimeError("unexpected native model shape: verify checkpoint/deployment")
        sham = sham_coordinate(root_id)
        clean_fits = reconstruct_reader(clean["bank2"], clean["bank3"], clean["message"])
        if any(f.reconstruction_relative_error > 1e-3 for f in clean_fits):
            raise RuntimeError("clean reader reconstruction failed: inspect native hooks")
        traces = {"clean_bank2": clean["bank2"], "clean_bank3": clean["bank3"],
                  "clean_message": clean["message"]}
        for arm in ARMS:
            trial = intervention(
                model, prompt, call, arm, sham, clean["top2"][0],
                torch.as_tensor(clean["message"], dtype=torch.float32),
            )
            before, after = clean["bank3"], trial["bank3_after"]
            predicted = patch_vector(before, arm, sham)
            # BF16/native assignment can quantize. Check against realized displacement.
            if not np.allclose(predicted, after, rtol=0.02, atol=0.05):
                raise RuntimeError(f"native patch does not match registered arm: {root_id}/{arm}")
            if not np.allclose(trial["restored_bank3"], before, rtol=0, atol=0.05):
                raise RuntimeError(f"bank restoration failed: {root_id}/{arm}")
            fits = reconstruct_reader(clean["bank2"], after, trial["message_before_rescue"])
            if any(f.reconstruction_relative_error > 1e-3 for f in fits):
                raise RuntimeError(f"patched reader reconstruction failed: {root_id}/{arm}")
            traces[f"{arm}_bank3"] = after
            traces[f"{arm}_native_message"] = trial["message_before_rescue"]
            records.append({
                "root_id": root_id, "arm": arm,
                "reader_pass": config["reader_pass"], "reader_call": call,
                "sham_coordinate": sham, "reference_token_id": clean["top2"][0],
                "clean_top1_token_id": clean["top2"][0], "native_top1_token_id": trial["top1"],
                "clean_reference_logprob": clean["top2_logprobs"][0],
                "patched_reference_logprob": trial["reference_logprob"],
                "layer17_raw_coordinate_by_pass": clean["layer17_coordinate_by_pass"],
                "clean_beta3_qkvm": [f.beta3 for f in clean_fits],
                "patched_beta3_qkvm": [f.beta3 for f in fits],
                "reconstruction_relative_error": [f.reconstruction_relative_error for f in fits],
                **vector_metrics(before, after),
                "status": "development_only",
            })
        np.savez_compressed(out / f"vectors_{root_id}.npz", **traces)
    with (out / "records.jsonl").open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.iterdir() if p.is_file()}
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"roots": len(config["root_prompts"]), "intervention_rows": len(records),
                      "output": str(out), "verdict": "development_only_native_probe"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute native NDIF calls (requires NDIF_API_KEY)")
    parser.add_argument("--pass-number", type=int, choices=(2, 3), default=2)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/ink_route_01"))
    args = parser.parse_args()
    config = plan(args.pass_number, args.limit)
    if not args.run:
        print(json.dumps(config, indent=2))
    else:
        execute(args.out, config)


if __name__ == "__main__":
    main()
