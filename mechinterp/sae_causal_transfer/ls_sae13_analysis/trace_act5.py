"""LS-SAE13 microtrace: locate the first ACT5 B1/B2 projection mismatch.

Known-puzzle exploratory follow-up to the audited Intel Xeon Linux pilot.
Runs ACT1..4 once at B1 for each known board, then compares the exact same
ACT4 anchor at B1 and B2 through one complete ACT5 (3H * (6L + 1H)).
The first row is unedited and the second B2 row is its clean duplicate.
PyTorch forward hooks record the first row's inputs and outputs of all
qkv/o/gate_up/down projections in both TRM layers, in execution order.
A focused second pass records actual tensors around the earliest mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.active_reasoning_v1 import run_native as RN
from mechinterp.sae_causal_transfer.active_reasoning_v1 import schedule as S

SCHEMA = "LS-SAE13/ACT5-batch-1-vs-2-microtrace-v1"
POSITIONS = (23, 40)
PROJECTS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def cpu_f32_first(x):
    if not torch.is_tensor(x) or x.ndim < 2:
        raise ValueError("expected batched tensor")
    a = x[0].detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    if not np.isfinite(a).all():
        raise ArithmeticError("nonfinite hooked tensor")
    return a.copy()


def hsh(x):
    a = np.asarray(x, dtype="<f4")
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def anchor_after_act4(model, x):
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        q = RN.question(x)
        inp = TRM.encode_questions([q], "cpu")
        x_emb = model.encode_inputs(inp)
        zh, zl = model.init_states(1)
        for _ in range(4):
            zh, zl = model.act_step(zh, zl, x_emb)
    return {"zh": zh.detach().clone(), "zl": zl.detach().clone(),
            "x_emb": x_emb.detach().clone()}


def trace_once(model, anchor, batch: int, focus: int | None = None):
    if batch not in (1, 2):
        raise ValueError("only B1/B2 permitted")
    events = []
    focused = None
    handles = []

    def hook_for(name):
        def hook(_mod, inputs, output):
            nonlocal focused
            index = len(events)
            ins = cpu_f32_first(inputs[0])
            outs = cpu_f32_first(output)
            events.append({
                "index": index, "projection": name,
                "input_sha256": hsh(ins), "output_sha256": hsh(outs),
                "input_shape": list(ins.shape), "output_shape": list(outs.shape),
            })
            if index == focus:
                focused = {"input": ins, "output": outs, "projection": name}
        return hook

    for name, mod in model.named_modules():
        if name.startswith("layers.") and name.rsplit(".", 1)[-1] in PROJECTS:
            handles.append(mod.register_forward_hook(hook_for(name)))
    try:
        with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
            zh = anchor["zh"].repeat(batch, 1, 1)
            zl = anchor["zl"].repeat(batch, 1, 1)
            emb = anchor["x_emb"].repeat(batch, 1, 1)
            zh, zl = model.act_step(zh, zl, emb)
            logits = model.digit_logits(zh)
            out = {"zH_sha256": hsh(cpu_f32_first(zh)),
                   "zL_sha256": hsh(cpu_f32_first(zl)),
                   "logits_sha256": hsh(cpu_f32_first(logits)),
                   "logits": cpu_f32_first(logits)}
    finally:
        for h in handles:
            h.remove()
    expected = model.h_cycles * (model.l_cycles + 1) * len(model.layers) * len(PROJECTS)
    if len(events) != expected:
        raise ArithmeticError(f"expected {expected} projection calls; got {len(events)}")
    if focus is not None and focused is None:
        raise ArithmeticError("focused projection not captured")
    return events, out, focused


def diff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("mismatched focused projection dimensions")
    d = b - a
    return {"max_abs": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d * d))),
            "fro": float(np.linalg.norm(d.ravel())),
            "equal_float32_bits": hsh(a) == hsh(b)}


def trace_board(model, x, target):
    anchor = anchor_after_act4(model, x)
    one, final_one, _ = trace_once(model, anchor, 1)
    two, final_two, _ = trace_once(model, anchor, 2)
    if [e["projection"] for e in one] != [e["projection"] for e in two]:
        raise ArithmeticError("different hook execution order by batch")
    different_in = [i for i, (a, b) in enumerate(zip(one, two))
                    if a["input_sha256"] != b["input_sha256"]]
    different_out = [i for i, (a, b) in enumerate(zip(one, two))
                     if a["output_sha256"] != b["output_sha256"]]
    first = min(different_in + different_out) if different_in or different_out else None
    focused = None
    if first is not None:
        _, _, cf1 = trace_once(model, anchor, 1, focus=first)
        _, _, cf2 = trace_once(model, anchor, 2, focus=first)
        focused = {
            "index": first, "projection": one[first]["projection"],
            "same_input_bits": one[first]["input_sha256"] == two[first]["input_sha256"],
            "same_output_bits": one[first]["output_sha256"] == two[first]["output_sha256"],
            "input": diff(cf1["input"], cf2["input"]),
            "output": diff(cf1["output"], cf2["output"]),
        }
    logdiff = diff(final_one["logits"], final_two["logits"])
    blank = np.array(S.blanks(x), dtype=np.int64)
    flips = np.argmax(final_one["logits"][blank], axis=1) != np.argmax(final_two["logits"][blank], axis=1)
    return {
        "anchor_zH_sha256": hsh(cpu_f32_first(anchor["zh"])),
        "anchor_zL_sha256": hsh(cpu_f32_first(anchor["zl"])),
        "input_embedding_sha256": hsh(cpu_f32_first(anchor["x_emb"])),
        "projection_call_count": len(one),
        "first_input_mismatch_index": different_in[0] if different_in else None,
        "first_output_mismatch_index": different_out[0] if different_out else None,
        "first_mismatch_focus": focused,
        "projection_digest_pairs": [
            {"index": a["index"], "name": a["projection"],
             "B1_in": a["input_sha256"], "B2_in": b["input_sha256"],
             "B1_out": a["output_sha256"], "B2_out": b["output_sha256"]}
            for a, b in zip(one, two)],
        "ACT5": {"logits_diff": logdiff,
                 "blank_argmax_flips": int(np.sum(flips)),
                 "B1_zH": final_one["zH_sha256"], "B2_zH": final_two["zH_sha256"],
                 "B1_zL": final_one["zL_sha256"], "B2_zL": final_two["zL_sha256"],
                 "B1_logits": final_one["logits_sha256"], "B2_logits": final_two["logits_sha256"]},
    }


def run(source: Path, out: Path):
    src, _ = RN.load_source(source)
    by_pos = {x["pool_position"]: x for x in src["pool"]}
    model, checkpoint_sha = RN.load_model()
    boards = []
    for pos in POSITIONS:
        row = by_pos[pos]
        report = trace_board(model, row["x"], row["target"])
        boards.append({"pool_position": pos, "source_file": row["source_file"], **report})
        print("LS_SAE13_MICROTRACE", pos, "calls", report["projection_call_count"],
              "first_input", report["first_input_mismatch_index"],
              "first_output", report["first_output_mismatch_index"],
              "focus", report["first_mismatch_focus"],
              "ACT5_max_delta", report["ACT5"]["logits_diff"]["max_abs"],
              flush=True)
    data = {"schema": SCHEMA, "scope": "exploratory_engineering",
            "platform": RN.platform_fingerprint(), "checkpoint_sha256": checkpoint_sha,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "positions": list(POSITIONS), "boards": boards}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print("LS_SAE13_MICROTRACE_SHA256", hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.out)


if __name__ == "__main__":
    main()
