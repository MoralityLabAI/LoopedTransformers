"""Isolate LS-SAE13's first ACT5 B1/B2 down-projection rounding discrepancy.

Exploratory diagnostic on the two previously observed positions (23,40).
Capture the first actual layers.0.down_proj input/output at ACT5 for B1/B2
from an identical ACT4 anchor, then replay the *same input* at B1 and B2
with controlled float32 CPU thread counts and MKLDNN toggles.

Outputs an auditable JSON report and an NPZ of actual float32 input/weight
fixtures. Its claim is numerical integrity, not structural edit efficacy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mechinterp.sae_causal_transfer.active_reasoning_v1 import run_native as RN
from mechinterp.sae_causal_transfer.ls_sae13_analysis.trace_act5 import anchor_after_act4

SCHEMA = "LS-SAE13/down-projection-isolated-B1-B2-v1"
POSITIONS = (23, 40)


def f32(x):
    if not torch.is_tensor(x):
        raise TypeError("expected torch.Tensor")
    a = x.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return a.copy()


def hash_f32(x):
    a = f32(x) if torch.is_tensor(x) else np.asarray(x, dtype="<f4")
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def difference(x, y):
    a, b = f32(x).astype(np.float64), f32(y).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("incompatible arrays")
    d = b - a
    return {
        "bit_identical": bool(np.array_equal(f32(x), f32(y))),
        "number_float32_different": int(np.count_nonzero(f32(x) != f32(y))),
        "max_abs": float(np.abs(d).max()),
        "rms": float(np.mean(d*d)**0.5),
        "fro": float(np.linalg.norm(d.ravel())),
        "sha_first": hash_f32(x),
        "sha_second": hash_f32(y),
    }


def first_downproj(model, anchor, batch):
    """Capture the very first down-projection call in ACT5 in this batch."""
    cap = {}
    def hook(_m, args, y):
        if not cap:
            cap.update({"input":args[0].detach().clone(),
                        "output":y.detach().clone(),
                        "input_shape":list(args[0].shape),
                        "output_shape":list(y.shape),
                        "input_strides":list(args[0].stride()),
                        "output_strides":list(y.stride())})
    h = model.layers[0].down_proj.register_forward_hook(hook)
    try:
        with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
            zh = anchor["zh"].repeat(batch, 1, 1)
            zl = anchor["zl"].repeat(batch, 1, 1)
            emb = anchor["x_emb"].repeat(batch, 1, 1)
            model.act_step(zh, zl, emb)
    finally:
        h.remove()
    if not cap:
        raise ArithmeticError("first down-projection was not captured")
    return cap


def replay(model, c1, c2):
    a, b = c1["input"], c2["input"]
    if a.shape != (1,97,1536) or b.shape != (2,97,1536):
        raise ArithmeticError("unexpected projection shape")
    if not torch.equal(a[0],b[0]):
        raise ArithmeticError("first-row projection inputs differ")
    weight = model.layers[0].down_proj.weight.detach().clone()
    if weight.shape != (512,1536):
        raise ArithmeticError("unexpected output projection weight")
    results=[]
    original_threads = torch.get_num_threads()
    original_mkldnn = torch.backends.mkldnn.enabled
    try:
        for threads in (1,4):
            torch.set_num_threads(threads)
            for mkldnn in (False,True):
                torch.backends.mkldnn.enabled=mkldnn
                # Actual captured tensors, and a deliberately repeated copy
                # of the identical B1 first-row input as a B2 input.
                with torch.no_grad():
                    ya = F.linear(a, weight)[0]
                    yb = F.linear(b, weight)[0]
                    yb_duplicate = F.linear(a.repeat(2,1,1), weight)[0]
                    yflat = F.linear(a[0], weight)
                    yref = F.linear(a.double(), weight.double())[0].float()
                r={
                    "num_threads":threads, "mkldnn_enabled":bool(mkldnn),
                    "B1_vs_B2_original_input": difference(ya,yb),
                    "B1_vs_B2_duplicated_input":difference(ya,yb_duplicate),
                    "B1_vs_2D_flat":difference(ya,yflat),
                    "B1_vs_float64_accumulation_reference":difference(ya,yref),
                    "B2_vs_float64_accumulation_reference":difference(yb,yref),
                    "B1_equals_original_hook_output":bool(torch.equal(ya, c1["output"][0])),
                    "B2_equals_original_hook_output":bool(torch.equal(yb, c2["output"][0])),
                }
                r["float64_reference_sha256"]=hash_f32(yref)
                results.append(r)
    finally:
        torch.backends.mkldnn.enabled=original_mkldnn
        torch.set_num_threads(original_threads)
    return weight, results


def run(source_path, out, npz_path):
    src, src_sha = RN.load_source(source_path)
    by_pos={x["pool_position"]:x for x in src["pool"]}
    model, ckpt=RN.load_model()
    records=[]
    arrays={}
    for pos in POSITIONS:
        src_row=by_pos[pos]
        anchor=anchor_after_act4(model,src_row["x"])
        one=first_downproj(model,anchor,1)
        two=first_downproj(model,anchor,2)
        weight,modes=replay(model,one,two)
        entry={
            "position":pos,"source_file":src_row["source_file"],
            "B1_input_shape":one["input_shape"],"B2_input_shape":two["input_shape"],
            "B1_input_strides":one["input_strides"],"B2_input_strides":two["input_strides"],
            "B1_output_shape":one["output_shape"],"B2_output_shape":two["output_shape"],
            "source_input_byte_identical_first_row":bool(torch.equal(one["input"][0],two["input"][0])),
            "weights_sha256":hash_f32(weight),"input_sha256":hash_f32(one["input"][0]),
            "original_projection_output_difference":difference(one["output"][0],two["output"][0]),
            "replays":modes,
        }
        if not entry["source_input_byte_identical_first_row"]:
            raise ArithmeticError("unmatched model projection inputs")
        records.append(entry)
        arrays[f"position{pos}_input"]=f32(one["input"][0])
        arrays[f"position{pos}_weight"]=f32(weight)
        arrays[f"position{pos}_output_B1"]=f32(one["output"][0])
        arrays[f"position{pos}_output_B2"]=f32(two["output"][0])
        print("LS_SAE13_DOWNPROJ",pos,"original",entry["original_projection_output_difference"]["max_abs"],flush=True)
        for r in modes:
            print("LS_SAE13_DOWNPROJ_MODE",pos,r["num_threads"],"mkldnn",r["mkldnn_enabled"],
                  "B1_B2",r["B1_vs_B2_original_input"]["max_abs"],
                  "B1vs2D",r["B1_vs_2D_flat"]["max_abs"],
                  "B1vsFP64",r["B1_vs_float64_accumulation_reference"]["max_abs"],
                  "B2vsFP64",r["B2_vs_float64_accumulation_reference"]["max_abs"],
                  "same_hook",r["B1_equals_original_hook_output"],r["B2_equals_original_hook_output"],
                  flush=True)
    npz_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(npz_path,**arrays)
    npz_sha=hashlib.sha256(npz_path.read_bytes()).hexdigest()
    data={"schema":SCHEMA,"scope":"exploratory_debugging",
          "source_receipt_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
          "checkpoint_sha256":ckpt,"platform":RN.platform_fingerprint(),
          "npz_sha256":npz_sha,
          "positions":list(POSITIONS),"records":records}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,sort_keys=True,indent=1,allow_nan=False)+"\n")
    print("LS_SAE13_DOWNPROJ_PASS",hashlib.sha256(out.read_bytes()).hexdigest(),npz_sha,flush=True)
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    args=p.parse_args()
    run(args.source,args.out,args.fixture)


if __name__=="__main__":
    main()
