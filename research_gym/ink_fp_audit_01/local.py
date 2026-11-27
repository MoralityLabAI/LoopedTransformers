"""Local packed-expert checkpoint depth audit. No training, no network, development only."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

from .core import UPSTREAM, digest, grade, roots
from research_gym.inkling_routing_01.checkpoint import (
    FILE_HASHES, TRANSFORMERS_VERSION, WEIGHTS, load_model, verify_metadata,
)


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def run(args):
    import numpy as np
    import torch
    import psutil
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(f"requires transformers=={TRANSFORMERS_VERSION}")
    if subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise RuntimeError("upstream wrapper commit mismatch")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("upstream checkout must be clean")
    sys.path.insert(0, str(args.upstream.resolve()))
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.checkpoint == "base":
        verify_metadata(args.snapshot)
        if args.sha256 != FILE_HASHES[WEIGHTS]:
            raise RuntimeError("base weight hash differs from INK-ROUTE-01")
    if digest(args.snapshot / "tokenizer.json") != FILE_HASHES["tokenizer.json"]:
        raise RuntimeError("tokenizer mismatch")
    if digest(args.snapshot / WEIGHTS) != args.sha256:
        raise RuntimeError("weight hash mismatch")
    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(args.device)
        budget = int(args.cuda_budget_gib * 1024**3)
        if free < budget + 2 * 1024**3:
            raise RuntimeError("insufficient CUDA headroom")
        torch.cuda.set_per_process_memory_fraction(budget / total, args.device)
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("insufficient host RAM")

    plan = roots(n=args.roots)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "ink-fp-audit-01.dev.v1", "checkpoint": args.checkpoint,
                "weight_sha256": args.sha256, "upstream_commit": UPSTREAM,
                "tokenizer_sha256": FILE_HASHES["tokenizer.json"], "transformers": TRANSFORMERS_VERSION,
                "torch": torch.__version__, "device": args.device, "roots": plan,
                "depths": args.depths, "max_new_tokens": args.max_new_tokens,
                "sealed_readout": False, "unit": "generated source mapping",
                "n_units": len(plan), "planned_comparisons": len(args.depths),
                "scientific_effect_floor": 0.10, "matched_sham_floor": None,
                "verdict": "development_only", "readout_selection_exposure": "public development",
                "primary_contrast": "native 3 vs same checkpoint wrapper 3 parity, then descriptive depth sweep",
                "causal_controls": "no interventions in this stage; causal claims blocked",
                "power": "not estimated; independent sham development and preflight required",
                "dose": "number of recurrent passes; no causal activation intervention"}
    write_json(args.out / "manifest.json", manifest)
    (args.out / "records.jsonl").touch()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
        with safe_open(str(args.snapshot / WEIGHTS), framework="pt", device="cpu") as weight_file:
            model, packed_count = load_model(args.snapshot, weight_file, args.device)
            decoder = model.model
            native_forward = decoder.forward
            if len(decoder.layers) != 24:
                raise RuntimeError("expected 24 physical layers")
            tokenizer.padding_side = "left"

            def inputs(root):
                messages = [{"role": "user", "content": root["prompt"]}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                return {k: v.to(args.device) for k, v in enc.items()}

            def probe(enc, expected_visits):
                trace = []
                def hook(_m, _args, output):
                    if not isinstance(output, torch.Tensor):
                        raise RuntimeError("unexpected writer output")
                    trace.append(output[0, -1].detach().float().cpu().numpy().copy())
                handle = decoder.layers[17].register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
                finally:
                    handle.remove()
                if len(trace) != expected_visits:
                    raise RuntimeError(f"writer count {len(trace)} != {expected_visits}")
                if not torch.isfinite(logits).all() or not np.isfinite(trace).all():
                    raise RuntimeError("nonfinite prefill")
                return logits, np.stack(trace)

            # A wrapper must reproduce native three-pass outputs before depth sweeps.
            # Compare full logits and all writer visits on two public development roots.
            parity = []
            for root in plan[:2]:
                enc = inputs(root)
                decoder.forward = native_forward
                a, av = probe(enc, 3)
                install_midstack_loop_forward(model, LoopConfig(num_passes=3))
                b, bv = probe(enc, 3)
                delta = float((a-b).abs().max())
                relative = float(np.linalg.norm(av-bv) / max(np.linalg.norm(av), 1e-12))
                parity.append({"root_id": root["root_id"], "max_logit_delta": delta,
                               "relative_writer_delta": relative, "same_top1": bool(a.argmax()==b.argmax())})
            write_json(args.out / "native_wrapper_parity.json", parity)
            if any(not p["same_top1"] or p["max_logit_delta"] > 0.05 or p["relative_writer_delta"] > 0.005 for p in parity):
                raise RuntimeError("native/wrapper parity failed: do not interpret depth results")

            arms = [("native3", 3)] + [(f"wrapper{d}", d) for d in args.depths]
            for name, depth in arms:
                if name == "native3":
                    decoder.forward = native_forward
                    decoder.config.loop_num_passes = 3
                    model.config.loop_num_passes = 3
                else:
                    install_midstack_loop_forward(model, LoopConfig(num_passes=depth))
                for root in plan:
                    enc = inputs(root)
                    start = time.perf_counter()
                    logits, vectors = probe(enc, depth)
                    # Cache-free decoding is intentional until the upstream damped cache
                    # snapshot/restore contract is repaired and separately qualified.
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                             do_sample=False, use_cache=False)
                    tail = out[0, enc["input_ids"].shape[1]:]
                    raw = tokenizer.decode(tail, skip_special_tokens=False)
                    cosine = [float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
                              for a,b in zip(vectors[:-1],vectors[1:])]
                    artifact = args.out / f"{root['root_id']}_{name}.npz"
                    np.savez_compressed(artifact, layer17=vectors, logits=logits.numpy())
                    rec = {"root_id": root["root_id"], "arm": name, "depth": depth,
                           "answer": root["answer"], "raw": raw, **grade(raw, root["answer"]),
                           "writer_cosine": cosine, "tokens": int(tail.numel()),
                           "seconds": time.perf_counter()-start,
                           "vector_sha256": digest(artifact), "packed_count": packed_count,
                           "rss_bytes": psutil.Process().memory_info().rss,
                           "cuda_peak_allocated": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0}
                    with (args.out / "records.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, allow_nan=False)+"\n")
        write_json(args.out / "complete.json", {"expected_records": len(arms)*len(plan),
                   "records_sha256": digest(args.out / "records.jsonl")})
    except Exception as exc:
        write_json(args.out / "failure.json", {"error": repr(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--checkpoint", choices=("base", "inkblot", "fpsolve"), required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cuda-budget-gib", type=float, default=5)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--roots", type=int, default=2)
    p.add_argument("--depths", type=int, nargs="+", default=[3,4])
    p.add_argument("--max-new-tokens", type=int, default=128)
    a = p.parse_args()
    if not 2 <= a.roots <= 8 or len(set(a.depths)) != len(a.depths) or any(not 1 <= d <= 8 for d in a.depths):
        p.error("require 2..8 roots and unique depths 1..8")
    if not 1 <= a.max_new_tokens <= 512 or a.threads < 1 or a.cuda_budget_gib <= 0:
        p.error("invalid resource limits")
    run(a)


if __name__ == "__main__":
    main()
