"""Trace the downstream K-clamp contrast and test a localized router swap."""

import argparse
from contextlib import ExitStack
from functools import partial
from pathlib import Path
import time

import numpy as np

from ..inkling_routing_01.checkpoint import MODEL_ID, MODEL_REVISION, file_sha256
from ..inkling_routing_01.core import HF_READER_LAYER
from ..inkling_routing_01.hooks import as_numpy, reader_call
from ..inkling_routing_01.local import execute, write_json
from .k_only import (ARMS, PASS, SMOKE, TREATMENTS, UNCLAMPED, KOnlyTrace,
                     score_forward, total_variation)
from .native import NATIVE_PACKAGES, REPLAYS, digest, headroom
from .receipts import verified_run


def router_weights(module, logits, indices):
    scores = module.score_fn(logits).gather(1, indices)
    return scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) * module.routed_scaling_factor


class DownstreamTrace:
    """Capture only states at or after the pass-3 K delivery."""

    def __init__(self, reader, seq_len, swap_layer=None, swap_arm="ablation"):
        self.reader = reader
        self.seq_len = seq_len
        self.swap_layer = swap_layer
        self.swap_arm = swap_arm
        self.states = {}
        self.routes = {}
        self.handles = ExitStack()

    def __enter__(self):
        with ExitStack() as pending:
            for layer_index, layer in enumerate(self.reader.decoder.layers):
                if layer_index < HF_READER_LAYER:
                    continue
                prefix = f"hf{layer_index:02}"
                for name, module in (("attention", layer.self_attn), ("mlp", layer.mlp), ("layer", layer)):
                    hook = partial(self.save_state, f"{prefix}_{name}")
                    pending.callback(module.register_forward_hook(hook).remove)
                pending.callback(layer.mlp.gate.register_forward_hook(partial(self.save_router, layer_index)).remove)
                if layer_index == HF_READER_LAYER:
                    for name, module in (("k_gate", layer.k_message_gate),
                                         ("k_projection", layer.self_attn.k_proj)):
                        hook = partial(self.save_state, f"{prefix}_{name}")
                        pending.callback(module.register_forward_hook(hook).remove)
                    pending.callback(layer.self_attn.register_forward_pre_hook(
                        partial(self.save_key_input, f"{prefix}_k_cast"), with_kwargs=True).remove)
            pending.callback(self.reader.decoder.norm.register_forward_hook(
                partial(self.save_state, "final_norm")).remove)
            self.handles = pending.pop_all()
        return self

    def __exit__(self, *exc):
        self.handles.close()

    def active(self):
        return self.reader.visits > reader_call(PASS)

    def save_key_input(self, name, module, args, kwargs):
        self.save_state(name, module, args, kwargs["key_hidden_states"])

    def save_state(self, name, module, args, output):
        if self.active():
            tensor = output[0] if isinstance(output, tuple) else output
            if name in self.states or tensor.ndim < 3:
                raise RuntimeError(f"unexpected downstream state at {name}")
            token_axis = 2 if name.endswith("k_projection") and tensor.ndim == 4 else 1
            self.states[name] = as_numpy(tensor.select(token_axis, self.seq_len - 1))

    def save_router(self, layer_index, module, args, output):
        import torch

        if not self.active():
            return None
        logits, weights, indices = output
        if layer_index in self.routes or logits.shape[0] != self.seq_len * len(self.reader.lanes):
            raise RuntimeError(f"unexpected downstream router layout at HF layer {layer_index}")
        last = slice(self.seq_len - 1, None, self.seq_len)
        raw = logits[last]
        selected = indices[last]
        routing_scores = module.score_fn(raw) + module.e_score_correction_bias
        cutoff = routing_scores.topk(selected.shape[-1] + 1).values
        route = {"indices": as_numpy(selected).astype(np.int64), "weights": as_numpy(weights[last]),
                 "scores": as_numpy(routing_scores), "margin": as_numpy(cutoff[:, -2] - cutoff[:, -1])}
        self.routes[layer_index] = route
        if layer_index != self.swap_layer:
            return None
        # Exchange only the expert sets for the two last-token treatment lanes.
        # Use each recipient's original logits to compute its counterfactual weights.
        pair = int(self.swap_arm == "vector")
        chosen = [self.reader.lanes[PASS, arm] for arm in (TREATMENTS[pair], UNCLAMPED[pair])]
        tokens = [lane * self.seq_len + self.seq_len - 1 for lane in chosen]
        scores = logits[tokens]
        if not torch.allclose(router_weights(module, scores, indices[tokens]), weights[tokens], rtol=1e-6, atol=1e-8):
            raise RuntimeError("recipient routing formula failed native verification")
        if np.array_equal(np.sort(route["indices"][chosen[0]]), np.sort(route["indices"][chosen[1]])):
            route["swapped"] = route["indices"].copy()
            return None
        patched_indices, patched_weights = indices.clone(), weights.clone()
        patched_indices[tokens] = indices[tokens[::-1]]
        patched_weights[tokens] = router_weights(module, scores, patched_indices[tokens])
        route["swapped"] = as_numpy(patched_indices[last]).astype(np.int64)
        return logits, patched_weights, patched_indices


def stage_rows(trace, root_id, replay, lanes):
    zeros = [lanes[PASS, arm] for arm in (TREATMENTS[0], UNCLAMPED[0])]
    vectors = [lanes[PASS, arm] for arm in (TREATMENTS[1], UNCLAMPED[1])]
    for name, state in trace.states.items():
        yield {"root_id": root_id, "replay": replay, "stage": name,
               "ablation_clamp_delta_l2": float(np.linalg.norm(state[zeros[0]] - state[zeros[1]])),
               "vector_clamp_delta_l2": float(np.linalg.norm(state[vectors[0]] - state[vectors[1]]))}
    for layer, info in trace.routes.items():
        r = {"root_id": root_id, "replay": replay, "stage": f"hf{layer:02}_router"}
        for label, pair in (("ablation", zeros), ("vector", vectors)):
            a, b = info["indices"][pair]
            r[f"{label}_clamped_experts"] = sorted(a.tolist())
            r[f"{label}_unclamped_experts"] = sorted(b.tolist())
            r[f"{label}_expert_set_changed"] = set(a) != set(b)
            r[f"{label}_cutoff_margins"] = info["margin"][pair].tolist()
            leaving, entering = sorted(set(b) - set(a)), sorted(set(a) - set(b))
            r[f"{label}_changed_expert_scores"] = [
                {"expert": int(i), "clamped": float(info["scores"][pair[0], i]),
                 "unclamped": float(info["scores"][pair[1], i])}
                for i in (*leaving, *entering)
            ]
        if "swapped" in info:
            r["swap_arm"] = trace.swap_arm
            r["swap_delivered_indices"] = info["swapped"][vectors if trace.swap_arm == "vector" else zeros].tolist()
        yield r


def probe(model, tokenizer, prompt, root_id, passes, device, out, *, reserve_gib, swap_layer, swap_arm):
    import torch
    import json

    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    rows, outcomes = [], []
    for replay in range(REPLAYS):
        headroom(device, reserve_gib=reserve_gib)
        reader = KOnlyTrace(model.model, root_id, clamp_beta=True)
        trace = DownstreamTrace(reader, len(ids), swap_layer, swap_arm)
        batch = {name: value.to(device).expand(len(reader.lanes), -1).contiguous() for name, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode(), reader, trace:
            logp = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1).cpu().numpy()
        elapsed = time.perf_counter() - started
        if (set(trace.routes) != set(range(HF_READER_LAYER, len(model.model.layers)))
                or "final_norm" not in trace.states or (swap_layer is not None and "swapped" not in trace.routes[swap_layer])):
            raise RuntimeError("incomplete downstream trace")
        arrays = {f"state_{key}": value for key, value in trace.states.items()}
        arrays.update({f"router_hf{layer:02}_{name}": value
                       for layer, route in trace.routes.items() for name, value in route.items()})
        np.savez_compressed(out / f"k_trace_{root_id}_replay{replay}.npz", **arrays, **reader.readers[PASS],
                            writer_last_token=np.asarray(reader.writers), input_ids=np.asarray(ids), logp=logp)
        with (out / "stages.jsonl").open("a", encoding="utf-8") as records:
            for row in stage_rows(trace, root_id, replay, reader.lanes):
                records.write(json.dumps(row, allow_nan=False) + "\n")
        rows.extend(score_forward(reader, logp, root_id, replay, elapsed))
        outcomes.append(logp)
    for row in rows:
        lane = row["lane"]
        row["tv_same_arm_cross_replay"] = total_variation(outcomes[0][lane], outcomes[1][lane])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--swap-layer", type=int)
    parser.add_argument("--swap-arm", choices=("ablation", "vector"), default="ablation")
    parser.add_argument("--offload-vocabulary", action="store_true")
    args = parser.parse_args()
    if args.swap_layer is not None and not HF_READER_LAYER <= args.swap_layer < 24:
        parser.error("swap layer must be a downstream decoder layer")
    args.device, args.threads = "cuda:0", 2
    args.cuda_budget_gib, reserve_gib = (1.85, 1) if args.offload_vocabulary else (4.2, 2)
    config = {
        "schema": "ink-route-02.k-clamp-downstream-trace.v1", "development_only": True,
        "sealed_evaluations_accessed": False, "root_prompts": SMOKE, "reader_passes": [PASS],
        "arms": ("clean", *ARMS, *UNCLAMPED), "replays": REPLAYS,
        "swap_experts_at_hf_layer": args.swap_layer, "swap_arm": args.swap_arm if args.swap_layer is not None else None,
        "model_id": MODEL_ID,
        "checkpoint_revision": MODEL_REVISION, "offload_vocabulary": args.offload_vocabulary,
        "source_sha256": {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("k_only.py"))},
    }
    config["manifest_sha256"] = digest(config)

    def preflight(software):
        if software["packages"] != NATIVE_PACKAGES or software["pytorch_cuda"] != "12.8":
            raise RuntimeError("native trace requires the pinned software set")
        headroom(args.device, args.cuda_budget_gib, reserve_gib=reserve_gib,
                 load_ram_gib=8 if args.offload_vocabulary else 10)

    execute(args.out, config, args,
            probe_root=partial(probe, reserve_gib=reserve_gib, swap_layer=args.swap_layer, swap_arm=args.swap_arm),
            check_preflight=preflight)
    verified_run(args.out, tensor_test=True)


if __name__ == "__main__":
    main()
