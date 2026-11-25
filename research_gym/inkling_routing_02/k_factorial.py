"""Fixed-set 2×2 test of HF layers 14 and 16 on the two K-clamp dev roots."""

import argparse
from functools import partial
from itertools import product
from pathlib import Path
import time

import numpy as np

from ..inkling_routing_01.checkpoint import MODEL_ID, MODEL_REVISION, file_sha256
from ..inkling_routing_01.hooks import as_numpy
from ..inkling_routing_01.local import execute, write_json
from .k_only import (ARMS, PASS, SCIENTIFIC_MINIMUM_TV, SMOKE, UNCLAMPED, KOnlyTrace,
                     score_forward, total_variation)
from .k_trace import DownstreamTrace, router_weights
from .native import NATIVE_PACKAGES, REPLAYS, digest, headroom
from .receipts import verified_run

LAYERS = (14, 16)
ARMS_IN_ORDER = ("clean", *ARMS, *UNCLAMPED)
VECTOR = ARMS_IN_ORDER.index("vector_k")
NATIVE_VECTOR = ARMS_IN_ORDER.index("vector_k_unclamped")
ABLATION = ARMS_IN_ORDER.index("zero_k")
CONDITIONS = dict(zip(("CC", "CU", "UC", "UU"), product("CU", repeat=2)))
ALL_CONDITIONS = ("native", *CONDITIONS)
REPLAY_TV_TOLERANCE = 1e-6


def fixed_sets(source):
    manifest, rows, failures = verified_run(source, tensor_test=True)
    if (manifest["schema"] != "ink-route-02.k-clamp-downstream-trace.v1" or manifest["root_prompts"] != SMOKE
            or manifest["development_only"] is not True or manifest["sealed_evaluations_accessed"] is not False
            or manifest["checkpoint_revision"] != MODEL_REVISION or manifest["offload_vocabulary"]
            or manifest.get("swap_experts_at_hf_layer") is not None
            or manifest.get("swap_ablation_experts_at_hf_layer") is not None or not rows or failures):
        raise ValueError("expected the verified all-CUDA unmodified two-root K trace")
    recipes = {}
    for root in SMOKE:
        recipes[root] = {}
        for replay in range(REPLAYS):
            with np.load(source / f"k_trace_{root}_replay{replay}.npz", allow_pickle=False) as z:
                for layer in LAYERS:
                    ids = z[f"router_hf{layer:02}_indices"]
                    choices = {"C": ids[VECTOR].tolist(), "U": ids[NATIVE_VECTOR].tolist()}
                    if replay and recipes[root][layer] != choices:
                        raise ValueError(f"{root} layer {layer} source expert order drifted")
                    recipes[root][layer] = choices
    return recipes


class FactorialTrace(DownstreamTrace):
    def __init__(self, reader, seq_len, forced):
        super().__init__(reader, seq_len)
        self.forced = forced

    def override_router(self, layer_index, module, output, last, route):
        import torch

        if layer_index not in self.forced:
            return None
        logits, weights, indices = output
        if self.reader.lanes[PASS, "vector_k"] != VECTOR:
            raise RuntimeError("unexpected vector-lane layout")
        token = VECTOR * self.seq_len + self.seq_len - 1
        recipient = logits[token:token + 1]
        native = indices[token:token + 1]
        chosen = torch.as_tensor(self.forced[layer_index], device=indices.device, dtype=indices.dtype).unsqueeze(0)
        if chosen.shape != native.shape or chosen.unique().numel() != chosen.numel():
            raise ValueError("invalid recorded expert set")
        if not torch.allclose(router_weights(module, recipient, native), weights[token:token + 1], rtol=1e-6, atol=1e-8):
            raise RuntimeError("recipient router weights do not reconstruct")
        if torch.equal(chosen.sort().values, native.sort().values):
            chosen = native
        patched_indices, patched_weights = indices.clone(), weights.clone()
        patched_indices[token:token + 1] = chosen
        patched_weights[token:token + 1] = router_weights(module, recipient, chosen)
        route["delivered"] = as_numpy(patched_indices[last]).astype(np.int64)
        route["delivered_weights"] = as_numpy(patched_weights[last])
        route["recipient_logits"] = as_numpy(recipient[0])
        return logits, patched_weights, patched_indices


def probe(recipes, model, tokenizer, prompt, root_id, passes, device, out):
    import torch

    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    rows = []
    for condition in ALL_CONDITIONS:
        forced = {layer: recipes[root_id][layer][value]
                  for layer, value in zip(LAYERS, CONDITIONS.get(condition, ()))}
        outcomes, condition_rows = [], []
        for replay in range(REPLAYS):
            headroom(device)
            reader = KOnlyTrace(model.model, root_id, clamp_beta=True)
            trace = FactorialTrace(reader, len(ids), forced)
            batch = {name: value.to(device).expand(len(reader.lanes), -1).contiguous()
                     for name, value in inputs.items()}
            started = time.perf_counter()
            with torch.inference_mode(), reader, trace:
                logp = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1).cpu().numpy()
            elapsed = time.perf_counter() - started
            if not set(LAYERS) <= trace.routes.keys() or any("delivered" not in trace.routes[layer] for layer in forced):
                raise RuntimeError("did not observe both factorial routers")
            arrays = {f"router_hf{layer:02}_{key}": value for layer in LAYERS
                      for key, value in trace.routes[layer].items()}
            np.savez_compressed(out / f"factorial_{root_id}_{condition}_r{replay}.npz",
                                **arrays, **reader.readers[PASS], writer_last_token=np.asarray(reader.writers),
                                input_ids=np.asarray(ids), logp=logp)
            for row in score_forward(reader, logp, root_id, replay, elapsed):
                row["condition"] = condition
                condition_rows.append(row)
            outcomes.append(logp)
        for row in condition_rows:
            lane = row["lane"]
            row["tv_same_arm_cross_replay"] = total_variation(outcomes[0][lane], outcomes[1][lane])
        rows.extend(condition_rows)
        print(f"{root_id} {condition}: completed {REPLAYS} native forwards", flush=True)
    return rows


def check_forced(z, layer, expected):
    prefix = f"router_hf{layer:02}"
    natural, delivered = z[f"{prefix}_indices"], z[f"{prefix}_delivered"]
    if sorted(delivered[VECTOR].tolist()) != sorted(expected):
        raise ValueError(f"layer {layer} delivered expert set mismatch")
    untouched = np.arange(len(ARMS_IN_ORDER)) != VECTOR
    if (not np.array_equal(natural[untouched], delivered[untouched])
            or not np.array_equal(z[f"{prefix}_weights"][untouched], z[f"{prefix}_delivered_weights"][untouched])):
        raise ValueError(f"layer {layer} changed an untargeted lane")


def compare_replay(path, source, root, replay, recipes):
    receipt = lambda condition: path / f"factorial_{root}_{condition}_r{replay}.npz"
    untouched = np.arange(len(ARMS_IN_ORDER)) != VECTOR
    keys = ("input_ids", "bank2", "original", "bank3", "delivered", "shared_beta3")
    tvs, max_other_tv = {}, 0.0
    with (np.load(receipt("native"), allow_pickle=False) as native,
          np.load(source / f"k_trace_{root}_replay{replay}.npz", allow_pickle=False) as prior):
        for key in keys:
            if not np.array_equal(native[key], prior[key]):
                raise ValueError(f"{root} native replay drifted in {key}")
        source_tv = max(total_variation(a, b) for a, b in zip(native["logp"], prior["logp"]))
        if source_tv > REPLAY_TV_TOLERANCE:
            raise ValueError(f"{root} native replay failed source TV tolerance")
        for layer in LAYERS:
            key = f"router_hf{layer:02}_indices"
            if not np.array_equal(native[key], prior[key]):
                raise ValueError(f"{root} native router changed before the intervention at layer {layer}")
        for condition in ALL_CONDITIONS:
            with np.load(receipt(condition), allow_pickle=False) as z:
                if any(not np.array_equal(z[key], native[key]) for key in keys):
                    raise ValueError(f"{root} {condition} K delivery drift")
                other = max(total_variation(a, b)
                            for a, b in zip(z["logp"][untouched], native["logp"][untouched]))
                max_other_tv = max(max_other_tv, other)
                if other > REPLAY_TV_TOLERANCE:
                    raise ValueError(f"{root} {condition} changed an untargeted output")
                for layer, state in zip(LAYERS, CONDITIONS.get(condition, ())):
                    check_forced(z, layer, recipes[layer][state])
                tvs[condition] = total_variation(z["logp"][ABLATION], z["logp"][VECTOR])
    if abs(tvs["CC"] - tvs["native"]) > REPLAY_TV_TOLERANCE:
        raise ValueError(f"{root} forcing both native sets changed the baseline")
    effect_at_c14, effect_at_u14 = tvs["CU"] - tvs["CC"], tvs["UU"] - tvs["UC"]
    return {"replay": replay, "tv_by_condition": tvs, "layer16_effect_at_C14": effect_at_c14,
            "layer16_effect_at_U14": effect_at_u14, "interaction_difference_in_differences": effect_at_u14 - effect_at_c14,
            "both_restored_below_threshold": tvs["UU"] < SCIENTIFIC_MINIMUM_TV,
            "max_native_source_tv": source_tv, "max_unmodified_lane_tv": max_other_tv}


def summarize(out, source, recipes):
    _, rows, failures = verified_run(out, tensor_test=True)
    keys = {(r["root_id"], r["condition"], r["replay"], r["arm"]) for r in rows}
    if failures or len(keys) != len(rows) or keys != set(product(SMOKE, ALL_CONDITIONS, range(REPLAYS), ARMS_IN_ORDER)):
        raise ValueError("factorial experiment receipt grid is incomplete")
    return {
        "development_only": True, "sealed_evaluations_accessed": False, "tv_threshold": SCIENTIFIC_MINIMUM_TV,
        "row_count": len(rows), "maximum_recovered_beta_gap": max(r["delivered_beta3_gap"] for r in rows),
        "maximum_clean_rescue_tv": max(r["tv_from_clean"] for r in rows if r["arm"].endswith("_rescue")),
        "any_top_token_flip": any(r["top_token_flip"] for r in rows),
        "roots": {root: [compare_replay(out, source, root, replay, recipes[root]) for replay in range(REPLAYS)]
                  for root in SMOKE},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.device, args.threads, args.cuda_budget_gib = "cuda:0", 2, 4.2
    recipes = fixed_sets(args.discovery)
    config = {
        "schema": "ink-route-02.k-clamp-expert-factorial.v1", "development_only": True,
        "sealed_evaluations_accessed": False, "root_prompts": SMOKE, "reader_passes": [PASS],
        "arms": ARMS_IN_ORDER, "layers": LAYERS, "conditions": CONDITIONS, "replays": REPLAYS,
        "recorded_expert_sets": recipes, "source_receipt_sha256": file_sha256(args.discovery / "sha256.json"),
        "model_id": MODEL_ID, "checkpoint_revision": MODEL_REVISION, "tv_threshold": SCIENTIFIC_MINIMUM_TV,
        "source_sha256": {p.name: file_sha256(p) for p in
                          (Path(__file__), Path(__file__).with_name("k_trace.py"), Path(__file__).with_name("k_only.py"))},
    }
    config["manifest_sha256"] = digest(config)

    def preflight(software):
        if software["packages"] != NATIVE_PACKAGES or software["pytorch_cuda"] != "12.8":
            raise RuntimeError("native factorial requires the pinned software set")
        headroom(args.device, args.cuda_budget_gib)

    execute(args.out, config, args, probe_root=partial(probe, recipes), check_preflight=preflight)
    write_json(args.out.with_name(args.out.name + "-summary.json"), summarize(args.out, args.discovery, recipes))


if __name__ == "__main__":
    main()
