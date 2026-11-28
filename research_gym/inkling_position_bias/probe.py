"""Inkling 2 Mini hidden-state-conditioned relative-position probe.

This is a selective, local safetensors reader. It does not load the 26.4B
parameter model or infer attention from a positional-bias curve alone.

Reference: huggingface/transformers
  src/transformers/models/inkling2_loop/modeling_inkling2_loop.py
  Inkling2LoopRelativePositionBias, Inkling2LoopScaledL2Clip,
  Inkling2LoopSlidingDecoderLayer, Inkling2LoopTextModel.
Checkpoint tensor names checked against thinkingmachines/Inkling-2-Mini's
model.safetensors.index.json on 2026-11-22.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOKENS = (
    "the", "of", "in", "on", "to", "from", "with", "former", "latter",
    "he", "she", "they", "it", "that", ".", ",", ":", "?",
)
EXPECTED_HEADS = 24
EXPECTED_DREL = 16
DEFAULT_EXTENT = 1024
DEFAULT_HIDDEN = 2304
EPS = 1e-6
REL_CLIP = 8.0
LOG_FLOOR = 8192
LOG_ALPHA = 0.1
CHECKPOINT_NAME = "thinkingmachines/Inkling-2-Mini"


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Reference RMSNorm: cast to float32, normalize, apply learned gain."""
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    if x.shape[-1] != gamma.size:
        raise ValueError("RMSNorm dimension mismatch")
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * gamma


def layer0_inputs(embeddings: np.ndarray, embedding_gamma: np.ndarray,
                  attn_gamma: np.ndarray) -> np.ndarray:
    """The actual layer-0 bias input: embed RMSNorm then layer-0 attention RMSNorm.

    Layer 0 has no block-attention-residual mixer or cross-pass message.
    """
    return rms_norm(rms_norm(embeddings, embedding_gamma), attn_gamma)


def relative_curves(states: np.ndarray, wr: np.ndarray, rel_scale: np.ndarray,
                    basis: np.ndarray, *, clip: float = REL_CLIP,
                    position_ids: np.ndarray | None = None,
                    log_scaling: bool = False, log_floor: int = LOG_FLOOR,
                    log_alpha: float = LOG_ALPHA) -> np.ndarray:
    """Return [queries, heads, backward_distances] raw additive logit biases.

    Follows the released relative_position_bias: learned projection,
    per-head scaled-L2 clipping, shared relative basis, and optional
    (full-attention only) position-dependent logarithmic scale.
    """
    states = np.asarray(states, dtype=np.float32)
    wr = np.asarray(wr, dtype=np.float32)
    rel_scale = np.asarray(rel_scale, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if states.ndim != 2 or wr.ndim != 2 or rel_scale.ndim != 2 or basis.ndim != 2:
        raise ValueError("expected 2D states, wr, rel_scale and basis")
    q, dim = states.shape
    heads, d_rel = rel_scale.shape
    extent = basis.shape[1]
    if wr.shape != (heads * d_rel, dim) or basis.shape != (d_rel, extent):
        raise ValueError("relative projection and basis shapes do not agree")
    if not (0 < clip and extent > 0 and log_floor > 0):
        raise ValueError("invalid relative-bias parameters")
    if any(not np.all(np.isfinite(t)) for t in (states, wr, rel_scale, basis)):
        raise ValueError("nonfinite inputs")
    features = (states @ wr.T).reshape(q, heads, d_rel)
    features *= rel_scale[None, :, :]
    lengths = np.linalg.norm(features, axis=-1, keepdims=True)
    features *= np.minimum(1.0, clip / (lengths + 1e-6))
    if log_scaling:
        if position_ids is None:
            raise ValueError("full-attention curves require absolute query positions")
        position_ids = np.asarray(position_ids)
        if position_ids.shape != (q,) or np.any(position_ids < 0):
            raise ValueError("position_ids must be nonnegative with shape [queries]")
        tau = 1 + log_alpha * np.log(np.maximum((position_ids + 1) / log_floor, 1))
        features *= np.asarray(tau, dtype=np.float32)[:, None, None]
    return features @ basis


def bias_at_distances(curves: np.ndarray, distances: np.ndarray,
                      *, window: int | None = DEFAULT_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """Map learned profiles to key positions; return [N,H,K] and causal/window mask.

    Bias is exactly zero for backward distances beyond the learned basis,
    even when full attention still permits those distant keys.
    """
    curves = np.asarray(curves, dtype=np.float32)
    distances = np.asarray(distances)
    if curves.ndim != 3 or distances.ndim != 2 or distances.shape[0] != curves.shape[0]:
        raise ValueError("expected curves [N,H,E] and distances [N,K]")
    if not np.issubdtype(distances.dtype, np.integer):
        raise ValueError("distances must be integral")
    extent = curves.shape[-1]
    mask = distances >= 0
    if window is not None:
        if window <= 0:
            raise ValueError("window must be positive")
        mask &= distances < window
    gathered = np.take_along_axis(
        curves, np.broadcast_to(distances.clip(0, extent - 1)[:, None, :],
                                (curves.shape[0], curves.shape[1], distances.shape[1])), axis=-1
    )
    gathered *= ((distances >= 0) & (distances < extent))[:, None, :]
    return gathered, mask


def attention_delta(content_logits: np.ndarray, curves: np.ndarray,
                    distances: np.ndarray, sink_logits: np.ndarray | None,
                    *, window: int | None) -> dict[str, Any]:
    """Matched content/sink counterfactual: set ONLY relative-position bias to zero.

    Captured QK logits are required. This is a conditional frozen-logit
    intervention, not a recomputed model-forward causal effect.
    """
    content_logits = np.asarray(content_logits, dtype=np.float64)
    if content_logits.ndim != 3 or content_logits.shape[:2] != curves.shape[:2]:
        raise ValueError("content_logits must be [N,H,K] with matching queries/heads")
    bias, mask = bias_at_distances(curves, distances, window=window)
    if content_logits.shape != bias.shape:
        raise ValueError("content and positional-bias logit shapes differ")
    if not np.all(np.isfinite(content_logits)) or not np.any(mask, axis=1).all():
        raise ValueError("content logits must be finite; each query needs a visible key")
    if sink_logits is not None:
        sink_logits = np.asarray(sink_logits, dtype=np.float64)
        if sink_logits.shape != content_logits.shape[:2] or not np.all(np.isfinite(sink_logits)):
            raise ValueError("sink_logits must be finite [N,H]")

    def probabilities(key_logits: np.ndarray) -> np.ndarray:
        safe = np.where(mask[:, None, :], key_logits, -np.inf)
        if sink_logits is not None:
            safe = np.concatenate((safe, sink_logits[..., None]), axis=-1)
        maximum = np.max(safe, axis=-1, keepdims=True)
        ex = np.exp(safe - maximum)
        return ex / np.sum(ex, axis=-1, keepdims=True)

    clean = probabilities(content_logits + bias)
    flat = probabilities(content_logits)
    l1 = np.sum(np.abs(clean - flat), axis=-1)
    top_changed = np.argmax(clean, axis=-1) != np.argmax(flat, axis=-1)
    outcome = {
        "kind": "frozen_qk_and_sink_logits_bias_zero_ablation",
        "n_query_head_cells": int(l1.size),
        "mean_l1_probability_change": float(np.mean(l1)),
        "max_l1_probability_change": float(np.max(l1)),
        "top_attention_destination_changes": int(np.count_nonzero(top_changed)),
    }
    if sink_logits is not None:
        outcome["mean_absolute_sink_probability_change"] = float(
            np.mean(np.abs(clean[..., -1] - flat[..., -1]))
        )
    return outcome


def _from_tensor(tensor: Any) -> np.ndarray:
    """Safetensors framework=pt supports BF16; NumPy does not directly."""
    return tensor.to(dtype=__import__("torch").float32).cpu().numpy().copy()


def checkpoint_tensors(checkpoint: Path, layer: int, *, ids: list[int] | None,
                       require_embeddings: bool) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read exactly the relevant tensor slices from a LOCAL safetensors file."""
    from safetensors import safe_open

    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    prefix = f"model.llm.layers.{layer}."
    keys = {
        "wr": prefix + "attn.wr_du.weight",
        "rel_scale": prefix + "attn.rel_l2_clip.scale",
        "basis": prefix + "attn.rel_logits_proj.proj",
    }
    if require_embeddings:
        keys.update({
            "embed": "model.llm.embed.weight",
            "embedding_gamma": "model.llm.embed_norm.weight",
            "attn_gamma": prefix + "attn_norm.weight",
        })
    tensors: dict[str, np.ndarray] = {}
    tensor_hashes: dict[str, str] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        available = set(file.keys())
        absent = [key for key in keys.values() if key not in available]
        if absent:
            raise KeyError(f"required checkpoint tensors absent: {absent}")
        for name, key in keys.items():
            if name == "embed":
                if ids is None:
                    raise ValueError("embedding extraction needs token IDs")
                slice_handle = file.get_slice(key)
                shape = slice_handle.get_shape()
                if len(shape) != 2 or any(not 0 <= idx < shape[0] for idx in ids):
                    raise ValueError("token ID outside embedding vocabulary")
                tensor = np.stack([_from_tensor(slice_handle[idx:idx + 1])[0] for idx in ids])
            else:
                tensor = _from_tensor(file.get_tensor(key))
            tensors[name] = tensor
            tensor_hashes[key] = hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()
        if layer == 0:
            for name in ("k_sconv", "v_sconv"):
                key = prefix + f"attn.{name}.weight"
                if key in available:
                    data = _from_tensor(file.get_tensor(key))
                    tensors[name] = data
                    tensor_hashes[key] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    return tensors, tensor_hashes


def tokens_to_ids(tokenizer_path: Path, tokens: tuple[str, ...], form: str) -> tuple[list[str], list[int], list[str]]:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    labels, ids, skipped = [], [], []
    for text in tokens:
        variants = (text, " " + text) if form == "both" else ((text,) if form == "bare" else (" " + text,))
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False).ids
            if len(token_ids) != 1:
                skipped.append(f"{variant!r}: {len(token_ids)} tokenizer IDs")
                continue
            labels.append(variant)
            ids.append(int(token_ids[0]))
    if not ids:
        raise ValueError("no probe strings resolved to one tokenizer ID")
    return labels, ids, skipped


def conv_taps(weights: dict[str, np.ndarray]) -> dict[str, Any]:
    """Descriptive static K/V tap magnitudes for the first sliding layer."""
    out = {}
    for label in ("k_sconv", "v_sconv"):
        if label not in weights:
            continue
        w = weights[label]
        if w.size % 4:
            raise ValueError(f"{label} not divisible into four taps")
        rows = w.reshape(-1, 4)
        absmean = np.mean(np.abs(rows), axis=0)
        out[label] = {
            "mean_abs_taps_oldest_to_current": [float(v) for v in absmean],
            "normalized_abs_share_oldest_to_current": [float(v) for v in absmean / (absmean.sum() + 1e-12)],
            "n_channels": int(rows.shape[0]),
        }
    return out


def _save_plot(path: Path, curves: np.ndarray, labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(labels), 1, figsize=(14, max(4, 2 * len(labels))),
                             squeeze=False, constrained_layout=True)
    bound = float(np.max(np.abs(curves)))
    for i, label in enumerate(labels):
        axes[i, 0].imshow(curves[i], aspect="auto", interpolation="nearest",
                          vmin=-bound, vmax=bound, cmap="coolwarm")
        axes[i, 0].set_ylabel(f"{label!r}\nhead")
        axes[i, 0].set_xlabel("backward distance (tokens)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="local 19.5 GB safetensors file")
    parser.add_argument("--tokenizer", type=Path, help="local tokenizer.json (layer 0)")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--hidden-npz", type=Path, help="for later layers: arrays states [N,2304], labels [N]")
    parser.add_argument("--tokens", nargs="*", default=list(DEFAULT_TOKENS))
    parser.add_argument("--form", choices=("bare", "space", "both"), default="both")
    parser.add_argument("--position", type=int, default=4096,
                        help="query position, for causal interpretation and full-attention log scaling")
    parser.add_argument("--model-revision", default="unverified", help="record pinned HF commit SHA")
    parser.add_argument("--content-npz", type=Path,
                        help="optional captured content_logits [N,H,K], distances [N,K], sink_logits [N,H]")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.layer < 24 or args.position < 0:
        parser.error("layer must be in [0,23] and position nonnegative")
    if args.layer != 0 and args.hidden_npz is None:
        parser.error("layer > 0 requires captured post-attention-norm hidden states")
    if args.layer == 0 and (args.tokenizer is None or args.hidden_npz is not None):
        parser.error("layer 0 requires --tokenizer and uses exact embedding reconstruction")
    is_full = (args.layer + 1) % 6 == 0
    if args.layer == 0:
        labels, ids, skipped = tokens_to_ids(args.tokenizer, tuple(args.tokens), args.form)
        weights, tensor_hashes = checkpoint_tensors(args.checkpoint, 0, ids=ids, require_embeddings=True)
        states = layer0_inputs(weights["embed"], weights["embedding_gamma"], weights["attn_gamma"])
        position_ids = np.full(len(ids), args.position, dtype=np.int64)
        provenance = "exact_first_layer_weight_only_embedding_path"
    else:
        with np.load(args.hidden_npz, allow_pickle=False) as src:
            states = np.asarray(src["states"], dtype=np.float32)
            labels = [str(v) for v in src["labels"]]
            position_ids = np.asarray(src["position_ids"] if "position_ids" in src else
                                      np.full(len(labels), args.position), dtype=np.int64)
        if states.ndim != 2 or len(labels) != len(states) or len(position_ids) != len(labels):
            raise ValueError("captured activations must align with labels and position_ids")
        weights, tensor_hashes = checkpoint_tensors(args.checkpoint, args.layer, ids=None, require_embeddings=False)
        ids, skipped = [], []
        provenance = "captured_attention_input_activations_required_for_claims"
    curves = relative_curves(states, weights["wr"], weights["rel_scale"],
                             weights["basis"], position_ids=position_ids,
                             log_scaling=is_full)
    if (curves.shape[1] != EXPECTED_HEADS or weights["basis"].shape[0] != EXPECTED_DREL
            or curves.shape[-1] != DEFAULT_EXTENT or states.shape[1] != DEFAULT_HIDDEN):
        raise ValueError("checkpoint differs from registered Inkling 2 Mini architecture")
    summary = {
        "schema": "inkling2mini.relative_position_probe.v1", "model": CHECKPOINT_NAME,
        "model_revision": args.model_revision,
        "checkpoint_filename": args.checkpoint.name,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "layer": args.layer, "layer_type": "full" if is_full else "sliding",
        "provenance": provenance,
        "n_queries": len(labels), "n_heads": curves.shape[1],
        "d_rel": weights["basis"].shape[0], "extent": curves.shape[-1],
        "labels": labels, "token_ids": ids, "query_positions": position_ids.tolist(),
        "skipped_non_single_token_forms": skipped,
        "per_head_peak_distance": np.argmax(curves, axis=-1).tolist(),
        "per_head_peak_bias": np.max(curves, axis=-1).tolist(),
        "per_head_valley_distance": np.argmin(curves, axis=-1).tolist(),
        "selected_tensor_sha256": tensor_hashes,
        "position_bias_only": True,
        "limitations": [
            "Bias depends on query hidden state, not identity of the attended key.",
            "Layer-0 bias input is computed from checkpoint embeddings; deeper layers need native traces.",
            "A raw bias curve does not show QK content scores, sink logits, or causal effects.",
            "Computations use float32; exact BF16 runtime rounding may differ.",
        ],
    }
    if args.layer == 0:
        summary["static_four_tap_convolutions"] = conv_taps(weights)
    if args.content_npz is not None:
        with np.load(args.content_npz, allow_pickle=False) as captured:
            content = captured["content_logits"]
            distances = captured["distances"]
            sinks = captured["sink_logits"] if "sink_logits" in captured else None
        summary["captured_content_npz_sha256"] = hashlib.sha256(args.content_npz.read_bytes()).hexdigest()
        summary["frozen_attention_ablation"] = attention_delta(
            content, curves, distances, sinks, window=None if is_full else DEFAULT_EXTENT
        )
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "relative_curves.npz", curves=curves,
                        labels=np.asarray(labels), token_ids=np.asarray(ids, dtype=np.int64),
                        position_ids=position_ids)
    if args.plot:
        _save_plot(args.out / "relative_curves.png", curves, labels)
    summary["curves_npz_sha256"] = hashlib.sha256((args.out / "relative_curves.npz").read_bytes()).hexdigest()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"out": str(args.out), "labels": labels, "shape": list(curves.shape),
                      "source": provenance}, sort_keys=True))


if __name__ == "__main__":
    main()
