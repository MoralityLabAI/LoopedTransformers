"""Resumable streaming Inkling-2-Mini NVFP4/MXFP8 -> GGML GGUF tensor-bundle converter.

Streams one physical expert (or one backbone tensor stripe) at a time with bounded
RAM, reuses the verified SilicoExperiments PR #3 native dequantization, uses actual
GGML C quantization routines (ggml_quantize_chunk), exports 24 independently
downloadable expert layer shards + a separately quantized text-only backbone +
full 3,072-expert manifest, and supports incremental Hugging Face uploads and
disconnect-safe resuming on Google Colab CPU.

NOTE: Produced GGUF files are standalone GGML tensor bundles (general.type="tensors",
inkynano.llama_cpp_dropin_compatible=False), NOT drop-in llama.cpp models.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gguf_backend import (
    GGMLCBackend,
    PINNED_LLAMA_CPP_COMMIT,
    SyntheticRefBackend,
    build_ggml_cpu,
    discover_ggml_lib_dir,
    encode_unquantized,
)
from gguf_dequant import (
    CHECKPOINT_SHARD,
    CHECKPOINT_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MODEL,
    NUM_EXPERTS_PER_LAYER,
    NUM_PHYSICAL_LAYERS,
    REV,
    TRANSFORMERS_REV,
    SafetensorsStreamSource,
    dequantize_expert,
    dequantize_projection_standalone,
    dequantize_projection_transformers,
    has_upstream_transformers_dequant,
    open_source,
)
from gguf_format import (
    GGUF_DEFAULT_ALIGNMENT,
    GGMLType,
    GGUFBundleReader,
    GGUFValueType,
    StreamingGGUFWriter,
    TensorSpec,
    ggml_nbytes,
    parse_ggml_type,
)
from gguf_profiles import (
    BACKBONE_PROFILES,
    EXPERT_PROFILES,
    ExpertQuantProfile,
    classify_backbone_tensor,
    estimate_backbone_bytes,
    is_text_backbone_tensor,
    rtx3050_residency_table,
)
from native_expert import activation as native_activation
from native_expert import expert as native_expert_fn
from stage1_memory import GIB, MIB, available_memory, process_memory

PILOT_LAYERS = (0, 6, 17, 23)
PILOT_SEED = 17
DEFAULT_HF_REPO = "AlephFunk/InkyNano-GGUF"
CONVERTER_SCHEMA_VERSION = "inkynano-gguf-v1"
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * MIB), b""):
            h.update(chunk)
    return h.hexdigest()


def array_error_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    diff = p - r
    ref_norm = float(np.linalg.norm(r))
    diff_norm = float(np.linalg.norm(diff))
    pred_norm = float(np.linalg.norm(p))
    rel_l2 = diff_norm / max(ref_norm, 1e-20)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    cos = float(np.dot(p, r) / max(pred_norm * ref_norm, 1e-20))
    return {"relative_l2": rel_l2, "max_abs": max_abs, "cosine_sim": cos}


def pad_last_axis_zeros(arr: np.ndarray, target_ne0: int) -> np.ndarray:
    if arr.shape[-1] == target_ne0:
        return arr
    if target_ne0 < arr.shape[-1]:
        raise ValueError(f"target_ne0={target_ne0} < current {arr.shape[-1]}")
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_ne0 - arr.shape[-1])]
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def build_deterministic_probes(
    hidden_size: int = HIDDEN_SIZE,
    n_samples: int = 32,
    seed: int = PILOT_SEED,
) -> Dict[str, torch.Tensor]:
    """Deterministic multi-regime input probes exercising linear and soft-capped SwiGLU regimes."""
    rng = np.random.default_rng(seed)
    # 1. Unit-RMS (typical post-RMSNorm pre-MLP activation scale)
    x_unit = rng.standard_normal((n_samples, hidden_size)).astype(np.float32)
    x_unit /= np.sqrt(np.mean(x_unit**2, axis=-1, keepdims=True)) + 1e-6
    # 2. Small scale (linear region below gate_cap=4)
    x_small = (x_unit * 0.25).astype(np.float32)
    # 3. Heavy-tailed / outlier-injected (exercises gate_cap=4.0 and up_cap=25.0 saturation)
    x_heavy = x_unit.copy()
    mask = rng.random(x_heavy.shape) < 0.02
    x_heavy[mask] *= 6.0
    return {
        "unit_rms": torch.from_numpy(x_unit).to(torch.bfloat16),
        "small_0p25": torch.from_numpy(x_small).to(torch.bfloat16),
        "heavy_tail": torch.from_numpy(x_heavy).to(torch.bfloat16),
    }


@torch.no_grad()
def run_softcapped_swiglu(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Execute exact native Inkling-2-Mini soft-capped SwiGLU (gate_cap=4, up_cap=25, raw sigmoid, FP32 accum)."""
    if gate.dtype == x.dtype and up.dtype == x.dtype and down.dtype == x.dtype:
        g = F.linear(x, gate)
        u = F.linear(x, up)
        act = native_activation(g, u, gate_cap=GATE_SOFTCAP, up_cap=UP_SOFTCAP)
        return F.linear(act, down)
    g = F.linear(x.float(), gate.float())
    u = F.linear(x.float(), up.float())
    act = native_activation(g, u, gate_cap=GATE_SOFTCAP, up_cap=UP_SOFTCAP)
    out = F.linear(act.float(), down.float())
    return out.to(x.dtype)


def pilot_expert_plan(seed: int = PILOT_SEED, layers: Sequence[int] = PILOT_LAYERS, per_layer: int = 4) -> Dict[int, List[int]]:
    """Exact same 16-expert stratified roster used in SilicoExperiments PR #3."""
    rng = torch.Generator().manual_seed(seed)
    return {int(layer): torch.randperm(NUM_EXPERTS_PER_LAYER, generator=rng)[:per_layer].tolist() for layer in layers}


def get_backend(lib_dir: Optional[Path] = None, allow_ci_fallback: bool = False):
    found = discover_ggml_lib_dir(lib_dir)
    if found is not None:
        return GGMLCBackend(lib_dir=found)
    if allow_ci_fallback:
        return SyntheticRefBackend()
    raise FileNotFoundError(
        "Compiled GGML C library (libggml-base / ggml-base.dll) is required for real GGML quantization. "
        "Build it with `python InkyNano/gguf_convert.py build-ggml --dest /content` or pass --ggml-lib-dir."
    )


def _check_rss_guard(max_rss_gib: Optional[float]) -> Tuple[int, int]:
    rss, peak = process_memory()
    if max_rss_gib is not None and rss > int(max_rss_gib * GIB):
        raise MemoryError(f"RSS {rss / GIB:.2f} GiB exceeded safety ceiling {max_rss_gib:.2f} GiB")
    return rss, peak


def _bundle_common_metadata(
    bundle_kind: str,
    name: str,
    description: str,
    backend_info: Dict[str, Any],
    quant_profile: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kv: Dict[str, Any] = {
        "general.architecture": "inkling2_loop_bundle",
        "general.type": "tensors",
        "general.name": name,
        "general.description": description,
        "general.license": "apache-2.0",
        "inkynano.schema_version": CONVERTER_SCHEMA_VERSION,
        "inkynano.bundle_kind": bundle_kind,
        "inkynano.llama_cpp_dropin_compatible": False,
        "inkynano.source_model": MODEL,
        "inkynano.source_revision": REV,
        "inkynano.transformers_revision": TRANSFORMERS_REV,
        "inkynano.ggml_commit": str(backend_info.get("ggml_commit", "unknown")),
        "inkynano.ggml_backend": str(backend_info.get("backend_kind", "unknown")),
        "inkynano.quant_profile": quant_profile,
        "inkynano.activation": "softcapped_swiglu",
        "inkynano.mlp_gate_softcap": (GGUFValueType.FLOAT32, GATE_SOFTCAP),
        "inkynano.mlp_up_softcap": (GGUFValueType.FLOAT32, UP_SOFTCAP),
        "inkynano.mlp_sigmoid_on_raw_gate": True,
        "inkynano.mlp_accum_dtype": "float32",
        "inkynano.hidden_size": (GGUFValueType.UINT32, HIDDEN_SIZE),
        "inkynano.intermediate_size": (GGUFValueType.UINT32, INTERMEDIATE_SIZE),
        "inkynano.num_physical_layers": (GGUFValueType.UINT32, NUM_PHYSICAL_LAYERS),
        "inkynano.num_experts_per_layer": (GGUFValueType.UINT32, NUM_EXPERTS_PER_LAYER),
        "inkynano.num_experts_per_tok": (GGUFValueType.UINT32, 8),
        "inkynano.route_scale": (GGUFValueType.FLOAT32, 8.0),
        "inkynano.loop_core_start": (GGUFValueType.UINT32, 6),
        "inkynano.loop_core_end": (GGUFValueType.UINT32, 18),
        "inkynano.loop_num_passes": (GGUFValueType.UINT32, 3),
        "inkynano.loader_cast": True,
    }
    if extra:
        kv.update(extra)
    return kv


def expert_tensor_names(layer: int, expert_id: int) -> Tuple[str, str, str]:
    prefix = f"blk.{layer}.exp.{expert_id:03d}"
    return (
        f"{prefix}.ffn_gate.weight",
        f"{prefix}.ffn_up.weight",
        f"{prefix}.ffn_down.weight",
    )


def shard_relpath(layer: int) -> str:
    return f"shards/inkynano_experts_layer_{layer:02d}.gguf"


def shard_receipt_relpath(layer: int) -> str:
    return f"receipts/shard_receipt_layer_{layer:02d}.json"


BACKBONE_RELPATH = "backbone/inkynano_backbone_text.gguf"
BACKBONE_RECEIPT_RELPATH = "receipts/backbone_receipt.json"
BENCHMARK_RECEIPT_RELPATH = "receipts/benchmark_16_experts.json"
CHECKPOINT_STATE_RELPATH = "receipts/checkpoint_state.json"
MANIFEST_RELPATH = "manifest.json"


# ---------------------------------------------------------------------------
# 16-Expert Pilot Benchmark (Pure Q4_K vs Mixed Q4_K/Q6_K)
# ---------------------------------------------------------------------------

def run_benchmark_16(
    source: SafetensorsStreamSource,
    backend,
    out_dir: Path,
    profile_names: Sequence[str] = (
        "pure_q4_k",
        "pure_q4_k_q4_0",
        "pure_q4_k_padded",
        "mixed_q4_k_q6_k",
        "mixed_q4_k_q6_k_down_q8",
        "mixed_q4_k_q6_k_padded",
    ),
    dequant_backend: str = "auto",
    n_threads: int = 0,
    max_rss_gib: Optional[float] = 8.0,
) -> Dict[str, Any]:
    """Benchmark 16 stratified experts (layers 0, 6, 17, 23 x 4) across pure and mixed profiles."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = pilot_expert_plan()
    probes = build_deterministic_probes(n_samples=32, seed=PILOT_SEED)

    # Verify native dequantization parity between upstream Transformers and standalone mirror
    parity_checks: List[Dict[str, Any]] = []
    uncast_diffs: List[Dict[str, Any]] = []
    per_profile_records: Dict[str, List[Dict[str, Any]]] = {p: [] for p in profile_names}
    per_profile_mix: Dict[str, List[Dict[str, Any]]] = {p: [] for p in profile_names}
    dequant_times: List[float] = []

    t_bench_start = time.perf_counter()
    tf_available = has_upstream_transformers_dequant()

    for layer, expert_ids in plan.items():
        native_fmt = source.native_layer_format(layer)
        # Keep quantized dequantized pair for 2-expert mixture check on first two experts per layer
        pair_refs: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        pair_quants: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {p: [] for p in profile_names}

        for idx_in_layer, eid in enumerate(expert_ids):
            _check_rss_guard(max_rss_gib)
            t_dq0 = time.perf_counter()
            raw_w13, keys_w13 = source.read_expert_raw_inputs(layer, eid, "w13")
            raw_w2, keys_w2 = source.read_expert_raw_inputs(layer, eid, "w2")

            use_tf = tf_available and dequant_backend in ("auto", "transformers")
            fn = dequantize_projection_transformers if use_tf else dequantize_projection_standalone
            gate_up_bf16 = fn(raw_w13, keys_w13, "w13", dtype=torch.bfloat16, device="cpu", loader_cast=True)
            down_bf16 = fn(raw_w2, keys_w2, "w2", dtype=torch.bfloat16, device="cpu", loader_cast=True)
            t_dq1 = time.perf_counter()
            dequant_times.append(t_dq1 - t_dq0)

            # Cross-check standalone mirror against pinned upstream transformers on first expert per layer
            if idx_in_layer == 0 and tf_available:
                st_w13 = dequantize_projection_standalone(raw_w13, keys_w13, "w13", dtype=torch.bfloat16, device="cpu", loader_cast=True)
                st_w2 = dequantize_projection_standalone(raw_w2, keys_w2, "w2", dtype=torch.bfloat16, device="cpu", loader_cast=True)
                err13 = array_error_metrics(st_w13.float().numpy(), gate_up_bf16.float().numpy())
                err2 = array_error_metrics(st_w2.float().numpy(), down_bf16.float().numpy())
                parity_checks.append(
                    {
                        "layer": layer,
                        "expert": eid,
                        "native_format": native_fmt,
                        "w13_transformers_vs_standalone": err13,
                        "w2_transformers_vs_standalone": err2,
                    }
                )
                if err13["max_abs"] != 0.0 or err2["max_abs"] != 0.0:
                    raise RuntimeError(f"standalone vs transformers parity mismatch at L{layer}/E{eid}: {err13}, {err2}")
                del st_w13, st_w2

            # Measure loader_cast=True vs loader_cast=False on first expert per layer (matching Silico PR #3)
            if idx_in_layer == 0:
                uc_w13 = fn(raw_w13, keys_w13, "w13", dtype=torch.bfloat16, device="cpu", loader_cast=False)
                uc_w2 = fn(raw_w2, keys_w2, "w2", dtype=torch.bfloat16, device="cpu", loader_cast=False)
                uncast_diffs.append(
                    {
                        "layer": layer,
                        "expert": eid,
                        "native_format": native_fmt,
                        "w13_uncast_vs_loader_cast": array_error_metrics(uc_w13.float().numpy(), gate_up_bf16.float().numpy()),
                        "w2_uncast_vs_loader_cast": array_error_metrics(uc_w2.float().numpy(), down_bf16.float().numpy()),
                    }
                )
                del uc_w13, uc_w2

            del raw_w13, raw_w2

            gate_bf16, up_bf16 = gate_up_bf16.chunk(2, dim=0)
            gate_f32 = gate_bf16.contiguous().float().numpy()
            up_f32 = up_bf16.contiguous().float().numpy()
            down_f32 = down_bf16.contiguous().float().numpy()

            # Verify native_expert.expert() matches explicit gate/up/down SwiGLU
            ref_outputs = {}
            for pname, x in probes.items():
                y_native = native_expert_fn(x, gate_up_bf16, down_bf16)
                y_split = run_softcapped_swiglu(x, gate_bf16, up_bf16, down_bf16)
                split_err = array_error_metrics(y_split.float().numpy(), y_native.float().numpy())
                if split_err["relative_l2"] > 1e-2:
                    raise RuntimeError(f"split SwiGLU parity error too high: {split_err}")
                ref_outputs[pname] = y_native.float().numpy()

            if idx_in_layer < 2:
                pair_refs.append((eid, gate_up_bf16.clone(), down_bf16.clone(), probes["unit_rms"][:8]))

            for prof_name in profile_names:
                prof = EXPERT_PROFILES[prof_name]
                p_gate, p_up, p_down = prof.layer_plan(layer)

                t_q0 = time.perf_counter()
                q_gate = backend.quantize(pad_last_axis_zeros(gate_f32, p_gate.stored_shape[1]), p_gate.effective_type, n_threads=n_threads)
                q_up = backend.quantize(pad_last_axis_zeros(up_f32, p_up.stored_shape[1]), p_up.effective_type, n_threads=n_threads)
                q_down = backend.quantize(pad_last_axis_zeros(down_f32, p_down.stored_shape[1]), p_down.effective_type, n_threads=n_threads)
                t_q1 = time.perf_counter()

                t_d0 = time.perf_counter()
                dq_gate = backend.dequantize(q_gate, p_gate.stored_shape, p_gate.effective_type)[:, : p_gate.logical_shape[1]]
                dq_up = backend.dequantize(q_up, p_up.stored_shape, p_up.effective_type)[:, : p_up.logical_shape[1]]
                dq_down = backend.dequantize(q_down, p_down.stored_shape, p_down.effective_type)[:, : p_down.logical_shape[1]]
                t_d1 = time.perf_counter()

                w_err = {
                    "gate": array_error_metrics(dq_gate, gate_f32),
                    "up": array_error_metrics(dq_up, up_f32),
                    "down": array_error_metrics(dq_down, down_f32),
                }

                gt = torch.from_numpy(dq_gate)
                ut = torch.from_numpy(dq_up)
                dt = torch.from_numpy(dq_down)
                swiglu_err = {}
                for pname, x in probes.items():
                    y_q = run_softcapped_swiglu(x, gt, ut, dt).float().numpy()
                    swiglu_err[pname] = array_error_metrics(y_q, ref_outputs[pname])

                if idx_in_layer < 2:
                    pair_quants[prof_name].append((gt.clone(), ut.clone(), dt.clone()))

                per_profile_records[prof_name].append(
                    {
                        "layer": layer,
                        "expert": eid,
                        "native_format": native_fmt,
                        "quant_types": {
                            "gate": p_gate.effective_type.name,
                            "up": p_up.effective_type.name,
                            "down": p_down.effective_type.name,
                        },
                        "stored_bytes": len(q_gate) + len(q_up) + len(q_down),
                        "dequant_source_seconds": t_dq1 - t_dq0,
                        "ggml_quant_seconds": t_q1 - t_q0,
                        "ggml_dequant_seconds": t_d1 - t_d0,
                        "weight_error": w_err,
                        "swiglu_error": swiglu_err,
                    }
                )
                del q_gate, q_up, q_down, dq_gate, dq_up, dq_down, gt, ut, dt

            del gate_up_bf16, gate_bf16, up_bf16, down_bf16, gate_f32, up_f32, down_f32
            gc.collect()

        # Weighted 2-expert mixture check (0.37 / 0.63) matching SilicoExperiments PR #3
        if len(pair_refs) == 2:
            (_, gu0, d0, xpair), (_, gu1, d1, _) = pair_refs
            ref_mix = (0.37 * native_expert_fn(xpair, gu0, d0).float() + 0.63 * native_expert_fn(xpair, gu1, d1).float()).to(xpair.dtype).float().numpy()
            for prof_name in profile_names:
                (g0, u0, dn0), (g1, u1, dn1) = pair_quants[prof_name]
                q_mix = (0.37 * run_softcapped_swiglu(xpair, g0, u0, dn0).float() + 0.63 * run_softcapped_swiglu(xpair, g1, u1, dn1).float()).to(xpair.dtype).float().numpy()
                per_profile_mix[prof_name].append(
                    {"layer": layer, "native_format": native_fmt, **array_error_metrics(q_mix, ref_mix)}
                )
        del pair_refs, pair_quants
        gc.collect()

    # Estimate backbone bytes directly from source header entries
    header_entries = {k: {"shape": list(e.shape), "dtype": e.dtype} for k, e in source.entries.items()}
    backbone_pure_est = estimate_backbone_bytes(header_entries, "backbone_pure_q4_k")
    backbone_mixed_est = estimate_backbone_bytes(header_entries, "backbone_mixed_q4_k_q6_k")

    # Summarize each profile
    summaries: Dict[str, Any] = {}
    for prof_name in profile_names:
        recs = per_profile_records[prof_name]
        prof = EXPERT_PROFILES[prof_name]
        nv_recs = [r for r in recs if r["native_format"] == "nvfp4"]
        mx_recs = [r for r in recs if r["native_format"] == "mxfp8"]

        def _mean(items, fn):
            return float(np.mean([fn(x) for x in items])) if items else 0.0

        def _max(items, fn):
            return float(np.max([fn(x) for x in items])) if items else 0.0

        # Weighted 3,072-expert projection (2,560 NVFP4 + 512 MXFP8)
        nv_total_s = _mean(nv_recs, lambda r: r["dequant_source_seconds"] + r["ggml_quant_seconds"])
        mx_total_s = _mean(mx_recs, lambda r: r["dequant_source_seconds"] + r["ggml_quant_seconds"])
        nv_q_s = _mean(nv_recs, lambda r: r["ggml_quant_seconds"])
        mx_q_s = _mean(mx_recs, lambda r: r["ggml_quant_seconds"])
        proj_3072_s = 2560 * nv_total_s + 512 * mx_total_s
        proj_3072_quant_only_s = 2560 * nv_q_s + 512 * mx_q_s

        summaries[prof_name] = {
            "description": prof.description,
            "n_experts_benchmarked": len(recs),
            "nvfp4_quant_types": {p.projection: p.effective_type.name for p in prof.layer_plan(0)},
            "mxfp8_quant_types": {p.projection: p.effective_type.name for p in prof.layer_plan(23)},
            "nvfp4_expert_bytes": prof.expert_payload_bytes(0),
            "mxfp8_expert_bytes": prof.expert_payload_bytes(23),
            "full_bank_payload_bytes": prof.bank_payload_bytes(),
            "full_bank_payload_gib": prof.bank_payload_bytes() / GIB,
            "weight_rel_l2_mean": {
                proj: _mean(recs, lambda r, pr=proj: r["weight_error"][pr]["relative_l2"])
                for proj in ("gate", "up", "down")
            },
            "weight_rel_l2_nvfp4_mean": {
                proj: _mean(nv_recs, lambda r, pr=proj: r["weight_error"][pr]["relative_l2"])
                for proj in ("gate", "up", "down")
            },
            "weight_rel_l2_mxfp8_mean": {
                proj: _mean(mx_recs, lambda r, pr=proj: r["weight_error"][pr]["relative_l2"])
                for proj in ("gate", "up", "down")
            },
            "swiglu_unit_rms_rel_l2_mean": _mean(recs, lambda r: r["swiglu_error"]["unit_rms"]["relative_l2"]),
            "swiglu_unit_rms_rel_l2_max": _max(recs, lambda r: r["swiglu_error"]["unit_rms"]["relative_l2"]),
            "swiglu_unit_rms_nvfp4_mean": _mean(nv_recs, lambda r: r["swiglu_error"]["unit_rms"]["relative_l2"]),
            "swiglu_unit_rms_mxfp8_mean": _mean(mx_recs, lambda r: r["swiglu_error"]["unit_rms"]["relative_l2"]),
            "swiglu_heavy_tail_rel_l2_mean": _mean(recs, lambda r: r["swiglu_error"]["heavy_tail"]["relative_l2"]),
            "swiglu_unit_rms_cosine_mean": _mean(recs, lambda r: r["swiglu_error"]["unit_rms"]["cosine_sim"]),
            "mixture_2expert_rel_l2_mean": _mean(per_profile_mix[prof_name], lambda r: r["relative_l2"]),
            "mean_dequant_seconds_per_expert": float(np.mean(dequant_times)),
            "mean_quant_seconds_per_expert": _mean(recs, lambda r: r["ggml_quant_seconds"]),
            "throughput_experts_per_second_quant_only": 1.0 / max(_mean(recs, lambda r: r["ggml_quant_seconds"]), 1e-9),
            "projected_3072_total_seconds": proj_3072_s,
            "projected_3072_total_minutes": proj_3072_s / 60.0,
            "projected_3072_quant_only_seconds": proj_3072_quant_only_s,
            "rtx3050_fit_with_mixed_backbone": rtx3050_residency_table(prof, backbone_mixed_est["total_payload_bytes"]),
            "rtx3050_fit_with_pure_backbone": rtx3050_residency_table(prof, backbone_pure_est["total_payload_bytes"]),
        }

    rss, peak_rss = process_memory()
    # Profiling-driven CPU vs GPU decision: GGML K-quant routines are CPU-native;
    # GPU could only accelerate NVFP4/MXFP8 dequantization. Request GPU only if
    # dequantization dominates (>50% of runtime) AND total projected CPU time > 45 min.
    ref_summary = summaries.get("mixed_q4_k_q6_k") or next(iter(summaries.values()))
    mean_dq = ref_summary["mean_dequant_seconds_per_expert"]
    mean_q = ref_summary["mean_quant_seconds_per_expert"]
    dq_share = mean_dq / max(mean_dq + mean_q, 1e-9)
    gpu_recommended = bool(ref_summary["projected_3072_total_minutes"] > 45.0 and dq_share > 0.5)

    receipt = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "source_model": MODEL,
        "source_revision": REV,
        "transformers_revision": TRANSFORMERS_REV,
        "upstream_transformers_used": bool(tf_available and dequant_backend in ("auto", "transformers")),
        "backend": backend.info(),
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "n_threads": n_threads if n_threads > 0 else max(1, min(os.cpu_count() or 1, 8)),
            "rss_bytes": rss,
            "peak_rss_bytes": peak_rss,
            "available_ram_bytes": available_memory(),
        },
        "pilot_plan": {str(k): v for k, v in plan.items()},
        "elapsed_seconds": time.perf_counter() - t_bench_start,
        "source_bytes_read": source.bytes_read,
        "parity_checks_transformers_vs_standalone": parity_checks,
        "loader_cast_vs_uncast_diagnostics": uncast_diffs,
        "backbone_estimates": {
            "backbone_pure_q4_k": backbone_pure_est,
            "backbone_mixed_q4_k_q6_k": backbone_mixed_est,
        },
        "profile_summaries": summaries,
        "cpu_vs_gpu_profiling": {
            "dequant_time_share": dq_share,
            "projected_3072_cpu_minutes": ref_summary["projected_3072_total_minutes"],
            "gpu_recommended": gpu_recommended,
            "rationale": (
                "GGML K-quantization (ggml_quantize_chunk) executes natively on CPU SIMD threads; "
                "stay on CPU runtime."
                if not gpu_recommended
                else "Source dequantization dominates CPU runtime (>50%) and projected runtime exceeds 45 min; GPU runtime warranted."
            ),
        },
        "per_profile_experts": per_profile_records,
        "per_profile_mixtures": per_profile_mix,
    }
    dump_json_atomic(out_dir / BENCHMARK_RECEIPT_RELPATH, receipt)
    return receipt


# ---------------------------------------------------------------------------
# Streaming Layer Shard Conversion (1 expert at a time, O(1) RAM)
# ---------------------------------------------------------------------------

def convert_layer_shard(
    source: SafetensorsStreamSource,
    backend,
    layer: int,
    profile_name: str,
    out_dir: Path,
    dequant_backend: str = "auto",
    n_threads: int = 0,
    max_rss_gib: Optional[float] = 8.0,
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
    num_experts: int = NUM_EXPERTS_PER_LAYER,
    verify_after_write: bool = True,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Stream one physical layer (experts 0..num_experts-1) into a standalone GGUF shard."""
    prof = EXPERT_PROFILES[profile_name]
    p_gate, p_up, p_down = prof.layer_plan(layer)
    native_fmt = source.native_layer_format(layer)

    out_dir = Path(out_dir)
    shard_path = out_dir / shard_relpath(layer)
    receipt_path = out_dir / shard_receipt_relpath(layer)

    # Declare all 3 * num_experts tensors up front so StreamingGGUFWriter can write
    # exact headers and offsets in pass 1 and stream one expert at a time in pass 2.
    specs: List[TensorSpec] = []
    for e in range(num_experts):
        n_gate, n_up, n_down = expert_tensor_names(layer, e)
        specs.append(TensorSpec(n_gate, p_gate.stored_shape, p_gate.effective_type, p_gate.logical_shape))
        specs.append(TensorSpec(n_up, p_up.stored_shape, p_up.effective_type, p_up.logical_shape))
        specs.append(TensorSpec(n_down, p_down.stored_shape, p_down.effective_type, p_down.logical_shape))

    kv = _bundle_common_metadata(
        bundle_kind="expert_layer_shard",
        name=f"InkyNano-ExpertShard-L{layer:02d}",
        description=(
            f"Standalone GGML-quantized expert tensor bundle for Inkling-2-Mini physical layer {layer} "
            f"(128 individually addressable experts, native {native_fmt.upper()}, profile {profile_name}). "
            "NOT a complete drop-in llama.cpp model; designed for selective per-expert loading and "
            "LDT-controlled residency with soft-capped SwiGLU."
        ),
        backend_info=backend.info(),
        quant_profile=profile_name,
        extra={
            "inkynano.physical_layer": (GGUFValueType.UINT32, layer),
            "inkynano.native_quant_format": native_fmt,
            "inkynano.incompat_policy": prof.incompat_policy,
            "inkynano.gate_ggml_type": p_gate.effective_type.name,
            "inkynano.up_ggml_type": p_up.effective_type.name,
            "inkynano.down_ggml_type": p_down.effective_type.name,
            "inkynano.down_logical_ne0": (GGUFValueType.UINT32, p_down.logical_shape[1]),
            "inkynano.down_stored_ne0": (GGUFValueType.UINT32, p_down.stored_shape[1]),
        },
    )

    probe_x = build_deterministic_probes(hidden_size=p_gate.logical_shape[1], n_samples=16, seed=PILOT_SEED + layer)["unit_rms"]
    t0 = time.perf_counter()
    expert_entries: List[Dict[str, Any]] = []
    weight_rel_l2 = {"gate": [], "up": [], "down": []}
    swiglu_rel_l2: List[float] = []
    dq_backend_used = None

    writer = StreamingGGUFWriter(shard_path, metadata=kv, tensors=specs, alignment=alignment)
    try:
        for e in range(num_experts):
            _check_rss_guard(max_rss_gib)
            exp = dequantize_expert(
                source,
                layer=layer,
                expert_id=e,
                dtype=torch.bfloat16,
                device="cpu",
                backend=dequant_backend,
                loader_cast=True,
            )
            dq_backend_used = exp["dequant_backend"]
            gate_t: torch.Tensor = exp["gate"]  # type: ignore[assignment]
            up_t: torch.Tensor = exp["up"]  # type: ignore[assignment]
            down_t: torch.Tensor = exp["down"]  # type: ignore[assignment]

            gate_f32 = gate_t.float().numpy()
            up_f32 = up_t.float().numpy()
            down_f32 = down_t.float().numpy()

            ref_y = run_softcapped_swiglu(probe_x, gate_t, up_t, down_t).float().numpy()

            n_gate, n_up, n_down = expert_tensor_names(layer, e)
            q_gate = backend.quantize(pad_last_axis_zeros(gate_f32, p_gate.stored_shape[1]), p_gate.effective_type, n_threads=n_threads)
            q_up = backend.quantize(pad_last_axis_zeros(up_f32, p_up.stored_shape[1]), p_up.effective_type, n_threads=n_threads)
            q_down = backend.quantize(pad_last_axis_zeros(down_f32, p_down.stored_shape[1]), p_down.effective_type, n_threads=n_threads)

            rec_g = writer.write_tensor(n_gate, q_gate)
            rec_u = writer.write_tensor(n_up, q_up)
            rec_d = writer.write_tensor(n_down, q_down)

            # Numerical fidelity receipt using actual GGML C dequantization
            dq_gate = backend.dequantize(q_gate, p_gate.stored_shape, p_gate.effective_type)[:, : p_gate.logical_shape[1]]
            dq_up = backend.dequantize(q_up, p_up.stored_shape, p_up.effective_type)[:, : p_up.logical_shape[1]]
            dq_down = backend.dequantize(q_down, p_down.stored_shape, p_down.effective_type)[:, : p_down.logical_shape[1]]

            eg = array_error_metrics(dq_gate, gate_f32)
            eu = array_error_metrics(dq_up, up_f32)
            ed = array_error_metrics(dq_down, down_f32)
            q_y = run_softcapped_swiglu(probe_x, torch.from_numpy(dq_gate), torch.from_numpy(dq_up), torch.from_numpy(dq_down)).float().numpy()
            es = array_error_metrics(q_y, ref_y)

            weight_rel_l2["gate"].append(eg["relative_l2"])
            weight_rel_l2["up"].append(eu["relative_l2"])
            weight_rel_l2["down"].append(ed["relative_l2"])
            swiglu_rel_l2.append(es["relative_l2"])

            span_start = rec_g["file_offset"]
            span_end = rec_d["file_offset"] + rec_d["nbytes"]
            expert_entries.append(
                {
                    "layer": layer,
                    "expert": e,
                    "native_format": native_fmt,
                    "span_file_offset": span_start,
                    "span_bytes": span_end - span_start,
                    "payload_bytes": rec_g["nbytes"] + rec_u["nbytes"] + rec_d["nbytes"],
                    "tensors": {
                        "gate": rec_g,
                        "up": rec_u,
                        "down": rec_d,
                    },
                    "fidelity": {
                        "gate_rel_l2": round(eg["relative_l2"], 6),
                        "up_rel_l2": round(eu["relative_l2"], 6),
                        "down_rel_l2": round(ed["relative_l2"], 6),
                        "swiglu_rel_l2": round(es["relative_l2"], 6),
                        "swiglu_cosine": round(es["cosine_sim"], 6),
                    },
                }
            )

            del exp, gate_t, up_t, down_t, gate_f32, up_f32, down_f32
            del q_gate, q_up, q_down, dq_gate, dq_up, dq_down, ref_y, q_y
            if (e + 1) % 16 == 0:
                gc.collect()

        file_info = writer.finalize()
    except Exception:
        writer.abort()
        raise

    if verify_after_write:
        reader = GGUFBundleReader(shard_path)
        reader.verify_layout()
        if len(reader.tensors) != 3 * num_experts:
            raise RuntimeError(f"shard tensor count mismatch: {len(reader.tensors)} != {3 * num_experts}")
        # Spot-check first, middle, last expert byte-offset + SHA-256 reads
        for spot_e in sorted({0, num_experts // 2, num_experts - 1}):
            entry = expert_entries[spot_e]
            for proj in ("gate", "up", "down"):
                tinfo = entry["tensors"][proj]
                reader.read_tensor_bytes(tinfo["name"], verify_sha256=tinfo["sha256"])

    elapsed = time.perf_counter() - t0
    rss, peak_rss = process_memory()
    receipt = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "source_model": MODEL,
        "source_revision": REV,
        "transformers_revision": TRANSFORMERS_REV,
        "dequant_backend": dq_backend_used,
        "backend": backend.info(),
        "quant_profile": profile_name,
        "layer": layer,
        "native_format": native_fmt,
        "num_experts": num_experts,
        "shard_relpath": shard_relpath(layer),
        "bytes": file_info["bytes"],
        "sha256": file_info["sha256"],
        "alignment": file_info["alignment"],
        "data_section_offset": file_info["data_section_offset"],
        "tensor_count": file_info["tensor_count"],
        "uniform_expert_payload_bytes": prof.expert_payload_bytes(layer),
        "uniform_expert_span_bytes": expert_entries[0]["span_bytes"],
        "elapsed_seconds": elapsed,
        "experts_per_second": num_experts / max(elapsed, 1e-9),
        "rss_bytes": rss,
        "peak_rss_bytes": peak_rss,
        "fidelity_summary": {
            "gate_rel_l2_mean": float(np.mean(weight_rel_l2["gate"])),
            "gate_rel_l2_max": float(np.max(weight_rel_l2["gate"])),
            "up_rel_l2_mean": float(np.mean(weight_rel_l2["up"])),
            "up_rel_l2_max": float(np.max(weight_rel_l2["up"])),
            "down_rel_l2_mean": float(np.mean(weight_rel_l2["down"])),
            "down_rel_l2_max": float(np.max(weight_rel_l2["down"])),
            "swiglu_rel_l2_mean": float(np.mean(swiglu_rel_l2)),
            "swiglu_rel_l2_p95": float(np.percentile(swiglu_rel_l2, 95)),
            "swiglu_rel_l2_max": float(np.max(swiglu_rel_l2)),
        },
        "experts": expert_entries,
    }
    dump_json_atomic(receipt_path, receipt)
    return shard_path, receipt_path, receipt


# ---------------------------------------------------------------------------
# Streaming Text-Only Backbone Conversion (bounded RAM via row striping)
# ---------------------------------------------------------------------------

def _quantize_large_tensor_striped(
    source: SafetensorsStreamSource,
    key: str,
    qtype: GGMLType,
    backend,
    n_threads: int = 0,
    stripe_rows: int = 4096,
) -> Tuple[bytes, Dict[str, float]]:
    """Quantize a potentially huge 2D BF16 matrix (e.g., [201024, 2304] embed/unembed) in bounded RAM."""
    entry = source.entries[key]
    rows, cols = entry.shape
    row_raw_bytes = cols * entry.elem_size
    out_chunks: List[bytes] = []

    diff_sq = 0.0
    ref_sq = 0.0
    pred_sq = 0.0
    dot_pr = 0.0
    max_abs = 0.0

    for r0 in range(0, rows, stripe_rows):
        r1 = min(rows, r0 + stripe_rows)
        n_r = r1 - r0
        raw_stripe = source.read_range(entry.data_start + r0 * row_raw_bytes, n_r * row_raw_bytes)
        if entry.dtype == "BF16":
            u16 = np.frombuffer(raw_stripe, dtype="<u2").reshape(n_r, cols)
            f32 = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        elif entry.dtype == "F32":
            f32 = np.frombuffer(raw_stripe, dtype="<f4").reshape(n_r, cols).copy()
        elif entry.dtype == "F16":
            f32 = np.frombuffer(raw_stripe, dtype="<f2").reshape(n_r, cols).astype(np.float32)
        else:
            raise ValueError(f"unexpected backbone tensor dtype {entry.dtype} for {key}")

        qbytes = backend.quantize(f32, qtype, n_threads=n_threads)
        dq = backend.dequantize(qbytes, (n_r, cols), qtype)
        diff = dq.astype(np.float64) - f32.astype(np.float64)
        diff_sq += float(np.sum(diff * diff))
        ref_sq += float(np.sum(f32.astype(np.float64) ** 2))
        pred_sq += float(np.sum(dq.astype(np.float64) ** 2))
        dot_pr += float(np.sum(dq.astype(np.float64) * f32.astype(np.float64)))
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        out_chunks.append(qbytes)
        del raw_stripe, f32, dq, diff

    metrics = {
        "relative_l2": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-20),
        "max_abs": max_abs,
        "cosine_sim": dot_pr / max(math.sqrt(pred_sq) * math.sqrt(ref_sq), 1e-20),
    }
    return b"".join(out_chunks), metrics


def convert_text_backbone(
    source: SafetensorsStreamSource,
    backend,
    out_dir: Path,
    backbone_profile: str = "backbone_mixed_q4_k_q6_k",
    n_threads: int = 0,
    max_rss_gib: Optional[float] = 8.0,
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
    verify_after_write: bool = True,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Export separately quantized text-only backbone preserving sensitive routing/recurrent tensors."""
    if backbone_profile not in BACKBONE_PROFILES:
        raise ValueError(f"unknown backbone profile: {backbone_profile!r}")

    out_dir = Path(out_dir)
    backbone_path = out_dir / BACKBONE_RELPATH
    receipt_path = out_dir / BACKBONE_RECEIPT_RELPATH

    selected_keys = sorted(k for k in source.keys() if is_text_backbone_tensor(k))
    if not selected_keys:
        raise ValueError("no text-only backbone tensors found in source")
    # Explicitly verify zero vision and zero routed-expert leakage
    if any(k.startswith("model.visual.") or ".mlp.experts." in k for k in selected_keys):
        raise AssertionError("non-backbone tensor leaked into text-only backbone selection")

    plan: List[Tuple[str, TensorSpec, str, str]] = []
    for k in selected_keys:
        entry = source.entries[k]
        qtype, cat = classify_backbone_tensor(k, entry.shape, entry.dtype, backbone_profile)
        plan.append((k, TensorSpec(k, entry.shape, qtype, entry.shape), cat, entry.dtype))

    kv = _bundle_common_metadata(
        bundle_kind="text_only_backbone",
        name=f"InkyNano-TextBackbone-{backbone_profile}",
        description=(
            f"Separately quantized text-only backbone for Inkling-2-Mini (profile {backbone_profile}). "
            "Excludes vision tower, MTP, and all 3,072 routed experts. Preserves all routing "
            "(mlp.gate.*), recurrent message controllers (controllers.*, middle_cycle_cross_pass*), "
            "block attention residuals (attn_res.*), short convolutions (*sconv*), norms, and clip scales "
            "at exact native BF16/F32 precision. NOT a complete drop-in llama.cpp model."
        ),
        backend_info=backend.info(),
        quant_profile=backbone_profile,
        extra={
            "inkynano.vocab_size": (GGUFValueType.UINT32, 201024),
            "inkynano.unpadded_vocab_size": (GGUFValueType.UINT32, 200058),
            "inkynano.final_logit_softcapping": (GGUFValueType.FLOAT32, 30.0),
            "inkynano.vision_excluded": True,
            "inkynano.mtp_excluded": True,
            "inkynano.experts_excluded": True,
        },
    )

    t0 = time.perf_counter()
    tensor_receipts: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = {}

    writer = StreamingGGUFWriter(backbone_path, metadata=kv, tensors=[p[1] for p in plan], alignment=alignment)
    try:
        for key, spec, cat, src_dtype in plan:
            _check_rss_guard(max_rss_gib)
            entry = source.entries[key]
            is_verbatim = (
                (src_dtype == "BF16" and spec.ggml_type == GGMLType.BF16)
                or (src_dtype == "F32" and spec.ggml_type == GGMLType.F32)
                or (src_dtype == "F16" and spec.ggml_type == GGMLType.F16)
            )
            if is_verbatim:
                raw_bytes, _ = source.read_raw_tensor(key)
                metrics = {"relative_l2": 0.0, "max_abs": 0.0, "cosine_sim": 1.0, "verbatim_native": True}
            elif len(entry.shape) == 2 and entry.shape[0] > 4096:
                raw_bytes, metrics = _quantize_large_tensor_striped(
                    source, key, spec.ggml_type, backend, n_threads=n_threads
                )
                metrics["verbatim_native"] = False
            else:
                t = source.read_tensor(key).float().numpy()
                raw_bytes = backend.quantize(t, spec.ggml_type, n_threads=n_threads)
                dq = backend.dequantize(raw_bytes, spec.pt_shape, spec.ggml_type)
                metrics = array_error_metrics(dq, t)
                metrics["verbatim_native"] = False
                del t, dq

            rec = writer.write_tensor(key, raw_bytes)
            rec["category"] = cat
            rec["source_dtype"] = src_dtype
            rec["fidelity"] = metrics
            tensor_receipts.append(rec)

            cs = category_stats.setdefault(cat, {"tensors": 0, "bytes": 0, "rel_l2s": []})
            cs["tensors"] += 1
            cs["bytes"] += rec["nbytes"]
            cs["rel_l2s"].append(metrics["relative_l2"])

            del raw_bytes
            if entry.nbytes >= 32 * MIB:
                gc.collect()

        file_info = writer.finalize()
    except Exception:
        writer.abort()
        raise

    if verify_after_write:
        reader = GGUFBundleReader(backbone_path)
        reader.verify_layout()
        if len(reader.tensors) != len(plan):
            raise RuntimeError(f"backbone tensor count mismatch: {len(reader.tensors)} != {len(plan)}")
        for spot in (0, len(tensor_receipts) // 2, len(tensor_receipts) - 1):
            tinfo = tensor_receipts[spot]
            reader.read_tensor_bytes(tinfo["name"], verify_sha256=tinfo["sha256"])

    elapsed = time.perf_counter() - t0
    rss, peak_rss = process_memory()
    summary_by_cat = {
        cat: {
            "tensors": v["tensors"],
            "bytes": v["bytes"],
            "rel_l2_mean": float(np.mean(v["rel_l2s"])),
            "rel_l2_max": float(np.max(v["rel_l2s"])),
        }
        for cat, v in sorted(category_stats.items())
    }
    preserved_bytes = sum(v["bytes"] for k, v in summary_by_cat.items() if k.startswith("preserved_"))
    quantized_bytes = sum(v["bytes"] for k, v in summary_by_cat.items() if not k.startswith("preserved_"))

    receipt = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "source_model": MODEL,
        "source_revision": REV,
        "transformers_revision": TRANSFORMERS_REV,
        "backend": backend.info(),
        "backbone_profile": backbone_profile,
        "backbone_relpath": BACKBONE_RELPATH,
        "bytes": file_info["bytes"],
        "sha256": file_info["sha256"],
        "alignment": file_info["alignment"],
        "data_section_offset": file_info["data_section_offset"],
        "tensor_count": file_info["tensor_count"],
        "preserved_payload_bytes": preserved_bytes,
        "quantized_payload_bytes": quantized_bytes,
        "category_summary": summary_by_cat,
        "elapsed_seconds": elapsed,
        "rss_bytes": rss,
        "peak_rss_bytes": peak_rss,
        "tensors": tensor_receipts,
    }
    dump_json_atomic(receipt_path, receipt)
    return backbone_path, receipt_path, receipt


# ---------------------------------------------------------------------------
# Master Manifest Builder & HF Model Card Generator
# ---------------------------------------------------------------------------

def build_master_manifest(
    out_dir: Path,
    expected_layers: Sequence[int] = tuple(range(NUM_PHYSICAL_LAYERS)),
) -> Dict[str, Any]:
    """Assemble manifest.json mapping every expert and backbone tensor to offsets, shapes, types & SHA-256."""
    out_dir = Path(out_dir)
    shards: Dict[str, Any] = {}
    experts_index: Dict[str, Dict[str, Any]] = {}
    per_layer_expert_bytes: List[int] = []
    per_layer_span_bytes: List[int] = []
    total_shard_bytes = 0
    total_experts = 0
    quant_profile = None
    backend_info = None

    for layer in expected_layers:
        rpath = out_dir / shard_receipt_relpath(layer)
        if not rpath.is_file():
            raise FileNotFoundError(f"missing shard receipt for layer {layer}: {rpath}")
        rec = json.loads(rpath.read_text(encoding="utf8"))
        if quant_profile is None:
            quant_profile = rec["quant_profile"]
            backend_info = rec["backend"]
        elif rec["quant_profile"] != quant_profile:
            raise ValueError(f"mixed quant_profile across shards: {rec['quant_profile']} vs {quant_profile}")

        shards[str(layer)] = {
            "layer": layer,
            "path": rec["shard_relpath"],
            "receipt_path": shard_receipt_relpath(layer),
            "bytes": rec["bytes"],
            "sha256": rec["sha256"],
            "alignment": rec["alignment"],
            "data_section_offset": rec["data_section_offset"],
            "tensor_count": rec["tensor_count"],
            "native_format": rec["native_format"],
            "num_experts": rec["num_experts"],
            "uniform_expert_payload_bytes": rec["uniform_expert_payload_bytes"],
            "uniform_expert_span_bytes": rec["uniform_expert_span_bytes"],
            "fidelity_summary": rec["fidelity_summary"],
        }
        per_layer_expert_bytes.append(int(rec["uniform_expert_payload_bytes"]))
        per_layer_span_bytes.append(int(rec["uniform_expert_span_bytes"]))
        total_shard_bytes += int(rec["bytes"])
        total_experts += int(rec["num_experts"])

        layer_map: Dict[str, Any] = {}
        for ex in rec["experts"]:
            eid = int(ex["expert"])
            layer_map[str(eid)] = {
                "shard": rec["shard_relpath"],
                "native_format": ex["native_format"],
                "span_file_offset": ex["span_file_offset"],
                "span_bytes": ex["span_bytes"],
                "payload_bytes": ex["payload_bytes"],
                "tensors": {
                    proj: {
                        "name": t["name"],
                        "shape": t["shape"],
                        "logical_shape": t["logical_shape"],
                        "ggml_type": t["ggml_type"],
                        "file_offset": t["file_offset"],
                        "data_offset": t["data_offset"],
                        "nbytes": t["nbytes"],
                        "sha256": t["sha256"],
                    }
                    for proj, t in ex["tensors"].items()
                },
            }
        experts_index[str(layer)] = layer_map

    bb_rpath = out_dir / BACKBONE_RECEIPT_RELPATH
    backbone_summary = None
    if bb_rpath.is_file():
        bbrec = json.loads(bb_rpath.read_text(encoding="utf8"))
        backbone_summary = {
            "path": bbrec["backbone_relpath"],
            "receipt_path": BACKBONE_RECEIPT_RELPATH,
            "backbone_profile": bbrec["backbone_profile"],
            "bytes": bbrec["bytes"],
            "sha256": bbrec["sha256"],
            "alignment": bbrec["alignment"],
            "data_section_offset": bbrec["data_section_offset"],
            "tensor_count": bbrec["tensor_count"],
            "preserved_payload_bytes": bbrec["preserved_payload_bytes"],
            "quantized_payload_bytes": bbrec["quantized_payload_bytes"],
            "category_summary": bbrec["category_summary"],
            "tensors": [
                {
                    "name": t["name"],
                    "shape": t["shape"],
                    "ggml_type": t["ggml_type"],
                    "file_offset": t["file_offset"],
                    "data_offset": t["data_offset"],
                    "nbytes": t["nbytes"],
                    "sha256": t["sha256"],
                    "category": t["category"],
                }
                for t in bbrec["tensors"]
            ],
        }

    manifest = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "bundle_format": "gguf_v3_standalone_tensor_bundles",
        "llama_cpp_dropin_compatible": False,
        "llama_cpp_note": (
            "These files are standalone GGUF v3 GGML-quantized tensor bundles (24 per-layer expert shards + "
            "1 text-only backbone) designed for selective per-expert byte-offset loading and LDT residency. "
            "They are NOT a complete drop-in llama.cpp llama-cli model."
        ),
        "source_model": MODEL,
        "source_revision": REV,
        "transformers_revision": TRANSFORMERS_REV,
        "license": "apache-2.0",
        "backend": backend_info,
        "quant_profile": quant_profile,
        "activation": {
            "type": "softcapped_swiglu",
            "gate_softcap": GATE_SOFTCAP,
            "up_softcap": UP_SOFTCAP,
            "sigmoid_on_raw_gate": True,
            "accumulation_dtype": "float32",
        },
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "shared_cache_key": "(physical_layer, expert)"},
        "num_layers": len(expected_layers),
        "total_experts": total_experts,
        "total_shard_bytes": total_shard_bytes,
        "per_layer_expert_payload_bytes": per_layer_expert_bytes,
        "per_layer_expert_span_bytes": per_layer_span_bytes,
        "shards": shards,
        "backbone": backbone_summary,
        "experts": experts_index,
    }
    dump_json_atomic(out_dir / MANIFEST_RELPATH, manifest)
    return manifest


def render_hf_readme(manifest: Dict[str, Any], benchmark_receipt: Optional[Dict[str, Any]] = None) -> str:
    """Render an accurate Hugging Face README.md distinguishing tensor bundles from llama.cpp models."""
    bb = manifest.get("backbone") or {}
    bb_bytes = int(bb.get("bytes", 0))
    total_bank = int(manifest.get("total_shard_bytes", 0))
    prof_name = manifest.get("quant_profile", "unknown")
    per_layer_bytes = manifest.get("per_layer_expert_payload_bytes", [])

    k8 = sum(8 * b for b in per_layer_bytes)
    k16 = sum(16 * b for b in per_layer_bytes)
    k32 = sum(32 * b for b in per_layer_bytes)

    lines = [
        "---",
        "license: apache-2.0",
        "base_model: thinkingmachines/Inkling-2-Mini",
        "tags:",
        "  - gguf",
        "  - ggml",
        "  - moe",
        "  - inkynano",
        "  - tensor-bundle",
        "---",
        "",
        "# InkyNano-GGUF (Standalone GGML Expert Shards & Text-Only Backbone)",
        "",
        "> **Important — Standalone Tensor Bundles, NOT a drop-in `llama.cpp` model:**",
        "> Every `.gguf` file in this repository is a standard GGUF v3 container holding actual GGML",
        "> block-quantized tensors (`general.type = \"tensors\"`, `inkynano.llama_cpp_dropin_compatible = false`).",
        "> Upstream `llama.cpp` does not implement Inkling-2-Loop's 3-pass recurrent core, block attention",
        "> residuals, short convolutions, or soft-capped SwiGLU (`gate_cap=4.0`, `up_cap=25.0`, raw-gate sigmoid).",
        "> Load these bundles via `InkyNano/gguf_loader.py` for selective per-expert byte-offset fetching",
        "> and LDT-controlled residency on RTX 3050 (4 GiB).",
        "",
        "## Provenance",
        "",
        f"- Source checkpoint: `{manifest['source_model']}@{manifest['source_revision']}` (Apache-2.0)",
        f"- Native dequantization reference: `huggingface/transformers@{manifest['transformers_revision']}` (reusing verified SilicoExperiments PR #3 parity path, `loader_cast=True`)",
        f"- GGML quantization commit: `ggml-org/llama.cpp@{(manifest.get('backend') or {}).get('ggml_commit', 'unknown')}`",
        f"- Expert quantization profile: `{prof_name}`",
        f"- Backbone quantization profile: `{bb.get('backbone_profile', 'n/a')}`",
        "",
        "## Repository Layout",
        "",
        f"- `shards/inkynano_experts_layer_{{00..23}}.gguf` — 24 independently downloadable layer shards ({manifest['total_experts']} physical experts, {total_bank / GIB:.3f} GiB total).",
        f"- `backbone/inkynano_backbone_text.gguf` — separately quantized text-only backbone ({bb_bytes / GIB:.3f} GiB; excludes vision tower, MTP, and experts; preserves all routing `mlp.gate.*`, recurrent `controllers.*`, `middle_cycle_cross_pass*`, `attn_res.*`, `*sconv*`, norms and clip scales verbatim at native BF16/F32).",
        "- `manifest.json` — full index mapping all 3,072 experts and all backbone tensors to shapes, GGML types, exact file offsets, sizes, contiguous per-expert spans, and SHA-256 hashes.",
        "- `receipts/` — 16-expert benchmark comparison (`pure_q4_k` vs `mixed_q4_k_q6_k`), per-shard conversion fidelity receipts, backbone receipt, and checkpoint state.",
        "",
        "## RTX 3050 (4 GiB) Payload Residency Estimates",
        "",
        "| Resident slots / layer (`k`) | Resident experts | Expert payload | Backbone + experts | Headroom before KV/activations/CUDA |",
        "|---:|---:|---:|---:|---:|",
        f"| 8 | 192 | {k8 / GIB:.3f} GiB | {(bb_bytes + k8) / GIB:.3f} GiB | {(4 * GIB - bb_bytes - k8) / GIB:.3f} GiB |",
        f"| 16 | 384 | {k16 / GIB:.3f} GiB | {(bb_bytes + k16) / GIB:.3f} GiB | {(4 * GIB - bb_bytes - k16) / GIB:.3f} GiB |",
        f"| 32 | 768 | {k32 / GIB:.3f} GiB | {(bb_bytes + k32) / GIB:.3f} GiB | {(4 * GIB - bb_bytes - k32) / GIB:.3f} GiB |",
        "",
    ]
    if benchmark_receipt and "profile_summaries" in benchmark_receipt:
        lines.extend(
            [
                "## 16-Expert Pilot Benchmark Summary",
                "",
                "| Profile | NVFP4 types | MXFP8 types | Full bank (GiB) | SwiGLU unit-RMS rel-L2 | SwiGLU heavy-tail rel-L2 | Projected 3,072 CPU min |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for pname, s in benchmark_receipt["profile_summaries"].items():
            nvt = "/".join(s["nvfp4_quant_types"][k] for k in ("gate", "up", "down"))
            mxt = "/".join(s["mxfp8_quant_types"][k] for k in ("gate", "up", "down"))
            lines.append(
                f"| `{pname}` | `{nvt}` | `{mxt}` | {s['full_bank_payload_gib']:.3f} | "
                f"{s['swiglu_unit_rms_rel_l2_mean']:.5f} | {s['swiglu_heavy_tail_rel_l2_mean']:.5f} | "
                f"{s['projected_3072_total_minutes']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Incremental Hugging Face Sync & Disconnect-Safe Resume Orchestrator
# ---------------------------------------------------------------------------

def _get_hf_api(token: Optional[str] = None):
    from huggingface_hub import HfApi

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return HfApi(token=tok), tok


def _remote_file_index(api, repo_id: str) -> Dict[str, Dict[str, Any]]:
    """Map remote repo relative paths to {'size': int, 'sha256': Optional[str]}."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        entries = api.list_repo_tree(repo_id=repo_id, repo_type="model", recursive=True)
    except Exception:
        return out
    for item in entries:
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)
        if path is None or size is None:
            continue
        lfs = getattr(item, "lfs", None)
        sha = None
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        out[path] = {"size": int(size), "sha256": sha}
    return out


def _download_remote_json(api, repo_id: str, relpath: str, dest: Path, token: Optional[str]) -> Optional[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(repo_id=repo_id, filename=relpath, repo_type="model", token=token)
        data = json.loads(Path(cached).read_text(encoding="utf8"))
        dump_json_atomic(dest, data)
        return data
    except Exception:
        return None


def _receipt_matches_job(rec: Dict[str, Any], profile_key: str, expected_profile: str) -> bool:
    return (
        rec.get("source_model") == MODEL
        and rec.get("source_revision") == REV
        and rec.get("transformers_revision") == TRANSFORMERS_REV
        and rec.get(profile_key) == expected_profile
    )


def _local_artifact_verified(artifact_path: Path, receipt: Dict[str, Any]) -> bool:
    if not artifact_path.is_file():
        return False
    if artifact_path.stat().st_size != int(receipt.get("bytes", -1)):
        return False
    return sha256_file(artifact_path) == receipt.get("sha256")


def _remote_artifact_verified(remote_index: Dict[str, Dict[str, Any]], relpath: str, receipt: Dict[str, Any]) -> bool:
    meta = remote_index.get(relpath)
    if meta is None:
        return False
    if meta["size"] != int(receipt.get("bytes", -1)):
        return False
    if meta.get("sha256") and meta["sha256"] != receipt.get("sha256"):
        return False
    return True


def _upload_commit(api, repo_id: str, files: Sequence[Tuple[Path, str]], message: str) -> str:
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path)) for local_path, repo_path in files]
    info = api.create_commit(repo_id=repo_id, repo_type="model", operations=ops, commit_message=message)
    return getattr(info, "oid", "") or str(info)


def run_all_resumable(
    source: SafetensorsStreamSource,
    backend,
    out_dir: Path,
    profile_name: str = "mixed_q4_k_q6_k",
    backbone_profile: str = "backbone_mixed_q4_k_q6_k",
    layers: Sequence[int] = tuple(range(NUM_PHYSICAL_LAYERS)),
    hf_repo: Optional[str] = DEFAULT_HF_REPO,
    hf_token: Optional[str] = None,
    private_repo: bool = True,
    keep_local_shards: bool = False,
    dequant_backend: str = "auto",
    n_threads: int = 0,
    max_rss_gib: Optional[float] = 8.0,
    num_experts: int = NUM_EXPERTS_PER_LAYER,
    source_config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert all requested shards + text backbone with disconnect-safe resume & incremental HF upload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(l) for l in layers]

    api = None
    tok = None
    remote_idx: Dict[str, Dict[str, Any]] = {}
    if hf_repo:
        api, tok = _get_hf_api(hf_token)
        api.create_repo(repo_id=hf_repo, repo_type="model", private=private_repo, exist_ok=True)
        remote_idx = _remote_file_index(api, hf_repo)

    state: Dict[str, Any] = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "updated_utc": utc_now(),
        "source_model": MODEL,
        "source_revision": REV,
        "transformers_revision": TRANSFORMERS_REV,
        "quant_profile": profile_name,
        "backbone_profile": backbone_profile,
        "hf_repo": hf_repo,
        "completed_layers": [],
        "skipped_layers": [],
        "converted_layers": [],
        "backbone_status": "pending",
    }

    # 1. Resumable layer-by-layer loop
    for layer in layers:
        s_rel = shard_relpath(layer)
        r_rel = shard_receipt_relpath(layer)
        s_path = out_dir / s_rel
        r_path = out_dir / r_rel

        # Check remote HF completion first (survives full Colab VM wipe!)
        if hf_repo and s_rel in remote_idx and r_rel in remote_idx:
            rec = None
            if r_path.is_file():
                try:
                    rec = json.loads(r_path.read_text(encoding="utf8"))
                except Exception:
                    rec = None
            if rec is None:
                rec = _download_remote_json(api, hf_repo, r_rel, r_path, tok)
            if rec and _receipt_matches_job(rec, "quant_profile", profile_name) and _remote_artifact_verified(remote_idx, s_rel, rec):
                print(f"[RESUME] Layer {layer:02d}: verified on HF ({rec['bytes'] / MIB:.1f} MiB, sha256={rec['sha256'][:12]}...) -> SKIP")
                state["completed_layers"].append(layer)
                state["skipped_layers"].append(layer)
                dump_json_atomic(out_dir / CHECKPOINT_STATE_RELPATH, state)
                continue

        # Check local completion (e.g. local-only run or interrupted before upload)
        local_ready = False
        rec = None
        if r_path.is_file():
            try:
                rec = json.loads(r_path.read_text(encoding="utf8"))
                if _receipt_matches_job(rec, "quant_profile", profile_name) and _local_artifact_verified(s_path, rec):
                    local_ready = True
            except Exception:
                rec = None

        if local_ready and rec is not None:
            print(f"[RESUME] Layer {layer:02d}: verified locally ({rec['bytes'] / MIB:.1f} MiB) -> skip conversion")
            state["skipped_layers"].append(layer)
        else:
            print(f"[CONVERT] Layer {layer:02d}/{NUM_PHYSICAL_LAYERS - 1} ({source.native_layer_format(layer).upper()}, profile={profile_name})...")
            s_path, r_path, rec = convert_layer_shard(
                source=source,
                backend=backend,
                layer=layer,
                profile_name=profile_name,
                out_dir=out_dir,
                dequant_backend=dequant_backend,
                n_threads=n_threads,
                max_rss_gib=max_rss_gib,
                num_experts=num_experts,
            )
            state["converted_layers"].append(layer)
            print(
                f"[DONE]    Layer {layer:02d}: {rec['bytes'] / MIB:.1f} MiB in {rec['elapsed_seconds']:.1f}s "
                f"(SwiGLU rel_l2 mean={rec['fidelity_summary']['swiglu_rel_l2_mean']:.5f}, "
                f"peak_rss={rec['peak_rss_bytes'] / GIB:.2f} GiB)"
            )

        state["completed_layers"].append(layer)
        state["updated_utc"] = utc_now()
        dump_json_atomic(out_dir / CHECKPOINT_STATE_RELPATH, state)

        if hf_repo and not _remote_artifact_verified(remote_idx, s_rel, rec):
            print(f"[UPLOAD]  Layer {layer:02d} -> {hf_repo} ...")
            oid = _upload_commit(
                api,
                hf_repo,
                [
                    (s_path, s_rel),
                    (r_path, r_rel),
                    (out_dir / CHECKPOINT_STATE_RELPATH, CHECKPOINT_STATE_RELPATH),
                ],
                message=f"Add InkyNano GGUF expert shard layer {layer:02d} ({profile_name})",
            )
            remote_idx = _remote_file_index(api, hf_repo)
            if not _remote_artifact_verified(remote_idx, s_rel, rec):
                raise RuntimeError(f"remote verification failed after upload for layer {layer:02d}")
            print(f"[SYNCED]  Layer {layer:02d} commit={oid[:12]}")
            if not keep_local_shards:
                s_path.unlink(missing_ok=True)

    # 2. Resumable Text-Only Backbone
    bb_path = out_dir / BACKBONE_RELPATH
    bb_rpath = out_dir / BACKBONE_RECEIPT_RELPATH
    bb_rec = None
    bb_skipped = False

    if hf_repo and BACKBONE_RELPATH in remote_idx and BACKBONE_RECEIPT_RELPATH in remote_idx:
        if bb_rpath.is_file():
            try:
                bb_rec = json.loads(bb_rpath.read_text(encoding="utf8"))
            except Exception:
                bb_rec = None
        if bb_rec is None:
            bb_rec = _download_remote_json(api, hf_repo, BACKBONE_RECEIPT_RELPATH, bb_rpath, tok)
        if bb_rec and _receipt_matches_job(bb_rec, "backbone_profile", backbone_profile) and _remote_artifact_verified(remote_idx, BACKBONE_RELPATH, bb_rec):
            print(f"[RESUME] Backbone: verified on HF ({bb_rec['bytes'] / GIB:.3f} GiB) -> SKIP")
            bb_skipped = True
            state["backbone_status"] = "skipped_remote"

    if not bb_skipped:
        local_bb_ok = False
        if bb_rpath.is_file():
            try:
                bb_rec = json.loads(bb_rpath.read_text(encoding="utf8"))
                if _receipt_matches_job(bb_rec, "backbone_profile", backbone_profile) and _local_artifact_verified(bb_path, bb_rec):
                    local_bb_ok = True
            except Exception:
                bb_rec = None
        if local_bb_ok and bb_rec is not None:
            print(f"[RESUME] Backbone: verified locally ({bb_rec['bytes'] / GIB:.3f} GiB) -> skip conversion")
            state["backbone_status"] = "skipped_local"
        else:
            print(f"[CONVERT] Text-only backbone (profile={backbone_profile})...")
            bb_path, bb_rpath, bb_rec = convert_text_backbone(
                source=source,
                backend=backend,
                out_dir=out_dir,
                backbone_profile=backbone_profile,
                n_threads=n_threads,
                max_rss_gib=max_rss_gib,
            )
            state["backbone_status"] = "converted"
            print(
                f"[DONE]    Backbone: {bb_rec['bytes'] / GIB:.3f} GiB in {bb_rec['elapsed_seconds']:.1f}s "
                f"(preserved={bb_rec['preserved_payload_bytes'] / MIB:.1f} MiB, peak_rss={bb_rec['peak_rss_bytes'] / GIB:.2f} GiB)"
            )

        state["updated_utc"] = utc_now()
        dump_json_atomic(out_dir / CHECKPOINT_STATE_RELPATH, state)

        if hf_repo and not _remote_artifact_verified(remote_idx, BACKBONE_RELPATH, bb_rec):
            print(f"[UPLOAD]  Backbone -> {hf_repo} ...")
            oid = _upload_commit(
                api,
                hf_repo,
                [
                    (bb_path, BACKBONE_RELPATH),
                    (bb_rpath, BACKBONE_RECEIPT_RELPATH),
                    (out_dir / CHECKPOINT_STATE_RELPATH, CHECKPOINT_STATE_RELPATH),
                ],
                message=f"Add InkyNano quantized text-only backbone ({backbone_profile})",
            )
            remote_idx = _remote_file_index(api, hf_repo)
            if not _remote_artifact_verified(remote_idx, BACKBONE_RELPATH, bb_rec):
                raise RuntimeError("remote verification failed after backbone upload")
            print(f"[SYNCED]  Backbone commit={oid[:12]}")
            if not keep_local_shards:
                bb_path.unlink(missing_ok=True)

    # 3. Build Master Manifest & README & Upload Final Metadata
    manifest = build_master_manifest(out_dir, expected_layers=layers)
    bench_path = out_dir / BENCHMARK_RECEIPT_RELPATH
    bench_rec = json.loads(bench_path.read_text(encoding="utf8")) if bench_path.is_file() else None
    readme_text = render_hf_readme(manifest, bench_rec)
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf8")

    state["manifest_sha256"] = sha256_file(out_dir / MANIFEST_RELPATH)
    state["manifest_bytes"] = (out_dir / MANIFEST_RELPATH).stat().st_size
    state["status"] = "complete"
    state["updated_utc"] = utc_now()
    dump_json_atomic(out_dir / CHECKPOINT_STATE_RELPATH, state)

    if hf_repo:
        final_files: List[Tuple[Path, str]] = [
            (out_dir / MANIFEST_RELPATH, MANIFEST_RELPATH),
            (readme_path, "README.md"),
            (out_dir / CHECKPOINT_STATE_RELPATH, CHECKPOINT_STATE_RELPATH),
        ]
        if bench_path.is_file():
            final_files.append((bench_path, BENCHMARK_RECEIPT_RELPATH))
        if source_config_dir is not None:
            for meta_name in ("config.json", "hf_quant_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "generation_config.json"):
                cand = Path(source_config_dir) / meta_name
                if cand.is_file():
                    final_files.append((cand, meta_name))
        oid = _upload_commit(
            api,
            hf_repo,
            final_files,
            message=f"Publish InkyNano GGUF manifest, README and receipts ({profile_name})",
        )
        print(f"[PUBLISHED] {hf_repo} manifest + README commit={oid[:12]}")

    return state


def parse_layer_spec(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="InkyNano resumable GGML GGUF expert & backbone converter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-ggml", help="Build pinned CPU-only libggml-base + libggml-cpu")
    p_build.add_argument("--dest", type=Path, required=True)
    p_build.add_argument("--commit", default=PINNED_LLAMA_CPP_COMMIT)
    p_build.add_argument("--jobs", type=int, default=0)

    def _add_common(p):
        p.add_argument("--source", type=Path, default=None, help="Local model-00001-of-00001.safetensors (omit for HTTP Range streaming)")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--ggml-lib-dir", type=Path, default=None)
        p.add_argument("--dequant-backend", choices=("auto", "transformers", "standalone"), default="auto")
        p.add_argument("--threads", type=int, default=0)
        p.add_argument("--max-rss-gib", type=float, default=8.0)

    p_bench = sub.add_parser("benchmark", help="Run 16-expert pilot benchmark (pure Q4_K vs mixed Q4_K/Q6_K)")
    _add_common(p_bench)
    p_bench.add_argument(
        "--profiles",
        default="pure_q4_k,pure_q4_k_q4_0,pure_q4_k_padded,mixed_q4_k_q6_k,mixed_q4_k_q6_k_down_q8,mixed_q4_k_q6_k_padded",
    )

    p_layer = sub.add_parser("convert-layer", help="Convert one physical layer shard")
    _add_common(p_layer)
    p_layer.add_argument("--layer", type=int, required=True)
    p_layer.add_argument("--profile", choices=tuple(EXPERT_PROFILES), default="mixed_q4_k_q6_k")

    p_bb = sub.add_parser("convert-backbone", help="Convert separately quantized text-only backbone")
    _add_common(p_bb)
    p_bb.add_argument("--backbone-profile", choices=tuple(BACKBONE_PROFILES), default="backbone_mixed_q4_k_q6_k")

    p_all = sub.add_parser("run-all", help="Resumable full 24-shard + backbone conversion & HF upload")
    _add_common(p_all)
    p_all.add_argument("--profile", choices=tuple(EXPERT_PROFILES), default="mixed_q4_k_q6_k")
    p_all.add_argument("--backbone-profile", choices=tuple(BACKBONE_PROFILES), default="backbone_mixed_q4_k_q6_k")
    p_all.add_argument("--layers", default="0-23")
    p_all.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HF repo id (pass empty string to disable upload)")
    p_all.add_argument("--keep-local-shards", action="store_true")
    p_all.add_argument("--source-config-dir", type=Path, default=None)

    p_man = sub.add_parser("build-manifest", help="Rebuild manifest.json from local receipts")
    p_man.add_argument("--out", type=Path, required=True)
    p_man.add_argument("--layers", default="0-23")

    args = ap.parse_args()

    if args.cmd == "build-ggml":
        receipt = build_ggml_cpu(args.dest, commit=args.commit, jobs=args.jobs)
        print(json.dumps(receipt, indent=2))
        return

    if args.cmd == "build-manifest":
        man = build_master_manifest(args.out, expected_layers=parse_layer_spec(args.layers))
        print(json.dumps({"total_experts": man["total_experts"], "total_shard_bytes": man["total_shard_bytes"]}, indent=2))
        return

    backend = get_backend(args.ggml_lib_dir)
    with open_source(args.source) as src:
        if args.cmd == "benchmark":
            profs = [p.strip() for p in args.profiles.split(",") if p.strip()]
            rec = run_benchmark_16(
                src,
                backend,
                args.out,
                profile_names=profs,
                dequant_backend=args.dequant_backend,
                n_threads=args.threads,
                max_rss_gib=args.max_rss_gib,
            )
            print(json.dumps({k: {m: v[m] for m in ("full_bank_payload_gib", "swiglu_unit_rms_rel_l2_mean", "projected_3072_total_minutes")} for k, v in rec["profile_summaries"].items()}, indent=2))
        elif args.cmd == "convert-layer":
            _, _, rec = convert_layer_shard(
                src,
                backend,
                layer=args.layer,
                profile_name=args.profile,
                out_dir=args.out,
                dequant_backend=args.dequant_backend,
                n_threads=args.threads,
                max_rss_gib=args.max_rss_gib,
            )
            print(json.dumps({"layer": args.layer, "bytes": rec["bytes"], "sha256": rec["sha256"], "fidelity": rec["fidelity_summary"]}, indent=2))
        elif args.cmd == "convert-backbone":
            _, _, rec = convert_text_backbone(
                src,
                backend,
                out_dir=args.out,
                backbone_profile=args.backbone_profile,
                n_threads=args.threads,
                max_rss_gib=args.max_rss_gib,
            )
            print(json.dumps({"bytes": rec["bytes"], "sha256": rec["sha256"], "categories": rec["category_summary"]}, indent=2))
        elif args.cmd == "run-all":
            hf_repo = args.hf_repo.strip() or None
            state = run_all_resumable(
                src,
                backend,
                out_dir=args.out,
                profile_name=args.profile,
                backbone_profile=args.backbone_profile,
                layers=parse_layer_spec(args.layers),
                hf_repo=hf_repo,
                keep_local_shards=args.keep_local_shards,
                dequant_backend=args.dequant_backend,
                n_threads=args.threads,
                max_rss_gib=args.max_rss_gib,
                source_config_dir=args.source_config_dir,
            )
            print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
