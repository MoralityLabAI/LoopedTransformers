"""Quantization profiles for Inkling-2-Mini expert shards and text-only backbone.

Explicitly encodes:
  1. Pure Q4_K vs mixed Q4_K/Q6_K expert profiles (including handling of
     down_proj ne[0] = 1152 which is divisible by 32 but not QK_K = 256).
  2. Text-only backbone classification preserving ALL sensitive routing and
     recurrent control tensors (mlp.gate, controllers.*, middle_cycle_cross_pass*,
     attn_res.*, *sconv*, norms, L2 clips, sinks) at native BF16/F32 precision.
  3. Exact byte and RTX 3050 4 GiB residency budgets for k = 8, 16, 32 slots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from gguf_backend import resolve_ggml_type
from gguf_dequant import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS_PER_LAYER, NUM_PHYSICAL_LAYERS
from gguf_format import GGMLType, ggml_nbytes, parse_ggml_type

NVFP4_LAYERS = frozenset(range(0, 20))
MXFP8_LAYERS = frozenset(range(20, 24))


@dataclass(frozen=True)
class ExpertProjectionPlan:
    projection: str  # "gate" | "up" | "down"
    logical_shape: Tuple[int, int]
    stored_shape: Tuple[int, int]
    requested_type: GGMLType
    effective_type: GGMLType
    nbytes: int


@dataclass(frozen=True)
class ExpertQuantProfile:
    name: str
    description: str
    nvfp4_types: Tuple[GGMLType, GGMLType, GGMLType]  # (gate, up, down)
    mxfp8_types: Tuple[GGMLType, GGMLType, GGMLType]  # (gate, up, down)
    incompat_policy: str = "llama_cpp_fallback"

    def layer_plan(self, layer: int) -> Tuple[ExpertProjectionPlan, ExpertProjectionPlan, ExpertProjectionPlan]:
        if not 0 <= layer < NUM_PHYSICAL_LAYERS:
            raise ValueError(f"layer out of range: {layer}")
        req_gate, req_up, req_down = self.mxfp8_types if layer in MXFP8_LAYERS else self.nvfp4_types
        specs = (
            ("gate", (INTERMEDIATE_SIZE, HIDDEN_SIZE), req_gate),
            ("up", (INTERMEDIATE_SIZE, HIDDEN_SIZE), req_up),
            ("down", (HIDDEN_SIZE, INTERMEDIATE_SIZE), req_down),
        )
        plans: List[ExpertProjectionPlan] = []
        for proj, logical_shape, req_t in specs:
            eff_t, stored_ne0 = resolve_ggml_type(req_t, logical_shape[1], incompat_policy=self.incompat_policy)
            stored_shape = (logical_shape[0], stored_ne0)
            plans.append(
                ExpertProjectionPlan(
                    projection=proj,
                    logical_shape=logical_shape,
                    stored_shape=stored_shape,
                    requested_type=req_t,
                    effective_type=eff_t,
                    nbytes=ggml_nbytes(stored_shape, eff_t),
                )
            )
        return plans[0], plans[1], plans[2]

    def expert_payload_bytes(self, layer: int) -> int:
        return sum(p.nbytes for p in self.layer_plan(layer))

    def layer_payload_bytes(self, layer: int) -> int:
        return NUM_EXPERTS_PER_LAYER * self.expert_payload_bytes(layer)

    def bank_payload_bytes(self) -> int:
        return sum(self.layer_payload_bytes(l) for l in range(NUM_PHYSICAL_LAYERS))

    def per_layer_expert_bytes(self) -> List[int]:
        return [self.expert_payload_bytes(l) for l in range(NUM_PHYSICAL_LAYERS)]


EXPERT_PROFILES: Dict[str, ExpertQuantProfile] = {
    "pure_q4_k": ExpertQuantProfile(
        name="pure_q4_k",
        description=(
            "Pure Q4_K with canonical llama.cpp tensor_type_fallback: gate/up=Q4_K, "
            "down=Q5_0 (exact unpadded [2304,1152] since 1152 % 256 = 128)."
        ),
        nvfp4_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        mxfp8_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        incompat_policy="llama_cpp_fallback",
    ),
    "pure_q4_k_q4_0": ExpertQuantProfile(
        name="pure_q4_k_q4_0",
        description=(
            "Strict 4.5-bpw unpadded Q4_K/Q4_0 baseline: gate/up=Q4_K, down=Q4_0 "
            "(exact unpadded [2304,1152])."
        ),
        nvfp4_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        mxfp8_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        incompat_policy="q4_0_fallback",
    ),
    "pure_q4_k_padded": ExpertQuantProfile(
        name="pure_q4_k_padded",
        description=(
            "Literal Q4_K on all three projections by zero-padding down_proj ne[0] "
            "from 1152 to 1280 (5 * 256); logical_shape remains [2304,1152]."
        ),
        nvfp4_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        mxfp8_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        incompat_policy="pad_to_block",
    ),
    "mixed_q4_k_q6_k": ExpertQuantProfile(
        name="mixed_q4_k_q6_k",
        description=(
            "Format-matched mixed Q4_K/Q6_K with canonical llama.cpp fallback: "
            "native NVFP4 layers 0-19 use gate/up=Q4_K, down=Q5_0; "
            "native MXFP8 layers 20-23 use gate/up=Q6_K, down=Q8_0 (unpadded)."
        ),
        nvfp4_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q4_K),
        mxfp8_types=(GGMLType.Q6_K, GGMLType.Q6_K, GGMLType.Q6_K),
        incompat_policy="llama_cpp_fallback",
    ),
    "mixed_q4_k_q6_k_down_q8": ExpertQuantProfile(
        name="mixed_q4_k_q6_k_down_q8",
        description=(
            "Q4_K_M-style mixed Q4_K/Q6_K with canonical llama.cpp fallback: "
            "layers 0-19 gate/up=Q4_K, down=Q8_0 (fallback for Q6_K); "
            "layers 20-23 gate/up=Q6_K, down=Q8_0 (unpadded)."
        ),
        nvfp4_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q6_K),
        mxfp8_types=(GGMLType.Q6_K, GGMLType.Q6_K, GGMLType.Q6_K),
        incompat_policy="llama_cpp_fallback",
    ),
    "mixed_q4_k_q6_k_padded": ExpertQuantProfile(
        name="mixed_q4_k_q6_k_padded",
        description=(
            "Literal mixed Q4_K/Q6_K via down_proj padding 1152->1280: "
            "layers 0-19 gate/up=Q4_K, down=Q6_K; layers 20-23 gate/up/down=Q6_K."
        ),
        nvfp4_types=(GGMLType.Q4_K, GGMLType.Q4_K, GGMLType.Q6_K),
        mxfp8_types=(GGMLType.Q6_K, GGMLType.Q6_K, GGMLType.Q6_K),
        incompat_policy="pad_to_block",
    ),
}

# Patterns that MUST be preserved at native precision (BF16 / F32) in the
# text-only backbone: routers, recurrent loop controllers (message gates),
# cross-pass queries, block attention residuals, short convolutions, norms,
# L2 clip scales, attention sinks, and relative logit projections.
SENSITIVE_BACKBONE_PATTERNS: Tuple[str, ...] = (
    ".mlp.gate.",
    ".controllers.",
    "middle_cycle_cross_pass",
    ".attn_res.",
    "sconv",
    "norm.",
    "_l2_clip.",
    "attn_sink_proj.",
    "rel_logits_proj.",
)

# Large eligible 2D linear matrices in the text-only backbone.
BACKBONE_PROFILES: Dict[str, Dict[str, str]] = {
    "backbone_pure_q4_k": {
        "description": "Pure Q4_K for large attention, embed and unembed; all routing, recurrent, conv, residual and norm tensors preserved at native BF16/F32.",
        "embed": "Q4_K",
        "unembed": "Q4_K",
        "attn_q_gate": "Q4_K",
        "attn_kv_r": "Q4_K",
        "attn_o": "Q4_K",
    },
    "backbone_mixed_q4_k_q6_k": {
        "description": "Mixed Q4_K/Q6_K backbone (Q4_K_M-style): unembed, wo_ud, wk_dv, wv_dv, wr_du in Q6_K; embed, wq_du, gate_attn in Q4_K; all routing, recurrent controllers, cross-pass queries, attn_res, sconv, norms and clips preserved verbatim at native BF16/F32.",
        "embed": "Q4_K",
        "unembed": "Q6_K",
        "attn_q_gate": "Q4_K",
        "attn_kv_r": "Q6_K",
        "attn_o": "Q6_K",
    },
    "backbone_q6_k_embed_unembed": {
        "description": "Higher-precision embed+unembed (both Q6_K), wo/kv/r in Q6_K, wq/gate_attn in Q4_K; all routing and recurrent controls preserved verbatim at native BF16/F32.",
        "embed": "Q6_K",
        "unembed": "Q6_K",
        "attn_q_gate": "Q4_K",
        "attn_kv_r": "Q6_K",
        "attn_o": "Q6_K",
    },
}


def is_text_backbone_tensor(name: str) -> bool:
    """True iff tensor belongs to the text-only backbone (excludes vision, MTP and routed experts)."""
    return name.startswith("model.llm.") and ".mlp.experts." not in name


def is_sensitive_backbone_tensor(name: str) -> bool:
    """True iff tensor is a sensitive routing, recurrent, conv, residual or norm control tensor."""
    return any(pat in name for pat in SENSITIVE_BACKBONE_PATTERNS)


def classify_backbone_tensor(
    name: str,
    shape: Sequence[int],
    source_dtype: str,
    backbone_profile: str = "backbone_mixed_q4_k_q6_k",
) -> Tuple[GGMLType, str]:
    """Return (target_ggml_type, category) for one text-only backbone tensor."""
    if not is_text_backbone_tensor(name):
        raise ValueError(f"not a text-only backbone tensor: {name}")
    if backbone_profile not in BACKBONE_PROFILES:
        raise ValueError(f"unknown backbone profile: {backbone_profile!r}")
    cfg = BACKBONE_PROFILES[backbone_profile]

    native_type = GGMLType.F32 if source_dtype == "F32" else GGMLType.BF16 if source_dtype == "BF16" else GGMLType.F16

    if is_sensitive_backbone_tensor(name):
        if ".mlp.gate." in name:
            return native_type, "preserved_router"
        if ".controllers." in name or "middle_cycle_cross_pass" in name:
            return native_type, "preserved_recurrent_control"
        if ".attn_res." in name:
            return native_type, "preserved_attn_res"
        if "sconv" in name:
            return native_type, "preserved_short_conv"
        return native_type, "preserved_norm_or_scale"

    if len(shape) != 2 or source_dtype != "BF16":
        return native_type, "preserved_small_or_non2d"

    ne0 = int(shape[-1])
    if name == "model.llm.embed.weight":
        req = parse_ggml_type(cfg["embed"])
        eff, _ = resolve_ggml_type(req, ne0, incompat_policy="llama_cpp_fallback")
        return eff, "quantized_embed"
    if name == "model.llm.unembed.weight":
        req = parse_ggml_type(cfg["unembed"])
        eff, _ = resolve_ggml_type(req, ne0, incompat_policy="llama_cpp_fallback")
        return eff, "quantized_unembed"
    if ".attn.wo_ud.weight" in name:
        req = parse_ggml_type(cfg["attn_o"])
        eff, _ = resolve_ggml_type(req, ne0, incompat_policy="llama_cpp_fallback")
        return eff, "quantized_attn_o"
    if any(tok in name for tok in (".attn.wk_dv.weight", ".attn.wv_dv.weight", ".attn.wr_du.weight")):
        req = parse_ggml_type(cfg["attn_kv_r"])
        eff, _ = resolve_ggml_type(req, ne0, incompat_policy="llama_cpp_fallback")
        return eff, "quantized_attn_kv_r"
    if any(tok in name for tok in (".attn.wq_du.weight", ".attn.gate_attn.weight")):
        req = parse_ggml_type(cfg["attn_q_gate"])
        eff, _ = resolve_ggml_type(req, ne0, incompat_policy="llama_cpp_fallback")
        return eff, "quantized_attn_q_gate"

    return native_type, "preserved_unclassified"


def estimate_backbone_bytes(
    tensor_entries: Mapping[str, Mapping[str, object]],
    backbone_profile: str = "backbone_mixed_q4_k_q6_k",
) -> Dict[str, object]:
    """Compute exact GGML tensor payload bytes for a text-only backbone profile."""
    by_category: Dict[str, Dict[str, int]] = {}
    total = 0
    preserved_total = 0
    quantized_total = 0
    for name, meta in sorted(tensor_entries.items()):
        if not is_text_backbone_tensor(name):
            continue
        shape = tuple(int(d) for d in meta["shape"])  # type: ignore[index]
        dtype = str(meta["dtype"])  # type: ignore[index]
        qtype, cat = classify_backbone_tensor(name, shape, dtype, backbone_profile)
        nb = ggml_nbytes(shape, qtype)
        bucket = by_category.setdefault(cat, {"tensors": 0, "bytes": 0})
        bucket["tensors"] += 1
        bucket["bytes"] += nb
        total += nb
        if cat.startswith("preserved_"):
            preserved_total += nb
        else:
            quantized_total += nb
    return {
        "backbone_profile": backbone_profile,
        "total_payload_bytes": total,
        "total_payload_gib": total / (1 << 30),
        "preserved_payload_bytes": preserved_total,
        "quantized_payload_bytes": quantized_total,
        "by_category": by_category,
    }


def rtx3050_residency_table(
    profile: ExpertQuantProfile,
    backbone_bytes: int,
    capacities: Sequence[int] = (8, 16, 32),
    vram_budget_bytes: int = 4 * (1 << 30),
) -> Dict[str, object]:
    """Compute RTX 3050 4 GiB payload footprint across resident expert capacities k."""
    per_layer = profile.per_layer_expert_bytes()
    rows = {}
    for k in capacities:
        resident_bytes = sum(k * b for b in per_layer)
        combined = backbone_bytes + resident_bytes
        rows[f"k_{k}"] = {
            "resident_experts_per_layer": k,
            "total_resident_experts": k * NUM_PHYSICAL_LAYERS,
            "resident_expert_bytes": resident_bytes,
            "resident_expert_gib": resident_bytes / (1 << 30),
            "backbone_plus_experts_bytes": combined,
            "backbone_plus_experts_gib": combined / (1 << 30),
            "headroom_before_kv_activations_gib": (vram_budget_bytes - combined) / (1 << 30),
        }
    return {
        "expert_profile": profile.name,
        "per_layer_expert_bytes": per_layer,
        "full_bank_payload_bytes": profile.bank_payload_bytes(),
        "full_bank_payload_gib": profile.bank_payload_bytes() / (1 << 30),
        "backbone_payload_bytes": backbone_bytes,
        "backbone_payload_gib": backbone_bytes / (1 << 30),
        "rtx3050_4gib_scenarios": rows,
    }
