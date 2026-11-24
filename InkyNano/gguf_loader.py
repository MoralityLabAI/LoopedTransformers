"""Selective per-expert byte-offset loader & LDT-controlled residency manager.

Designed for RTX 3050 (4 GiB) + CPU-backed / disk-backed GGUF expert shards:
  - Reads ONLY manifest.json on startup; never loads full shards or the full bank.
  - Fetches any single physical expert (layer, expert_id) in one contiguous seek+read
    using exact manifest byte offsets and verifies per-tensor SHA-256 on demand.
  - Executes exact native Inkling-2-Mini soft-capped SwiGLU (gate_cap=4.0, up_cap=25.0,
    raw-gate sigmoid, FP32 accumulation).
  - Integrates directly with InkyNano/expert_lattice.py (ExpertLattice) so one physical
    (layer, expert) slot is shared across loop passes 0-2 under strict byte accounting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from expert_lattice import ExpertLattice
from gguf_backend import GGMLCBackend, SyntheticRefBackend, discover_ggml_lib_dir
from gguf_format import GGMLType, GGUFBundleReader, parse_ggml_type
from native_expert import activation as native_activation


@dataclass(frozen=True)
class RawExpertPayload:
    layer: int
    expert: int
    gate_bytes: bytes
    up_bytes: bytes
    down_bytes: bytes
    gate_type: GGMLType
    up_type: GGMLType
    down_type: GGMLType
    gate_shape: Tuple[int, int]
    up_shape: Tuple[int, int]
    down_shape: Tuple[int, int]
    down_logical_shape: Tuple[int, int]
    span_bytes: int

    @property
    def payload_bytes(self) -> int:
        return len(self.gate_bytes) + len(self.up_bytes) + len(self.down_bytes)


class SelectiveExpertStore:
    """Byte-offset selective reader for the 24 GGUF expert layer shards."""

    def __init__(self, manifest_path: Path, root_dir: Optional[Path] = None, backend=None) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir is not None else self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf8"))
        if self.manifest.get("llama_cpp_dropin_compatible") is not False:
            raise ValueError("manifest must explicitly record llama_cpp_dropin_compatible=False")
        if backend is not None:
            self.backend = backend
        else:
            lib = discover_ggml_lib_dir()
            self.backend = GGMLCBackend(lib_dir=lib) if lib is not None else SyntheticRefBackend()

        act = self.manifest.get("activation", {})
        self.gate_softcap = float(act.get("gate_softcap", 4.0))
        self.up_softcap = float(act.get("up_softcap", 25.0))
        self.disk_bytes_read = 0
        self.expert_reads = 0

    @property
    def num_layers(self) -> int:
        return int(self.manifest["num_layers"])

    def layer_expert_bytes(self) -> List[int]:
        """Exact per-layer expert payload bytes for stage1.py --expert-bytes and ExpertLattice."""
        return [int(b) for b in self.manifest["per_layer_expert_payload_bytes"]]

    def expert_entry(self, layer: int, expert_id: int) -> Dict[str, object]:
        return self.manifest["experts"][str(layer)][str(expert_id)]

    def read_expert_raw(self, layer: int, expert_id: int, verify_sha256: bool = True) -> RawExpertPayload:
        """Fetch one physical expert in a SINGLE contiguous disk read via manifest byte offsets."""
        ex = self.expert_entry(layer, expert_id)
        shard_path = self.root_dir / str(ex["shard"])
        span_off = int(ex["span_file_offset"])
        span_len = int(ex["span_bytes"])

        with shard_path.open("rb") as fh:
            fh.seek(span_off)
            blob = fh.read(span_len)
        if len(blob) != span_len:
            raise EOFError(f"short read for L{layer}/E{expert_id}: wanted {span_len}, got {len(blob)}")
        self.disk_bytes_read += span_len
        self.expert_reads += 1

        tensors = ex["tensors"]
        slices: Dict[str, bytes] = {}
        for proj in ("gate", "up", "down"):
            t = tensors[proj]
            rel = int(t["file_offset"]) - span_off
            nb = int(t["nbytes"])
            raw = blob[rel : rel + nb]
            if len(raw) != nb:
                raise EOFError(f"truncated slice for {t['name']}")
            if verify_sha256:
                got = hashlib.sha256(raw).hexdigest()
                if got != t["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {t['name']}: expected {t['sha256']}, got {got}")
            slices[proj] = raw

        tg, tu, td = tensors["gate"], tensors["up"], tensors["down"]
        return RawExpertPayload(
            layer=layer,
            expert=expert_id,
            gate_bytes=slices["gate"],
            up_bytes=slices["up"],
            down_bytes=slices["down"],
            gate_type=parse_ggml_type(tg["ggml_type"]),
            up_type=parse_ggml_type(tu["ggml_type"]),
            down_type=parse_ggml_type(td["ggml_type"]),
            gate_shape=(int(tg["shape"][0]), int(tg["shape"][1])),
            up_shape=(int(tu["shape"][0]), int(tu["shape"][1])),
            down_shape=(int(td["shape"][0]), int(td["shape"][1])),
            down_logical_shape=(int(td["logical_shape"][0]), int(td["logical_shape"][1])),
            span_bytes=span_len,
        )

    def dequantize_expert(self, payload: RawExpertPayload, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dequantize a single resident expert payload into float32 (gate, up, down) tiles."""
        gate = self.backend.dequantize(payload.gate_bytes, payload.gate_shape, payload.gate_type)
        up = self.backend.dequantize(payload.up_bytes, payload.up_shape, payload.up_type)
        down = self.backend.dequantize(payload.down_bytes, payload.down_shape, payload.down_type)
        down = down[:, : payload.down_logical_shape[1]]
        return (
            torch.from_numpy(np.ascontiguousarray(gate)).to(device),
            torch.from_numpy(np.ascontiguousarray(up)).to(device),
            torch.from_numpy(np.ascontiguousarray(down)).to(device),
        )

    @torch.no_grad()
    def forward_expert(self, x: torch.Tensor, payload: RawExpertPayload) -> torch.Tensor:
        """Apply one selectively loaded expert with exact native soft-capped SwiGLU semantics."""
        gate, up, down = self.dequantize_expert(payload, device=str(x.device))
        g = F.linear(x.float(), gate)
        u = F.linear(x.float(), up)
        act = native_activation(g, u, gate_cap=self.gate_softcap, up_cap=self.up_softcap)
        out = F.linear(act.float(), down)
        return out.to(x.dtype)


class LDTResidencyManager:
    """LDT-driven bounded resident expert cache shared across loop passes 0-2.

    Uses InkyNano/expert_lattice.py (ExpertLattice.admit) to decide deterministic
    cache transitions per physical layer while tracking exact resident bytes,
    admissions, evictions, and disk/staging transfer bytes.
    """

    def __init__(
        self,
        store: SelectiveExpertStore,
        capacity_per_layer: int,
        max_resident_bytes: Optional[int] = None,
        verify_sha256: bool = True,
        experts_per_layer: int = 128,
    ) -> None:
        self.store = store
        self.capacity = int(capacity_per_layer)
        self.max_resident_bytes = max_resident_bytes
        self.verify_sha256 = verify_sha256
        self.lattice = ExpertLattice(experts=experts_per_layer)
        self._resident: Dict[int, Dict[int, RawExpertPayload]] = {}
        self.total_admissions = 0
        self.total_evictions = 0
        self.total_transfer_bytes = 0

    @property
    def resident_bytes(self) -> int:
        return sum(p.payload_bytes for layer_map in self._resident.values() for p in layer_map.values())

    def resident_ids(self, layer: int) -> FrozenSet[int]:
        return frozenset(self._resident.get(layer, ()))

    def admit_layer(self, layer: int, scores: np.ndarray, swap_budget: int) -> Dict[str, object]:
        """Apply an LDT controller score vector via ExpertLattice and load only newly admitted experts."""
        prior_map = self._resident.setdefault(layer, {})
        prior_set = frozenset(prior_map.keys()) if prior_map else None
        chosen = self.lattice.admit(scores, prior_set, capacity=self.capacity, budget=swap_budget)

        to_evict = sorted((prior_set or frozenset()) - chosen)
        to_admit = sorted(chosen - (prior_set or frozenset()))

        for eid in to_evict:
            del prior_map[eid]
            self.total_evictions += 1

        admitted_bytes = 0
        for eid in to_admit:
            payload = self.store.read_expert_raw(layer, eid, verify_sha256=self.verify_sha256)
            if self.max_resident_bytes is not None and self.resident_bytes + payload.payload_bytes > self.max_resident_bytes:
                raise MemoryError(
                    f"admitting L{layer}/E{eid} ({payload.payload_bytes} B) exceeds max_resident_bytes={self.max_resident_bytes}"
                )
            prior_map[eid] = payload
            admitted_bytes += payload.payload_bytes
            self.total_admissions += 1
            self.total_transfer_bytes += payload.payload_bytes

        return {
            "layer": layer,
            "admitted": to_admit,
            "evicted": to_evict,
            "admitted_bytes": admitted_bytes,
            "resident_count": len(prior_map),
            "total_resident_bytes": self.resident_bytes,
        }

    def get_resident(self, layer: int, expert_id: int) -> RawExpertPayload:
        layer_map = self._resident.get(layer, {})
        if expert_id not in layer_map:
            raise KeyError(f"expert ({layer}, {expert_id}) is not resident in LDT cache")
        return layer_map[expert_id]

    @torch.no_grad()
    def forward_resident_mixture(
        self,
        layer: int,
        x: torch.Tensor,
        selected_experts: Sequence[int],
        weights: Sequence[float],
    ) -> torch.Tensor:
        """Compute weighted FP32-accumulated output over resident selected experts (shared across loop passes)."""
        accum = torch.zeros_like(x, dtype=torch.float32)
        for eid, w in zip(selected_experts, weights):
            payload = self.get_resident(layer, int(eid))
            accum = accum + self.store.forward_expert(x, payload).float() * float(w)
        return accum.to(x.dtype)


class BackboneBundleReader:
    """Selective reader for the separately quantized text-only backbone GGUF bundle."""

    def __init__(self, backbone_path: Path, backend=None) -> None:
        self.reader = GGUFBundleReader(backbone_path)
        self.reader.verify_layout()
        if self.reader.metadata.get("inkynano.bundle_kind") != "text_only_backbone":
            raise ValueError(f"unexpected bundle_kind: {self.reader.metadata.get('inkynano.bundle_kind')}")
        if backend is not None:
            self.backend = backend
        else:
            lib = discover_ggml_lib_dir()
            self.backend = GGMLCBackend(lib_dir=lib) if lib is not None else SyntheticRefBackend()

    def load_tensor(self, name: str, verify_sha256: Optional[str] = None) -> np.ndarray:
        lay = self.reader.tensors[name]
        raw = self.reader.read_tensor_bytes(name, verify_sha256=verify_sha256)
        return self.backend.dequantize(raw, lay.pt_shape, lay.ggml_type)

    def verify_sensitive_precision(self) -> Dict[str, int]:
        """Assert every router, recurrent controller, attn_res, sconv and norm tensor is BF16/F32."""
        from gguf_profiles import is_sensitive_backbone_tensor

        preserved = 0
        quantized = 0
        for name, lay in self.reader.tensors.items():
            if is_sensitive_backbone_tensor(name):
                if lay.ggml_type not in (GGMLType.BF16, GGMLType.F32, GGMLType.F16):
                    raise ValueError(f"sensitive tensor {name} was unexpectedly quantized to {lay.ggml_type.name}")
                preserved += 1
            else:
                quantized += 1
        return {"preserved_sensitive_tensors": preserved, "other_tensors": quantized}
