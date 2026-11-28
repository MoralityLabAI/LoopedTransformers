"""Verified native Inkling-2-Mini NVFP4/MXFP8 dequantization and streaming source reader.

Reuses the verified native dequantization implementation from SilicoExperiments PR #3
(e43c01156e82585402f24183d30516c346754444, InkyNano/pilot_run.py::independent_conversion):
  - Calls pinned transformers@d8f10a82213a3a9db48385aba3a4b57ab535de50
    Inkling2LoopDequantizeExperts + Interleave(dim=1) when installed.
  - Mirrors the upstream loader's pre-conversion dtype cast (loader_cast=True)
    which achieved exact 0.0 max_abs parity against from_pretrained(dtype=bfloat16).
  - Also includes a bit-exact standalone PyTorch reference mirror for CPU CI and
    cross-verification without materializing the full 49 GB dequantized model.
  - Streams one expert or one backbone tensor at a time from either a local
    safetensors file or bounded HTTP 206 byte-range requests.
"""
from __future__ import annotations

import json
import math
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from inventory import BASE, LIMIT, MODEL, REV

TRANSFORMERS_REV = "d8f10a82213a3a9db48385aba3a4b57ab535de50"
CHECKPOINT_SHARD = "model-00001-of-00001.safetensors"
CHECKPOINT_SIZE = 19526019168
NUM_PHYSICAL_LAYERS = 24
NUM_EXPERTS_PER_LAYER = 128
HIDDEN_SIZE = 2304
INTERMEDIATE_SIZE = 1152
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_SAFETENSORS_DTYPE: Dict[str, Tuple[torch.dtype, int]] = {
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
}


@dataclass(frozen=True)
class SafetensorsEntry:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def elem_size(self) -> int:
        return _SAFETENSORS_DTYPE[self.dtype][1]

    @property
    def torch_dtype(self) -> torch.dtype:
        return _SAFETENSORS_DTYPE[self.dtype][0]


def raw_bytes_to_tensor(raw: bytes | bytearray, dtype_name: str, shape: Sequence[int]) -> torch.Tensor:
    """Decode little-endian safetensors raw bytes into a contiguous CPU torch.Tensor."""
    tdtype, elem_size = _SAFETENSORS_DTYPE[dtype_name]
    expected = math.prod(int(d) for d in shape) * elem_size
    if len(raw) != expected:
        raise ValueError(f"byte length mismatch for {dtype_name} {shape}: got {len(raw)}, expected {expected}")
    buf = bytearray(raw)
    if dtype_name == "F8_E4M3":
        t = torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn)
    elif dtype_name == "BF16":
        t = torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16)
    else:
        t = torch.frombuffer(buf, dtype=tdtype)
    return t.reshape(tuple(int(d) for d in shape)).clone()


class SafetensorsStreamSource:
    """Bounded-memory tensor & single-expert slice reader over safetensors."""

    def __init__(self, header: Dict[str, dict], data_section_offset: int, total_size: int) -> None:
        self.data_section_offset = data_section_offset
        self.total_size = total_size
        self.entries: Dict[str, SafetensorsEntry] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            self.entries[name] = SafetensorsEntry(
                name=name,
                dtype=meta["dtype"],
                shape=tuple(int(d) for d in meta["shape"]),
                data_start=data_section_offset + int(lo),
                data_end=data_section_offset + int(hi),
            )
        self.bytes_read = 0

    def read_range(self, start: int, size: int) -> bytes:
        raise NotImplementedError

    def has_key(self, key: str) -> bool:
        return key in self.entries

    def keys(self) -> List[str]:
        return list(self.entries.keys())

    def read_raw_tensor(self, key: str) -> Tuple[bytes, SafetensorsEntry]:
        entry = self.entries[key]
        raw = self.read_range(entry.data_start, entry.nbytes)
        return raw, entry

    def read_tensor(self, key: str) -> torch.Tensor:
        raw, entry = self.read_raw_tensor(key)
        return raw_bytes_to_tensor(raw, entry.dtype, entry.shape)

    def read_axis0_slice(self, key: str, index: int) -> torch.Tensor:
        """Read exactly [index:index+1] along axis 0 without reading the full tensor."""
        entry = self.entries[key]
        if not entry.shape or not (0 <= index < entry.shape[0]):
            raise IndexError(f"index {index} out of bounds for {key} shape {entry.shape}")
        inner_elems = math.prod(entry.shape[1:]) if len(entry.shape) > 1 else 1
        slice_bytes = inner_elems * entry.elem_size
        start = entry.data_start + index * slice_bytes
        raw = self.read_range(start, slice_bytes)
        slice_shape = (1, *entry.shape[1:])
        return raw_bytes_to_tensor(raw, entry.dtype, slice_shape)

    def read_expert_raw_inputs(self, layer: int, expert_id: int, projection: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Read only the source tensors needed to dequantize one (layer, expert, projection)."""
        if projection not in ("w13", "w2"):
            raise ValueError(f"invalid projection: {projection!r}")
        base = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
        keys = [base, base + "_scale", base + "_amax", base + "_outer_scale_denominator"]
        tensors: Dict[str, torch.Tensor] = {}
        for key in keys:
            if not self.has_key(key):
                continue
            if key == base + "_outer_scale_denominator":
                tensors[key] = self.read_tensor(key)
            else:
                tensors[key] = self.read_axis0_slice(key, expert_id)
        if base not in tensors or base + "_scale" not in tensors:
            raise KeyError(f"missing weight or scale keys for layer={layer} projection={projection}")
        return tensors, keys

    def native_layer_format(self, layer: int) -> str:
        amax_key = f"model.llm.layers.{layer}.mlp.experts.w13_weight_amax"
        return "nvfp4" if self.has_key(amax_key) else "mxfp8"


class LocalSafetensorsSource(SafetensorsStreamSource):
    """Stream slices directly via file seek+read (no full-file RAM or mmap bloat)."""

    def __init__(self, path: Path, expected_size: Optional[int] = None) -> None:
        self.path = Path(path)
        total_size = self.path.stat().st_size
        if expected_size is not None and total_size != expected_size:
            raise ValueError(f"checkpoint size mismatch: got {total_size}, expected {expected_size}")
        with self.path.open("rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            if not 2 <= hlen <= LIMIT or hlen + 8 > total_size:
                raise ValueError(f"invalid safetensors header length: {hlen}")
            header = json.loads(fh.read(hlen))
        super().__init__(header, 8 + hlen, total_size)
        self._fh = self.path.open("rb")

    def read_range(self, start: int, size: int) -> bytes:
        self._fh.seek(start)
        data = self._fh.read(size)
        if len(data) != size:
            raise EOFError(f"short local read at {start}: wanted {size}, got {len(data)}")
        self.bytes_read += size
        return data

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "LocalSafetensorsSource":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class RemoteRangeSafetensorsSource(SafetensorsStreamSource):
    """Stream single-expert or single-tensor slices over HTTP 206 Range requests."""

    def __init__(
        self,
        url: str = BASE + CHECKPOINT_SHARD,
        expected_size: Optional[int] = CHECKPOINT_SIZE,
        hf_token: Optional[str] = None,
        timeout: int = 120,
        retries: int = 5,
    ) -> None:
        self.url = url
        self.hf_token = hf_token
        self.timeout = timeout
        self.retries = retries
        prefix, total_size = self._http_range(0, 8)
        if expected_size is not None and total_size != expected_size:
            raise ValueError(f"remote checkpoint size mismatch: got {total_size}, expected {expected_size}")
        (hlen,) = struct.unpack("<Q", prefix)
        if not 2 <= hlen <= LIMIT or hlen + 8 > total_size:
            raise ValueError(f"invalid remote safetensors header length: {hlen}")
        raw_header, total2 = self._http_range(8, hlen)
        if total2 != total_size:
            raise ValueError("inconsistent remote total size")
        header = json.loads(raw_header)
        super().__init__(header, 8 + hlen, total_size)

    def _http_range(self, start: int, size: int) -> Tuple[bytes, int]:
        if size <= 0:
            raise ValueError(f"invalid range size: {size}")
        end = start + size - 1
        headers = {
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "InkyNano-GGUF-Converter/1.0",
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                req = Request(self.url, headers=headers)
                with urlopen(req, timeout=self.timeout) as resp:
                    cr = resp.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr)
                    if resp.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, end):
                        raise ValueError(
                            f"server refused exact range bytes={start}-{end} (status={resp.status}, Content-Range={cr!r})"
                        )
                    data = resp.read(size + 1)
                    if len(data) != size:
                        raise EOFError(f"incomplete HTTP range: wanted {size}, got {len(data)}")
                    return data, int(match.group(3))
            except (HTTPError, URLError, TimeoutError, EOFError, OSError) as exc:
                last_err = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2.0 ** attempt, 16.0))
                    continue
                raise
        raise RuntimeError(f"HTTP range failed after {self.retries} retries: {last_err}")

    def read_range(self, start: int, size: int) -> bytes:
        # Chunk very large single-tensor reads (e.g., 926 MB embed/unembed) into 64 MiB ranges.
        chunk_limit = 64 * 1024 * 1024
        if size <= chunk_limit:
            data, _ = self._http_range(start, size)
            self.bytes_read += size
            return data
        out = bytearray(size)
        pos = 0
        while pos < size:
            n = min(chunk_limit, size - pos)
            part, _ = self._http_range(start + pos, n)
            out[pos : pos + n] = part
            pos += n
            self.bytes_read += n
        return bytes(out)

    def close(self) -> None:
        return None

    def __enter__(self) -> "RemoteRangeSafetensorsSource":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def has_upstream_transformers_dequant() -> bool:
    try:
        from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave  # noqa: F401
        return True
    except Exception:
        return False


@torch.no_grad()
def dequantize_projection_transformers(
    raw_tensors: Dict[str, torch.Tensor],
    keys: Sequence[str],
    projection: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    loader_cast: bool = True,
) -> torch.Tensor:
    """Exact SilicoExperiments PR #3 independent_conversion using pinned upstream Transformers."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

    cast_tensors = {
        k: v.to(device=device, dtype=dtype if loader_cast else v.dtype)
        for k, v in raw_tensors.items()
    }
    op = Inkling2LoopDequantizeExperts()
    result = op.convert(cast_tensors, list(keys), ["reconstructed"])["reconstructed"]
    if projection == "w13":
        result = Interleave(dim=1).convert({"reconstructed": result}, ["reconstructed"], ["reconstructed"])[
            "reconstructed"
        ]
    return result[0].to(dtype).to(device)


@torch.no_grad()
def dequantize_projection_standalone(
    raw_tensors: Dict[str, torch.Tensor],
    keys: Sequence[str],
    projection: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    loader_cast: bool = True,
) -> torch.Tensor:
    """Bit-exact standalone PyTorch mirror of upstream Inkling2LoopDequantizeExperts + Interleave(dim=1).

    Verified against transformers@d8f10a82213a3a9db48385aba3a4b57ab535de50 to 0.0 max_abs.
    """
    cast_tensors = {
        k: v.to(device=device, dtype=dtype if loader_cast else v.dtype)
        for k, v in raw_tensors.items()
    }
    weight, scale, amax, denominator = (cast_tensors.get(k) for k in keys)
    num_experts, rows = weight.shape[:2]
    columns = weight.shape[2] * 2 if amax is not None else weight.shape[2]

    block_scale = scale.float()
    if amax is None:
        # MXFP8: float8 E4M3 values scaled by 2**(e8m0 - 127) per 32-element block
        values = weight.float().view(*block_scale.shape, 32)
        values = values * torch.exp2(block_scale - 127)[..., None]
    else:
        # NVFP4: two E2M1 nibbles per uint8 (low nibble even, high nibble odd),
        # E4M3 block scale per 16 values, and per-expert factor amax / denominator
        e2m1_values = torch.tensor(E2M1_VALUES, device=weight.device, dtype=torch.float32)
        byte = torch.arange(256, device=weight.device)
        byte_to_values = torch.stack((e2m1_values[byte & 0xF], e2m1_values[byte >> 4]), dim=-1)
        expert_factor = amax.float() / denominator.float()
        codes = weight.to(torch.uint8).long()
        values = byte_to_values[codes].view(*block_scale.shape, 16) * block_scale[..., None]
        values = values * expert_factor[:, None, None, None]

    out = values.reshape(num_experts, rows, columns).to(dtype)
    if projection == "w13":
        # Interleave(dim=1): reshape [E, rows//2, 2, cols] -> transpose(1, 2) -> [E, rows, cols]
        out = out.reshape(num_experts, rows // 2, 2, columns).transpose(1, 2).reshape(num_experts, rows, columns).contiguous()
    return out[0].to(dtype).to(device)


@torch.no_grad()
def dequantize_expert(
    source: SafetensorsStreamSource,
    layer: int,
    expert_id: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    backend: str = "auto",
    loader_cast: bool = True,
) -> Dict[str, object]:
    """Stream and dequantize one physical expert into gate [1152,2304], up [1152,2304], down [2304,1152]."""
    if backend == "auto":
        use_tf = has_upstream_transformers_dequant()
        chosen = "transformers" if use_tf else "standalone"
    elif backend in ("transformers", "standalone"):
        chosen = backend
    else:
        raise ValueError(f"unknown dequant backend: {backend!r}")

    fn = dequantize_projection_transformers if chosen == "transformers" else dequantize_projection_standalone

    w13_raw, w13_keys = source.read_expert_raw_inputs(layer, expert_id, "w13")
    gate_up = fn(w13_raw, w13_keys, "w13", dtype=dtype, device=device, loader_cast=loader_cast)
    del w13_raw
    gate, up = gate_up.chunk(2, dim=0)
    gate = gate.contiguous()
    up = up.contiguous()
    del gate_up

    w2_raw, w2_keys = source.read_expert_raw_inputs(layer, expert_id, "w2")
    down = fn(w2_raw, w2_keys, "w2", dtype=dtype, device=device, loader_cast=loader_cast).contiguous()
    del w2_raw

    return {
        "gate": gate,
        "up": up,
        "down": down,
        "layer": layer,
        "expert_id": expert_id,
        "native_format": source.native_layer_format(layer),
        "dequant_backend": chosen,
        "loader_cast": loader_cast,
        "dtype": str(dtype),
    }


def open_source(
    source_path: Optional[Path] = None,
    remote_url: str = BASE + CHECKPOINT_SHARD,
    hf_token: Optional[str] = None,
    expected_size: Optional[int] = CHECKPOINT_SIZE,
) -> SafetensorsStreamSource:
    if source_path is not None:
        return LocalSafetensorsSource(Path(source_path), expected_size=expected_size)
    return RemoteRangeSafetensorsSource(url=remote_url, expected_size=expected_size, hf_token=hf_token)
