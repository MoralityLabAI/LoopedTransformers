"""Streaming GGUF v3 writer and selective random-access reader.

Writes standard GGUF v3 binary files in two passes so arbitrarily large shards
can be streamed one quantized tensor at a time with O(1) memory overhead.
Distinguishes standalone tensor bundles (general.type = "tensors") from full
llama.cpp-compatible inference models.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_MAX_NAME = 64


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30
    MXFP4 = 39


# (blck_size, type_size) from upstream ggml.c / ggml-quants.c
GGML_TYPE_TRAITS: Dict[GGMLType, Tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 36),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
    GGMLType.MXFP4: (32, 17),
}

GGML_TYPE_BY_NAME: Dict[str, GGMLType] = {t.name: t for t in GGMLType}


def parse_ggml_type(value: GGMLType | str | int) -> GGMLType:
    if isinstance(value, GGMLType):
        return value
    if isinstance(value, int):
        return GGMLType(value)
    key = str(value).strip().upper()
    if key in GGML_TYPE_BY_NAME:
        return GGML_TYPE_BY_NAME[key]
    raise ValueError(f"unsupported GGML type: {value!r}")


def ggml_blck_size(qtype: GGMLType) -> int:
    return GGML_TYPE_TRAITS[parse_ggml_type(qtype)][0]


def ggml_type_size(qtype: GGMLType) -> int:
    return GGML_TYPE_TRAITS[parse_ggml_type(qtype)][1]


def ggml_row_size(qtype: GGMLType, ne0: int) -> int:
    blck, tsize = GGML_TYPE_TRAITS[parse_ggml_type(qtype)]
    if ne0 <= 0 or ne0 % blck != 0:
        raise ValueError(f"ne0={ne0} not divisible by blck_size={blck} for {qtype.name}")
    return (ne0 // blck) * tsize


def ggml_nbytes(pt_shape: Sequence[int], qtype: GGMLType) -> int:
    """Compute exact GGML byte size for a row-major PyTorch/NumPy tensor shape."""
    if not pt_shape:
        raise ValueError("scalar (0-D) tensors are not supported in GGUF")
    if any(int(d) <= 0 for d in pt_shape):
        raise ValueError(f"invalid tensor shape: {pt_shape}")
    ne0 = int(pt_shape[-1])
    nrows = math.prod(int(d) for d in pt_shape[:-1]) if len(pt_shape) > 1 else 1
    return nrows * ggml_row_size(qtype, ne0)


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FMT: Dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.BOOL: "<?",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}


def align_up(offset: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of two, got {alignment}")
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_value(vtype: GGUFValueType, value: Any) -> bytes:
    if vtype == GGUFValueType.STRING:
        return _pack_string(str(value))
    if vtype == GGUFValueType.ARRAY:
        elem_type, items = value
        out = [struct.pack("<IQ", int(elem_type), len(items))]
        for item in items:
            out.append(_pack_value(elem_type, item))
        return b"".join(out)
    if vtype == GGUFValueType.BOOL:
        return struct.pack("<B", 1 if value else 0)
    return struct.pack(_SCALAR_FMT[vtype], value)


def _infer_kv_type(value: Any) -> Tuple[GGUFValueType, Any]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], GGUFValueType):
        return value
    if isinstance(value, bool):
        return GGUFValueType.BOOL, value
    if isinstance(value, int):
        if 0 <= value < 2**32:
            return GGUFValueType.UINT32, value
        if -(2**31) <= value < 2**31:
            return GGUFValueType.INT32, value
        if 0 <= value < 2**64:
            return GGUFValueType.UINT64, value
        return GGUFValueType.INT64, value
    if isinstance(value, float):
        return GGUFValueType.FLOAT32, value
    if isinstance(value, str):
        return GGUFValueType.STRING, value
    if isinstance(value, (list, tuple)):
        if not value:
            return GGUFValueType.ARRAY, (GGUFValueType.STRING, [])
        first_t, _ = _infer_kv_type(value[0])
        if first_t == GGUFValueType.ARRAY:
            raise ValueError("nested GGUF arrays are not supported")
        return GGUFValueType.ARRAY, (first_t, list(value))
    raise TypeError(f"unsupported GGUF KV value type: {type(value).__name__}")


@dataclass(frozen=True)
class TensorSpec:
    """Descriptor for one tensor written to a GGUF bundle."""

    name: str
    pt_shape: Tuple[int, ...]
    ggml_type: GGMLType
    logical_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        encoded = self.name.encode("utf-8")
        if not encoded or len(encoded) >= GGML_MAX_NAME:
            raise ValueError(
                f"GGUF tensor name must be 1..{GGML_MAX_NAME - 1} UTF-8 bytes, got {len(encoded)}: {self.name!r}"
            )
        # Validate shape divisibility early before writing anything.
        ggml_nbytes(self.pt_shape, self.ggml_type)

    @property
    def ggml_ne(self) -> Tuple[int, ...]:
        """GGML stores dimensions fastest-varying first (reversed from PyTorch)."""
        return tuple(int(d) for d in reversed(self.pt_shape))

    @property
    def nbytes(self) -> int:
        return ggml_nbytes(self.pt_shape, self.ggml_type)


@dataclass(frozen=True)
class TensorLayout:
    name: str
    pt_shape: Tuple[int, ...]
    logical_shape: Tuple[int, ...]
    ggml_ne: Tuple[int, ...]
    ggml_type: GGMLType
    data_offset: int
    file_offset: int
    nbytes: int
    padded_nbytes: int


class StreamingGGUFWriter:
    """Two-pass O(1)-RAM GGUF v3 file writer with incremental SHA-256 hashing."""

    def __init__(
        self,
        path: Path,
        metadata: Dict[str, Any],
        tensors: Sequence[TensorSpec],
        alignment: int = GGUF_DEFAULT_ALIGNMENT,
    ) -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.alignment = int(alignment)
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(f"invalid alignment: {alignment}")
        if not tensors:
            raise ValueError("GGUF file must contain at least one tensor")

        seen = set()
        for spec in tensors:
            if spec.name in seen:
                raise ValueError(f"duplicate tensor name: {spec.name}")
            seen.add(spec.name)
        self.specs = list(tensors)

        kv = dict(metadata)
        kv["general.alignment"] = (GGUFValueType.UINT32, self.alignment)
        self.metadata = kv

        header_bytes, self.data_section_offset, self.layouts = self._build_header_and_layout()
        self._layout_by_name = {lay.name: lay for lay in self.layouts}
        self._next_idx = 0
        self._file_sha = hashlib.sha256()
        self._tensor_sha: Dict[str, str] = {}
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.tmp_path.open("wb")
        try:
            self._write_raw(header_bytes)
        except Exception:
            self._fh.close()
            self.tmp_path.unlink(missing_ok=True)
            raise

    def _build_header_and_layout(self) -> Tuple[bytes, int, List[TensorLayout]]:
        kv_chunks: List[bytes] = []
        for key, raw_val in self.metadata.items():
            vtype, val = _infer_kv_type(raw_val)
            kv_chunks.append(_pack_string(key) + struct.pack("<I", int(vtype)) + _pack_value(vtype, val))

        tensor_chunks: List[bytes] = []
        rel_offsets: List[Tuple[int, int]] = []
        cursor = 0
        for spec in self.specs:
            offset = align_up(cursor, self.alignment)
            nbytes = spec.nbytes
            padded = align_up(nbytes, self.alignment)
            rel_offsets.append((offset, padded))
            cursor = offset + padded

            ne = spec.ggml_ne
            tensor_chunks.append(
                _pack_string(spec.name)
                + struct.pack("<I", len(ne))
                + struct.pack(f"<{len(ne)}Q", *ne)
                + struct.pack("<IQ", int(spec.ggml_type), offset)
            )

        prefix = (
            GGUF_MAGIC
            + struct.pack("<IQQ", GGUF_VERSION, len(self.specs), len(self.metadata))
            + b"".join(kv_chunks)
            + b"".join(tensor_chunks)
        )
        data_section_offset = align_up(len(prefix), self.alignment)
        header_pad = b"\x00" * (data_section_offset - len(prefix))

        layouts: List[TensorLayout] = []
        for spec, (rel_off, padded) in zip(self.specs, rel_offsets):
            layouts.append(
                TensorLayout(
                    name=spec.name,
                    pt_shape=tuple(int(d) for d in spec.pt_shape),
                    logical_shape=tuple(int(d) for d in (spec.logical_shape or spec.pt_shape)),
                    ggml_ne=spec.ggml_ne,
                    ggml_type=spec.ggml_type,
                    data_offset=rel_off,
                    file_offset=data_section_offset + rel_off,
                    nbytes=spec.nbytes,
                    padded_nbytes=padded,
                )
            )
        return prefix + header_pad, data_section_offset, layouts

    def _write_raw(self, data: bytes | memoryview) -> None:
        self._fh.write(data)
        self._file_sha.update(data)

    def write_tensor(self, name: str, payload: bytes | bytearray | memoryview) -> Dict[str, Any]:
        """Stream the next tensor's raw GGML payload to disk in strict declared order."""
        if self._closed:
            raise RuntimeError("writer already finalized")
        if self._next_idx >= len(self.layouts):
            raise RuntimeError("all declared tensors have already been written")
        layout = self.layouts[self._next_idx]
        if name != layout.name:
            raise ValueError(f"expected tensor {layout.name!r} at index {self._next_idx}, got {name!r}")
        if len(payload) != layout.nbytes:
            raise ValueError(f"tensor {name!r}: expected {layout.nbytes} bytes, got {len(payload)}")

        cur_pos = self._fh.tell()
        if cur_pos != layout.file_offset:
            raise RuntimeError(f"internal offset mismatch for {name!r}: at {cur_pos}, expected {layout.file_offset}")

        t_sha = hashlib.sha256(payload).hexdigest()
        self._write_raw(payload)
        pad = layout.padded_nbytes - layout.nbytes
        if pad:
            self._write_raw(b"\x00" * pad)

        self._tensor_sha[name] = t_sha
        self._next_idx += 1
        return {
            "name": layout.name,
            "shape": list(layout.pt_shape),
            "logical_shape": list(layout.logical_shape),
            "ggml_ne": list(layout.ggml_ne),
            "ggml_type": layout.ggml_type.name,
            "ggml_type_id": int(layout.ggml_type),
            "data_offset": layout.data_offset,
            "file_offset": layout.file_offset,
            "nbytes": layout.nbytes,
            "padded_nbytes": layout.padded_nbytes,
            "sha256": t_sha,
        }

    def finalize(self) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("writer already finalized")
        try:
            if self._next_idx != len(self.layouts):
                raise RuntimeError(
                    f"incomplete GGUF write: wrote {self._next_idx}/{len(self.layouts)} tensors"
                )
            self._fh.flush()
            os.fsync(self._fh.fileno())
            total_bytes = self._fh.tell()
        finally:
            self._fh.close()
            self._closed = True

        self.tmp_path.replace(self.path)
        return {
            "path": str(self.path),
            "bytes": total_bytes,
            "sha256": self._file_sha.hexdigest(),
            "alignment": self.alignment,
            "data_section_offset": self.data_section_offset,
            "tensor_count": len(self.layouts),
            "tensors": [
                {
                    "name": lay.name,
                    "shape": list(lay.pt_shape),
                    "logical_shape": list(lay.logical_shape),
                    "ggml_ne": list(lay.ggml_ne),
                    "ggml_type": lay.ggml_type.name,
                    "ggml_type_id": int(lay.ggml_type),
                    "data_offset": lay.data_offset,
                    "file_offset": lay.file_offset,
                    "nbytes": lay.nbytes,
                    "padded_nbytes": lay.padded_nbytes,
                    "sha256": self._tensor_sha[lay.name],
                }
                for lay in self.layouts
            ],
        }

    def abort(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True
        self.tmp_path.unlink(missing_ok=True)


def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise EOFError(f"unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def _read_string(fh) -> str:
    (length,) = struct.unpack("<Q", _read_exact(fh, 8))
    if length > 16 * 1024 * 1024:
        raise ValueError(f"GGUF string length exceeds safety bound: {length}")
    return _read_exact(fh, length).decode("utf-8")


def _read_value(fh, vtype: GGUFValueType) -> Any:
    if vtype == GGUFValueType.STRING:
        return _read_string(fh)
    if vtype == GGUFValueType.ARRAY:
        elem_id, count = struct.unpack("<IQ", _read_exact(fh, 12))
        elem_type = GGUFValueType(elem_id)
        return [_read_value(fh, elem_type) for _ in range(count)]
    fmt = _SCALAR_FMT[vtype]
    val = struct.unpack(fmt, _read_exact(fh, struct.calcsize(fmt)))[0]
    return bool(val) if vtype == GGUFValueType.BOOL else val


class GGUFBundleReader:
    """Zero-copy header parser & selective random-access reader for GGUF v3 files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.file_size = self.path.stat().st_size
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorLayout] = {}
        self.tensor_order: List[str] = []
        self.version: int = 0
        self.alignment: int = GGUF_DEFAULT_ALIGNMENT
        self.data_section_offset: int = 0
        self._parse_header()

    def _parse_header(self) -> None:
        with self.path.open("rb") as fh:
            magic = _read_exact(fh, 4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"invalid GGUF magic {magic!r} in {self.path}")
            version, tensor_count, kv_count = struct.unpack("<IQQ", _read_exact(fh, 20))
            if version != GGUF_VERSION:
                raise ValueError(f"unsupported GGUF version {version} (expected {GGUF_VERSION})")
            self.version = version

            for _ in range(kv_count):
                key = _read_string(fh)
                (vtype_id,) = struct.unpack("<I", _read_exact(fh, 4))
                self.metadata[key] = _read_value(fh, GGUFValueType(vtype_id))

            self.alignment = int(self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))

            raw_infos: List[Tuple[str, Tuple[int, ...], GGMLType, int]] = []
            for _ in range(tensor_count):
                name = _read_string(fh)
                (ndims,) = struct.unpack("<I", _read_exact(fh, 4))
                ne = struct.unpack(f"<{ndims}Q", _read_exact(fh, 8 * ndims))
                qtype_id, rel_offset = struct.unpack("<IQ", _read_exact(fh, 12))
                raw_infos.append((name, tuple(int(d) for d in ne), GGMLType(qtype_id), int(rel_offset)))

            self.data_section_offset = align_up(fh.tell(), self.alignment)

            for name, ne, qtype, rel_offset in raw_infos:
                pt_shape = tuple(reversed(ne))
                nbytes = ggml_nbytes(pt_shape, qtype)
                padded = align_up(nbytes, self.alignment)
                file_offset = self.data_section_offset + rel_offset
                if file_offset + nbytes > self.file_size:
                    raise ValueError(
                        f"tensor {name!r} extends past EOF ({file_offset + nbytes} > {self.file_size})"
                    )
                self.tensors[name] = TensorLayout(
                    name=name,
                    pt_shape=pt_shape,
                    logical_shape=pt_shape,
                    ggml_ne=ne,
                    ggml_type=qtype,
                    data_offset=rel_offset,
                    file_offset=file_offset,
                    nbytes=nbytes,
                    padded_nbytes=padded,
                )
                self.tensor_order.append(name)

    def read_tensor_bytes(self, name: str, verify_sha256: Optional[str] = None) -> bytes:
        lay = self.tensors[name]
        with self.path.open("rb") as fh:
            fh.seek(lay.file_offset)
            data = _read_exact(fh, lay.nbytes)
        if verify_sha256 is not None:
            actual = hashlib.sha256(data).hexdigest()
            if actual != verify_sha256:
                raise ValueError(f"SHA-256 mismatch for {name!r}: expected {verify_sha256}, got {actual}")
        return data

    def read_span(self, file_offset: int, nbytes: int) -> bytes:
        if file_offset < self.data_section_offset or file_offset + nbytes > self.file_size:
            raise ValueError(f"invalid span [{file_offset}, {file_offset + nbytes}) for file size {self.file_size}")
        with self.path.open("rb") as fh:
            fh.seek(file_offset)
            return _read_exact(fh, nbytes)

    def verify_layout(self) -> Dict[str, Any]:
        """Verify monotonic non-overlapping offsets and alignment without reading full payloads."""
        prev_end = self.data_section_offset
        for name in self.tensor_order:
            lay = self.tensors[name]
            if lay.data_offset % self.alignment != 0 or lay.file_offset % self.alignment != 0:
                raise ValueError(f"unaligned tensor offset for {name!r}")
            if lay.file_offset < prev_end:
                raise ValueError(f"overlapping tensor layout at {name!r}")
            prev_end = lay.file_offset + lay.nbytes
        return {
            "path": str(self.path),
            "version": self.version,
            "alignment": self.alignment,
            "tensor_count": len(self.tensors),
            "file_size": self.file_size,
            "data_section_offset": self.data_section_offset,
            "bundle_kind": self.metadata.get("inkynano.bundle_kind"),
            "general_type": self.metadata.get("general.type"),
            "llama_cpp_dropin_compatible": self.metadata.get("inkynano.llama_cpp_dropin_compatible"),
        }
