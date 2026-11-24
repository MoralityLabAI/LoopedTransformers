"""Actual GGML C quantization/dequantization backend plus build & fallback helpers.

Primary production backend loads upstream ggml-org/llama.cpp shared libraries
(libggml-base + libggml-cpu) via ctypes and calls ggml_quantize_chunk and
ggml_get_type_traits(t)->to_float directly. ctypes releases the Python GIL, so
row-chunked ThreadPoolExecutor achieves multi-core bit-exact C quantization on
Colab CPU without GPU credits.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gguf_format import GGMLType, ggml_blck_size, ggml_nbytes, ggml_row_size, ggml_type_size, parse_ggml_type

PINNED_LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
PINNED_LLAMA_CPP_COMMIT = "ae6a7ad2c5754dd8db93d04c84fda70435864f6e"

# Canonical llama.cpp src/llama-quant.cpp fallback_type_for() mapping when
# ne[0] % 256 != 0 (e.g. Inkling-2-Mini down_proj ne[0] = 1152 = 36 * 32).
LLAMA_CPP_FALLBACK: Dict[GGMLType, GGMLType] = {
    GGMLType.Q2_K: GGMLType.Q4_0,
    GGMLType.Q3_K: GGMLType.Q4_0,
    GGMLType.Q4_K: GGMLType.Q5_0,
    GGMLType.Q5_K: GGMLType.Q5_1,
    GGMLType.Q6_K: GGMLType.Q8_0,
}


def resolve_ggml_type(
    requested: GGMLType | str,
    ne0: int,
    incompat_policy: str = "llama_cpp_fallback",
) -> Tuple[GGMLType, int]:
    """Resolve GGML quantization type and target ne0 (with optional zero-padding).

    Returns (effective_ggml_type, stored_ne0).
    Policies:
      - "llama_cpp_fallback": canonical llama.cpp fallback_type_for() when ne0 % blck != 0
      - "q4_0_fallback": demote incompatible 4-bit K-quants to Q4_0 (4.5 bpw), Q6_K to Q8_0
      - "pad_to_block": keep requested K-quant by zero-padding ne0 up to a multiple of blck_size
      - "strict": raise ValueError if ne0 % blck_size != 0
    """
    qtype = parse_ggml_type(requested)
    blck = ggml_blck_size(qtype)
    if ne0 % blck == 0:
        return qtype, ne0

    if incompat_policy == "strict":
        raise ValueError(f"ne0={ne0} not divisible by blck_size={blck} for {qtype.name}")
    if incompat_policy == "pad_to_block":
        padded = ((ne0 + blck - 1) // blck) * blck
        return qtype, padded
    if incompat_policy == "q4_0_fallback":
        fb = GGMLType.Q4_0 if qtype in (GGMLType.Q4_K, GGMLType.Q3_K, GGMLType.Q2_K) else LLAMA_CPP_FALLBACK.get(qtype)
    elif incompat_policy == "llama_cpp_fallback":
        fb = LLAMA_CPP_FALLBACK.get(qtype)
    else:
        raise ValueError(f"unknown incompat_policy: {incompat_policy!r}")

    if fb is None or ne0 % ggml_blck_size(fb) != 0:
        return GGMLType.F16, ne0
    return fb, ne0


def float32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Round-to-nearest-even FP32 -> little-endian BF16 raw bytes."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    # Handle NaNs cleanly, RNE rounding for finite values.
    nan_mask = (u32 & 0x7FFFFFFF) > 0x7F800000
    rounding = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16 = ((u32 + rounding) >> np.uint32(16)).astype(np.uint16)
    if np.any(nan_mask):
        bf16 = np.where(nan_mask, np.uint16(0x7FC0), bf16)
    return bf16.astype("<u2", copy=False).tobytes()


def bf16_bytes_to_float32(raw: bytes | bytearray | memoryview, shape: Sequence[int]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).copy()


def encode_unquantized(arr: np.ndarray, qtype: GGMLType) -> bytes:
    qtype = parse_ggml_type(qtype)
    if qtype == GGMLType.F32:
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if qtype == GGMLType.F16:
        return np.ascontiguousarray(arr, dtype="<f2").tobytes()
    if qtype == GGMLType.BF16:
        return float32_to_bf16_bytes(arr)
    raise ValueError(f"not an unquantized type: {qtype.name}")


def decode_unquantized(raw: bytes | bytearray | memoryview, shape: Sequence[int], qtype: GGMLType) -> np.ndarray:
    qtype = parse_ggml_type(qtype)
    if qtype == GGMLType.F32:
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if qtype == GGMLType.F16:
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if qtype == GGMLType.BF16:
        return bf16_bytes_to_float32(raw, shape)
    raise ValueError(f"not an unquantized type: {qtype.name}")


_TO_FLOAT_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
_FROM_FLOAT_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)


class _GgmlTypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", _TO_FLOAT_FN),
        ("from_float_ref", _FROM_FLOAT_FN),
    ]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_ggml_lib_dir(explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate a directory containing compiled libggml-base / ggml-base.dll."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("INKYNANO_GGML_LIB_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/content/llama.cpp/build/bin"),
            Path("/content/llama.cpp/build/ggml/src"),
            Path.cwd() / "llama.cpp" / "build" / "bin",
            Path.cwd() / "ggml_build" / "llama.cpp" / "build" / "bin",
        ]
    )
    base_names = ("ggml-base.dll",) if sys.platform == "win32" else ("libggml-base.so", "libggml-base.dylib")
    for cand in candidates:
        if cand.is_dir():
            for name in base_names:
                if (cand / name).is_file():
                    return cand
            # Also match versioned .so.0 on Linux.
            if sys.platform != "win32" and list(cand.glob("libggml-base.so*")):
                return cand
    return None


class GGMLCBackend:
    """Actual upstream GGML C quantization and dequantization backend via ctypes."""

    backend_kind = "ggml_c"

    def __init__(self, lib_dir: Optional[Path] = None, ggml_commit: Optional[str] = None) -> None:
        resolved = discover_ggml_lib_dir(lib_dir)
        if resolved is None:
            raise FileNotFoundError(
                "Could not locate compiled libggml-base (.so/.dll). "
                "Set INKYNANO_GGML_LIB_DIR or run build_ggml_cpu()."
            )
        self.lib_dir = resolved
        self.ggml_commit = ggml_commit or self._read_commit_receipt(resolved)

        if sys.platform == "win32":
            os.add_dll_directory(str(resolved))
            base_path = resolved / "ggml-base.dll"
            cpu_candidates = [resolved / "ggml-cpu-x64.dll", resolved / "ggml-cpu.dll"]
            cpu_candidates.extend(sorted(resolved.glob("ggml-cpu-*.dll")))
        else:
            so_matches = sorted(resolved.glob("libggml-base.so*"))
            if not so_matches:
                so_matches = sorted(resolved.glob("libggml-base.dylib"))
            base_path = so_matches[0]
            cpu_candidates = sorted(resolved.glob("libggml-cpu*.so*")) + sorted(resolved.glob("libggml-cpu*.dylib"))

        self.base_lib_path = base_path
        self.base_lib_sha256 = _sha256_file(base_path)
        self._base = ctypes.CDLL(str(base_path))
        self._cpu = None
        for cand in cpu_candidates:
            if cand.is_file():
                try:
                    lib = ctypes.CDLL(str(cand))
                    if hasattr(lib, "ggml_cpu_init"):
                        lib.ggml_cpu_init()
                    self._cpu = lib
                    self.cpu_lib_path = cand
                    break
                except OSError:
                    continue

        self._base.ggml_type_name.argtypes = [ctypes.c_int]
        self._base.ggml_type_name.restype = ctypes.c_char_p
        self._base.ggml_blck_size.argtypes = [ctypes.c_int]
        self._base.ggml_blck_size.restype = ctypes.c_int64
        self._base.ggml_type_size.argtypes = [ctypes.c_int]
        self._base.ggml_type_size.restype = ctypes.c_size_t
        self._base.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._base.ggml_row_size.restype = ctypes.c_size_t
        self._base.ggml_quantize_init.argtypes = [ctypes.c_int]
        self._base.ggml_quantize_init.restype = None
        self._base.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self._base.ggml_quantize_chunk.restype = ctypes.c_size_t
        self._base.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self._base.ggml_get_type_traits.restype = ctypes.POINTER(_GgmlTypeTraits)

        self._initialized_types: set[int] = set()
        self._verify_abi()

    @staticmethod
    def _read_commit_receipt(lib_dir: Path) -> str:
        for parent in (lib_dir, lib_dir.parent, lib_dir.parent.parent):
            receipt = parent / "ggml_build_receipt.json"
            if receipt.is_file():
                try:
                    return str(json.loads(receipt.read_text(encoding="utf8")).get("commit", "unknown"))
                except Exception:
                    pass
        return "unknown"

    def _verify_abi(self) -> None:
        """Sanity-check C struct ggml_type_traits against known Q4_K/Q6_K/Q5_0/Q8_0 invariants."""
        for qt in (GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.Q5_0, GGMLType.Q8_0, GGMLType.Q4_0):
            traits = self._base.ggml_get_type_traits(int(qt)).contents
            exp_blck, exp_ts = ggml_blck_size(qt), ggml_type_size(qt)
            if traits.blck_size != exp_blck or traits.type_size != exp_ts:
                raise RuntimeError(
                    f"GGML C ABI mismatch for {qt.name}: got ({traits.blck_size}, {traits.type_size}), "
                    f"expected ({exp_blck}, {exp_ts})"
                )

    def info(self) -> Dict[str, object]:
        return {
            "backend_kind": self.backend_kind,
            "lib_dir": str(self.lib_dir),
            "base_lib": self.base_lib_path.name,
            "base_lib_sha256": self.base_lib_sha256,
            "ggml_commit": self.ggml_commit,
            "platform": platform.platform(),
        }

    def quantize(self, array: np.ndarray, qtype: GGMLType | str, n_threads: int = 0) -> bytes:
        """Quantize a row-major float32 array using upstream C ggml_quantize_chunk."""
        qt = parse_ggml_type(qtype)
        if qt in (GGMLType.F32, GGMLType.F16, GGMLType.BF16):
            return encode_unquantized(array, qt)

        arr = np.ascontiguousarray(array, dtype=np.float32)
        if arr.ndim < 1:
            raise ValueError("cannot quantize 0-D array")
        ne0 = int(arr.shape[-1])
        nrows = int(arr.size // ne0)
        flat2d = arr.reshape(nrows, ne0)

        blck = int(self._base.ggml_blck_size(int(qt)))
        if ne0 % blck != 0:
            raise ValueError(f"ne0={ne0} not divisible by blck_size={blck} for {qt.name}")

        if int(qt) not in self._initialized_types:
            self._base.ggml_quantize_init(int(qt))
            self._initialized_types.add(int(qt))

        row_sz = int(self._base.ggml_row_size(int(qt), ne0))
        if row_sz != ggml_row_size(qt, ne0):
            raise RuntimeError(f"row_size mismatch for {qt.name}: C={row_sz} vs spec={ggml_row_size(qt, ne0)}")
        dst = np.empty((nrows, row_sz), dtype=np.uint8)

        workers = n_threads if n_threads > 0 else max(1, min(os.cpu_count() or 1, 8))
        if workers == 1 or nrows < workers * 4:
            written = self._base.ggml_quantize_chunk(
                int(qt), flat2d.ctypes.data, dst.ctypes.data, 0, nrows, ne0, None
            )
            if written != nrows * row_sz:
                raise RuntimeError(f"ggml_quantize_chunk wrote {written} bytes, expected {nrows * row_sz}")
            return dst.tobytes()

        # ctypes releases the GIL; splitting along rows is bit-exact with single-threaded.
        edges = np.linspace(0, nrows, workers + 1, dtype=np.int64)
        ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(workers) if edges[i + 1] > edges[i]]

        def _worker(span: Tuple[int, int]) -> int:
            r0, r1 = span
            src_slice = flat2d[r0:r1]
            dst_slice = dst[r0:r1]
            return int(
                self._base.ggml_quantize_chunk(
                    int(qt), src_slice.ctypes.data, dst_slice.ctypes.data, 0, r1 - r0, ne0, None
                )
            )

        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            written = sum(pool.map(_worker, ranges))
        if written != nrows * row_sz:
            raise RuntimeError(f"threaded ggml_quantize_chunk wrote {written} bytes, expected {nrows * row_sz}")
        return dst.tobytes()

    def dequantize(self, raw: bytes | bytearray | memoryview, pt_shape: Sequence[int], qtype: GGMLType | str) -> np.ndarray:
        """Dequantize GGML tensor bytes back to float32 using C ggml_get_type_traits->to_float."""
        qt = parse_ggml_type(qtype)
        expected = ggml_nbytes(pt_shape, qt)
        if len(raw) != expected:
            raise ValueError(f"buffer length {len(raw)} != expected {expected} for {qt.name} {pt_shape}")
        if qt in (GGMLType.F32, GGMLType.F16, GGMLType.BF16):
            return decode_unquantized(raw, pt_shape, qt)

        traits = self._base.ggml_get_type_traits(int(qt)).contents
        if not bool(traits.to_float):
            raise RuntimeError(f"GGML type {qt.name} has no C to_float routine")
        src = np.frombuffer(raw, dtype=np.uint8)
        if not src.flags.c_contiguous:
            src = np.ascontiguousarray(src)
        out = np.empty(tuple(int(d) for d in pt_shape), dtype=np.float32)
        traits.to_float(src.ctypes.data, out.ctypes.data, int(out.size))
        return out


class SyntheticRefBackend:
    """Pure-NumPy reference backend for CI unit tests when C libggml-base is absent.

    Supports exact F32/F16/BF16 and block-32 Q8_0 / Q4_0 GGML binary layouts so
    all streaming, manifest, resume and selective-loader tests run without C++.
    Refuses production K-quant conversion unless actual GGML C library is loaded.
    """

    backend_kind = "numpy_ci_test_only"

    def info(self) -> Dict[str, object]:
        return {"backend_kind": self.backend_kind, "ggml_commit": "ci_test_only"}

    def quantize(self, array: np.ndarray, qtype: GGMLType | str, n_threads: int = 0) -> bytes:
        qt = parse_ggml_type(qtype)
        if qt in (GGMLType.F32, GGMLType.F16, GGMLType.BF16):
            return encode_unquantized(array, qt)
        arr = np.ascontiguousarray(array, dtype=np.float32)
        if qt == GGMLType.Q8_0:
            if arr.shape[-1] % 32 != 0:
                raise ValueError("ne0 must be divisible by 32 for Q8_0")
            blocks = arr.reshape(-1, 32)
            amax = np.max(np.abs(blocks), axis=1)
            d = (amax / 127.0).astype("<f2")
            inv_d = np.where(d.astype(np.float32) != 0, 1.0 / d.astype(np.float32), 0.0)
            qs = np.clip(np.rint(blocks * inv_d[:, None]), -127, 127).astype(np.int8)
            out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
            out[:, :2] = d.view(np.uint8).reshape(-1, 2)
            out[:, 2:] = qs.view(np.uint8)
            return out.tobytes()
        if qt == GGMLType.Q4_0:
            if arr.shape[-1] % 32 != 0:
                raise ValueError("ne0 must be divisible by 32 for Q4_0")
            blocks = arr.reshape(-1, 32)
            # Match ggml quantize_row_q4_0_ref layout: low nibble = idx 0..15, high nibble = idx 16..31.
            max_idx = np.argmax(np.abs(blocks), axis=1)
            signed_max = blocks[np.arange(blocks.shape[0]), max_idx]
            d = (signed_max / -8.0).astype("<f2")
            inv_d = np.where(d.astype(np.float32) != 0, 1.0 / d.astype(np.float32), 0.0)
            q = np.clip(np.trunc(blocks * inv_d[:, None] + 8.5), 0, 15).astype(np.uint8)
            packed = (q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)
            out = np.empty((blocks.shape[0], 18), dtype=np.uint8)
            out[:, :2] = d.view(np.uint8).reshape(-1, 2)
            out[:, 2:] = packed
            return out.tobytes()
        raise RuntimeError(
            f"GGML type {qt.name} requires the actual compiled GGML C library (GGMLCBackend). "
            "Run build_ggml_cpu() or set INKYNANO_GGML_LIB_DIR."
        )

    def dequantize(self, raw: bytes | bytearray | memoryview, pt_shape: Sequence[int], qtype: GGMLType | str) -> np.ndarray:
        qt = parse_ggml_type(qtype)
        if qt in (GGMLType.F32, GGMLType.F16, GGMLType.BF16):
            return decode_unquantized(raw, pt_shape, qt)
        if qt == GGMLType.Q8_0:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
            d = np.ascontiguousarray(arr[:, :2]).view("<f2").reshape(-1, 1).astype(np.float32)
            qs = arr[:, 2:].view(np.int8).astype(np.float32)
            return (qs * d).reshape(pt_shape)
        if qt == GGMLType.Q4_0:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
            d = np.ascontiguousarray(arr[:, :2]).view("<f2").reshape(-1, 1).astype(np.float32)
            packed = arr[:, 2:]
            lo = (packed & 0x0F).astype(np.int16) - 8
            hi = ((packed >> 4) & 0x0F).astype(np.int16) - 8
            vals = np.concatenate([lo, hi], axis=1).astype(np.float32) * d
            return vals.reshape(pt_shape)
        raise RuntimeError(f"GGML type {qt.name} requires GGMLCBackend")


def build_ggml_cpu(
    dest_dir: Path,
    commit: str = PINNED_LLAMA_CPP_COMMIT,
    repo_url: str = PINNED_LLAMA_CPP_REPO,
    jobs: int = 0,
) -> Dict[str, object]:
    """Clone pinned llama.cpp and build CPU-only libggml-base + libggml-cpu shared libs."""
    dest_dir = Path(dest_dir)
    src_dir = dest_dir / "llama.cpp"
    build_dir = src_dir / "build"
    bin_dir = build_dir / "bin"
    receipt_path = build_dir / "ggml_build_receipt.json"

    if receipt_path.is_file() and discover_ggml_lib_dir(bin_dir) is not None:
        existing = json.loads(receipt_path.read_text(encoding="utf8"))
        if existing.get("commit") == commit:
            return existing

    dest_dir.mkdir(parents=True, exist_ok=True)
    if not (src_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", repo_url, str(src_dir)],
            check=True,
        )
    subprocess.run(["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(src_dir), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual_commit = subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != commit:
        raise RuntimeError(f"llama.cpp commit mismatch: expected {commit}, got {actual_commit}")

    cmake_args = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_CUDA=OFF",
        "-DGGML_VULKAN=OFF",
        "-DGGML_METAL=OFF",
        "-DGGML_NATIVE=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
    ]
    subprocess.run(cmake_args, check=True)
    n_jobs = jobs if jobs > 0 else max(1, os.cpu_count() or 2)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "ggml-base", "ggml-cpu", "-j", str(n_jobs)],
        check=True,
    )
    lib_dir = discover_ggml_lib_dir(bin_dir) or discover_ggml_lib_dir(build_dir / "ggml" / "src")
    if lib_dir is None:
        raise RuntimeError(f"GGML build succeeded but shared library not found under {build_dir}")

    backend = GGMLCBackend(lib_dir=lib_dir, ggml_commit=actual_commit)
    receipt = {
        "repo": repo_url,
        "commit": actual_commit,
        "lib_dir": str(lib_dir),
        "backend": backend.info(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return receipt
