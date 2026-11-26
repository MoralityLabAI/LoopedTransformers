"""Bounded, versioned local-observer maps for MOLT-07.

The cap covers retained serialized key and map bytes, not process RSS.
All sketches live in an orthonormal Euclidean intervention chart.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

HEADER = struct.Struct("<4sII")
DIGEST_BYTES = 32
MAX_DIMENSION = 1024


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
    h.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return h.hexdigest()


@dataclass(frozen=True)
class MapBinding:
    checkpoint: str
    frame: str
    epoch: str
    input_digest: str
    state_digest: str
    observer: str
    cut: int
    position: int
    support: str
    max_rank: int

    def key(self) -> bytes:
        for field in (self.checkpoint, self.input_digest, self.state_digest, self.observer, self.support):
            if len(field) != 64 or any(c not in "0123456789abcdef" for c in field):
                raise ValueError("content digest required")
        if not self.frame or not self.epoch or self.cut < 0 or self.position < 0 or self.max_rank < 1:
            raise ValueError("invalid map binding")
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).digest()


@dataclass(frozen=True)
class Sketch:
    basis: torch.Tensor
    singular_values: torch.Tensor

    @classmethod
    def build(cls, observer: torch.Tensor, max_rank: int) -> Sketch:
        if observer.ndim != 2 or not observer.numel() or not bool(torch.isfinite(observer).all()):
            raise ValueError("nonempty finite observer required")
        d = observer.shape[1]
        if not 1 <= max_rank <= d <= MAX_DIMENSION:
            raise ValueError("invalid rank/dimension")
        # Full V supplies an orthonormal completion even for a short observer.
        _, values, vh = torch.linalg.svd(observer.detach().cpu().double(), full_matrices=True)
        sigma = torch.zeros(d, dtype=torch.float64)
        sigma[:len(values)] = values
        return cls(vh[:max_rank].T.contiguous(), sigma)

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def max_rank(self) -> int:
        return int(self.basis.shape[1])

    def project(self, delta: torch.Tensor, rank: int) -> torch.Tensor:
        if delta.shape != (self.dimension,) or not 0 <= rank <= self.max_rank:
            raise ValueError("invalid projection")
        basis = self.basis[:, :rank]
        return basis @ (basis.T @ delta.detach().cpu().double())

    def tail(self, rank: int) -> float:
        if not 0 <= rank <= self.max_rank:
            raise ValueError("invalid rank")
        return float(self.singular_values[rank]) if rank < self.dimension else 0.0

    def encode(self) -> bytes:
        d, k = self.dimension, self.max_rank
        values = self.basis.reshape(-1).tolist() + self.singular_values.tolist()
        if len(values) != d * k + d or not all(math.isfinite(x) for x in values):
            raise ValueError("invalid sketch")
        return HEADER.pack(b"OMS1", d, k) + struct.pack(f"<{len(values)}d", *values)

    @classmethod
    def decode(cls, payload: bytes) -> Sketch:
        if len(payload) < HEADER.size:
            raise ValueError("truncated sketch")
        magic, d, k = HEADER.unpack_from(payload)
        count = d * k + d
        if magic != b"OMS1" or not 1 <= k <= d <= MAX_DIMENSION or len(payload) != HEADER.size + 8 * count:
            raise ValueError("invalid sketch header")
        values = struct.unpack_from(f"<{count}d", payload, HEADER.size)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite sketch")
        b = torch.tensor(values[:d * k], dtype=torch.float64).reshape(d, k)
        s = torch.tensor(values[d * k:], dtype=torch.float64)
        if not torch.allclose(b.T @ b, torch.eye(k, dtype=b.dtype), atol=1e-9, rtol=0):
            raise ValueError("nonorthogonal cached basis")
        if bool((s < 0).any()) or bool((s[1:] > s[:-1]).any()):
            raise ValueError("invalid spectrum")
        return cls(b, s)


class MapCache:
    """LRU of immutable serialized maps, with explicit scope invalidation.

    The dictionary's allocator overhead, temporary tensors, and returned
    decoded maps are outside retained_bytes. The host supplies and audits
    the binding; it must never reuse a binding for a different observer.
    """
    def __init__(self, cap_bytes: int):
        if type(cap_bytes) is not int or cap_bytes < 0:
            raise ValueError("nonnegative integer cap required")
        self.cap_bytes = cap_bytes
        self.entries: OrderedDict[bytes, bytes] = OrderedDict()
        self.scope: tuple[str, str] | None = None
        self.retained_bytes = 0
        self.peak_retained_bytes = 0
        self.hits = self.misses = self.builds = self.evictions = self.invalidations = self.bypasses = 0
        self.build_seconds = 0.0

    def reset_scope(self, checkpoint: str, epoch: str) -> None:
        scope = (checkpoint, epoch)
        if self.scope != scope:
            self.invalidations += len(self.entries)
            self.entries.clear()
            self.retained_bytes = 0
            self.scope = scope

    def get(self, binding: MapBinding, builder: Callable[[], Sketch]) -> Sketch:
        if self.scope != (binding.checkpoint, binding.epoch):
            raise ValueError("activate checkpoint/epoch scope first")
        key = binding.key()
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return Sketch.decode(self.entries[key])
        self.misses += 1
        start = time.perf_counter()
        value = builder()
        self.build_seconds += time.perf_counter() - start
        self.builds += 1
        if value.max_rank != binding.max_rank:
            raise ValueError("map rank disagrees with binding")
        size = DIGEST_BYTES + HEADER.size + 8 * value.dimension * (value.max_rank + 1)
        if size > self.cap_bytes:
            self.bypasses += 1
            return value
        payload = value.encode()
        if len(key) + len(payload) != size:
            raise RuntimeError("serialized size mismatch")
        while self.retained_bytes + size > self.cap_bytes:
            old_key, old_payload = self.entries.popitem(last=False)
            self.retained_bytes -= len(old_key) + len(old_payload)
            self.evictions += 1
        self.entries[key] = payload
        self.retained_bytes += size
        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        return value

    def receipt(self) -> dict:
        if self.retained_bytes != sum(len(k) + len(v) for k, v in self.entries.items()):
            raise RuntimeError("retained byte ledger mismatch")
        return {k: getattr(self, k) for k in (
            "cap_bytes", "retained_bytes", "peak_retained_bytes", "hits", "misses", "builds",
            "evictions", "invalidations", "bypasses", "build_seconds",
        )}
