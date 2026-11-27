"""Configurable RAM-bounded Mechanistic Memory Cache (MMS) and Immutable Receipt Ledger.

Design aligned with DataGoblin's GradientBank, MechanisticMapStore, and mms_utility
contracts:
  - Configurable budgets: 0, 64, 256, 1024, and 4096 MiB (strictly lazy allocation;
    never pre-allocates huge zero arrays; enforces allocated_bytes <= budget_bytes).
  - Explicitly accounts for cache construction (build_seconds), lookup (lookup_seconds),
    and maintenance (maintenance_seconds: insert, priority update, eviction).
  - Keeps ALL native Lean verifier receipts strictly immutable in an append-only
    SHA-256-chained ledger. The cache never mutates, fabricates, or skips verifier
    receipts or fresh kernel certification.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any

import numpy as np

VALID_MMS_BUDGETS_MIB = (0, 64, 256, 1024, 4096)
_ENTRY_OVERHEAD_BYTES = 128


def _canonical_digest(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _estimate_json_bytes(obj: Any) -> int:
    return len(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class ImmutableReceiptLedger:
    """Append-only, SHA-256-chained immutable store for native Lean verifier receipts.

    Every verifier RPC receipt (open, premises, step, cert, graph, lemma_graph, replay)
    is frozen on append and linked to the previous digest so tampering or silent edits
    are provably detectable.
    """

    def __init__(self, root_id: str) -> None:
        self._root_id = str(root_id)
        self._entries: list[MappingProxyType] = []
        self._hashes: list[str] = []
        self._chain_hash: str = _canonical_digest({"genesis": self._root_id})

    def append(self, op: str, receipt: dict[str, Any]) -> str:
        # Deep-copy via canonical JSON round-trip so external mutations cannot alias
        frozen_payload = json.loads(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
        record = {
            "seq": len(self._entries),
            "root_id": self._root_id,
            "op": str(op),
            "prev_hash": self._chain_hash,
            "receipt": frozen_payload,
        }
        entry_hash = _canonical_digest(record)
        record["entry_hash"] = entry_hash
        self._entries.append(MappingProxyType(record))
        self._hashes.append(entry_hash)
        self._chain_hash = entry_hash
        return entry_hash

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def chain_sha256(self) -> str:
        return self._chain_hash

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> MappingProxyType:
        return self._entries[idx]

    def verify_chain(self) -> bool:
        expected_prev = _canonical_digest({"genesis": self._root_id})
        for idx, entry in enumerate(self._entries):
            if entry["seq"] != idx or entry["prev_hash"] != expected_prev:
                return False
            body = {
                "seq": entry["seq"],
                "root_id": entry["root_id"],
                "op": entry["op"],
                "prev_hash": entry["prev_hash"],
                "receipt": entry["receipt"],
            }
            h = _canonical_digest(body)
            if h != entry["entry_hash"] or h != self._hashes[idx]:
                return False
            expected_prev = h
        return expected_prev == self._chain_hash

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entries]


@dataclass(frozen=True)
class EpisodicRecord:
    """One read-only memory record derived from a verified attempt or certified path."""

    key: str
    signature: np.ndarray  # float32 unit vector for similarity lookup
    tactic_template: str
    action_kind: str
    ok: bool
    closed: bool
    ngoals_delta: float
    hb_raw: int
    after_fail_streak: int
    step_idx: int

    @property
    def nbytes(self) -> int:
        return int(self.signature.nbytes + len(self.key) + len(self.tactic_template) + _ENTRY_OVERHEAD_BYTES)


@dataclass
class _SlotMeta:
    table: str
    key: str
    nbytes: int
    priority: float
    seq: int


class ProofMemoryCache:
    """Byte-budgeted, lazily allocated MMS cache for proof-frontier search & BPD maps.

    Tables:
      - ``lemma_graphs``: read-only parsed lemma term-graph subtrees keyed by (env_key, lemma).
      - ``state_features``: read-only feature packets / BPD component summaries keyed by exact path.
      - ``episodic_exact``: exact lookup statistics keyed by (goal_head_sig, tactic_template).
      - ``episodic_records``: vector-indexed episodic records for BPD/state similarity retrieval.
      - ``bpd_maps``: decomposed (V, U) factors & component stability/utility statistics.

    Enforces ``allocated_bytes <= budget_bytes`` across all tables combined.
    """

    TABLES = ("lemma_graphs", "state_features", "episodic_exact", "episodic_records", "bpd_maps")

    def __init__(
        self,
        budget_mib: float = 0.0,
        *,
        strict_registered_budget: bool = False,
        max_similarity_scan: int = 256,
    ) -> None:
        t0 = time.perf_counter()
        if not math.isfinite(budget_mib) or budget_mib < 0:
            raise ValueError(f"mms budget_mib must be finite and >= 0, got {budget_mib}")
        if strict_registered_budget and int(budget_mib) not in VALID_MMS_BUDGETS_MIB:
            raise ValueError(
                f"budget_mib={budget_mib} not in registered MMS budgets {VALID_MMS_BUDGETS_MIB}"
            )
        self.budget_mib = float(budget_mib)
        self.budget_bytes = int(round(self.budget_mib * (2**20)))
        self.enabled = self.budget_bytes > 0
        self.max_similarity_scan = max(1, int(max_similarity_scan))

        self._tables: dict[str, OrderedDict[str, Any]] = {t: OrderedDict() for t in self.TABLES}
        self._meta: dict[tuple[str, str], _SlotMeta] = {}
        self._allocated_bytes = 0
        self._peak_allocated_bytes = 0
        self._seq = 0

        # Accounting metrics
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0
        self.collisions = 0
        self.budget_rejections = 0
        self.build_seconds = 0.0
        self.lookup_seconds = 0.0
        self.maintenance_seconds = 0.0

        self.build_seconds += time.perf_counter() - t0

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    @property
    def peak_allocated_bytes(self) -> int:
        return self._peak_allocated_bytes

    @property
    def utilization_ratio(self) -> float:
        return 0.0 if self.budget_bytes <= 0 else self._allocated_bytes / self.budget_bytes

    def _evict_for(self, needed_bytes: int) -> bool:
        """Evict lowest-priority (oldest tie-break) entries until needed_bytes fits."""
        if needed_bytes > self.budget_bytes:
            return False
        while self._allocated_bytes + needed_bytes > self.budget_bytes and self._meta:
            # Find lowest priority, then oldest sequence number
            victim_id = min(self._meta, key=lambda k: (self._meta[k].priority, self._meta[k].seq))
            meta = self._meta.pop(victim_id)
            self._tables[meta.table].pop(meta.key, None)
            self._allocated_bytes -= meta.nbytes
            self.evictions += 1
        return self._allocated_bytes + needed_bytes <= self.budget_bytes

    def _admit(self, table: str, key: str, value: Any, nbytes: int, priority: float) -> bool:
        t0 = time.perf_counter()
        try:
            if not self.enabled or nbytes > self.budget_bytes:
                self.budget_rejections += 1
                return False
            slot = (table, key)
            if slot in self._meta:
                old = self._meta.pop(slot)
                self._tables[table].pop(key, None)
                self._allocated_bytes -= old.nbytes
                self.collisions += 1
            if not self._evict_for(nbytes):
                self.budget_rejections += 1
                return False
            self._seq += 1
            self._tables[table][key] = value
            self._meta[slot] = _SlotMeta(table=table, key=key, nbytes=nbytes, priority=priority, seq=self._seq)
            self._allocated_bytes += nbytes
            if self._allocated_bytes > self._peak_allocated_bytes:
                self._peak_allocated_bytes = self._allocated_bytes
            self.writes += 1
            return True
        finally:
            self.maintenance_seconds += time.perf_counter() - t0

    def _get(self, table: str, key: str) -> Any | None:
        t0 = time.perf_counter()
        try:
            if not self.enabled:
                self.misses += 1
                return None
            tbl = self._tables[table]
            if key in tbl:
                tbl.move_to_end(key)
                self.hits += 1
                return tbl[key]
            self.misses += 1
            return None
        finally:
            self.lookup_seconds += time.perf_counter() - t0

    # --- 1. Read-only lemma-graph subtree cache ---
    def put_lemma_graph(self, env_key: str, lemma_name: str, graph_obj: dict[str, Any]) -> bool:
        k = f"{env_key}|{lemma_name}"
        nbytes = _estimate_json_bytes(graph_obj) + _ENTRY_OVERHEAD_BYTES
        return self._admit("lemma_graphs", k, graph_obj, nbytes, priority=1.0)

    def get_lemma_graph(self, env_key: str, lemma_name: str) -> dict[str, Any] | None:
        return self._get("lemma_graphs", f"{env_key}|{lemma_name}")

    # --- 2. Read-only state feature cache (exact path key, never lossy PP dedup) ---
    def put_state_feature(self, exact_path_key: str, vec: np.ndarray) -> bool:
        arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float32))
        nbytes = int(arr.nbytes + len(exact_path_key) + _ENTRY_OVERHEAD_BYTES)
        return self._admit("state_features", exact_path_key, arr, nbytes, priority=1.1)

    def get_state_feature(self, exact_path_key: str) -> np.ndarray | None:
        return self._get("state_features", exact_path_key)

    # --- 3. Exact explicit-memory template outcome table ---
    def update_exact_stat(
        self,
        exact_key: str,
        *,
        ok: bool,
        closed: bool,
        hb_raw: int,
        after_fail_streak: int = 0,
    ) -> bool:
        t0 = time.perf_counter()
        cur = self._tables["episodic_exact"].get(exact_key) if self.enabled else None
        if cur is None:
            new_stat = {
                "n": 1,
                "ok": int(ok),
                "closed": int(closed),
                "hb_sum": int(hb_raw),
                "recovery_n": int(after_fail_streak >= 2),
                "recovery_ok": int(after_fail_streak >= 2 and (ok or closed)),
            }
        else:
            new_stat = {
                "n": cur["n"] + 1,
                "ok": cur["ok"] + int(ok),
                "closed": cur["closed"] + int(closed),
                "hb_sum": cur["hb_sum"] + int(hb_raw),
                "recovery_n": cur["recovery_n"] + int(after_fail_streak >= 2),
                "recovery_ok": cur["recovery_ok"] + int(after_fail_streak >= 2 and (ok or closed)),
            }
        self.maintenance_seconds += time.perf_counter() - t0
        prio = 1.5 + 2.0 * (new_stat["closed"] / max(1, new_stat["n"])) + 0.5 * (new_stat["ok"] / max(1, new_stat["n"]))
        nbytes = len(exact_key) + _ENTRY_OVERHEAD_BYTES + 48
        return self._admit("episodic_exact", exact_key, new_stat, nbytes, priority=prio)

    def get_exact_stat(self, exact_key: str) -> dict[str, int] | None:
        return self._get("episodic_exact", exact_key)

    # --- 4. Vector episodic memory (for BPD / proof-state similarity lookup) ---
    def put_episodic(self, rec: EpisodicRecord) -> bool:
        prio = 2.5 if rec.closed else (1.4 if rec.ok else 0.7)
        if rec.after_fail_streak >= 2 and (rec.ok or rec.closed):
            prio += 0.8
        return self._admit("episodic_records", rec.key, rec, rec.nbytes, priority=prio)

    def query_episodic(
        self,
        query_sig: np.ndarray,
        *,
        tactic_template: str | None = None,
        top_k: int = 8,
        min_cosine: float = 0.15,
    ) -> tuple[bool, float, float, int]:
        """Retrieve nearest episodic records; returns (hit, ok_rate, utility_est, n_matches)."""
        t0 = time.perf_counter()
        try:
            if not self.enabled or not self._tables["episodic_records"]:
                self.misses += 1
                return False, 0.0, 0.0, 0
            q = np.asarray(query_sig, dtype=np.float32).reshape(-1)
            qn = float(np.linalg.norm(q))
            if qn <= 1e-8 or not np.isfinite(q).all():
                self.misses += 1
                return False, 0.0, 0.0, 0
            q = q / qn
            # Scan most recent max_similarity_scan entries for bounded O(1) lookup latency
            items = list(self._tables["episodic_records"].values())[-self.max_similarity_scan :]
            scored: list[tuple[float, EpisodicRecord]] = []
            for rec in items:
                if tactic_template is not None and rec.tactic_template != tactic_template:
                    continue
                if rec.signature.shape != q.shape:
                    continue
                cos = float(np.dot(q, rec.signature))
                if cos >= min_cosine:
                    scored.append((cos, rec))
            if not scored:
                self.misses += 1
                return False, 0.0, 0.0, 0
            scored.sort(key=lambda x: -x[0])
            top = scored[:top_k]
            w_sum = sum(c for c, _ in top)
            ok_rate = sum(c * (1.0 if r.ok else 0.0) for c, r in top) / max(1e-8, w_sum)
            util = sum(
                c * (2.0 if r.closed else (0.5 + 0.25 * max(-2.0, min(2.0, r.ngoals_delta)) if r.ok else -0.25))
                for c, r in top
            ) / max(1e-8, w_sum)
            self.hits += 1
            return True, float(ok_rate), float(util), len(top)
        finally:
            self.lookup_seconds += time.perf_counter() - t0

    # --- 5. BPD mechanistic map cache ---
    def put_bpd_map(self, site_name: str, payload: dict[str, Any], *, nbytes: int | None = None) -> bool:
        t0 = time.perf_counter()
        if nbytes is None:
            total = _ENTRY_OVERHEAD_BYTES
            for v in payload.values():
                if isinstance(v, np.ndarray):
                    total += int(v.nbytes)
                else:
                    total += _estimate_json_bytes(v)
            nbytes = total
        self.build_seconds += time.perf_counter() - t0
        return self._admit("bpd_maps", site_name, payload, int(nbytes), priority=3.0)

    def get_bpd_map(self, site_name: str) -> dict[str, Any] | None:
        return self._get("bpd_maps", site_name)

    def receipt(self, ledger: ImmutableReceiptLedger | None = None) -> dict[str, Any]:
        table_bytes = {t: 0 for t in self.TABLES}
        for meta in self._meta.values():
            table_bytes[meta.table] += meta.nbytes
        table_counts = {t: len(self._tables[t]) for t in self.TABLES}
        out = {
            "budget_mib": self.budget_mib,
            "budget_bytes": self.budget_bytes,
            "allocated_bytes": self._allocated_bytes,
            "peak_allocated_bytes": self._peak_allocated_bytes,
            "utilization_ratio": round(self.utilization_ratio, 6),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / max(1, self.hits + self.misses), 4),
            "writes": self.writes,
            "evictions": self.evictions,
            "collisions": self.collisions,
            "budget_rejections": self.budget_rejections,
            "build_seconds": round(self.build_seconds, 6),
            "lookup_seconds": round(self.lookup_seconds, 6),
            "maintenance_seconds": round(self.maintenance_seconds, 6),
            "total_cache_seconds": round(self.build_seconds + self.lookup_seconds + self.maintenance_seconds, 6),
            "table_bytes": table_bytes,
            "table_counts": table_counts,
        }
        if ledger is not None:
            out["receipt_count"] = ledger.count
            out["receipt_chain_sha256"] = ledger.chain_sha256
            out["receipt_chain_valid"] = ledger.verify_chain()
        return out
