"""LBF-18: Incremental Relational Fresco, Hybrid Symbolic-Recurrent Readout,
2x2 Bank Interaction Factorial, Provenance Collision Stress Tests, and
Frozen Independent Sealed-Cohort Confirmation.

Stacked on merged PR #343 (`3d8243818858c19c5031c36e2bf7e328f831cf25`), freezing
all 24 LBF-17 checkpoints (`results/lbf17/`).
"""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
from dataclasses import dataclass
import functools
import gc
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import lbf15_meeting_fresco as m15
import lbf16_equivariant_fresco as m16
import lbf17_relational_fresco as m17

base = m17.base
WORK = m17.WORK
DEPTH = m17.DEPTH
BUDGET_CURVE = m17.BUDGET_CURVE
SEEDS = m17.SEEDS
STREAMS = m17.STREAMS
PASSES = m17.PASSES
FP_FIBERS = m17.FP_FIBERS
CHAR2ID = m17.CHAR2ID
LBF17_SEED = 20261227
LBF16_SEED = 20261221
LBF16_SUITE_MANIFEST_SHA256 = "ed10fcf5bc72c9cc0f0ba7847fa5ebc11d9ec04c6a3a98a3b17c0ef316957375"
LBF17_SUITE_MANIFEST_SHA256 = "a85f402f674423efd4967f6d3b2128450964a200a40f2b0d47f40ab2b7cfc8e2"

HERE = Path(__file__).resolve().parent
LBF17_RESULTS_DIR = HERE / "results" / "lbf17"
LBF16_RESULTS_DIR = HERE / "results" / "lbf16"
LEAN18_FILE = HERE / "lean" / "LBF18IncrementalAndCollisions.lean"
RUST_CRATE_DIR = HERE / "rust_trellis"

LBF18_MODES = (
    "normal",
    "scramble_window",
    "scramble_ribbon",
    "ablate_axial",
    "env_zero",
    "env_roll",
    "mat_zero",
    "mat_roll",
    "env_mat_zero",
    "env_mat_roll",
)

EXTENDED_ALPHABET = "abcdefghijklmnop"
EXT_CHAR2ID = {ch: idx for idx, ch in enumerate(EXTENDED_ALPHABET)}


class CacheInvalidationError(RuntimeError):
    """Raised fail-closed when a cached envelope or ribbon violates dependency invariants."""


# ============================================================================
# 1. Cross-Platform Process RSS / Peak Memory & Clean Timing Utilities
# ============================================================================
def current_and_peak_rss_bytes() -> tuple[int, int]:
    """Return `(current_rss_bytes, peak_rss_bytes)` via OS kernel counters."""
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize), int(pmc.PeakWorkingSetSize)
        return 0, 0
    status_path = Path("/proc/self/status")
    if status_path.exists():
        cur_kb = peak_kb = 0
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                cur_kb = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                peak_kb = int(line.split()[1])
        return cur_kb * 1024, peak_kb * 1024
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mult = 1 if sys.platform == "darwin" else 1024
        return int(ru * mult), int(ru * mult)
    except Exception:
        return 0, 0


def live_param_signature(model: nn.Module) -> tuple:
    """Fast O(#tensors) live mutation guard sensitive to in-place edits and reassignments."""
    sig = []
    with torch.no_grad():
        for p in model.parameters():
            flat = p.detach().reshape(-1)
            n = flat.numel()
            h0 = float(flat[0].item()) if n > 0 else 0.0
            hm = float(flat[n // 2].item()) if n > 1 else 0.0
            hl = float(flat[-1].item()) if n > 0 else 0.0
            s1 = float(flat.sum().item())
            sig.append((p.data_ptr(), int(p._version), n, round(h0, 7), round(hm, 7), round(hl, 7), round(s1, 5)))
    return tuple(sig)


@functools.lru_cache(maxsize=2048)
def canonical_rules_key(rules: tuple[tuple[str, str], ...]) -> str:
    return hashlib.sha256(
        json.dumps([list(r) for r in rules], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


# ============================================================================
# 2. Sparse Compiled Rust Trellis Bridge (`lbf18_trellis_rs`) + Safe Fallback
# ============================================================================
class _RelProvWeightsPtrs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "env_init_w", "env_init_b",
        "env_gate_wpos", "env_gate_gw", "env_gate_gb",
        "pos_r2_w1", "pos_r2_b1", "pos_r2_w2", "pos_r2_b2",
        "pos_r3_w1", "pos_r3_b1", "pos_r3_w2", "pos_r3_b2",
        "pos_res_w1", "pos_res_b1", "pos_res_w2", "pos_res_b2",
        "bi_gate_wpos", "bi_gate_gw", "bi_gate_gb",
        "bi_r2_w1", "bi_r2_b1", "bi_r2_w2", "bi_r2_b2",
        "bi_r3_w1", "bi_r3_b1", "bi_r3_w2", "bi_r3_b2",
    )]


class RustTrellisBackend:
    """Loads `lbf18_trellis_rs` cdylib via ctypes (building automatically if cargo is available)."""

    def __init__(self, auto_build: bool = True):
        self.lib = None
        self.lib_path: Path | None = None
        self._load(auto_build=auto_build)

    @staticmethod
    def _candidate_lib_names() -> tuple[str, ...]:
        if sys.platform == "win32":
            return ("lbf18_trellis_rs.dll", "liblbf18_trellis_rs.dll")
        if sys.platform == "darwin":
            return ("liblbf18_trellis_rs.dylib",)
        return ("liblbf18_trellis_rs.so",)

    def _find_existing(self) -> Path | None:
        rel_dir = RUST_CRATE_DIR / "target" / "release"
        for name in self._candidate_lib_names():
            cand = rel_dir / name
            if cand.exists():
                return cand
        for sub in (RUST_CRATE_DIR / "target").glob("*/release"):
            for name in self._candidate_lib_names():
                cand = sub / name
                if cand.exists():
                    return cand
        return None

    def _try_build(self) -> Path | None:
        manifest = RUST_CRATE_DIR / "Cargo.toml"
        if not manifest.exists():
            return None
        cargo = shutil.which("cargo")
        if cargo is None:
            home_cargo = Path.home() / ".cargo" / "bin" / ("cargo.exe" if sys.platform == "win32" else "cargo")
            if home_cargo.exists():
                cargo = str(home_cargo)
        if cargo is None:
            return None
        cmd = [cargo]
        if sys.platform == "win32":
            rustup = Path(cargo).with_name("rustup.exe")
            if rustup.exists():
                res = subprocess.run([str(rustup), "toolchain", "list"], capture_output=True, text=True)
                if "stable-x86_64-pc-windows-msvc" in res.stdout:
                    cmd.append("+stable-x86_64-pc-windows-msvc")
        cmd.extend(["build", "--release", "--manifest-path", str(manifest)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return self._find_existing()

    def _load(self, auto_build: bool) -> None:
        cand = self._find_existing()
        if cand is None and auto_build:
            cand = self._try_build()
        if cand is None:
            return
        lib = ctypes.CDLL(str(cand))
        lib.lbf18_sym_prov_adj2_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.lbf18_sym_prov_adj2_batch.restype = ctypes.c_uint64

        lib.lbf18_rel_prov_word_envelopes_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(_RelProvWeightsPtrs),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.lbf18_rel_prov_word_envelopes_f32.restype = ctypes.c_uint64
        self.lib = lib
        self.lib_path = cand

    @property
    def available(self) -> bool:
        return self.lib is not None

    def sym_prov_adj2_u64(
        self, words: list[str], rules: tuple[tuple[str, str], ...],
        *, alpha: int = 6, fp: int = FP_FIBERS, k_max: int = 3, char_map: dict[str, int] = CHAR2ID,
    ) -> tuple[torch.Tensor, int]:
        """Returns `(bitsets_i64, rule_checks)` where `bitsets_i64` has shape `(N, k_max+1, fp+1, L-1)`."""
        if not self.available or alpha > 8 or not words:
            raise RuntimeError("Rust u64 bitset kernel unavailable or alpha > 8")
        N = len(words)
        L = len(words[0])
        Lm1 = L - 1
        ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
        max_rl = max((len(u) for u, _ in ors), default=2)
        w_arr = torch.tensor([[char_map[ch] for ch in w] for w in words], dtype=torch.uint8).contiguous()
        rl_lhs = torch.zeros((max(1, len(ors)), max_rl), dtype=torch.uint8).contiguous()
        rl_rhs = torch.zeros((max(1, len(ors)), max_rl), dtype=torch.uint8).contiguous()
        rl_len = torch.zeros((max(1, len(ors)),), dtype=torch.uint8).contiguous()
        for idx, (u, v) in enumerate(ors):
            rl_len[idx] = len(u)
            for q, ch in enumerate(u):
                rl_lhs[idx, q] = char_map[ch]
            for q, ch in enumerate(v):
                rl_rhs[idx, q] = char_map[ch]
        out_bits = torch.zeros((N, k_max + 1, fp + 1, Lm1), dtype=torch.int64).contiguous()
        rc = int(self.lib.lbf18_sym_prov_adj2_batch(
            w_arr.data_ptr(), N, L, alpha,
            rl_lhs.data_ptr(), rl_rhs.data_ptr(), rl_len.data_ptr(),
            len(ors), max_rl, fp, k_max, out_bits.data_ptr(),
        ))
        return out_bits, rc

    def pack_rel_prov_weights(self, model: m17.RelationalProvenanceFrescoNet) -> tuple[_RelProvWeightsPtrs, list[torch.Tensor]]:
        keep: list[torch.Tensor] = []

        def ptr(t: torch.Tensor) -> int:
            tc = t.detach().contiguous().float()
            keep.append(tc)
            return int(tc.data_ptr())

        z1 = torch.zeros((1,), dtype=torch.float32)
        with torch.no_grad():
            if model.nonrecurrent:
                eg_wpos = eg_gw = eg_gb = bg_wpos = bg_gw = bg_gb = z1
            else:
                eg_wpos = F.softplus(model.env_gate.proj.weight) / math.sqrt(model.ce)
                eg_gw = model.env_gate.gate.weight
                eg_gb = model.env_gate.gate.bias
                bg_wpos = F.softplus(model.bi_gate.proj.weight) / math.sqrt(model.cb)
                bg_gw = model.bi_gate.gate.weight
                bg_gb = model.bi_gate.gate.bias

            st = _RelProvWeightsPtrs(
                ptr(model.env_init.weight), ptr(model.env_init.bias),
                ptr(eg_wpos), ptr(eg_gw), ptr(eg_gb),
                ptr(model.pos_cell.rule2_mlp[0].weight), ptr(model.pos_cell.rule2_mlp[0].bias),
                ptr(model.pos_cell.rule2_mlp[2].weight), ptr(model.pos_cell.rule2_mlp[2].bias),
                ptr(model.pos_cell.rule3_mlp[0].weight), ptr(model.pos_cell.rule3_mlp[0].bias),
                ptr(model.pos_cell.rule3_mlp[2].weight), ptr(model.pos_cell.rule3_mlp[2].bias),
                ptr(model.pos_cell.res_ff[0].weight), ptr(model.pos_cell.res_ff[0].bias),
                ptr(model.pos_cell.res_ff[2].weight), ptr(model.pos_cell.res_ff[2].bias),
                ptr(bg_wpos), ptr(bg_gw), ptr(bg_gb),
                ptr(model.bi_cell.rule2_mlp[0].weight), ptr(model.bi_cell.rule2_mlp[0].bias),
                ptr(model.bi_cell.rule2_mlp[2].weight), ptr(model.bi_cell.rule2_mlp[2].bias),
                ptr(model.bi_cell.rule3_mlp[0].weight), ptr(model.bi_cell.rule3_mlp[0].bias),
                ptr(model.bi_cell.rule3_mlp[2].weight), ptr(model.bi_cell.rule3_mlp[2].bias),
            )
        return st, keep

    def rel_prov_envelopes(
        self, words: list[str], rules: tuple[tuple[str, str], ...],
        *, fp: int, passes: int, env_mode_id: int, nonrecurrent: bool,
        wstruct: _RelProvWeightsPtrs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        N = len(words)
        L = len(words[0])
        Lm1 = L - 1
        w_arr = torch.tensor([[CHAR2ID[ch] for ch in w] for w in words], dtype=torch.uint8).contiguous()
        r2_t, n_r2, r3_t, n_r3 = _cached_compact_rule_bytes(rules)
        out_q = torch.empty((N, passes, L, 6, 20), dtype=torch.float32).contiguous()
        out_b = torch.empty((N, passes, L, 6, 20), dtype=torch.float32).contiguous()
        out_bi = torch.empty((N, passes, fp, Lm1, 6, 6, 4), dtype=torch.float32).contiguous()
        rc = int(self.lib.lbf18_rel_prov_word_envelopes_f32(
            w_arr.data_ptr(), N, L,
            r2_t.data_ptr(), n_r2,
            r3_t.data_ptr(), n_r3,
            fp, passes, env_mode_id, 1 if nonrecurrent else 0,
            ctypes.byref(wstruct),
            out_q.data_ptr(), out_b.data_ptr(), out_bi.data_ptr(),
        ))
        return out_q, out_b, out_bi, rc


@functools.lru_cache(maxsize=512)
def _cached_compact_rule_bytes(rules: tuple[tuple[str, str], ...]):
    r2_l, r3_l = [], []
    for a, b in rules:
        for src, dst in ((a, b), (b, a)):
            if src == dst:
                continue
            row = [CHAR2ID[ch] for ch in src + dst]
            if len(src) == 2:
                r2_l.append(row)
            else:
                r3_l.append(row)
    r2_t = torch.tensor(r2_l if r2_l else [[0, 0, 0, 0]], dtype=torch.uint8).contiguous()
    r3_t = torch.tensor(r3_l if r3_l else [[0, 0, 0, 0, 0, 0]], dtype=torch.uint8).contiguous()
    return r2_t, len(r2_l), r3_t, len(r3_l)


RUST_BACKEND = RustTrellisBackend(auto_build=True)


def unpack_u64_bitsets_to_float36(bitsets_i64: torch.Tensor) -> torch.Tensor:
    """Vectorized unpack of `(..., Lm1)` int64 bitmasks (low 36 bits) -> `(..., Lm1, 36)` float32."""
    shifts = torch.arange(36, dtype=torch.int64, device=bitsets_i64.device)
    return ((bitsets_i64.unsqueeze(-1) >> shifts) & 1).float()


# ============================================================================
# 3. Frozen LBF-17 Checkpoint Loader (SHA-Verified)
# ============================================================================
def load_frozen_lbf17_checkpoint(
    seed: int, arch: str = "rel_prov_opened", results_dir: Path = LBF17_RESULTS_DIR
) -> tuple[nn.Module, str, dict]:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = None
    for f in manifest["fits"]:
        if f["seed"] == seed and f["arch"] == arch:
            expected = f
            break
    if expected is None:
        raise ValueError(f"missing fit entry for ({seed}, {arch}) in {results_dir / 'manifest.json'}")
    ckpt_gz = results_dir / f"{seed}_{arch}" / "checkpoint.pt.gz"
    raw_bytes = gzip.decompress(ckpt_gz.read_bytes())
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != expected["checkpoint_sha256"]:
        raise ValueError(f"frozen checkpoint SHA mismatch for {seed}_{arch}: {actual_sha} != {expected['checkpoint_sha256']}")
    model, _ = m17.new_lbf17_model(seed, arch)
    payload = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
    model.load_state_dict(payload["weights"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    actual_fp = base._model_fingerprint(model)
    if actual_fp != expected["final"]:
        raise ValueError(f"frozen model parameter fingerprint mismatch for {seed}_{arch}")
    return model, actual_sha, expected


# ============================================================================
# 4. Exact Factored Incremental Controller for `RelationalProvenanceFrescoNet`
# ============================================================================
@dataclass
class WordEnvelopeEntry:
    w_ids: torch.Tensor          # (L,) long
    oh: torch.Tensor             # (L, 6, 1) float32
    oh_bi: torch.Tensor          # (Lm1, 6, 6, 1) float32
    pub_deg_norm: float
    pub_checks: int
    passes_ready: int
    qu: torch.Tensor             # (K, L, 6, ce)
    bu: torch.Tensor             # (K, L, 6, ce)
    bi_u: torch.Tensor           # (K, Fp, Lm1, 6, 6, cb)
    u_pos_L: torch.Tensor        # (K, L, 6, 6*ce)
    v_pos_R: torch.Tensor        # (K, L, 6, 6*ce)
    u_bi_L: torch.Tensor         # (K, Lm1, 36, 6*cb)
    v_bi_R: torch.Tensor         # (K, Lm1, 36, 6*cb)
    node_inv: torch.Tensor       # (K, 2*ce + 2*cb)
    sub_bright: torch.Tensor     # (K, L, 6)


@dataclass
class PairRibbonEntry:
    passes_ready: int
    rib_pool: torch.Tensor       # (K, 2*dm)
    dh: float
    lb: float
    span: float


@functools.lru_cache(maxsize=256)
def _calibrated_stage_flops(arch: str, L: int, passes: int) -> tuple[int, int]:
    """Exact per-word envelope FLOPs and per-pair static ribbon FLOPs via FlopCounterMode."""
    torch.manual_seed(107)
    net = m17.RelationalProvenanceFrescoNet(arch=arch, passes=passes).cpu().float().eval()
    dummy_rules = (("ab", "cd"), ("bc", "ef"), ("abc", "def"), ("bcd", "efa"))
    r2_p, r3_p = m17._cached_pos_rules(dummy_rules)
    r2_b, r3_b = m17.FastProvenanceBigramCell.encode_bigram_rules(dummy_rules)
    w_str = ("abcdef" * ((L // 6) + 2))[:L]
    fc_w = FlopCounterMode(display=False)
    with torch.no_grad(), fc_w:
        oh, oh_bi, h, _ = net.init_words([w_str])
        b = torch.zeros_like(h)
        Fp = FP_FIBERS if net.use_prov else 1
        bi = torch.zeros((1, Fp, L - 1, 6, 6, net.cb))
        for k in range(net.passes):
            if net.nonrecurrent:
                _, b, _, _ = net.pos_cell(oh, h, torch.zeros_like(h), b, r2_p, r3_p)
                bi, _ = net.bi_cell(oh_bi, 0, torch.zeros_like(bi), bi, r2_b, r3_b)
            else:
                g, _ = net.env_gate(h, b)
                h, b, _, _ = net.pos_cell(oh, h, g, b, r2_p, r3_p)
                gbi = torch.zeros_like(bi) if k == 0 else net.bi_gate(oh_bi.unsqueeze(1).expand_as(bi), bi)[0]
                bi, _ = net.bi_cell(oh_bi, k, gbi, bi, r2_b, r3_b)
    word_flops = int(fc_w.get_total_flops())

    fc_p = FlopCounterMode(display=False)
    with torch.no_grad(), fc_p:
        for _ in range(net.passes):
            up = torch.zeros((L, 6 * net.ce, 1, 6))
            vp = torch.zeros((L, 6 * net.ce, 6, 1))
            _ = up @ vp
            ub = torch.zeros((L - 1, 6 * net.cb, 1, 36))
            vb = torch.zeros((L - 1, 6 * net.cb, 36, 1))
            _ = ub @ vb
            rib = net.ribbon_in(torch.zeros((1, 1, L, 5 * net.ce + 5 * net.cb + 1)))
            _ = net.ribbon_conv(rib.reshape(1, L, net.dm).transpose(1, 2))
    pair_flops = int(fc_p.get_total_flops())
    return word_flops, pair_flops


@functools.lru_cache(maxsize=1024)
def _calibrated_stage_c_flops(arch: str, m0: int, m1: int, passes_exec: int) -> int:
    torch.manual_seed(107)
    net = m17.RelationalProvenanceFrescoNet(arch=arch).cpu().float().eval()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        bm = torch.zeros((m0, m1, net.dm))
        infeas = torch.zeros((m0, m1), dtype=torch.bool)
        h_mat = None
        for k in range(passes_exec):
            pbase = net.pair_in(torch.zeros((m0, m1, 2 * net.dm + 12)))
            pbase = pbase + net.pair_prelude(pbase)
            if net.nonrecurrent:
                h_mat, bm = net.fresco_block(pbase, bm, infeas)
                if arch == "param_nonrecurrent":
                    h_mat = h_mat + net.nr_pair_ff(h_mat)
                _ = net.pot_head(h_mat)
            else:
                if k == 0:
                    gm, _ = net.fresco_gate(pbase, bm)
                    h_mat, bm = net.fresco_block(pbase + gm, bm, infeas)
                    _ = net.pot_head(h_mat)
                else:
                    gm, _ = net.fresco_gate(h_mat, bm)
                    h_mat, bm = net.fresco_block(h_mat + 0.5 * pbase + gm, bm, infeas)
                    _ = net.pot_head(h_mat - pbase)
            node_d = net.node_head[0].in_features
            _ = net.node_head(torch.zeros((m0, node_d)))
            _ = net.node_head(torch.zeros((m1, node_d)))
    return int(fc.get_total_flops())


class IncrementalFrescoController:
    """Dependency-keyed incremental relational-provenance fresco evaluator.

    Caches:
      1. Word-level multi-pass relational envelopes keyed by `(ckpt_sha, rules_key, env_mode, word)`
      2. Static pairwise subterm ribbon pools keyed by `(ckpt_sha, rules_key, env_mode, u_word, v_word)`
    Rejects cache reuse fail-closed if model parameters mutate or if `mode` introduces
    frontier-cardinality-dependent spatial permutations (`scramble_window`, `scramble_ribbon`).
    """

    SUPPORTED_ARCHS = ("rel_prov_opened", "rel_prov_standard", "adj2_noprov_opened", "param_nonrecurrent")

    def __init__(
        self,
        model: m17.RelationalProvenanceFrescoNet,
        checkpoint_sha: str,
        *,
        use_rust: bool = True,
        cache_pair_ribbons: bool = True,
    ):
        if model.arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"IncrementalFrescoController supports {self.SUPPORTED_ARCHS}, got {model.arch}")
        self.model = model
        self.checkpoint_sha = checkpoint_sha
        self.bound_signature = live_param_signature(model)
        self.use_rust = bool(use_rust and RUST_BACKEND.available)
        self.cache_pair_ribbons = cache_pair_ribbons
        self.fp = FP_FIBERS if model.use_prov else 1
        self._rust_wstruct = None
        self._rust_keep: list[torch.Tensor] = []
        if self.use_rust:
            self._rust_wstruct, self._rust_keep = RUST_BACKEND.pack_rel_prov_weights(model)

        self.word_cache: dict[tuple[str, str, str, str], WordEnvelopeEntry] = {}
        self.pair_cache: dict[tuple[str, str, str, str, str], PairRibbonEntry] = {}
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.word_hits = 0
        self.word_misses = 0
        self.pair_hits = 0
        self.pair_misses = 0
        self.cache_rejections = 0
        self.rule_window_checks = 0
        self.neural_flops = 0

    def clear_caches(self) -> None:
        self.word_cache.clear()
        self.pair_cache.clear()

    def verify_integrity(self) -> None:
        if live_param_signature(self.model) != self.bound_signature:
            raise CacheInvalidationError(
                "model parameters were mutated after binding IncrementalFrescoController; cache reuse rejected"
            )

    @staticmethod
    def env_mode_for(mode: str) -> tuple[str, int]:
        if mode in ("normal", "ablate_axial", "mat_zero", "mat_roll"):
            return "normal", 0
        if mode in ("env_zero", "env_mat_zero"):
            return "env_zero", 1
        if mode in ("env_roll", "env_mat_roll"):
            return "env_roll", 2
        raise CacheInvalidationError(f"mode {mode} violates frontier-size invariance; cache reuse rejected")

    def _compute_missing_words(
        self, case, missing_words: list[str], r_key: str, env_mode: str, env_mode_id: int, target_passes: int
    ) -> None:
        if not missing_words:
            return
        m = self.model
        N = len(missing_words)
        L = len(missing_words[0])
        Lm1 = L - 1
        K = target_passes

        oh_all, oh_bi_all, h_all, ids_all = m.init_words(missing_words)
        if self.use_rust:
            q_k, b_k, bi_k, rc = RUST_BACKEND.rel_prov_envelopes(
                missing_words, case.rules,
                fp=self.fp, passes=K, env_mode_id=env_mode_id,
                nonrecurrent=m.nonrecurrent, wstruct=self._rust_wstruct,
            )
            self.rule_window_checks += rc
        else:
            r2_pos, r3_pos = m17._cached_pos_rules(case.rules)
            r2_bi, r3_bi = m17.FastProvenanceBigramCell.encode_bigram_rules(case.rules)
            n_r2 = 0 if r2_pos is None else r2_pos[0].shape[0]
            n_r3 = 0 if r3_pos is None else r3_pos[0].shape[0]
            win_per_word = n_r2 * max(0, L - 1) + n_r3 * max(0, L - 2)
            b_cur = torch.zeros_like(h_all)
            bi_cur = torch.zeros((N, self.fp, Lm1, 6, 6, m.cb), dtype=torch.float32)
            q_list, b_list, bi_list = [], [], []
            for k in range(K):
                self.rule_window_checks += N * win_per_word * (1 + (1 if k == 0 else self.fp))
                if m.nonrecurrent:
                    h_all, b_cur, _, q_cur = m.pos_cell(oh_all, h_all, torch.zeros_like(h_all), b_cur, r2_pos, r3_pos)
                    bi_cur, _ = m.bi_cell(oh_bi_all, 0, torch.zeros_like(bi_cur), bi_cur, r2_bi, r3_bi)
                else:
                    m_in = b_cur
                    mbi_in = bi_cur
                    if k > 0:
                        if env_mode_id == 1:
                            m_in = torch.zeros_like(m_in)
                            mbi_in = torch.zeros_like(mbi_in)
                        elif env_mode_id == 2:
                            m_in = m_in.roll(1, dims=1)
                            mbi_in = mbi_in.roll(1, dims=2)
                    g_all, _ = m.env_gate(h_all, m_in)
                    h_all, b_cur, _, q_cur = m.pos_cell(oh_all, h_all, g_all, m_in, r2_pos, r3_pos)
                    gbi_all = torch.zeros_like(mbi_in) if k == 0 else m.bi_gate(oh_bi_all.unsqueeze(1).expand_as(mbi_in), mbi_in)[0]
                    bi_cur, _ = m.bi_cell(oh_bi_all, k, gbi_all, mbi_in, r2_bi, r3_bi)
                q_list.append(q_cur)
                b_list.append(b_cur)
                bi_list.append(bi_cur)
            q_k = torch.stack(q_list, dim=1)
            b_k = torch.stack(b_list, dim=1)
            bi_k = torch.stack(bi_list, dim=1)

        wf_unit, _ = _calibrated_stage_flops(m.arch, L, K)
        self.neural_flops += N * wf_unit

        # Precompute per-word static overlap factors across all K passes once!
        b_anc = b_k[:, 0:1].expand_as(b_k)
        b_nov = F.relu(b_k - b_anc)
        oh_exp = oh_all.unsqueeze(1).expand_as(b_k)
        u_pos_L = torch.cat([q_k, b_anc, b_nov, 0.5 * b_anc, b_k, oh_exp], dim=-1)
        v_pos_R = torch.cat([q_k, b_anc, b_nov + 0.5 * b_anc, b_nov, oh_exp, b_k], dim=-1)

        c_bi = bi_k.amax(dim=2).reshape(N, K, Lm1, 36, m.cb)
        a_bi = c_bi[:, 0:1].expand_as(c_bi)
        n_bi = F.relu(c_bi - a_bi)
        ob0 = oh_bi_all.reshape(N, 1, Lm1, 36, 1).expand(N, K, Lm1, 36, m.cb)
        u_bi_L = torch.cat([ob0, a_bi, n_bi, 0.5 * a_bi, c_bi, ob0], dim=-1)
        v_bi_R = torch.cat([ob0, a_bi, n_bi + 0.5 * a_bi, n_bi, ob0, c_bi], dim=-1)

        bi_anc_full = bi_k[:, 0:1].expand_as(bi_k)
        node_inv = torch.cat([
            b_anc.mean(dim=(2, 3)),
            b_nov.amax(dim=(2, 3)),
            bi_anc_full.mean(dim=(2, 3, 4, 5)),
            F.relu(bi_k - bi_anc_full).amax(dim=(2, 3, 4, 5)),
        ], dim=-1)

        sub_list = [m.subterm_brightness(oh_all, b_k[:, k], bi_k[:, k]) for k in range(K)]
        sub_bright = torch.stack(sub_list, dim=1)

        for idx, w in enumerate(missing_words):
            dg, ch = m17._cached_pub_matches(case.rules, w)
            self.rule_window_checks += ch
            w_key = (self.checkpoint_sha, r_key, env_mode, w)
            self.word_cache[w_key] = WordEnvelopeEntry(
                w_ids=ids_all[idx],
                oh=oh_all[idx],
                oh_bi=oh_bi_all[idx],
                pub_deg_norm=dg / 24.0,
                pub_checks=ch,
                passes_ready=K,
                qu=q_k[idx],
                bu=b_k[idx],
                bi_u=bi_k[idx],
                u_pos_L=u_pos_L[idx],
                v_pos_R=v_pos_R[idx],
                u_bi_L=u_bi_L[idx],
                v_bi_R=v_bi_R[idx],
                node_inv=node_inv[idx],
                sub_bright=sub_bright[idx],
            )

    def _compute_pair_block(
        self, u_entries: list[WordEnvelopeEntry], v_entries: list[WordEnvelopeEntry], K: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(rib_pool (du, dv, K, 2*dm), dh (du, dv), lb (du, dv), span (du, dv))` for a block."""
        m = self.model
        du, dv = len(u_entries), len(v_entries)
        L = u_entries[0].w_ids.shape[0]
        Lm1 = L - 1

        u_ids = torch.stack([e.w_ids for e in u_entries], dim=0)
        v_ids = torch.stack([e.w_ids for e in v_entries], dim=0)
        mismatch = (u_ids[:, None, :] != v_ids[None, :, :])
        dh_mat = mismatch.sum(dim=-1).float()
        lb = torch.div(dh_mat.long() + 2, 3, rounding_mode="floor").float()
        pos_idx = torch.arange(L, dtype=torch.float32).view(1, 1, L)
        first_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
        last_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
        span = torch.clamp(last_mm - first_mm + 1.0, min=0.0)

        # Stack across (du, K, ...) and evaluate all K passes in one batched GEMM + Conv1d!
        up = torch.stack([e.u_pos_L[:K] for e in u_entries], dim=0).permute(1, 2, 4, 0, 3).contiguous()  # (K, L, 6*ce, du, 6)
        vp = torch.stack([e.v_pos_R[:K] for e in v_entries], dim=0).permute(1, 2, 4, 3, 0).contiguous()  # (K, L, 6*ce, 6, dv)
        ov_all6 = (up @ vp).permute(0, 3, 4, 1, 2)  # (K, du, dv, L, 6*ce)
        ov_q, ov_1hop, ov_k1, ov_k2, ov_u2v, ov_v2u = ov_all6.split(m.ce, dim=-1)
        ov_khop = ov_k1 + ov_k2

        ub = torch.stack([e.u_bi_L[:K] for e in u_entries], dim=0).permute(1, 2, 4, 0, 3).contiguous()   # (K, Lm1, 6*cb, du, 36)
        vb = torch.stack([e.v_bi_R[:K] for e in v_entries], dim=0).permute(1, 2, 4, 3, 0).contiguous()   # (K, Lm1, 6*cb, 36, dv)
        bi_all6 = (ub @ vb).permute(0, 3, 4, 1, 2)  # (K, du, dv, Lm1, 6*cb)
        b_0, b_1hop, b_k1, b_k2, b_u2v, b_v2u = bi_all6.split(m.cb, dim=-1)
        b_khop = b_k1 + b_k2
        bi_node = m.pad_edges_to_nodes(torch.cat([b_0, b_1hop, b_khop, b_u2v, b_v2u], dim=-1))

        mm_f = mismatch.float().unsqueeze(0).unsqueeze(-1).expand(K, du, dv, L, 1)
        rib_raw = torch.cat([ov_q, ov_1hop, ov_khop, ov_u2v, ov_v2u, bi_node, mm_f], dim=-1)
        rib = m.ribbon_in(rib_raw).reshape(K * du * dv, L, m.dm).transpose(1, 2)
        rib_c = F.gelu(m.ribbon_conv(rib)).transpose(1, 2).reshape(K, du, dv, L, m.dm)

        mm_cnt = mm_f.sum(dim=3).clamp_min(1.0)
        rib_pool = torch.cat([(rib_c * mm_f).sum(dim=3) / mm_cnt, rib_c.amax(dim=3)], dim=-1)  # (K, du, dv, 2*dm)
        _, pf_unit = _calibrated_stage_flops(m.arch, L, K)
        self.neural_flops += du * dv * pf_unit
        return rib_pool.permute(1, 2, 0, 3).contiguous(), dh_mat, lb, span

    def _compute_sparse_pairs(
        self, u_entries: list[WordEnvelopeEntry], v_entries: list[WordEnvelopeEntry],
        coords: list[tuple[int, int]], K: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat vector evaluation for non-rectangular missing pair subsets (e.g. partial warm cache)."""
        m = self.model
        P = len(coords)
        L = u_entries[0].w_ids.shape[0]
        Lm1 = L - 1

        u_ids = torch.stack([u_entries[i].w_ids for i, _ in coords], dim=0)
        v_ids = torch.stack([v_entries[j].w_ids for _, j in coords], dim=0)
        mismatch = (u_ids != v_ids)
        dh = mismatch.sum(dim=-1).float()
        lb = torch.div(dh.long() + 2, 3, rounding_mode="floor").float()
        pos_idx = torch.arange(L, dtype=torch.float32).view(1, L)
        first_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
        last_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
        span = torch.clamp(last_mm - first_mm + 1.0, min=0.0)

        up = torch.stack([u_entries[i].u_pos_L[:K] for i, _ in coords], dim=0)  # (P, K, L, 6, 6*ce)
        vp = torch.stack([v_entries[j].v_pos_R[:K] for _, j in coords], dim=0)
        ov_all6 = (up * vp).sum(dim=3)  # (P, K, L, 6*ce)
        ov_q, ov_1hop, ov_k1, ov_k2, ov_u2v, ov_v2u = ov_all6.split(m.ce, dim=-1)
        ov_khop = ov_k1 + ov_k2

        ub = torch.stack([u_entries[i].u_bi_L[:K] for i, _ in coords], dim=0)   # (P, K, Lm1, 36, 6*cb)
        vb = torch.stack([v_entries[j].v_bi_R[:K] for _, j in coords], dim=0)
        bi_all6 = (ub * vb).sum(dim=3)  # (P, K, Lm1, 6*cb)
        b_0, b_1hop, b_k1, b_k2, b_u2v, b_v2u = bi_all6.split(m.cb, dim=-1)
        b_khop = b_k1 + b_k2
        bi_node = m.pad_edges_to_nodes(torch.cat([b_0, b_1hop, b_khop, b_u2v, b_v2u], dim=-1))

        mm_f = mismatch.float().unsqueeze(1).unsqueeze(-1).expand(P, K, L, 1)
        rib_raw = torch.cat([ov_q, ov_1hop, ov_khop, ov_u2v, ov_v2u, bi_node, mm_f], dim=-1)
        rib = m.ribbon_in(rib_raw).reshape(P * K, L, m.dm).transpose(1, 2)
        rib_c = F.gelu(m.ribbon_conv(rib)).transpose(1, 2).reshape(P, K, L, m.dm)
        mm_cnt = mm_f.sum(dim=2).clamp_min(1.0)
        rib_pool = torch.cat([(rib_c * mm_f).sum(dim=2) / mm_cnt, rib_c.amax(dim=2)], dim=-1)
        _, pf_unit = _calibrated_stage_flops(m.arch, L, K)
        self.neural_flops += P * pf_unit
        return rib_pool, dh, lb, span

    def forward(
        self,
        case,
        s: m15.FringeState,
        rem_budget: int,
        *,
        mode: str = "normal",
        depth_cap: int = DEPTH,
        adaptive_margins: tuple[float, float] | None = None,
    ) -> list[dict]:
        if mode not in LBF18_MODES:
            raise ValueError(f"unsupported LBF-18 mode {mode}")
        self.verify_integrity()

        # Fail-closed rejection when spatial scrambles depend on dynamic `(m0, m1)`
        if mode in ("scramble_window", "scramble_ribbon"):
            self.cache_rejections += 1
            outs, rc = self.model(case, s, rem_budget, mode=mode, depth_cap=depth_cap)
            self.rule_window_checks += rc
            self.neural_flops += m17.cached_decision_flops(self.model, self.model.arch, case, s, rem_budget, mode=mode)
            return outs

        env_mode, env_mode_id = self.env_mode_for(mode)
        r_key = canonical_rules_key(case.rules)
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)
        max_K = self.model.passes

        def prepare_stages_ab(req_K: int, count_hits: bool):
            needed_words = []
            seen_needed = set()
            for w in u_words + v_words:
                wk = (self.checkpoint_sha, r_key, env_mode, w)
                ent = self.word_cache.get(wk)
                if ent is not None and ent.passes_ready >= req_K:
                    if count_hits:
                        self.word_hits += 1
                else:
                    if count_hits:
                        self.word_misses += 1
                    if w not in seen_needed:
                        seen_needed.add(w)
                        needed_words.append(w)
            if needed_words:
                self._compute_missing_words(case, needed_words, r_key, env_mode, env_mode_id, req_K)

            u_e = [self.word_cache[(self.checkpoint_sha, r_key, env_mode, w)] for w in u_words]
            v_e = [self.word_cache[(self.checkpoint_sha, r_key, env_mode, w)] for w in v_words]

            if self.cache_pair_ribbons:
                miss_coords: list[tuple[int, int]] = []
                miss_rows = set()
                miss_cols = set()
                for i, uw in enumerate(u_words):
                    for j, vw in enumerate(v_words):
                        pk = (self.checkpoint_sha, r_key, env_mode, uw, vw)
                        pe = self.pair_cache.get(pk)
                        if pe is not None and pe.passes_ready >= req_K:
                            if count_hits:
                                self.pair_hits += 1
                        else:
                            if count_hits:
                                self.pair_misses += 1
                            miss_coords.append((i, j))
                            miss_rows.add(i)
                            miss_cols.add(j)

                if miss_coords:
                    r_sorted = sorted(miss_rows)
                    c_sorted = sorted(miss_cols)
                    if len(miss_coords) == len(r_sorted) * len(c_sorted):
                        rp_blk, dh_blk, lb_blk, sp_blk = self._compute_pair_block(
                            [u_e[i] for i in r_sorted], [v_e[j] for j in c_sorted], req_K
                        )
                        for ri, i in enumerate(r_sorted):
                            uw = u_words[i]
                            for ci, j in enumerate(c_sorted):
                                vw = v_words[j]
                                self.pair_cache[(self.checkpoint_sha, r_key, env_mode, uw, vw)] = PairRibbonEntry(
                                    passes_ready=req_K,
                                    rib_pool=rp_blk[ri, ci],
                                    dh=float(dh_blk[ri, ci].item()),
                                    lb=float(lb_blk[ri, ci].item()),
                                    span=float(sp_blk[ri, ci].item()),
                                )
                    else:
                        rp_sp, dh_sp, lb_sp, sp_sp = self._compute_sparse_pairs(u_e, v_e, miss_coords, req_K)
                        for idx, (i, j) in enumerate(miss_coords):
                            self.pair_cache[(self.checkpoint_sha, r_key, env_mode, u_words[i], v_words[j])] = PairRibbonEntry(
                                passes_ready=req_K,
                                rib_pool=rp_sp[idx],
                                dh=float(dh_sp[idx].item()),
                                lb=float(lb_sp[idx].item()),
                                span=float(sp_sp[idx].item()),
                            )

                rg_t = torch.empty((m0, m1, req_K, 2 * self.model.dm), dtype=torch.float32)
                dh_t = torch.empty((m0, m1), dtype=torch.float32)
                lb_t = torch.empty((m0, m1), dtype=torch.float32)
                sp_t = torch.empty((m0, m1), dtype=torch.float32)
                for i, uw in enumerate(u_words):
                    for j, vw in enumerate(v_words):
                        pe = self.pair_cache[(self.checkpoint_sha, r_key, env_mode, uw, vw)]
                        rg_t[i, j] = pe.rib_pool[:req_K]
                        dh_t[i, j] = pe.dh
                        lb_t[i, j] = pe.lb
                        sp_t[i, j] = pe.span
            else:
                if count_hits:
                    self.pair_misses += m0 * m1
                rg_t, dh_t, lb_t, sp_t = self._compute_pair_block(u_e, v_e, req_K)
            return u_e, v_e, rg_t, dh_t, lb_t, sp_t

        init_K = 1 if (adaptive_margins is not None and max_K > 1) else max_K
        u_ent, v_ent, rib_grid, dh_mat, lb, span = prepare_stages_ab(init_K, count_hits=True)

        # Stage C: Dynamic Cross-Frontier Feasibility, Persistence, and Axial Set Competition
        g0 = torch.tensor([s.dist[0][w] for w in u_words], dtype=torch.float32)[:, None]
        g1 = torch.tensor([s.dist[1][w] for w in v_words], dtype=torch.float32)[None, :]
        tot_lb = g0 + g1 + torch.clamp(lb, min=1.0)
        infeas = tot_lb > depth_cap

        pers = torch.zeros((m0, m1), dtype=torch.float32)
        prev_v = torch.zeros((m0, m1), dtype=torch.float32)
        if s.prev_pot:
            for i, uw in enumerate(u_words):
                for j, vw in enumerate(v_words):
                    if (uw, vw) in s.prev_pot:
                        pers[i, j] = 1.0
                        prev_v[i, j] = s.prev_pot[(uw, vw)]

        b_frac = rem_budget / WORK
        lm_diff = (math.log1p(m0) - math.log1p(m1)) / math.log1p(WORK)
        pub = torch.stack([
            (g0 / depth_cap).expand(m0, m1),
            (g1 / depth_cap).expand(m0, m1),
            ((g0 + g1) / depth_cap).expand(m0, m1),
            dh_mat / L,
            lb / depth_cap,
            torch.clamp(depth_cap - tot_lb, min=0.0) / depth_cap,
            1.0 - infeas.float(),
            span / L,
            torch.full((m0, m1), b_frac),
            torch.full((m0, m1), lm_diff),
            pers,
            prev_v,
        ], dim=-1)

        def build_node_pub(entries: list[WordEnvelopeEntry], words: list[str], side: int) -> torch.Tensor:
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            g_ids = torch.tensor([CHAR2ID[ch] for ch in goal], dtype=torch.long)
            w_ids = torch.stack([e.w_ids for e in entries], dim=0)
            dh_goal = (w_ids != g_ids[None, :]).sum(dim=-1).float() / L
            degs = torch.tensor([e.pub_deg_norm for e in entries], dtype=torch.float32)
            gs = torch.tensor([s.dist[side][w] / depth_cap for w in words], dtype=torch.float32)
            return torch.stack([
                gs, degs,
                torch.full((ms,), math.log1p(ms) / 5.0),
                torch.full((ms,), math.log1p(mo) / 5.0),
                torch.full((ms,), b_frac),
                dh_goal,
            ], dim=-1)

        np0 = build_node_pub(u_ent, u_words, 0)
        np1 = build_node_pub(v_ent, v_words, 1)

        u_ninv = torch.stack([e.node_inv[:init_K] for e in u_ent], dim=0)
        v_ninv = torch.stack([e.node_inv[:init_K] for e in v_ent], dim=0)
        u_sub = torch.stack([e.sub_bright[:init_K] for e in u_ent], dim=0)
        v_sub = torch.stack([e.sub_bright[:init_K] for e in v_ent], dim=0)

        ablate_ax = (mode == "ablate_axial")
        mat_zero = mode in ("mat_zero", "env_mat_zero")
        mat_roll = mode in ("mat_roll", "env_mat_roll")

        bm = torch.zeros((m0, m1, self.model.dm), dtype=torch.float32)
        h_mat = pot_accum = None
        outs = []
        tau = F.softplus(self.model.log_tau) + 0.1
        feas_u = ~infeas.all(dim=1)
        feas_v = ~infeas.all(dim=0)
        n_feas_actions = int(feas_u.sum().item() + feas_v.sum().item())
        cur_K = init_K

        for k in range(max_K):
            if k >= cur_K:
                cur_K = max_K
                u_ent, v_ent, rib_grid, _, _, _ = prepare_stages_ab(cur_K, count_hits=False)
                u_ninv = torch.stack([e.node_inv[:cur_K] for e in u_ent], dim=0)
                v_ninv = torch.stack([e.node_inv[:cur_K] for e in v_ent], dim=0)
                u_sub = torch.stack([e.sub_bright[:cur_K] for e in u_ent], dim=0)
                v_sub = torch.stack([e.sub_bright[:cur_K] for e in v_ent], dim=0)
            mm_in = bm
            if k > 0:
                if mat_zero:
                    mm_in = torch.zeros_like(mm_in)
                elif mat_roll:
                    mm_in = mm_in.roll(shifts=(1, 1), dims=(0, 1))

            pbase = self.model.pair_in(torch.cat([rib_grid[:, :, k, :], pub], dim=-1))
            pbase = pbase + self.model.pair_prelude(pbase)

            if self.model.nonrecurrent:
                h_mat, bm = self.model.fresco_block(pbase, bm, infeas, ablate_axial=ablate_ax)
                if self.model.arch == "param_nonrecurrent":
                    h_mat = h_mat + self.model.nr_pair_ff(h_mat)
                pot = self.model.pot_head(h_mat).squeeze(-1)
            else:
                if k == 0:
                    gm, _ = self.model.fresco_gate(pbase, mm_in)
                    h_mat, bm = self.model.fresco_block(pbase + gm, mm_in, infeas, ablate_axial=ablate_ax)
                    pot_accum = self.model.pot_head(h_mat).squeeze(-1)
                else:
                    gm, _ = self.model.fresco_gate(h_mat, mm_in)
                    h_mat, bm = self.model.fresco_block(h_mat + 0.5 * pbase + gm, mm_in, infeas, ablate_axial=ablate_ax)
                    pot_accum = pot_accum + self.model.delta_scale * self.model.pot_head(h_mat - pbase).squeeze(-1)
                pot = pot_accum

            pot_masked = pot.masked_fill(infeas, -12.0)
            nu_k = torch.cat([u_ninv[:, k, :], np0], dim=-1)
            nv_k = torch.cat([v_ninv[:, k, :], np1], dim=-1)
            s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
                  + self.model.node_head(nu_k).squeeze(-1)).masked_fill(~feas_u, -12.0)
            s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
                  + self.model.node_head(nv_k).squeeze(-1)).masked_fill(~feas_v, -12.0)
            outs.append({
                "pot": pot,
                "pot_masked": pot_masked,
                "s0": s0,
                "s1": s1,
                "infeas": infeas,
                "sub_u": u_sub[:, k],
                "sub_v": v_sub[:, k],
            })

            if adaptive_margins is not None and k + 1 < max_K:
                if n_feas_actions <= 1:
                    break
                all_scores = torch.cat([s0[feas_u], s1[feas_v]], dim=0)
                if all_scores.numel() >= 2:
                    top2 = torch.topk(all_scores, k=2).values
                    margin = float((top2[0] - top2[1]).item())
                    thresh = adaptive_margins[0] if k == 0 else adaptive_margins[1]
                    if margin >= thresh:
                        break

        self.neural_flops += _calibrated_stage_c_flops(self.model.arch, m0, m1, len(outs))
        return outs


# ============================================================================
# 5. Incremental Rust-Bitset Acceleration for `sym_prov_readout` & `HybridSymRecFrescoNet`
# ============================================================================
@dataclass
class SymBitsetWordEntry:
    w_ids: torch.Tensor       # (L,)
    ub0: torch.Tensor         # (Lm1, 36)
    up_horizons: torch.Tensor # (3, Lm1, 36) for horizons k=1,2,3
    uf_horizons: torch.Tensor # (3, 4, Lm1, 36)
    pos_horizons: torch.Tensor # (4, L, 6) for horizons k=0,1,2,3
    pub_deg_norm: float


def compute_sym_bitset_word_entries(
    case, words: list[str], *, use_rust: bool = True
) -> tuple[list[SymBitsetWordEntry], int]:
    L = len(words[0])
    Lm1 = L - 1
    N = len(words)
    if use_rust and RUST_BACKEND.available:
        bits, rc = RUST_BACKEND.sym_prov_adj2_u64(words, case.rules, alpha=6, fp=FP_FIBERS, k_max=3)
        # bits: (N, 4, 5, Lm1) -> unpack
        unpacked = unpack_u64_bitsets_to_float36(bits)  # (N, 4, 5, Lm1, 36)
    else:
        rc = 0
        rows = []
        ors = [(u, v) for u, v in case.rules if u != v] + [(v, u) for u, v in case.rules if u != v]
        n_r2 = sum(1 for u, _ in ors if len(u) == 2)
        n_r3 = sum(1 for u, _ in ors if len(u) == 3)
        for w in words:
            rc += 3 * (n_r2 * Lm1 + n_r3 * max(0, Lm1 - 1))
            hist = m17.symbolic_prov4_adj2(case.rules, w, k_max=3, fp=FP_FIBERS)
            kh = []
            for k in range(4):
                b0, fibs = hist[k]
                kh.append(torch.stack([b0] + fibs, dim=0).reshape(FP_FIBERS + 1, Lm1, 36).float())
            rows.append(torch.stack(kh, dim=0))
        unpacked = torch.stack(rows, dim=0)

    # Derive positionwise marginals from unpacked (N, 4, 5, Lm1, 6, 6)
    u66 = unpacked.reshape(N, 4, FP_FIBERS + 1, Lm1, 6, 6).amax(dim=2)  # (N, 4, Lm1, 6, 6)
    left_s = F.pad(u66.amax(dim=-1), (0, 0, 0, 1))
    right_s = F.pad(u66.amax(dim=-2), (0, 0, 1, 0))
    pos_h = (torch.maximum(left_s, right_s) > 0.5).float()  # (N, 4, L, 6)

    entries = []
    for idx, w in enumerate(words):
        dg, ch = m17._cached_pub_matches(case.rules, w)
        rc += ch
        w_ids = torch.tensor([CHAR2ID[c] for c in w], dtype=torch.long)
        ub0 = unpacked[idx, 0, 0]            # (Lm1, 36)
        uf_h = unpacked[idx, 1:4, 1:5]       # (3, 4, Lm1, 36)
        up_h = (unpacked[idx, 1:4, :].amax(dim=1) > 0.5).float()  # Wait: note in LBF-17 sym_cell fiber 0..3 did NOT union base_a2!
        uf_only_max = uf_h.amax(dim=1)
        entries.append(SymBitsetWordEntry(
            w_ids=w_ids,
            ub0=ub0,
            up_horizons=uf_only_max,
            uf_horizons=uf_h,
            pos_horizons=pos_h[idx],
            pub_deg_norm=dg / 24.0,
        ))
    return entries, rc


class HybridSymRecFrescoNet(nn.Module):
    """Parameter-matched (`53,073` params) Hybrid Symbolic-Envelope / 3-Pass Recurrent-Readout Fresco.

    Uses exact zero-FLOP multi-horizon (`k = 1, 2, 3`) `u64`-bitset `prov4_adj2` + `pos1`
    envelopes as progressive horizon inputs into a 3-pass recurrent `FactoredAxialFrescoBlock`.
    """

    def __init__(self, dm: int = 38, passes: int = PASSES, init_gate_bias: float = -1.5):
        super().__init__()
        self.arch = "hybrid_sym_rec_opened"
        self.dm = dm
        self.passes = passes
        self.nonrecurrent = False
        # 14 S6-equivariant symbolic bridge features per horizon k=1,2,3 + learned pass scale
        self.ribbon_in = nn.Linear(14, dm)
        self.pass_emb = nn.Parameter(torch.zeros(passes, 19))
        self.pass_proj = nn.Linear(19, dm, bias=False)
        self.ribbon_conv1 = nn.Conv1d(dm, dm, kernel_size=3, padding=1)
        self.ribbon_conv2 = nn.Conv1d(dm, dm, kernel_size=3, padding=1)
        self.pair_in = nn.Linear(2 * dm + 12, dm)
        self.pair_prelude = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        self.fresco_gate = m16.MessageGate(dm, init_bias=init_gate_bias)
        self.fresco_block = m17.FactoredAxialFrescoBlock(dm)
        self.hyb_pair_ff = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 143), nn.GELU(), nn.Linear(143, dm))
        self.delta_scale = nn.Parameter(torch.tensor(0.35))
        self.pot_head = nn.Linear(dm, 1)
        self.node_head = nn.Sequential(nn.Linear(10, dm), nn.GELU(), nn.Linear(dm, 1))
        self.log_tau = nn.Parameter(torch.tensor(-0.7))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def pair_horizon_features(
        u_ent: list[SymBitsetWordEntry], v_ent: list[SymBitsetWordEntry]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(feat14 (3, du, dv, L, 14), mismatch, dh, lb, span)` from cached bitset entries."""
        du, dv = len(u_ent), len(v_ent)
        L = u_ent[0].w_ids.shape[0]
        Lm1 = L - 1
        u_ids = torch.stack([e.w_ids for e in u_ent], dim=0)
        v_ids = torch.stack([e.w_ids for e in v_ent], dim=0)
        mismatch = (u_ids[:, None, :] != v_ids[None, :, :])
        dh = mismatch.sum(dim=-1).float()
        lb = torch.div(dh.long() + 2, 3, rounding_mode="floor").float()
        pos_idx = torch.arange(L, dtype=torch.float32).view(1, 1, L)
        first_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
        last_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
        span = torch.clamp(last_mm - first_mm + 1.0, min=0.0)

        ub0 = torch.stack([e.ub0 for e in u_ent], dim=0)              # (du, Lm1, 36)
        vb0 = torch.stack([e.ub0 for e in v_ent], dim=0)              # (dv, Lm1, 36)
        up_h = torch.stack([e.up_horizons for e in u_ent], dim=0)     # (du, 3, Lm1, 36)
        vp_h = torch.stack([e.up_horizons for e in v_ent], dim=0)     # (dv, 3, Lm1, 36)
        uf_h = torch.stack([e.uf_horizons for e in u_ent], dim=0)     # (du, 3, 4, Lm1, 36)
        vf_h = torch.stack([e.uf_horizons for e in v_ent], dim=0)     # (dv, 3, 4, Lm1, 36)
        upos = torch.stack([e.pos_horizons for e in u_ent], dim=0)    # (du, 4, L, 6)
        vpos = torch.stack([e.pos_horizons for e in v_ent], dim=0)    # (dv, 4, L, 6)

        ov0 = torch.einsum("ipm,jpm->ijp", ub0, vb0)
        ov1 = torch.einsum("ipm,jpm->ijp", up_h[:, 0], vp_h[:, 0])
        u2v1 = torch.einsum("ipm,jpm->ijp", up_h[:, 0], vb0)
        v2u1 = torch.einsum("ipm,jpm->ijp", ub0, vp_h[:, 0])
        pos_ov0 = torch.einsum("ips,jps->ijp", upos[:, 0], vpos[:, 0])
        pos_ov1 = torch.einsum("ips,jps->ijp", upos[:, 1], vpos[:, 1])

        pass_feats = []
        mm_f = mismatch.float().unsqueeze(-1)
        for k in range(3):
            upk = up_h[:, k]
            vpk = vp_h[:, k]
            unov = F.relu(upk - up_h[:, 0]) if k > 0 else upk
            vnov = F.relu(vpk - vp_h[:, 0]) if k > 0 else vpk
            ovk = (torch.einsum("ipm,jpm->ijp", unov, vnov)
                   + 0.5 * torch.einsum("ipm,jpm->ijp", unov, vp_h[:, 0])
                   + 0.5 * torch.einsum("ipm,jpm->ijp", up_h[:, 0], vnov))
            u2vk = torch.einsum("ipm,jpm->ijp", upk, vb0)
            v2uk = torch.einsum("ipm,jpm->ijp", ub0, vpk)
            fib_pk = torch.einsum("ifpm,jgpm->ijfgp", uf_h[:, k], vf_h[:, k]).amax(dim=(2, 3))
            supp_k = 0.5 * (upk.mean(dim=-1)[:, None, :] + vpk.mean(dim=-1)[None, :, :])
            edge8 = torch.stack([ov0, ov1, ovk, u2v1, v2u1, u2vk, v2uk, fib_pk], dim=-1)
            edge9 = torch.cat([edge8, supp_k.unsqueeze(-1)], dim=-1)
            node9 = m17.RelationalProvenanceFrescoNet.pad_edges_to_nodes(edge9)

            pos_ovk = torch.einsum("ips,jps->ijp", upos[:, k + 1], vpos[:, k + 1])
            pos_u2vk = torch.einsum("ips,jps->ijp", upos[:, k + 1], vpos[:, 0])
            pos_v2uk = torch.einsum("ips,jps->ijp", upos[:, 0], vpos[:, k + 1])
            pos4 = torch.stack([pos_ov0, pos_ov1, pos_ovk, 0.5 * (pos_u2vk + pos_v2uk)], dim=-1)
            pass_feats.append(torch.cat([node9, pos4, mm_f], dim=-1))
        return torch.stack(pass_feats, dim=0), mismatch, dh, lb, span

    def forward_from_pools(
        self, case, s: m15.FringeState, rem_budget: int,
        u_ent: list[SymBitsetWordEntry], v_ent: list[SymBitsetWordEntry],
        rib_pools: torch.Tensor, dh: torch.Tensor, lb: torch.Tensor, span: torch.Tensor,
        *, depth_cap: int = DEPTH, mode: str = "normal",
    ) -> list[dict]:
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)

        g0 = torch.tensor([s.dist[0][w] for w in u_words], dtype=torch.float32)[:, None]
        g1 = torch.tensor([s.dist[1][w] for w in v_words], dtype=torch.float32)[None, :]
        tot_lb = g0 + g1 + torch.clamp(lb, min=1.0)
        infeas = tot_lb > depth_cap

        pers = torch.zeros((m0, m1), dtype=torch.float32)
        prev_v = torch.zeros((m0, m1), dtype=torch.float32)
        if s.prev_pot:
            for i, uw in enumerate(u_words):
                for j, vw in enumerate(v_words):
                    if (uw, vw) in s.prev_pot:
                        pers[i, j] = 1.0
                        prev_v[i, j] = s.prev_pot[(uw, vw)]

        b_frac = rem_budget / WORK
        lm_diff = (math.log1p(m0) - math.log1p(m1)) / math.log1p(WORK)
        pub = torch.stack([
            (g0 / depth_cap).expand(m0, m1),
            (g1 / depth_cap).expand(m0, m1),
            ((g0 + g1) / depth_cap).expand(m0, m1),
            dh / L, lb / depth_cap,
            torch.clamp(depth_cap - tot_lb, min=0.0) / depth_cap,
            1.0 - infeas.float(), span / L,
            torch.full((m0, m1), b_frac),
            torch.full((m0, m1), lm_diff),
            pers, prev_v,
        ], dim=-1)

        def node_feat(entries: list[SymBitsetWordEntry], words: list[str], side: int, k: int) -> torch.Tensor:
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            g_ids = torch.tensor([CHAR2ID[c] for c in goal], dtype=torch.long)
            w_ids = torch.stack([e.w_ids for e in entries], dim=0)
            dh_goal = (w_ids != g_ids[None, :]).sum(dim=-1).float() / L
            degs = torch.tensor([e.pub_deg_norm for e in entries], dtype=torch.float32)
            gs = torch.tensor([s.dist[side][w] / depth_cap for w in words], dtype=torch.float32)
            s1_cov = torch.tensor([float(e.up_horizons[0].mean().item()) for e in entries], dtype=torch.float32)
            sk_cov = torch.tensor([float(e.up_horizons[k].mean().item()) for e in entries], dtype=torch.float32)
            p1_cov = torch.tensor([float(e.pos_horizons[1].mean().item()) for e in entries], dtype=torch.float32)
            pk_cov = torch.tensor([float(e.pos_horizons[k + 1].mean().item()) for e in entries], dtype=torch.float32)
            return torch.stack([
                gs, degs,
                torch.full((ms,), math.log1p(ms) / 5.0),
                torch.full((ms,), math.log1p(mo) / 5.0),
                torch.full((ms,), b_frac),
                dh_goal, s1_cov, sk_cov, p1_cov, pk_cov,
            ], dim=-1)

        ablate_ax = (mode == "ablate_axial")
        mat_zero = mode in ("mat_zero", "env_mat_zero")
        mat_roll = mode in ("mat_roll", "env_mat_roll")
        bm = torch.zeros((m0, m1, self.dm), dtype=torch.float32)
        h_mat = pot_accum = None
        tau = F.softplus(self.log_tau) + 0.1
        feas_u = ~infeas.all(dim=1)
        feas_v = ~infeas.all(dim=0)
        outs = []

        for k in range(self.passes):
            mm_in = bm
            if k > 0:
                if mat_zero:
                    mm_in = torch.zeros_like(mm_in)
                elif mat_roll:
                    mm_in = mm_in.roll(shifts=(1, 1), dims=(0, 1))
            pbase = self.pair_in(torch.cat([rib_pools[:, :, k, :], pub], dim=-1))
            pbase = pbase + self.pair_prelude(pbase)
            if k == 0:
                gm, _ = self.fresco_gate(pbase, mm_in)
                h_mat, bm = self.fresco_block(pbase + gm, mm_in, infeas, ablate_axial=ablate_ax)
                h_mat = h_mat + self.hyb_pair_ff(h_mat)
                pot_accum = self.pot_head(h_mat).squeeze(-1)
            else:
                gm, _ = self.fresco_gate(h_mat, mm_in)
                h_mat, bm = self.fresco_block(h_mat + 0.5 * pbase + gm, mm_in, infeas, ablate_axial=ablate_ax)
                h_mat = h_mat + self.hyb_pair_ff(h_mat)
                pot_accum = pot_accum + self.delta_scale * self.pot_head(h_mat - pbase).squeeze(-1)
            pot = pot_accum
            pot_masked = pot.masked_fill(infeas, -12.0)
            s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
                  + self.node_head(node_feat(u_ent, u_words, 0, k)).squeeze(-1)).masked_fill(~feas_u, -12.0)
            s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
                  + self.node_head(node_feat(v_ent, v_words, 1, k)).squeeze(-1)).masked_fill(~feas_v, -12.0)
            outs.append({"pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas})
        return outs

    def compute_rib_pools(
        self, u_ent: list[SymBitsetWordEntry], v_ent: list[SymBitsetWordEntry]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat14, mismatch, dh, lb, span = self.pair_horizon_features(u_ent, v_ent)
        K, du, dv, L, _ = feat14.shape
        pshift = self.pass_proj(self.pass_emb[:K])[:, None, None, None, :]
        rib = (self.ribbon_in(feat14) + pshift).reshape(K * du * dv, L, self.dm).transpose(1, 2)
        rc = F.gelu(self.ribbon_conv2(F.gelu(self.ribbon_conv1(rib)))).transpose(1, 2).reshape(K, du, dv, L, self.dm)
        mm_f = mismatch.float().unsqueeze(0).unsqueeze(-1).expand(K, du, dv, L, 1)
        mm_cnt = mm_f.sum(dim=3).clamp_min(1.0)
        pools = torch.cat([(rc * mm_f).sum(dim=3) / mm_cnt, rc.amax(dim=3)], dim=-1)  # (K, du, dv, 2*dm)
        return pools.permute(1, 2, 0, 3).contiguous(), dh, lb, span

    def forward(
        self, case, s: m15.FringeState, rem_budget: int, *, mode: str = "normal",
        depth_cap: int = DEPTH, use_rust: bool = True,
    ) -> tuple[list[dict], int]:
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0 = len(u_words)
        all_ent, rc = compute_sym_bitset_word_entries(case, u_words + v_words, use_rust=use_rust)
        u_ent, v_ent = all_ent[:m0], all_ent[m0:]
        pools, dh, lb, span = self.compute_rib_pools(u_ent, v_ent)
        return self.forward_from_pools(case, s, rem_budget, u_ent, v_ent, pools, dh, lb, span, depth_cap=depth_cap, mode=mode), rc


class IncrementalSymController:
    """Incremental controller for `HybridSymRecFrescoNet` and frozen `sym_prov_readout`."""

    def __init__(self, model: nn.Module, checkpoint_sha: str, *, use_rust: bool = True):
        self.model = model
        self.checkpoint_sha = checkpoint_sha
        self.bound_signature = live_param_signature(model)
        self.use_rust = bool(use_rust and RUST_BACKEND.available)
        self.is_hybrid = isinstance(model, HybridSymRecFrescoNet)
        self.word_cache: dict[tuple[str, str, str], SymBitsetWordEntry] = {}
        self.pair_cache: dict[tuple[str, str, str, str], PairRibbonEntry] = {}
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.word_hits = 0
        self.word_misses = 0
        self.pair_hits = 0
        self.pair_misses = 0
        self.rule_window_checks = 0
        self.neural_flops = 0

    def clear_caches(self) -> None:
        self.word_cache.clear()
        self.pair_cache.clear()

    def forward(self, case, s: m15.FringeState, rem_budget: int, *, depth_cap: int = DEPTH) -> list[dict]:
        if live_param_signature(self.model) != self.bound_signature:
            raise CacheInvalidationError("symbolic-readout model mutated; cache reuse rejected")
        r_key = canonical_rules_key(case.rules)
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)

        missing_w = []
        seen_w = set()
        for w in u_words + v_words:
            wk = (self.checkpoint_sha, r_key, w)
            if wk in self.word_cache:
                self.word_hits += 1
            else:
                self.word_misses += 1
                if w not in seen_w:
                    seen_w.add(w)
                    missing_w.append(w)
        if missing_w:
            ents, rc = compute_sym_bitset_word_entries(case, missing_w, use_rust=self.use_rust)
            self.rule_window_checks += rc
            for w, ent in zip(missing_w, ents):
                self.word_cache[(self.checkpoint_sha, r_key, w)] = ent

        u_ent = [self.word_cache[(self.checkpoint_sha, r_key, w)] for w in u_words]
        v_ent = [self.word_cache[(self.checkpoint_sha, r_key, w)] for w in v_words]

        miss_rows, miss_cols, miss_coords = set(), set(), []
        for i, uw in enumerate(u_words):
            for j, vw in enumerate(v_words):
                pk = (self.checkpoint_sha, r_key, uw, vw)
                if pk in self.pair_cache:
                    self.pair_hits += 1
                else:
                    self.pair_misses += 1
                    miss_coords.append((i, j))
                    miss_rows.add(i)
                    miss_cols.add(j)

        K = self.model.passes if self.is_hybrid else 1
        if miss_coords:
            r_sorted = sorted(miss_rows)
            c_sorted = sorted(miss_cols)
            sub_u = [u_ent[i] for i in r_sorted]
            sub_v = [v_ent[j] for j in c_sorted]
            if self.is_hybrid:
                pools_blk, dh_blk, lb_blk, sp_blk = self.model.compute_rib_pools(sub_u, sub_v)
                pair_flop_est = len(r_sorted) * len(c_sorted) * K * L * (2 * 14 * 38 + 2 * 2 * 3 * 38 * 38)
            else:
                # Exact `sym_prov_readout` static ribbon pool
                m = self.model
                du_s, dv_s = len(sub_u), len(sub_v)
                u_ids = torch.stack([e.w_ids for e in sub_u], dim=0)
                v_ids = torch.stack([e.w_ids for e in sub_v], dim=0)
                mismatch = (u_ids[:, None, :] != v_ids[None, :, :])
                dh_blk = mismatch.sum(dim=-1).float()
                lb_blk = torch.div(dh_blk.long() + 2, 3, rounding_mode="floor").float()
                pos_idx = torch.arange(L, dtype=torch.float32).view(1, 1, L)
                fmm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
                lmm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
                sp_blk = torch.clamp(lmm - fmm + 1.0, min=0.0)

                ub0 = torch.stack([e.ub0 for e in sub_u], dim=0)
                vb0 = torch.stack([e.ub0 for e in sub_v], dim=0)
                up1 = torch.stack([e.up_horizons[0] for e in sub_u], dim=0)
                vp1 = torch.stack([e.up_horizons[0] for e in sub_v], dim=0)
                up3 = torch.stack([e.up_horizons[2] for e in sub_u], dim=0)
                vp3 = torch.stack([e.up_horizons[2] for e in sub_v], dim=0)
                uf3 = torch.stack([e.uf_horizons[2] for e in sub_u], dim=0)
                vf3 = torch.stack([e.uf_horizons[2] for e in sub_v], dim=0)
                unov = F.relu(up3 - up1)
                vnov = F.relu(vp3 - vp1)
                ov0 = torch.einsum("ipk,jpk->ijp", ub0, vb0).unsqueeze(-1)
                ov1 = torch.einsum("ipk,jpk->ijp", up1, vp1).unsqueeze(-1)
                ovk = (torch.einsum("ipk,jpk->ijp", unov, vnov)
                       + 0.5 * torch.einsum("ipk,jpk->ijp", unov, vp1)
                       + 0.5 * torch.einsum("ipk,jpk->ijp", up1, vnov)).unsqueeze(-1)
                u2v1 = torch.einsum("ipk,jpk->ijp", up1, vb0).unsqueeze(-1)
                v2u1 = torch.einsum("ipk,jpk->ijp", ub0, vp1).unsqueeze(-1)
                u2v3 = torch.einsum("ipk,jpk->ijp", up3, vb0).unsqueeze(-1)
                v2u3 = torch.einsum("ipk,jpk->ijp", ub0, vp3).unsqueeze(-1)
                fib_p = torch.einsum("ifpk,jgpk->ijfgp", uf3, vf3).amax(dim=(2, 3)).unsqueeze(-1)
                supp = 0.5 * (up3.mean(dim=-1)[:, None, :, None] + vp3.mean(dim=-1)[None, :, :, None])
                edge9 = torch.cat([ov0, ov1, ovk, u2v1, v2u1, u2v3, v2u3, fib_p, supp], dim=-1)
                node9 = m.pad_edges_to_nodes(edge9)
                sym_feat = torch.cat([node9, mismatch.float().unsqueeze(-1)], dim=-1)
                rib = m.ribbon_in(sym_feat).reshape(du_s * dv_s, L, m.dm).transpose(1, 2)
                rc_t = F.gelu(m.sym_rib_conv2(F.gelu(m.ribbon_conv(rib)))).transpose(1, 2).reshape(du_s, dv_s, L, m.dm)
                mm_m = mismatch.float().unsqueeze(-1)
                pools_blk = torch.cat([(rc_t * mm_m).sum(dim=2) / mm_m.sum(dim=2).clamp_min(1.0), rc_t.amax(dim=2)], dim=-1).unsqueeze(2)
                pair_flop_est = du_s * dv_s * L * (2 * 10 * 38 + 2 * 2 * 3 * 38 * 38)
            self.neural_flops += int(pair_flop_est)

            for ri, i in enumerate(r_sorted):
                uw = u_words[i]
                for ci, j in enumerate(c_sorted):
                    vw = v_words[j]
                    pk = (self.checkpoint_sha, r_key, uw, vw)
                    if pk not in self.pair_cache:
                        self.pair_cache[pk] = PairRibbonEntry(
                            passes_ready=K,
                            rib_pool=pools_blk[ri, ci],
                            dh=float(dh_blk[ri, ci].item()),
                            lb=float(lb_blk[ri, ci].item()),
                            span=float(sp_blk[ri, ci].item()),
                        )

        rib_grid = torch.empty((m0, m1, K, 2 * self.model.dm), dtype=torch.float32)
        dh = torch.empty((m0, m1), dtype=torch.float32)
        lb = torch.empty((m0, m1), dtype=torch.float32)
        span = torch.empty((m0, m1), dtype=torch.float32)
        for i, uw in enumerate(u_words):
            for j, vw in enumerate(v_words):
                pe = self.pair_cache[(self.checkpoint_sha, r_key, uw, vw)]
                rib_grid[i, j] = pe.rib_pool
                dh[i, j] = pe.dh
                lb[i, j] = pe.lb
                span[i, j] = pe.span

        if self.is_hybrid:
            outs = self.model.forward_from_pools(case, s, rem_budget, u_ent, v_ent, rib_grid, dh, lb, span, depth_cap=depth_cap)
            self.neural_flops += _calibrated_stage_c_flops("rel_prov_opened", m0, m1, len(outs))
            return outs

        # `sym_prov_readout` Stage C
        m = self.model
        g0 = torch.tensor([s.dist[0][w] for w in u_words], dtype=torch.float32)[:, None]
        g1 = torch.tensor([s.dist[1][w] for w in v_words], dtype=torch.float32)[None, :]
        tot_lb = g0 + g1 + torch.clamp(lb, min=1.0)
        infeas = tot_lb > depth_cap
        pers = torch.zeros((m0, m1), dtype=torch.float32)
        prev_v = torch.zeros((m0, m1), dtype=torch.float32)
        if s.prev_pot:
            for i, uw in enumerate(u_words):
                for j, vw in enumerate(v_words):
                    if (uw, vw) in s.prev_pot:
                        pers[i, j] = 1.0
                        prev_v[i, j] = s.prev_pot[(uw, vw)]
        b_frac = rem_budget / WORK
        lm_diff = (math.log1p(m0) - math.log1p(m1)) / math.log1p(WORK)
        pub = torch.stack([
            (g0 / depth_cap).expand(m0, m1), (g1 / depth_cap).expand(m0, m1),
            ((g0 + g1) / depth_cap).expand(m0, m1), dh / L, lb / depth_cap,
            torch.clamp(depth_cap - tot_lb, min=0.0) / depth_cap,
            1.0 - infeas.float(), span / L,
            torch.full((m0, m1), b_frac), torch.full((m0, m1), lm_diff), pers, prev_v,
        ], dim=-1)

        def sym_np(entries: list[SymBitsetWordEntry], words: list[str], side: int) -> torch.Tensor:
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            g_ids = torch.tensor([CHAR2ID[c] for c in goal], dtype=torch.long)
            w_ids = torch.stack([e.w_ids for e in entries], dim=0)
            dh_g = (w_ids != g_ids[None, :]).sum(dim=-1).float() / L
            degs = torch.tensor([e.pub_deg_norm for e in entries], dtype=torch.float32)
            gs = torch.tensor([s.dist[side][w] / depth_cap for w in words], dtype=torch.float32)
            return torch.stack([gs, degs, torch.full((ms,), math.log1p(ms) / 5.0),
                                torch.full((ms,), math.log1p(mo) / 5.0), torch.full((ms,), b_frac), dh_g], dim=-1)

        pbase = m.pair_in(torch.cat([rib_grid[:, :, 0, :], pub], dim=-1))
        pbase = pbase + m.pair_prelude(pbase)
        h_mat, _ = m.fresco_block(pbase, torch.zeros((m0, m1, m.dm)), infeas)
        h_mat = h_mat + m.sym_pair_ff(h_mat)
        pot = m.pot_head(h_mat).squeeze(-1)
        pot_masked = pot.masked_fill(infeas, -12.0)
        tau = F.softplus(m.log_tau) + 0.1
        s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
              + m.node_head(sym_np(u_ent, u_words, 0)).squeeze(-1)).masked_fill(infeas.all(dim=1), -12.0)
        s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
              + m.node_head(sym_np(v_ent, v_words, 1)).squeeze(-1)).masked_fill(infeas.all(dim=0), -12.0)
        self.neural_flops += _calibrated_stage_c_flops("param_nonrecurrent", m0, m1, 1)
        return [{"pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas}]


# ============================================================================
# 6. Clean Inference Episode Runner (Cold & Warm Cache, Zero Profiling Hooks)
# ============================================================================
def run_clean_episode(
    rg,
    case,
    *,
    controller: IncrementalFrescoController | IncrementalSymController | None = None,
    uncached_model: nn.Module | None = None,
    mode: str = "normal",
    budget: int = WORK,
    depth: int = DEPTH,
    cold_cache: bool = True,
    adaptive_margins: tuple[float, float] | None = None,
) -> dict:
    """Clean inference runner free of FlopCounterMode, tensor cloning, or sham hooks."""
    if controller is not None and cold_cache:
        controller.clear_caches()
    if controller is not None:
        w_hits_0, w_miss_0 = controller.word_hits, controller.word_misses
        p_hits_0, p_miss_0 = controller.pair_hits, controller.pair_misses
        rc_0, fl_0 = controller.rule_window_checks, controller.neural_flops
    else:
        w_hits_0 = w_miss_0 = p_hits_0 = p_miss_0 = rc_0 = fl_0 = 0

    s = m15.FringeState.start(case)
    calls = pair_evals = rec_passes = decisions = uncached_rc = uncached_flops = 0
    decision_latencies_ms: list[float] = []

    t_ep0 = time.perf_counter()
    with torch.no_grad():
        while s.live(budget):
            decisions += 1
            m0, m1 = len(s.fronts[0]), len(s.fronts[1])
            rem = budget - s.work
            t_d0 = time.perf_counter()
            if controller is not None:
                if isinstance(controller, IncrementalFrescoController):
                    outs = controller.forward(case, s, rem, mode=mode, depth_cap=depth, adaptive_margins=adaptive_margins)
                else:
                    outs = controller.forward(case, s, rem, depth_cap=depth)
            else:
                if uncached_model is None:
                    raise ValueError("either controller or uncached_model is required")
                outs, sc = uncached_model(case, s, rem, mode=mode, depth_cap=depth)
                uncached_rc += sc
                wf, pf = _calibrated_stage_flops(uncached_model.arch, len(case.source), uncached_model.passes)
                uncached_flops += (m0 + m1) * wf + (m0 * m1) * pf + _calibrated_stage_c_flops(
                    uncached_model.arch, m0, m1, len(outs)
                )
            decision_latencies_ms.append((time.perf_counter() - t_d0) * 1000.0)

            calls += 1
            rec_passes += len(outs)
            pair_evals += m0 * m1
            last = outs[-1]
            psig = last["pot"].sigmoid()
            pm = float(psig.mean().item())
            s.prev_pot = {
                (u, v): float(psig[i, j].item()) - pm
                for i, u in enumerate(s.fronts[0])
                for j, v in enumerate(s.fronts[1])
            }
            s0_l = last["s0"].tolist()
            s1_l = last["s1"].tolist()
            cands = [(s0_l[i], -m0, 0, -i) for i in range(m0)] + [(s1_l[j], -m1, -1, -j) for j in range(m1)]
            best = max(cands)
            side = 0 if best[2] == 0 else 1
            idx = -best[3]
            m15.expand_node(rg, case, s, side, idx, budget, depth)
    elapsed_ms = (time.perf_counter() - t_ep0) * 1000.0
    cur_rss, peak_rss = current_and_peak_rss_bytes()

    res = rg._result(case, "learned", s.path, s.work, s.expanded, calls, budget, depth)
    if controller is not None:
        wh = controller.word_hits - w_hits_0
        wm = controller.word_misses - w_miss_0
        ph = controller.pair_hits - p_hits_0
        pmiss = controller.pair_misses - p_miss_0
        rc_ep = controller.rule_window_checks - rc_0
        fl_ep = controller.neural_flops - fl_0
    else:
        wh = ph = 0
        wm = sum(1 for _ in range(calls))  # not cached
        pmiss = pair_evals
        rc_ep = uncached_rc
        fl_ep = uncached_flops

    return {
        **res,
        "pair_evaluations": pair_evals,
        "substring_checks": rc_ep,
        "recurrent_passes": rec_passes,
        "fringe_decisions": decisions,
        "neural_flops": fl_ep,
        "wall_clock_ms": elapsed_ms,
        "decision_latencies_ms": decision_latencies_ms,
        "word_hits": wh,
        "word_misses": wm,
        "pair_hits": ph,
        "pair_misses": pmiss,
        "rss_bytes": cur_rss,
        "peak_rss_bytes": peak_rss,
    }


# ============================================================================
# 7. Train-Only Hybrid Fitting & Train-Only Adaptive Threshold Calibration
# ============================================================================
def fit_hybrid_sym_rec_model(rg, train_frames: list[dict], draws_rows: list[list[int]], seed: int) -> tuple[HybridSymRecFrescoNet, dict]:
    """Fit `HybridSymRecFrescoNet` strictly on `20261227/train` using the exact LBF-17 draw schedule."""
    torch.manual_seed(seed)
    model = HybridSymRecFrescoNet(dm=38, passes=PASSES, init_gate_bias=-1.5).cpu().float()
    pcount = model.parameter_count()
    if not (53000 <= pcount <= 53200):
        raise ValueError(f"HybridSymRecFrescoNet parameter count {pcount} outside [53000, 53200]")
    init_fp = base._model_fingerprint(model)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(draws_rows), eta_min=2e-4)

    # Pre-cache train-frame bitset word envelopes once (zero leakage, pure deterministic rules+word)
    word_bank: dict[tuple[str, str], SymBitsetWordEntry] = {}
    t0 = time.perf_counter()
    model.train()
    pw_weights = [0.20, 0.30, 0.50]

    for step_idx, batch_indices in enumerate(draws_rows):
        opt.zero_grad()
        bsz = max(1, len(batch_indices))
        for fi in batch_indices:
            fr = train_frames[fi]
            c = fr["case"]
            s = fr["state"]
            rk = canonical_rules_key(c.rules)
            miss = [w for w in (s.fronts[0] + s.fronts[1]) if (rk, w) not in word_bank]
            if miss:
                ents, _ = compute_sym_bitset_word_entries(c, list(dict.fromkeys(miss)), use_rust=True)
                for w, ent in zip(dict.fromkeys(miss), ents):
                    word_bank[(rk, w)] = ent
            u_ent = [word_bank[(rk, w)] for w in s.fronts[0]]
            v_ent = [word_bank[(rk, w)] for w in s.fronts[1]]
            pools, dh, lb, sp = model.compute_rib_pools(u_ent, v_ent)
            outs = model.forward_from_pools(c, s, fr["rem_budget"], u_ent, v_ent, pools, dh, lb, sp)

            tgt_act = fr["action_targets"]
            tgt_pair = fr["pair_targets"]
            loss_item = torch.zeros((), dtype=torch.float32)
            for k, out in enumerate(outs):
                scores = torch.cat([out["s0"], out["s1"]], dim=0)
                log_p = F.log_softmax(scores, dim=0)
                kl = -(tgt_act * log_p).sum()
                feas = ~out["infeas"]
                if bool(feas.any()):
                    bce = F.binary_cross_entropy_with_logits(out["pot"][feas], tgt_pair[feas])
                else:
                    bce = torch.zeros((), dtype=torch.float32)
                loss_item = loss_item + pw_weights[k] * (kl + 0.35 * bce)
            (loss_item / bsz).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    fit_sec = time.perf_counter() - t0
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    final_fp = base._model_fingerprint(model)
    return model, {
        "arch": "hybrid_sym_rec_opened",
        "seed": seed,
        "parameters": pcount,
        "passes": PASSES,
        "initial": init_fp,
        "final": final_fp,
        "fit_wall_clock_sec": round(fit_sec, 3),
    }


def calibrate_adaptive_margins_on_train(
    models_by_seed: dict[int, tuple[m17.RelationalProvenanceFrescoNet, str]],
    train_frames: list[dict],
    target_agreement: float = 0.985,
) -> dict:
    """Select `(gamma1, gamma2)` strictly on `20261227/train` frames to preserve >=98.5% Pass-3 top-action agreement."""
    rec1, rec2 = [], []
    with torch.no_grad():
        for seed, (model, ckpt_sha) in sorted(models_by_seed.items()):
            ctrl = IncrementalFrescoController(model, ckpt_sha, use_rust=True)
            for fr in train_frames:
                c, s, rem = fr["case"], fr["state"], fr["rem_budget"]
                outs = ctrl.forward(c, s, rem, mode="normal")
                infeas = outs[-1]["infeas"]
                fu = ~infeas.all(dim=1)
                fv = ~infeas.all(dim=0)
                if int(fu.sum().item() + fv.sum().item()) <= 1:
                    continue
                sc3 = torch.cat([outs[2]["s0"], outs[2]["s1"]], dim=0)
                top3 = int(sc3.argmax().item())
                for k, bucket in ((0, rec1), (1, rec2)):
                    sck = torch.cat([outs[k]["s0"], outs[k]["s1"]], dim=0)
                    topk = int(sck.argmax().item())
                    feas_vals = torch.cat([outs[k]["s0"][fu], outs[k]["s1"][fv]], dim=0)
                    t2 = torch.topk(feas_vals, k=2).values
                    margin = float((t2[0] - t2[1]).item())
                    bucket.append((margin, topk == top3))

    def pick_gamma(records: list[tuple[float, bool]], default_hi: float = 2.5) -> tuple[float, float, float]:
        candidates = sorted({round(m, 3) for m, _ in records})
        for g in candidates:
            sub = [ok for m, ok in records if m >= g]
            if len(sub) >= max(15, int(0.10 * len(records))):
                acc = sum(sub) / len(sub)
                if acc >= target_agreement:
                    cov = len(sub) / max(1, len(records))
                    return float(g), float(acc), float(cov)
        return default_hi, 1.0, 0.0

    g1, acc1, cov1 = pick_gamma(rec1, default_hi=2.0)
    g2, acc2, cov2 = pick_gamma(rec2, default_hi=1.2)
    return {
        "gamma1": g1,
        "gamma2": g2,
        "train_target_agreement": target_agreement,
        "pass1_halt_agreement_train": round(acc1, 5),
        "pass1_halt_coverage_train": round(cov1, 5),
        "pass2_halt_agreement_train": round(acc2, 5),
        "pass2_halt_coverage_train": round(cov2, 5),
        "evaluated_train_frame_seed_pairs": len(rec1),
    }


# ============================================================================
# 8. Root-Level (`R` Independent Roots vs `S` Seeds) & 2x2 Factorial Analytics
# ============================================================================
def root_level_summary(
    roots: list[str],
    episodes_by_seed: dict[int, dict[str, dict]],
    budget: int = WORK,
) -> dict:
    """Distinguish `R` independent theorem roots from `S` repeated training-seed evaluations."""
    R = len(roots)
    seeds = sorted(episodes_by_seed.keys())
    S = len(seeds)
    c_mat = [[1.0 if episodes_by_seed[s][r]["closed"] else 0.0 for s in seeds] for r in roots]
    w_mat = [
        [
            float(episodes_by_seed[s][r]["work"] + (0 if episodes_by_seed[s][r]["closed"] else 1))
            for s in seeds
        ]
        for r in roots
    ]
    root_c = [sum(row) / S for row in c_mat]
    root_w = [sum(row) / S for row in w_mat]
    all_seeds_closed = sum(1 for row in c_mat if all(v == 1.0 for v in row))
    any_seed_closed = sum(1 for row in c_mat if any(v == 1.0 for v in row))

    # One-way random-effects ANOVA variance decomposition for penalized work across R roots x S seeds
    grand_w = sum(root_w) / max(1, R)
    if R > 1 and S > 1:
        ss_within = sum(sum((w_mat[i][j] - root_w[i]) ** 2 for j in range(S)) for i in range(R))
        ms_within = ss_within / (R * (S - 1))
        ss_between = S * sum((root_w[i] - grand_w) ** 2 for i in range(R))
        ms_between = ss_between / (R - 1)
        var_root = max(0.0, (ms_between - ms_within) / S)
        var_seed_within = ms_within
        tot_var = var_root + var_seed_within
        icc = (var_root / tot_var) if tot_var > 1e-12 else 0.0
    else:
        var_root = var_seed_within = icc = 0.0
    n_eff = (R * S) / (1.0 + (S - 1) * icc) if S > 0 else float(R)

    rng = random.Random(20261801)
    boot_c, boot_w = [], []
    for _ in range(2000):
        idxs = [rng.randrange(R) for _ in range(R)]
        boot_c.append(sum(root_c[i] for i in idxs) / R)
        boot_w.append(sum(root_w[i] for i in idxs) / R)
    boot_c.sort()
    boot_w.sort()

    return {
        "independent_roots": R,
        "seeds_per_root": S,
        "total_seed_episodes": R * S,
        "mean_closure_rate": sum(root_c) / R,
        "closure_root_95ci": [boot_c[int(0.025 * 2000)], boot_c[int(0.975 * 2000)]],
        "roots_closed_all_seeds": all_seeds_closed,
        "roots_closed_all_seeds_rate": all_seeds_closed / R,
        "roots_closed_any_seed": any_seed_closed,
        "roots_closed_any_seed_rate": any_seed_closed / R,
        "mean_penalized_work": grand_w,
        "penalized_work_root_95ci": [boot_w[int(0.025 * 2000)], boot_w[int(0.975 * 2000)]],
        "between_root_work_var": var_root,
        "within_root_seed_work_var": var_seed_within,
        "work_icc": icc,
        "effective_sample_size_n_eff": n_eff,
        "per_seed_closure_rate": {str(s): sum(c_mat[i][j] for i in range(R)) / R for j, s in enumerate(seeds)},
        "per_seed_penalized_work": {str(s): sum(w_mat[i][j] for i in range(R)) / R for j, s in enumerate(seeds)},
        "_root_c": root_c,
        "_root_w": root_w,
    }


def paired_root_contrast(
    roots: list[str],
    root_c_a: list[float],
    root_w_a: list[float],
    root_c_b: list[float],
    root_w_b: list[float],
) -> dict:
    """Paired contrast `A - B` strictly over the `R` independent theorem roots (plus sign-flip p-value)."""
    R = len(roots)
    dc = [root_c_a[i] - root_c_b[i] for i in range(R)]
    dw = [root_w_a[i] - root_w_b[i] for i in range(R)]
    mc = sum(dc) / R
    mw = sum(dw) / R

    rng = random.Random(20261802)
    bc, bw = [], []
    for _ in range(4000):
        idxs = [rng.randrange(R) for _ in range(R)]
        bc.append(sum(dc[i] for i in idxs) / R)
        bw.append(sum(dw[i] for i in idxs) / R)
    bc.sort()
    bw.sort()

    # Paired sign-flip randomization p-value (two-sided, 10,000 draws; exact if R <= 16!)
    if R <= 16:
        tot = 1 << R
        ext_c = ext_w = 0
        for mask in range(tot):
            sc = sw = 0.0
            for i in range(R):
                sgn = 1.0 if ((mask >> i) & 1) else -1.0
                sc += sgn * dc[i]
                sw += sgn * dw[i]
            if abs(sc / R) >= abs(mc) - 1e-12:
                ext_c += 1
            if abs(sw / R) >= abs(mw) - 1e-12:
                ext_w += 1
        p_c = ext_c / tot
        p_w = ext_w / tot
    else:
        n_perm = 10000
        ext_c = ext_w = 1
        for _ in range(n_perm - 1):
            sc = sw = 0.0
            for i in range(R):
                sgn = 1.0 if rng.getrandbits(1) else -1.0
                sc += sgn * dc[i]
                sw += sgn * dw[i]
            if abs(sc / R) >= abs(mc) - 1e-12:
                ext_c += 1
            if abs(sw / R) >= abs(mw) - 1e-12:
                ext_w += 1
        p_c = ext_c / n_perm
        p_w = ext_w / n_perm

    wins = sum(1 for i in range(R) if dc[i] > 1e-9 or (abs(dc[i]) <= 1e-9 and dw[i] < -1e-9))
    losses = sum(1 for i in range(R) if dc[i] < -1e-9 or (abs(dc[i]) <= 1e-9 and dw[i] > 1e-9))
    ties = R - wins - losses
    return {
        "independent_roots": R,
        "closure_mean": mc,
        "closure_95ci": [bc[int(0.025 * 4000)], bc[int(0.975 * 4000)]],
        "closure_signflip_p": p_c,
        "penalized_work_mean": mw,
        "penalized_work_95ci": [bw[int(0.025 * 4000)], bw[int(0.975 * 4000)]],
        "penalized_work_signflip_p": p_w,
        "root_wins_work_or_closure": wins,
        "root_losses_work_or_closure": losses,
        "root_ties": ties,
    }


def analyze_2x2_bank_factorial(
    roots: list[str],
    y11_r: list[float],
    y01_r: list[float],
    y10_r: list[float],
    y00_r: list[float],
    *,
    higher_is_better: bool = True,
) -> dict:
    """Exact 2x2 factorial interaction & Shapley decomposition over `R` independent root values."""
    R = len(roots)
    sgn = 1.0 if higher_is_better else -1.0
    # Convert to utility orientation where larger = better (`+AUC` or `-PenalizedWork`)
    u11 = [sgn * v for v in y11_r]
    u01 = [sgn * v for v in y01_r]
    u10 = [sgn * v for v in y10_r]
    u00 = [sgn * v for v in y00_r]

    def stats(u11_s, u01_s, u10_s, u00_s):
        m11 = sum(u11_s) / R
        m01 = sum(u01_s) / R
        m10 = sum(u10_s) / R
        m00 = sum(u00_s) / R
        joint = m11 - m00
        tau_e1 = m11 - m01
        tau_m1 = m11 - m10
        tau_e0 = m10 - m00
        tau_m0 = m01 - m00
        inter = m11 - m01 - m10 + m00
        phi_e = 0.5 * (tau_e1 + tau_e0)
        phi_m = 0.5 * (tau_m1 + tau_m0)
        return joint, tau_e1, tau_m1, tau_e0, tau_m0, inter, phi_e, phi_m

    joint, te1, tm1, te0, tm0, inter, pe, pm = stats(u11, u01, u10, u00)
    rng = random.Random(20261803)
    b_joint, b_inter, b_pe, b_pm = [], [], [], []
    for _ in range(4000):
        ix = [rng.randrange(R) for _ in range(R)]
        j_i, _, _, _, _, in_i, pe_i, pm_i = stats(
            [u11[k] for k in ix], [u01[k] for k in ix], [u10[k] for k in ix], [u00[k] for k in ix]
        )
        b_joint.append(j_i)
        b_inter.append(in_i)
        b_pe.append(pe_i)
        b_pm.append(pm_i)
    for arr in (b_joint, b_inter, b_pe, b_pm):
        arr.sort()

    naive_denom = te1 + tm1
    naive_env_share = (te1 / naive_denom) if abs(naive_denom) > 1e-12 else 0.5
    shapley_env_share = (pe / joint) if abs(joint) > 1e-12 else 0.5
    return {
        "cell_means_raw": {
            "E1_M1_intact": sum(y11_r) / R,
            "E0_M1_env_ablated": sum(y01_r) / R,
            "E1_M0_mat_ablated": sum(y10_r) / R,
            "E0_M0_both_ablated": sum(y00_r) / R,
        },
        "joint_recurrence_benefit": joint,
        "joint_95ci": [b_joint[int(0.025 * 4000)], b_joint[int(0.975 * 4000)]],
        "marginal_drop_env_from_intact": te1,
        "marginal_drop_mat_from_intact": tm1,
        "simple_gain_env_alone": te0,
        "simple_gain_mat_alone": tm0,
        "interaction_I_EM": inter,
        "interaction_95ci": [b_inter[int(0.025 * 4000)], b_inter[int(0.975 * 4000)]],
        "shapley_env": pe,
        "shapley_env_95ci": [b_pe[int(0.025 * 4000)], b_pe[int(0.975 * 4000)]],
        "shapley_mat": pm,
        "shapley_mat_95ci": [b_pm[int(0.025 * 4000)], b_pm[int(0.975 * 4000)]],
        "shapley_env_share": shapley_env_share,
        "shapley_mat_share": 1.0 - shapley_env_share,
        "naive_marginal_env_share": naive_env_share,
        "interaction_regime": (
            "superadditive_synergy" if inter > 0 else ("subadditive_redundancy" if inter < 0 else "exact_additive")
        ),
    }


# ============================================================================
# 9. Multi-Axis Scaling & Adversarial Provenance-Collision Stress Suite (Item 8)
# ============================================================================
def general_concrete_k_ball(rules: tuple[tuple[str, str], ...], w0: str, k_max: int, max_states: int = 8192) -> tuple[set[str], bool]:
    """Bounded concrete <=k_max reachable words for arbitrary alphabet, length, and rule widths."""
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    seen = {w0: 0}
    q = deque([w0])
    truncated = False
    L = len(w0)
    while q:
        u = q.popleft()
        d = seen[u]
        if d >= k_max:
            continue
        for lhs, rhs in ors:
            ll = len(lhs)
            start = 0
            while True:
                p = u.find(lhs, start)
                if p < 0:
                    break
                start = p + 1
                v = u[:p] + rhs + u[p + ll:]
                if v not in seen:
                    seen[v] = d + 1
                    if len(seen) >= max_states:
                        truncated = True
                        break
                    q.append(v)
            if truncated:
                break
        if truncated:
            break
    return set(seen.keys()), truncated


def general_prov_fp_adj2_pos_support(
    rules: tuple[tuple[str, str], ...], w0: str, *, alpha_str: str, fp: int, k_max: int
) -> torch.Tensor:
    """Exact `(L, A)` marginal symbol support of `prov_{fp} x adj2` for arbitrary `A`, `L`, rule widths, and `fp`."""
    A = len(alpha_str)
    cmap = {ch: i for i, ch in enumerate(alpha_str)}
    L = len(w0)
    Lm1 = L - 1
    if A <= 8 and L <= 64 and fp <= 32 and RUST_BACKEND.available:
        bits, _ = RUST_BACKEND.sym_prov_adj2_u64([w0], rules, alpha=A, fp=fp, k_max=k_max, char_map=cmap)
        # bits[0, k_max]: (fp + 1, Lm1) u64
        union_mask = torch.zeros((Lm1,), dtype=torch.int64)
        for f in range(fp + 1):
            union_mask |= bits[0, k_max, f]
        p_supp = torch.zeros((L, A), dtype=torch.bool)
        for p in range(Lm1):
            m = int(union_mask[p].item())
            for x in range(A):
                for y in range(A):
                    if (m >> (x * A + y)) & 1:
                        p_supp[p, x] = True
                        p_supp[p + 1, y] = True
        return p_supp

    # General tensor path for alphabets A > 8 (e.g. A = 12)
    def tf_adj2(a2: torch.Tensor) -> torch.Tensor:
        fwd = torch.zeros((Lm1 + 1, A), dtype=torch.bool)
        fwd[0] = a2[0].any(dim=1)
        for p in range(Lm1):
            fwd[p + 1] = (fwd[p, :, None] & a2[p]).any(dim=0)
        bwd = torch.zeros((Lm1 + 1, A), dtype=torch.bool)
        bwd[Lm1] = a2[Lm1 - 1].any(dim=0)
        for p in range(Lm1 - 1, -1, -1):
            bwd[p] = (a2[p] & bwd[p + 1, None, :]).any(dim=1)
        return fwd[:-1, :, None] & a2 & bwd[1:, None, :]

    def word_a2(w: str) -> torch.Tensor:
        t = torch.zeros((Lm1, A, A), dtype=torch.bool)
        for p in range(Lm1):
            t[p, cmap[w[p]], cmap[w[p + 1]]] = True
        return t

    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    base_a2 = word_a2(w0)
    fibs = [torch.zeros((Lm1, A, A), dtype=torch.bool) for _ in range(fp)]
    active = [False] * fp
    for lhs, rhs in ors:
        ll = len(lhs)
        for p in range(L - ll + 1):
            if w0[p:p + ll] == lhs:
                f = p % fp
                fibs[f] |= word_a2(w0[:p] + rhs + w0[p + ll:])
                active[f] = True
    fibs = [tf_adj2(fibs[f]) if active[f] else fibs[f] for f in range(fp)]
    for step_idx in range(1, k_max):
        nf = []
        for f in range(fp):
            if not active[f]:
                nf.append(fibs[f])
                continue
            ta = tf_adj2(fibs[f])
            nxt = ta.clone()
            for lhs, rhs in ors:
                ll = len(lhs)
                xs = [cmap[c] for c in lhs]
                ys = [cmap[c] for c in rhs]
                for p in range(L - ll + 1):
                    if step_idx == 1 and (p % fp == f) and w0[p:p + ll] == rhs:
                        continue
                    if all(bool(ta[p + q, xs[q], xs[q + 1]]) for q in range(ll - 1)):
                        for q in range(ll - 1):
                            nxt[p + q, ys[q], ys[q + 1]] = True
                        if p > 0:
                            nxt[p - 1, ta[p - 1, :, xs[0]], ys[0]] = True
                        re = p + ll - 1
                        if re < Lm1:
                            nxt[re, ys[-1], ta[re, xs[-1], :]] = True
            nf.append(tf_adj2(nxt))
        fibs = nf
    merged = base_a2.clone()
    for fa in fibs:
        merged |= fa
    p_supp = torch.zeros((L, A), dtype=torch.bool)
    p_supp[:-1] |= merged.any(dim=2)
    p_supp[1:] |= merged.any(dim=1)
    return p_supp


def run_stress_test_suite(seed: int = 20261818) -> dict:
    """Systematic 5-track stress test locating exact breakdown thresholds of F_p = 4."""
    rng = random.Random(seed)
    fp_grid = (1, 2, 4, 6, 8)

    def eval_random_family(n_cases: int, L: int, alpha_n: int, rule_lens: tuple[int, ...], k_max: int) -> dict:
        alpha_s = EXTENDED_ALPHABET[:alpha_n]
        tp = {fp: 0 for fp in fp_grid}
        fp_cnt = {fp: 0 for fp in fp_grid}
        fn_cnt = {fp: 0 for fp in fp_grid}
        pos1_fp = 0
        used = 0
        attempts = 0
        while used < n_cases and attempts < n_cases * 25:
            attempts += 1
            used_sides = set()
            rules = []
            for rl in rule_lens:
                for _ in range(4):
                    for _ in range(40):
                        a = "".join(rng.choice(alpha_s) for _ in range(rl))
                        b = "".join(rng.choice(alpha_s) for _ in range(rl))
                        if a != b and a not in used_sides and b not in used_sides:
                            used_sides.update((a, b))
                            rules.append((a, b))
                            break
            w0 = "".join(rng.choice(alpha_s) for _ in range(L))
            # Plant 2 active 1-step redexes so reachability is nontrivial
            if len(rules) >= 2 and L >= 2 * max(rule_lens) + 2:
                p0 = rng.randrange(0, L // 2 - len(rules[0][0]) + 1)
                w0 = w0[:p0] + rules[0][0] + w0[p0 + len(rules[0][0]):]
                p1 = rng.randrange(L // 2, L - len(rules[1][0]) + 1)
                w0 = w0[:p1] + rules[1][0] + w0[p1 + len(rules[1][0]):]
            rules_t = tuple(rules)
            cset, trunc = general_concrete_k_ball(rules_t, w0, k_max=k_max, max_states=4096)
            if trunc or len(cset) <= 1:
                continue
            used += 1
            cmap = {ch: i for i, ch in enumerate(alpha_s)}
            true_pos = torch.zeros((L, alpha_n), dtype=torch.bool)
            for w in cset:
                for p, ch in enumerate(w):
                    true_pos[p, cmap[ch]] = True
            # Pos1 baseline
            p1_s = true_pos.new_zeros((L, alpha_n))
            for p, ch in enumerate(w0):
                p1_s[p, cmap[ch]] = True
            ors = [(u, v) for u, v in rules_t] + [(v, u) for u, v in rules_t]
            for _ in range(k_max):
                nxt = p1_s.clone()
                for lhs, rhs in ors:
                    ll = len(lhs)
                    xs = [cmap[c] for c in lhs]
                    ys = [cmap[c] for c in rhs]
                    for p in range(L - ll + 1):
                        if all(bool(p1_s[p + q, xs[q]]) for q in range(ll)):
                            for q in range(ll):
                                nxt[p + q, ys[q]] = True
                p1_s = nxt
            pos1_fp += int((p1_s & ~true_pos).sum().item())

            for fval in fp_grid:
                ps = general_prov_fp_adj2_pos_support(rules_t, w0, alpha_str=alpha_s, fp=fval, k_max=k_max)
                tp[fval] += int((ps & true_pos).sum().item())
                fp_cnt[fval] += int((ps & ~true_pos).sum().item())
                fn_cnt[fval] += int((~ps & true_pos).sum().item())

        true_tot = tp[4] + fn_cnt[4]
        return {
            "evaluated_roots": used,
            "L": L,
            "alphabet_size": alpha_n,
            "rule_lengths": list(rule_lens),
            "horizon_k": k_max,
            "pos1_precision": float(true_tot / max(1, true_tot + pos1_fp)),
            "pos1_fp": pos1_fp,
            "precision_by_fp": {
                str(f): round(tp[f] / max(1, tp[f] + fp_cnt[f]), 5) for f in fp_grid
            },
            "fp_count_by_fp": {str(f): fp_cnt[f] for f in fp_grid},
            "fn_count_by_fp": {str(f): fn_cnt[f] for f in fp_grid},
        }

    # Track S5: Adversarial modulo-4 provenance collision generator (`p2 = p1 + 4`, overlapping width-5 or secondary fork)
    def eval_adversarial_mod4_suite(n_cases: int = 24) -> dict:
        alpha_s = EXTENDED_ALPHABET[:6]
        cmap = {ch: i for i, ch in enumerate(alpha_s)}
        tp = {fp: 0 for fp in fp_grid}
        fp_cnt = {fp: 0 for fp in fp_grid}
        fn_cnt = {fp: 0 for fp in fp_grid}
        chimera_bridge_hits = {fp: 0 for fp in fp_grid}
        for idx in range(n_cases):
            L = 13
            p1 = idx % 3  # p1 in {0, 1, 2}, p2 = p1 + 4 ( ALWAYS p1 ≡ p2 mod 4! )
            p2 = p1 + 4
            # Construct unique contention symbol 'e' at p2, flanked by 'b'/'c', with downstream detector rule on ('d','f','d')
            base_chars = [rng.choice("abc") for _ in range(L)]
            base_chars[p1:p1 + 5] = ["a", "b", "b", "b", "e"]
            base_chars[p2:p2 + 5] = ["e", "b", "b", "b", "c"]
            w0 = "".join(base_chars)
            rules_t = (
                ("abbbe", "dbbbf"),
                ("ebbbc", "fbbbd"),
                ("dbf", "eee"),
            )
            cset, _ = general_concrete_k_ball(rules_t, w0, k_max=3, max_states=2048)
            true_pos = torch.zeros((L, 6), dtype=torch.bool)
            for w in cset:
                for p, ch in enumerate(w):
                    true_pos[p, cmap[ch]] = True
            chimera_w = w0[:p1] + "dbbbfbbbd" + w0[p2 + 5:]
            assert chimera_w not in cset
            for fval in fp_grid:
                ps = general_prov_fp_adj2_pos_support(rules_t, w0, alpha_str=alpha_s, fp=fval, k_max=3)
                tp[fval] += int((ps & true_pos).sum().item())
                fp_cnt[fval] += int((ps & ~true_pos).sum().item())
                fn_cnt[fval] += int((~ps & true_pos).sum().item())
                # Check if abstract support simultaneously admits 'd' at p1 and 'd' at p2+4 under a single fiber!
                bits, _ = RUST_BACKEND.sym_prov_adj2_u64([w0], rules_t, alpha=6, fp=fval, k_max=2, char_map=cmap)
                admits_chimera = False
                for f_slot in range(1, fval + 1):
                    ok = True
                    for q in range(L - 1):
                        ab = cmap[chimera_w[q]] * 6 + cmap[chimera_w[q + 1]]
                        if ((int(bits[0, 2, f_slot, q].item()) >> ab) & 1) == 0:
                            ok = False
                            break
                    if ok:
                        admits_chimera = True
                        break
                if admits_chimera:
                    chimera_bridge_hits[fval] += 1
        return {
            "evaluated_roots": n_cases,
            "description": "Adversarial p2 = p1 + 4 overlapping width-5 contention (0 mod 4 fiber collision)",
            "precision_by_fp": {str(f): round(tp[f] / max(1, tp[f] + fp_cnt[f]), 5) for f in fp_grid},
            "fp_count_by_fp": {str(f): fp_cnt[f] for f in fp_grid},
            "fn_count_by_fp": {str(f): fn_cnt[f] for f in fp_grid},
            "spurious_chimera_word_rate_by_fp": {str(f): round(chimera_bridge_hits[f] / n_cases, 4) for f in fp_grid},
        }

    # Track S4b: Deeper search trajectories (`steps = 16`, `L = 15`, `DEPTH = 22`, `WORK = 192`) comparing frozen `rel_prov_opened` vs `bidir` & `geom_mm`
    deep_cases = []
    seen_lin = set()
    attempts = 0
    while len(deep_cases) < 16 and attempts < 800:
        attempts += 1
        pairs = rng.sample(["".join(x) for x in itertools.product("abcdef", repeat=2)], 16)
        triples = rng.sample(["".join(x) for x in itertools.product("abcdef", repeat=3)], 8)
        rules = tuple(list(zip(pairs[::2], pairs[1::2])) + list(zip(triples[::2], triples[1::2])))
        L_d = 15
        src = "".join(rng.choice("abcdef") for _ in range(L_d))
        # Walk 16 non-revisiting steps with high Hamming separation >= 6
        cur = src
        vis = {src}
        ok_walk = True
        for _ in range(16):
            opts = []
            for a, b in rules:
                for u_r, v_r in ((a, b), (b, a)):
                    for p in range(L_d - len(u_r) + 1):
                        if cur[p:p + len(u_r)] == u_r:
                            nw = cur[:p] + v_r + cur[p + len(u_r):]
                            if nw not in vis:
                                opts.append(nw)
            if not opts:
                ok_walk = False
                break
            cur = rng.choice(opts)
            vis.add(cur)
        if not ok_walk or sum(x != y for x, y in zip(src, cur)) < 6:
            continue
        lin = hashlib.sha256(repr(rules).encode()).hexdigest()
        if lin in seen_lin:
            continue
        seen_lin.add(lin)
        # Bypass Case.validate()'s L<=14 cap via simple duck-typed StressCase compatible with rg & FringeState
        deep_cases.append((f"stress_deep_{len(deep_cases):02d}", lin, src, cur, rules))

    return {
        "track_s1_word_length": [
            eval_random_family(24, L=L_v, alpha_n=6, rule_lens=(2, 3), k_max=3)
            for L_v in (10, 16, 24, 32)
        ],
        "track_s2_alphabet_size": [
            eval_random_family(24, L=14, alpha_n=A_v, rule_lens=(2, 3), k_max=3)
            for A_v in (6, 8, 12)
        ],
        "track_s3_rule_width_threshold": [
            eval_random_family(24, L=16, alpha_n=6, rule_lens=(2, rl_max), k_max=3)
            for rl_max in (3, 4, 5, 6)
        ],
        "track_s4_horizon_depth": [
            eval_random_family(24, L=14, alpha_n=6, rule_lens=(2, 3), k_max=kv)
            for kv in (2, 3, 4, 5)
        ],
        "track_s5_adversarial_mod4_collisions": eval_adversarial_mod4_suite(24),
        "_deep_search_specs": deep_cases,
    }


@dataclass(frozen=True)
class StressRewriteCase:
    """Length-unrestricted case wrapper (`L > 14`) compatible with `rg.neighbors/replay/search` and `FringeState`."""
    root: str
    lineage: str
    split: str
    family: str
    source: str
    target: str
    rules: tuple[tuple[str, str], ...]


def evaluate_deep_search_stress(rg, specs: list[tuple], models_by_seed: dict[int, tuple[nn.Module, str]]) -> dict:
    cases = [StressRewriteCase(r, lin, "calibration", "equality", src, tgt, rules) for r, lin, src, tgt, rules in specs]
    budget_d, depth_d = 192, 22
    bidir_c, geom_c = [], []
    bidir_w, geom_w = [], []
    for c in cases:
        rb = rg.bidir(c, budget=budget_d, depth=depth_d)
        bidir_c.append(1.0 if rb["closed"] else 0.0)
        bidir_w.append(float(rb["work"] + (0 if rb["closed"] else 1)))

        eg = m17.run_lbf17_episode(rg, c, "geom_mm", budget=budget_d, depth=depth_d)
        geom_c.append(1.0 if eg["closed"] else 0.0)
        geom_w.append(float(eg["work"] + (0 if eg["closed"] else 1)))

    ep_by_seed: dict[int, dict[str, dict]] = {}
    for seed, (model, sha) in sorted(models_by_seed.items()):
        ctrl = IncrementalFrescoController(model, sha, use_rust=True)
        ep_by_seed[seed] = {}
        for c in cases:
            ep_by_seed[seed][c.root] = run_clean_episode(rg, c, controller=ctrl, budget=budget_d, depth=depth_d, cold_cache=True)

    roots = [c.root for c in cases]
    rl = root_level_summary(roots, ep_by_seed, budget=budget_d)
    return {
        "roots": len(cases),
        "word_length": 15,
        "walk_steps": 16,
        "budget": budget_d,
        "depth_cap": depth_d,
        "bidir_closure": sum(bidir_c) / len(cases),
        "bidir_penalized_work": sum(bidir_w) / len(cases),
        "geom_mm_closure": sum(geom_c) / len(cases),
        "geom_mm_penalized_work": sum(geom_w) / len(cases),
        "rel_prov_inc_closure": rl["mean_closure_rate"],
        "rel_prov_inc_all_seeds_closure": rl["roots_closed_all_seeds_rate"],
        "rel_prov_inc_penalized_work": rl["mean_penalized_work"],
    }


# ============================================================================
# 10. Standalone Zero-Axiom Lean 4.19.0 Verifier for `LBF18IncrementalAndCollisions.lean`
# ============================================================================
def find_lean_exe() -> Path | None:
    env_home = os.environ.get("LCGRT_LEAN_HOME")
    if env_home:
        for nm in ("lean.exe", "lean"):
            cand = Path(env_home) / "bin" / nm
            if cand.exists():
                return cand
    which = shutil.which("lean")
    if which:
        return Path(which)
    elan = Path.home() / ".elan" / "toolchains" / "leanprover--lean4---v4.19.0" / "bin" / ("lean.exe" if sys.platform == "win32" else "lean")
    return elan if elan.exists() else None


def certify_single_thread(rg, items: list, out: Path, chunk_size: int = 350) -> dict:
    """Standalone zero-axiom Lean 4.19.0 certifier using `--threads=1` and bounded batch chunks
    so multi-agent Windows virtual commit charge stays < 400 MB."""
    import re
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    if not items:
        raise ValueError("nonempty native certificate list required")
    names = [n for _, _, n in items]
    if len(set(names)) != len(names):
        raise ValueError("duplicate certificate name")
    lean_exe = find_lean_exe()
    if lean_exe is None:
        raise RuntimeError("Lean 4.19.0 executable not found")
    out.mkdir(parents=True, exist_ok=True)
    ver = subprocess.check_output([str(lean_exe), "--version"], text=True).strip()
    if "version 4.19.0" not in ver:
        raise ValueError(f"pinned Lean 4.19.0 required, got {ver}")

    all_logs = []
    all_srcs = []
    passed_all: set[str] = set()
    for ci in range(0, len(items), chunk_size):
        chunk = items[ci:ci + chunk_size]
        c_names = [n for _, _, n in chunk]
        src_text = (
            "namespace LBF12Cert\n\n"
            + "\n\n".join(base.render_proof(rg, c, p, n) for c, p, n in chunk)
            + "\n\nend LBF12Cert\n"
            + "\n".join(f"#print axioms LBF12Cert.{n}" for n in c_names)
            + "\n"
        )
        pfile = out / (f"Proofs_{ci // chunk_size:03d}.lean" if len(items) > chunk_size else "Proofs.lean")
        pfile.write_text(src_text, encoding="utf-8")
        all_srcs.append(src_text)
        gc.collect()
        proc = None
        for attempt in range(3):
            proc = subprocess.run([str(lean_exe), "--threads=1", str(pfile)], capture_output=True, text=True, timeout=420)
            if proc.returncode == 0:
                break
            time.sleep(1.5 * (attempt + 1))
        log = (proc.stdout if proc else "") + (proc.stderr if proc else "")
        all_logs.append(log)
        passed = set(re.findall(r"'LBF12Cert\.([A-Za-z0-9_]+)' does not depend on any axioms", log))
        if proc is None or proc.returncode != 0 or passed != set(c_names):
            raise RuntimeError("independent standalone Lean certificate rejected: " + log[-2500:])
        passed_all.update(passed)

    full_log = "\n".join(all_logs)
    (out / "compiler.log").write_text(full_log, encoding="utf-8")
    src_concat_sha = hashlib.sha256("\n".join(all_srcs).encode("utf-8")).hexdigest()
    log_sha = hashlib.sha256(full_log.encode("utf-8")).hexdigest()
    exe_sha = hashlib.sha256(lean_exe.read_bytes()).hexdigest()
    m = {
        "schema": "lbf12.explicit_eq_trans_cert.v1",
        "actual_lean_compiler": True,
        "lean_version": ver,
        "lean_exe_sha256": exe_sha,
        "source_sha256": src_concat_sha,
        "log_sha256": log_sha,
        "compiled_axiom_free": len(passed_all),
        "declarations": names,
        "native_online_tactic_calls": 0,
    }
    (out / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return m


def verify_lbf18_lean_formalization(out_dir: Path | None = None) -> dict:
    lean_exe = find_lean_exe()
    if lean_exe is None:
        raise RuntimeError("Lean 4.19.0 executable not found")
    ver = subprocess.check_output([str(lean_exe), "--version"], text=True).strip()
    gc.collect()
    proc = None
    for attempt in range(3):
        proc = subprocess.run([str(lean_exe), "--threads=1", str(LEAN18_FILE)], capture_output=True, text=True)
        if proc.returncode == 0:
            break
        time.sleep(1.5 * (attempt + 1))
    if proc is None or proc.returncode != 0:
        raise RuntimeError(f"Lean 4 compilation failed:\n{proc.stdout if proc else ''}\n{proc.stderr if proc else ''}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    expected_theorems = [
        "keyed_cache_lookup_sound",
        "keyed_cache_rejects_mismatch",
        "incremental_word_map_equiv",
        "incremental_pair_map_equiv",
        "zero_preserving_gate_support",
        "shapley_sum_eq_twice_joint",
        "marginal_sum_minus_joint_eq_interaction",
        "mod_collision_disjoint_intervals",
        "disjoint_mod4_rewrites_commute_in_2_steps",
        "cex4a_fp4_spurious_fp5_exact",
        "cex4b_intrafiber_step2_fork",
        "cex4c_horizon_compression_long_word",
    ]
    for thm in expected_theorems:
        needle = f"'LBF18IncrementalAndCollisions.{thm}' does not depend on any axioms"
        if needle not in lines:
            raise RuntimeError(f"Theorem {thm} missing zero-axiom confirmation in Lean output: {lines}")
    src_sha = hashlib.sha256(LEAN18_FILE.read_bytes()).hexdigest()
    log_sha = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()
    res = {
        "lean_version": ver,
        "source_sha256": src_sha,
        "log_sha256": log_sha,
        "compiled_axiom_free": len(expected_theorems),
        "theorems": expected_theorems,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return res


# ============================================================================
# 11. Illustrated SVG Diagrams (Incremental Fresco Speedup, 2x2 Shapley, Sealed Confirmation & Collisions)
# ============================================================================
def _svg_esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_incremental_and_factorial_svg(cal_summary: dict, out_path: Path) -> str:
    bench = cal_summary["clean_inference_benchmark"]
    fac = cal_summary["factorial_2x2"]["zero_factorial_penalized_work_b128"]
    fac_auc = cal_summary["factorial_2x2"]["zero_factorial_auc"]

    arms = [
        ("uncached_py", "LBF-17 Uncached PyTorch (K=3)", "#94a3b8"),
        ("inc_py_cold", "LBF-18 Incremental PyTorch (Cold)", "#60a5fa"),
        ("inc_py_warm", "LBF-18 Incremental PyTorch (Warm)", "#3b82f6"),
        ("inc_rust_cold", "LBF-18 Incremental + Rust Bitset (Cold)", "#10b981"),
        ("inc_rust_warm", "LBF-18 Incremental + Rust Bitset (Warm)", "#059669"),
        ("rel_prov_adaptive_rust_cold", "LBF-18 Adaptive 1-3 Pass + Rust (Cold)", "#f59e0b"),
        ("sym_prov_readout_inc_rust_cold", "sym_prov_readout + Rust Bitset (K=1)", "#a78bfa"),
        ("hybrid_sym_rec_inc_rust_cold", "Hybrid Sym-Env + 3-Pass Axial + Rust", "#ec4899"),
    ]
    max_ms = max(bench[k]["mean_episode_ms"] for k, _, _ in arms)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 690" font-family="Inter, Segoe UI, system-ui, sans-serif">',
        '<rect width="1180" height="690" fill="#0b1020" rx="14"/>',
        '<text x="32" y="42" fill="#f8fafc" font-size="21" font-weight="700">'
        'LBF-18 Incremental Relational Fresco: Clean Inference Speedup &amp; 2x2 Bank Interaction</text>',
        '<text x="32" y="66" fill="#94a3b8" font-size="13">'
        'Frozen LBF-17 checkpoints (~53.1K params) on 48 independent calibration roots (x3 seeds) · Zero profiler overhead</text>',
        # Left panel: Clean latency & hit-rate bars
        '<rect x="28" y="84" width="700" height="350" fill="#111827" stroke="#1f2937" rx="10"/>',
        '<text x="46" y="112" fill="#e2e8f0" font-size="15" font-weight="600">'
        'A. Clean End-to-End Episode Latency (B = 128) &amp; Incremental Hit Rates</text>',
    ]
    for idx, (key, label, color) in enumerate(arms):
        row = bench[key]
        y = 132 + idx * 36
        ms = row["mean_episode_ms"]
        bar_w = max(4.0, (ms / max(1e-6, max_ms)) * 280.0)
        sp = bench["uncached_py"]["mean_episode_ms"] / max(1e-6, ms)
        wh = row["word_cache_hit_rate"] * 100.0
        ph = row["pair_cache_hit_rate"] * 100.0
        lines.append(f'<text x="46" y="{y + 14}" fill="#cbd5e1" font-size="12">{_svg_esc(label)}</text>')
        lines.append(f'<rect x="305" y="{y}" width="{bar_w:.1f}" height="20" fill="{color}" rx="4"/>')
        lines.append(
            f'<text x="{313 + bar_w:.1f}" y="{y + 14}" fill="#f8fafc" font-size="11.5" font-weight="600">'
            f'{ms:.1f} ms ({sp:.1f}x) · W-hit {wh:.0f}% · P-hit {ph:.0f}%</text>'
        )

    # Right panel: 2x2 Bank Interaction Matrix
    cm = fac["cell_means_raw"]
    ca = fac_auc["cell_means_raw"]
    lines.extend([
        '<rect x="748" y="84" width="404" height="350" fill="#111827" stroke="#1f2937" rx="10"/>',
        '<text x="766" y="112" fill="#e2e8f0" font-size="15" font-weight="600">'
        'B. 2x2 Recurrence Factorial (N = 48 Roots)</text>',
        '<text x="766" y="132" fill="#94a3b8" font-size="11.5">'
        'Envelope Bank (Bw, Bbi) x Cross-Frontier Matrix Bank (Bm)</text>',
        # 2x2 grid
        '<rect x="780" y="156" width="165" height="92" fill="#064e3b" stroke="#10b981" stroke-width="1.5" rx="8"/>',
        '<text x="794" y="178" fill="#a7f3d0" font-size="12" font-weight="700">E=1, M=1 (Intact)</text>',
        f'<text x="794" y="202" fill="#ffffff" font-size="16" font-weight="700">{cm["E1_M1_intact"]:.2f} work</text>',
        f'<text x="794" y="224" fill="#6ee7b7" font-size="12">5-Budget AUC: {ca["E1_M1_intact"]*100:.2f}%</text>',

        '<rect x="960" y="156" width="165" height="92" fill="#1e293b" stroke="#475569" rx="8"/>',
        '<text x="974" y="178" fill="#bfdbfe" font-size="12" font-weight="700">E=1, M=0 (mat_zero)</text>',
        f'<text x="974" y="202" fill="#f8fafc" font-size="16" font-weight="700">{cm["E1_M0_mat_ablated"]:.2f} work</text>',
        f'<text x="974" y="224" fill="#93c5fd" font-size="12">5-Budget AUC: {ca["E1_M0_mat_ablated"]*100:.2f}%</text>',

        '<rect x="780" y="262" width="165" height="92" fill="#1e293b" stroke="#475569" rx="8"/>',
        '<text x="794" y="284" fill="#fde68a" font-size="12" font-weight="700">E=0, M=1 (env_zero)</text>',
        f'<text x="794" y="308" fill="#f8fafc" font-size="16" font-weight="700">{cm["E0_M1_env_ablated"]:.2f} work</text>',
        f'<text x="794" y="330" fill="#fcd34d" font-size="12">5-Budget AUC: {ca["E0_M1_env_ablated"]*100:.2f}%</text>',

        '<rect x="960" y="262" width="165" height="92" fill="#3f1d2e" stroke="#f43f5e" rx="8"/>',
        '<text x="974" y="284" fill="#fecdd3" font-size="12" font-weight="700">E=0, M=0 (env_mat_zero)</text>',
        f'<text x="974" y="308" fill="#ffffff" font-size="16" font-weight="700">{cm["E0_M0_both_ablated"]:.2f} work</text>',
        f'<text x="974" y="330" fill="#fda4af" font-size="12">5-Budget AUC: {ca["E0_M0_both_ablated"]*100:.2f}%</text>',

        f'<text x="768" y="382" fill="#e2e8f0" font-size="12.5" font-weight="600">'
        f'Joint Recurrence Savings: -{fac["joint_recurrence_benefit"]:.2f} work '
        f'(I_EM = {fac["interaction_I_EM"]:+.2f})</text>',
        f'<text x="768" y="404" fill="#cbd5e1" font-size="12">'
        f'Shapley Credit: Envelope {fac["shapley_env_share"]*100:.1f}% (-{fac["shapley_env"]:.2f}) · '
        f'Matrix {fac["shapley_mat_share"]*100:.1f}% (-{fac["shapley_mat"]:.2f})</text>',
        f'<text x="768" y="423" fill="#94a3b8" font-size="11.5">'
        f'Naive marginal ratio was {fac["naive_marginal_env_share"]*100:.1f}% / {(1-fac["naive_marginal_env_share"])*100:.1f}%</text>',
    ])

    # Bottom panel: Three-stage Incremental Architecture & Resource Table
    lines.extend([
        '<rect x="28" y="452" width="1124" height="212" fill="#111827" stroke="#1f2937" rx="10"/>',
        '<text x="46" y="480" fill="#e2e8f0" font-size="15" font-weight="600">'
        'C. Three-Stage Incremental Factorization &amp; Verified Invariants</text>',
        '<rect x="48" y="498" width="335" height="142" fill="#0f172a" stroke="#10b981" rx="8"/>',
        '<text x="64" y="522" fill="#34d399" font-size="13" font-weight="700">Stage A: Word Envelope Cache C_word</text>',
        '<text x="64" y="544" fill="#cbd5e1" font-size="11.5">Key: (sha256(theta), ver, canon(R), env_mode, K, w)</text>',
        '<text x="64" y="564" fill="#cbd5e1" font-size="11.5">Rust u64 bitset skips 85-98% zero rewrite windows</text>',
        f'<text x="64" y="584" fill="#a7f3d0" font-size="11.5">Cold hit rate: {bench["inc_rust_cold"]["word_cache_hit_rate"]*100:.1f}% · Warm: {bench["inc_rust_warm"]["word_cache_hit_rate"]*100:.1f}%</text>',
        f'<text x="64" y="604" fill="#a7f3d0" font-size="11.5">Rule-window checks cut by {(1 - bench["inc_rust_cold"]["mean_rule_window_checks"]/max(1, bench["uncached_py"]["mean_rule_window_checks"]))*100:.1f}% (cold)</text>',
        '<text x="64" y="624" fill="#94a3b8" font-size="11">Max float diff vs PyTorch: &lt; 1e-6 (100% path identity)</text>',

        '<rect x="408" y="498" width="345" height="142" fill="#0f172a" stroke="#3b82f6" rx="8"/>',
        '<text x="424" y="522" fill="#60a5fa" font-size="13" font-weight="700">Stage B: Static Pair Ribbon Cache C_pair</text>',
        '<text x="424" y="544" fill="#cbd5e1" font-size="11.5">Key: (sha256(theta), ver, canon(R), env_mode, K, u, v)</text>',
        '<text x="424" y="564" fill="#cbd5e1" font-size="11.5">Caches Kx76 pooled Conv1d subterm bridges (912 B/pair)</text>',
        f'<text x="424" y="584" fill="#bfdbfe" font-size="11.5">Cold hit rate: {bench["inc_rust_cold"]["pair_cache_hit_rate"]*100:.1f}% · Warm: {bench["inc_rust_warm"]["pair_cache_hit_rate"]*100:.1f}%</text>',
        f'<text x="424" y="604" fill="#bfdbfe" font-size="11.5">Executed FLOPs cut by {(1 - bench["inc_rust_cold"]["mean_neural_flops"]/max(1, bench["uncached_py"]["mean_neural_flops"]))*100:.1f}% (cold)</text>',
        '<text x="424" y="624" fill="#94a3b8" font-size="11">Rejects cross-size reuse on scramble_window / ribbon</text>',

        '<rect x="778" y="498" width="354" height="142" fill="#0f172a" stroke="#f59e0b" rx="8"/>',
        '<text x="794" y="522" fill="#fbbf24" font-size="13" font-weight="700">Stage C: Live Cross-Frontier Axial Competition</text>',
        '<text x="794" y="544" fill="#cbd5e1" font-size="11.5">Dynamic infeas(g0, g1, D), rem_budget, lm_diff, prev_pot</text>',
        '<text x="794" y="564" fill="#cbd5e1" font-size="11.5">Adaptive 1-3 pass margin stopping + Hybrid 3-pass readout</text>',
        f'<text x="794" y="584" fill="#fde68a" font-size="11.5">Adaptive passes/dec: {bench["rel_prov_adaptive_rust_cold"]["mean_passes_per_decision"]:.2f} (closure {bench["rel_prov_adaptive_rust_cold"]["closure_rate_b128"]*100:.1f}%)</text>',
        f'<text x="794" y="604" fill="#fbcfe8" font-size="11.5">Hybrid 3-pass: {bench["hybrid_sym_rec_inc_rust_cold"]["mean_episode_ms"]:.1f} ms/ep · {bench["hybrid_sym_rec_inc_rust_cold"]["mean_penalized_work_b128"]:.2f} work</text>',
        f'<text x="794" y="624" fill="#94a3b8" font-size="11">Peak process RSS: {bench["inc_rust_cold"]["peak_rss_mb"]:.1f} MB (&lt;&lt; 2 GiB cap)</text>',
        '</svg>',
    ])
    svg = "\n".join(lines) + "\n"
    out_path.write_text(svg, encoding="utf-8")
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()


def render_confirmation_and_collisions_svg(conf_summary: dict, stress_summary: dict, out_path: Path) -> str:
    cohorts = conf_summary["cohorts"]
    s3 = stress_summary["track_s3_rule_width_threshold"]
    s5 = stress_summary["track_s5_adversarial_mod4_collisions"]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 680" font-family="Inter, Segoe UI, system-ui, sans-serif">',
        '<rect width="1180" height="680" fill="#0b1020" rx="14"/>',
        '<text x="32" y="42" fill="#f8fafc" font-size="21" font-weight="700">'
        'LBF-18 Independent Sealed-Cohort Confirmation &amp; Provenance Collision Thresholds</text>',
        '<text x="32" y="66" fill="#94a3b8" font-size="13">'
        '64 untouched sealed theorem roots across LBF-17 (20261227) &amp; LBF-16 (20261221) ordinary + OOD cohorts · Standalone Lean 4.19.0 certified</text>',
        '<rect x="28" y="84" width="630" height="330" fill="#111827" stroke="#1f2937" rx="10"/>',
        '<text x="46" y="112" fill="#e2e8f0" font-size="15" font-weight="600">'
        'A. Frozen Sealed Confirmation Closure at B = 128 (4 Disjoint Cohorts, R = 16 Each)</text>',
    ]

    cohort_keys = [
        ("lbf17_ordinary", "LBF-17 Ordinary (R=16)"),
        ("lbf17_ood", "LBF-17 OOD L=13,d=12 (R=16)"),
        ("lbf16_ordinary", "LBF-16 Ordinary (R=16)"),
        ("lbf16_ood", "LBF-16 OOD L=13,d=12 (R=16)"),
    ]
    pol_defs = [
        ("rel_prov_opened", "rel_prov_opened", "#10b981"),
        ("hybrid_sym_rec_opened", "hybrid_sym_rec", "#ec4899"),
        ("lbf16_pos1_opened", "lbf16_pos1", "#60a5fa"),
        ("geom_mm", "geom_mm", "#f59e0b"),
        ("bidir", "bidir", "#64748b"),
    ]

    for ci, (ck, clabel) in enumerate(cohort_keys):
        y0 = 132 + ci * 68
        lines.append(f'<text x="46" y="{y0 + 14}" fill="#e2e8f0" font-size="12" font-weight="600">{_svg_esc(clabel)}</text>')
        c_rates = cohorts[ck]["rates_b128"]
        for pi, (pk, plab, col) in enumerate(pol_defs):
            rate = c_rates[pk] * 100.0
            bx = 235 + pi * 80
            bh = max(3.0, rate * 0.36)
            by = y0 + 40 - bh
            lines.append(f'<rect x="{bx}" y="{by:.1f}" width="66" height="{bh:.1f}" fill="{col}" rx="3"/>')
            lines.append(f'<text x="{bx + 33}" y="{by - 4:.1f}" fill="#f8fafc" font-size="10.5" text-anchor="middle">{rate:.0f}%</text>')
            if ci == 3:
                lines.append(f'<text x="{bx + 33}" y="{y0 + 55}" fill="{col}" font-size="10" text-anchor="middle">{_svg_esc(plab)}</text>')

    # Panel B: Rule-Width & Adversarial Mod-4 Provenance Precision
    lines.extend([
        '<rect x="678" y="84" width="474" height="330" fill="#111827" stroke="#1f2937" rx="10"/>',
        '<text x="696" y="112" fill="#e2e8f0" font-size="15" font-weight="600">'
        'B. Where F_p = 4 Breaks: Rule Width &amp; Mod-4 Collisions</text>',
        '<text x="696" y="134" fill="#94a3b8" font-size="11.5">'
        'Track S3 (Random Widths l_max = 3..6) &amp; Track S5 (Adversarial p2 = p1 + 4)</text>',
    ])
    for ri, row in enumerate(s3):
        y = 160 + ri * 42
        rl_m = max(row["rule_lengths"])
        p1 = row["precision_by_fp"]["1"] * 100
        p4 = row["precision_by_fp"]["4"] * 100
        p6 = row["precision_by_fp"]["6"] * 100
        p8 = row["precision_by_fp"]["8"] * 100
        lines.append(
            f'<text x="696" y="{y}" fill="#e2e8f0" font-size="12" font-weight="600">'
            f'Max rule width l = {rl_m}:</text>'
            f'<text x="840" y="{y}" fill="#cbd5e1" font-size="12">'
            f'Fp=1: {p1:.1f}% · Fp=4: {p4:.1f}% · Fp=6: {p6:.1f}% · Fp=8: {p8:.1f}%</text>'
        )
    adv_p = s5["precision_by_fp"]
    adv_c = s5["spurious_chimera_word_rate_by_fp"]
    lines.extend([
        '<rect x="696" y="324" width="438" height="74" fill="#2e1020" stroke="#f43f5e" rx="7"/>',
        '<text x="710" y="345" fill="#fda4af" font-size="12" font-weight="700">'
        'Adversarial Mod-4 Contention (p2 = p1 + 4, l = 5 &gt; Fp = 4):</text>',
        f'<text x="710" y="366" fill="#fecdd3" font-size="11.5">'
        f'Precision: Fp=1 {adv_p["1"]*100:.1f}% · Fp=4 {adv_p["4"]*100:.1f}%  --&gt;  Fp=6 {adv_p["6"]*100:.1f}% · Fp=8 {adv_p["8"]*100:.1f}%</text>',
        f'<text x="710" y="385" fill="#fde68a" font-size="11.5">'
        f'Spurious chimera acceptance: Fp=4 {adv_c["4"]*100:.0f}% vs Fp=6 {adv_c["6"]*100:.0f}% &amp; Fp=8 {adv_c["8"]*100:.0f}%</text>',
    ])

    # Panel C: Lite-Brite Glyph Diagram of Counterexample 4A + Pooled Sealed Summary
    pooled = conf_summary["pooled_64_sealed_roots"]
    lines.extend([
        '<rect x="28" y="432" width="1124" height="222" fill="#111827" stroke="#1f2937" rx="10"/>',
        '<text x="46" y="460" fill="#e2e8f0" font-size="15" font-weight="600">'
        'C. Pooled 64-Root Independent Confirmation &amp; Counterexample 4A Lite-Brite Provenance Diagram</text>',
        '<rect x="46" y="476" width="540" height="160" fill="#0f172a" stroke="#10b981" rx="8"/>',
        f'<text x="64" y="502" fill="#34d399" font-size="13.5" font-weight="700">'
        f'Pooled Sealed Confirmation (64 Untouched Roots: 32 Ordinary + 32 OOD)</text>',
        f'<text x="64" y="526" fill="#f8fafc" font-size="12.5">'
        f'rel_prov_opened: {pooled["rel_prov_opened"]["mean_closure_rate"]*100:.2f}% closure · '
        f'{pooled["rel_prov_opened"]["mean_penalized_work"]:.2f} work · '
        f'{pooled["rel_prov_opened"]["roots_closed_all_seeds"]}/64 (3/3 seeds)</text>',
        f'<text x="64" y="548" fill="#fbcfe8" font-size="12.5">'
        f'hybrid_sym_rec_opened: {pooled["hybrid_sym_rec_opened"]["mean_closure_rate"]*100:.2f}% closure · '
        f'{pooled["hybrid_sym_rec_opened"]["mean_penalized_work"]:.2f} work</text>',
        f'<text x="64" y="570" fill="#fde68a" font-size="12.5">'
        f'rel_prov_adaptive: {pooled["rel_prov_adaptive"]["mean_closure_rate"]*100:.2f}% closure · '
        f'{pooled["rel_prov_adaptive"]["mean_penalized_work"]:.2f} work</text>',
        f'<text x="64" y="592" fill="#93c5fd" font-size="12">'
        f'Frozen LBF-16 (pos1): {pooled["lbf16_pos1_opened"]["mean_closure_rate"]*100:.2f}% ({pooled["lbf16_pos1_opened"]["mean_penalized_work"]:.2f} work) · '
        f'geom_mm: {pooled["geom_mm"]["mean_closure_rate"]*100:.2f}% · bidir: {pooled["bidir"]["mean_closure_rate"]*100:.2f}%</text>',
        f'<text x="64" y="616" fill="#a7f3d0" font-size="11.5">'
        f'Standalone zero-axiom Lean 4.19.0 sealed proof certificates compiled: {conf_summary["lean_sealed_certificates_compiled"]}</text>',

        '<rect x="606" y="476" width="526" height="160" fill="#0f172a" stroke="#f43f5e" rx="8"/>',
        '<text x="624" y="502" fill="#fda4af" font-size="13" font-weight="700">'
        'Counterexample 4A (Lean-Verified): w0 = "abbbebbbc" (L = 9)</text>',
        '<text x="624" y="525" fill="#cbd5e1" font-size="11.5">'
        'r1 @ p1=0: abbbe -&gt; dbbbf  (writes f at contended index 4; fiber 0 mod 4 = 0)</text>',
        '<text x="624" y="545" fill="#cbd5e1" font-size="11.5">'
        'r2 @ p2=4: ebbbc -&gt; fbbbd  (writes f at contended index 4; fiber 4 mod 4 = 0!)</text>',
        '<text x="624" y="569" fill="#fecdd3" font-size="12" font-weight="600">'
        'Fp = 4 Fiber 0 merges dbbbfbbbc + abbbfbbbd ==&gt; Spurious chimera "dbbbfbbbd"</text>',
        '<text x="624" y="593" fill="#6ee7b7" font-size="12" font-weight="600">'
        'Fp = 5 separates 0 mod 5 = 0 vs 4 mod 5 = 4 ==&gt; Exact {w0, w1, w2} (0 FPs)</text>',
        '<text x="624" y="616" fill="#94a3b8" font-size="11">'
        'Commutation Theorem (`mod_collision_disjoint_intervals`): Fp &gt;= l_max prevents all step-1 overlaps</text>',
        '</svg>',
    ])
    svg = "\n".join(lines) + "\n"
    out_path.write_text(svg, encoding="utf-8")
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()


# ============================================================================
# 12. Two-Phase Campaign Runner (`--phase calibrate` -> Registration -> `--phase confirm-sealed`)
# ============================================================================
def _load_gzipped_cases(suite_dir: Path, split: str, rg) -> list:
    gz_path = suite_dir / f"{split}.jsonl.gz"
    raw_path = suite_dir / f"{split}.jsonl"
    data = gzip.decompress(gz_path.read_bytes()) if gz_path.exists() else raw_path.read_bytes()
    m = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(data).hexdigest()
    if actual != m["artifacts"][f"{split}.jsonl"]:
        raise ValueError(f"artifact SHA mismatch for {split}.jsonl: {actual} != {m['artifacts'][f'{split}.jsonl']}")
    return [rg.Case.parse(json.loads(line)) for line in data.decode("utf-8").splitlines() if line.strip()]


def _summarize_clean_bench_arm(roots: list[str], eps_by_seed: dict[int, dict[str, dict]], budget_eps: dict[int, dict[int, dict[str, dict]]]) -> dict:
    rl_128 = root_level_summary(roots, eps_by_seed, budget=WORK)
    all_ep = [eps_by_seed[s][r] for s in sorted(eps_by_seed) for r in roots]
    ep_ms = sorted(e["wall_clock_ms"] for e in all_ep)
    dec_ms = sorted(x for e in all_ep for x in e["decision_latencies_ms"])
    tot_wh = sum(e["word_hits"] for e in all_ep)
    tot_wm = sum(e["word_misses"] for e in all_ep)
    tot_ph = sum(e["pair_hits"] for e in all_ep)
    tot_pm = sum(e["pair_misses"] for e in all_ep)
    tot_dec = sum(e["fringe_decisions"] for e in all_ep)
    tot_passes = sum(e["recurrent_passes"] for e in all_ep)
    tot_sec = sum(ep_ms) / 1000.0

    curve = {}
    for b in BUDGET_CURVE:
        rl_b = root_level_summary(roots, budget_eps[b], budget=b)
        curve[str(b)] = {
            "closure_rate": rl_b["mean_closure_rate"],
            "all_seeds_closure_rate": rl_b["roots_closed_all_seeds_rate"],
            "mean_penalized_work": rl_b["mean_penalized_work"],
        }
    auc = sum(curve[str(b)]["closure_rate"] for b in BUDGET_CURVE) / len(BUDGET_CURVE)

    return {
        "independent_roots": len(roots),
        "seed_episodes": len(all_ep),
        "closure_rate_b128": rl_128["mean_closure_rate"],
        "closure_root_95ci_b128": rl_128["closure_root_95ci"],
        "roots_closed_all_seeds_b128": rl_128["roots_closed_all_seeds"],
        "mean_penalized_work_b128": rl_128["mean_penalized_work"],
        "penalized_work_root_95ci_b128": rl_128["penalized_work_root_95ci"],
        "work_icc_b128": rl_128["work_icc"],
        "effective_sample_size_n_eff_b128": rl_128["effective_sample_size_n_eff"],
        "per_seed_closure_b128": rl_128["per_seed_closure_rate"],
        "per_seed_penalized_work_b128": rl_128["per_seed_penalized_work"],
        "budget_curve": curve,
        "budget_curve_auc": auc,
        "mean_episode_ms": sum(ep_ms) / len(ep_ms),
        "p50_episode_ms": ep_ms[len(ep_ms) // 2],
        "p95_episode_ms": ep_ms[min(len(ep_ms) - 1, int(0.95 * len(ep_ms)))],
        "mean_decision_ms": sum(dec_ms) / max(1, len(dec_ms)),
        "p50_decision_ms": dec_ms[len(dec_ms) // 2] if dec_ms else 0.0,
        "p95_decision_ms": dec_ms[min(len(dec_ms) - 1, int(0.95 * len(dec_ms)))] if dec_ms else 0.0,
        "throughput_decisions_per_sec": tot_dec / max(1e-9, tot_sec),
        "throughput_episodes_per_sec": len(all_ep) / max(1e-9, tot_sec),
        "word_cache_hit_rate": tot_wh / max(1, tot_wh + tot_wm),
        "pair_cache_hit_rate": tot_ph / max(1, tot_ph + tot_pm),
        "mean_rule_window_checks": sum(e["substring_checks"] for e in all_ep) / len(all_ep),
        "mean_neural_flops": sum(e["neural_flops"] for e in all_ep) / len(all_ep),
        "mean_neural_flops_per_decision": sum(e["neural_flops"] for e in all_ep) / max(1, tot_dec),
        "mean_passes_per_decision": tot_passes / max(1, tot_dec),
        "current_rss_mb": max(e["rss_bytes"] for e in all_ep) / (1024.0 * 1024.0),
        "peak_rss_mb": max(e["peak_rss_bytes"] for e in all_ep) / (1024.0 * 1024.0),
        "_root_c": rl_128["_root_c"],
        "_root_w": rl_128["_root_w"],
    }


def run_calibration_phase(rg, out_dir: Path, scratch_dir: Path) -> dict:
    """Phase 1: Calibration, Hybrid Train-Only Fit, Clean Benchmarks, 2x2 Factorial, and Stress Suite.
    Strictly forbidden from opening any `ordinary` or `ood` split."""
    torch.set_num_threads(1)
    base.certify = certify_single_thread
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Lean 4.19.0 formalization of LBF-18 theorems
    print("[LBF-18 Phase 1] Compiling standalone zero-axiom Lean 4.19.0 formalization...", flush=True)
    lean18_cert = verify_lbf18_lean_formalization(out_dir / "lean_theory_certificates")

    # 2. Load ONLY `train` and `calibration` of LBF-17 (`SEED = 20261227`)
    suite17_dir = LBF17_RESULTS_DIR / "suite"
    cal_cases = _load_gzipped_cases(suite17_dir, "calibration", rg)
    cal_roots = [c.root for c in cal_cases]

    # Reconstruct exact deterministic train teacher frames on scratch (train-only, SHA-verified!)
    train_suite_tmp = scratch_dir / "suite_20261227_train_only"
    if not train_suite_tmp.exists():
        rg.freeze(train_suite_tmp, seed=LBF17_SEED, counts=m17.COUNTS)
    if rg.file_sha(train_suite_tmp / "manifest.json") != LBF17_SUITE_MANIFEST_SHA256:
        raise ValueError("regenerated LBF-17 suite SHA mismatch")
    teacher_tmp = scratch_dir / "teacher_20261227"
    print("[LBF-18 Phase 1] Preparing train-only teacher frames (SHA-verified against LBF-17)...", flush=True)
    train_frames, teacher_man = m17.teacher(rg, train_suite_tmp, teacher_tmp, resume=True)
    lbf17_man = json.loads((LBF17_RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    if teacher_man["frames_summary_sha256"] != lbf17_man["training"]["frames_summary_sha256"]:
        raise ValueError("train teacher frames summary SHA mismatch vs LBF-17")

    # 3. Load frozen LBF-17 checkpoints (`rel_prov_opened`, `sym_prov_readout`, `lbf16_pos1_opened`)
    rel_models: dict[int, tuple[m17.RelationalProvenanceFrescoNet, str]] = {}
    sym_models: dict[int, tuple[m17.RelationalProvenanceFrescoNet, str]] = {}
    for seed in SEEDS:
        rm, rsha, _ = load_frozen_lbf17_checkpoint(seed, "rel_prov_opened")
        rel_models[seed] = (rm, rsha)
        sm, ssha, _ = load_frozen_lbf17_checkpoint(seed, "sym_prov_readout")
        sym_models[seed] = (sm, ssha)

    # 4. Fit `hybrid_sym_rec_opened` on `20261227/train` ONLY using archived LBF-17 draw schedules
    hyb_models: dict[int, tuple[HybridSymRecFrescoNet, str]] = {}
    hyb_fits = []
    for seed in SEEDS:
        ck_dir = out_dir / f"{seed}_hybrid_sym_rec_opened"
        ck_gz = ck_dir / "checkpoint.pt.gz"
        man_p = ck_dir / "manifest.json"
        if ck_gz.exists() and man_p.exists():
            fman = json.loads(man_p.read_text(encoding="utf-8"))
            raw = gzip.decompress(ck_gz.read_bytes())
            if hashlib.sha256(raw).hexdigest() != fman["checkpoint_sha256"]:
                raise ValueError("hybrid checkpoint SHA mismatch on resume")
            hm = HybridSymRecFrescoNet().cpu().float().eval()
            hm.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["weights"])
            for p in hm.parameters():
                p.requires_grad_(False)
            hyb_models[seed] = (hm, fman["checkpoint_sha256"])
            hyb_fits.append(fman)
            print(f"[LBF-18 Phase 1] Resumed {seed}_hybrid_sym_rec_opened ({fman['checkpoint_sha256'][:12]})", flush=True)
        else:
            draws_gz = LBF17_RESULTS_DIR / f"{seed}_rel_prov_opened" / "draws.jsonl.gz"
            draws_rows = [
                json.loads(ln)["frame_indices"]
                for ln in gzip.decompress(draws_gz.read_bytes()).decode("utf-8").splitlines()
                if ln.strip()
            ]
            print(f"[LBF-18 Phase 1] Fitting {seed}_hybrid_sym_rec_opened on 64 train roots (240 steps)...", flush=True)
            hm, fman = fit_hybrid_sym_rec_model(rg, train_frames, draws_rows, seed)
            ck_dir.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            torch.save({"arch": "hybrid_sym_rec_opened", "seed": seed, "parameters": hm.parameter_count(), "weights": hm.state_dict()}, buf)
            raw_b = buf.getvalue()
            csha = hashlib.sha256(raw_b).hexdigest()
            fman["checkpoint_sha256"] = csha
            fman["draws_sha256"] = hashlib.sha256(gzip.decompress(draws_gz.read_bytes())).hexdigest()
            ck_gz.write_bytes(gzip.compress(raw_b, compresslevel=9))
            man_p.write_text(json.dumps(fman, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            hyb_models[seed] = (hm, csha)
            hyb_fits.append(fman)

    # 5. Calibrate adaptive 1-3 pass stopping thresholds STRICTLY on `train_frames`
    print("[LBF-18 Phase 1] Calibrating adaptive 1-3 pass margins on train frames...", flush=True)
    adaptive_cal = calibrate_adaptive_margins_on_train(rel_models, train_frames, target_agreement=0.985)
    del train_frames
    gc.collect()
    adaptive_margins = (adaptive_cal["gamma1"], adaptive_cal["gamma2"])

    # 6. Warm up `_calibrated_stage_flops` & `_calibrated_stage_c_flops` outside clean timing!
    for arch_w in ("rel_prov_opened", "param_nonrecurrent"):
        for L_w in (10, 11, 13):
            _calibrated_stage_flops(arch_w, L_w, 3)
            _calibrated_stage_flops(arch_w, L_w, 1)
        for m0_w in range(1, 36):
            for m1_w in range(1, 36):
                for kp in (1, 2, 3):
                    _calibrated_stage_c_flops(arch_w, m0_w, m1_w, kp)

    # 7. Clean Inference Benchmark & Exact Equivalence Audit on Calibration (`48` roots x `3` seeds)
    print("[LBF-18 Phase 1] Running clean inference benchmark & exact equivalence audit on calibration...", flush=True)
    bench_arms = (
        "uncached_py",
        "inc_py_cold",
        "inc_py_warm",
        "inc_rust_cold",
        "inc_rust_warm",
        "rel_prov_adaptive_rust_cold",
        "sym_prov_readout_uncached_py",
        "sym_prov_readout_inc_rust_cold",
        "hybrid_sym_rec_inc_rust_cold",
    )
    arm_eps_128: dict[str, dict[int, dict[str, dict]]] = {a: {s: {} for s in SEEDS} for a in bench_arms}
    arm_budget_eps: dict[str, dict[int, dict[int, dict[str, dict]]]] = {
        a: {b: {s: {} for s in SEEDS} for b in BUDGET_CURVE} for a in bench_arms
    }

    for arm in bench_arms:
        gc.collect()
        for seed in SEEDS:
            rm, rsha = rel_models[seed]
            sm, ssha = sym_models[seed]
            hm, hsha = hyb_models[seed]
            if arm == "inc_py_cold":
                ctrl = IncrementalFrescoController(rm, rsha, use_rust=False)
            elif arm == "inc_py_warm":
                ctrl = IncrementalFrescoController(rm, rsha, use_rust=False)
            elif arm in ("inc_rust_cold", "inc_rust_warm", "rel_prov_adaptive_rust_cold"):
                ctrl = IncrementalFrescoController(rm, rsha, use_rust=True)
            elif arm == "sym_prov_readout_inc_rust_cold":
                ctrl = IncrementalSymController(sm, ssha, use_rust=True)
            elif arm == "hybrid_sym_rec_inc_rust_cold":
                ctrl = IncrementalSymController(hm, hsha, use_rust=True)
            else:
                ctrl = None

            # Pre-warm cache for `*_warm` arms with a single pass over the calibration frontier vocabulary
            is_warm = arm.endswith("_warm")
            if is_warm and ctrl is not None:
                for c in cal_cases:
                    _ = run_clean_episode(rg, c, controller=ctrl, budget=WORK, depth=DEPTH, cold_cache=False)
                ctrl.reset_telemetry()

            for b in BUDGET_CURVE:
                for c in cal_cases:
                    if arm == "uncached_py":
                        ep = run_clean_episode(rg, c, uncached_model=rm, budget=b, depth=DEPTH)
                    elif arm == "sym_prov_readout_uncached_py":
                        ep = run_clean_episode(rg, c, uncached_model=sm, budget=b, depth=DEPTH)
                    elif arm == "rel_prov_adaptive_rust_cold":
                        ep = run_clean_episode(
                            rg, c, controller=ctrl, budget=b, depth=DEPTH,
                            cold_cache=True, adaptive_margins=adaptive_margins,
                        )
                    else:
                        ep = run_clean_episode(rg, c, controller=ctrl, budget=b, depth=DEPTH, cold_cache=(not is_warm))
                    arm_budget_eps[arm][b][seed][c.root] = ep
                    if b == WORK:
                        arm_eps_128[arm][seed][c.root] = ep

    # Verify 100% trajectory, work, and proof-witness equivalence across all 144 `(seed, root)` episodes!
    traj_matches_py = traj_matches_rs = traj_matches_warm = 0
    for seed in SEEDS:
        for r in cal_roots:
            u_ep = arm_eps_128["uncached_py"][seed][r]
            if arm_eps_128["inc_py_cold"][seed][r]["path"] == u_ep["path"] and arm_eps_128["inc_py_cold"][seed][r]["work"] == u_ep["work"]:
                traj_matches_py += 1
            if arm_eps_128["inc_rust_cold"][seed][r]["path"] == u_ep["path"] and arm_eps_128["inc_rust_cold"][seed][r]["work"] == u_ep["work"]:
                traj_matches_rs += 1
            if arm_eps_128["inc_rust_warm"][seed][r]["path"] == u_ep["path"] and arm_eps_128["inc_rust_warm"][seed][r]["work"] == u_ep["work"]:
                traj_matches_warm += 1
    if not (traj_matches_py == 144 and traj_matches_rs == 144 and traj_matches_warm == 144):
        raise RuntimeError(
            f"incremental trajectory mismatch: py={traj_matches_py}/144, rs={traj_matches_rs}/144, warm={traj_matches_warm}/144"
        )

    # Certify all new `hybrid_sym_rec_opened` and `rel_prov_adaptive` closed calibration paths in Lean 4.19.0!
    new_cal_certs = []
    case_by_root = {c.root: c for c in cal_cases}
    for arm_tag in ("hybrid_sym_rec_inc_rust_cold", "rel_prov_adaptive_rust_cold"):
        for seed in SEEDS:
            for r in cal_roots:
                ep = arm_eps_128[arm_tag][seed][r]
                if ep["closed"]:
                    new_cal_certs.append((case_by_root[r], ep["path"], f"cal_{arm_tag}_{seed}_{r}"))
    print(f"[LBF-18 Phase 1] Certifying {len(new_cal_certs)} new calibration proof paths in Lean 4.19.0...", flush=True)
    cal_cert_man = base.certify(rg, new_cal_certs, scratch_dir / "lbf18_cal_certs")
    (out_dir / "calibration_new_certificates").mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_new_certificates" / "manifest.json").write_text(
        json.dumps(cal_cert_man, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    clean_bench_summary = {
        arm: _summarize_clean_bench_arm(cal_roots, arm_eps_128[arm], arm_budget_eps[arm])
        for arm in bench_arms
    }

    # 8. Joint 2x2 Envelope x Matrix Bank Recurrence Factorial (Zero & Roll)
    print("[LBF-18 Phase 1] Running 2x2 Envelope x Matrix Bank recurrence factorial...", flush=True)
    fac_modes = ("env_zero", "mat_zero", "env_mat_zero", "env_roll", "mat_roll", "env_mat_roll")
    fac_budget_eps: dict[str, dict[int, dict[int, dict[str, dict]]]] = {
        "normal": arm_budget_eps["inc_rust_cold"],
    }
    for md in fac_modes:
        fac_budget_eps[md] = {b: {s: {} for s in SEEDS} for b in BUDGET_CURVE}
        for seed in SEEDS:
            rm, rsha = rel_models[seed]
            ctrl = IncrementalFrescoController(rm, rsha, use_rust=True)
            for b in BUDGET_CURVE:
                for c in cal_cases:
                    fac_budget_eps[md][b][seed][c.root] = run_clean_episode(
                        rg, c, controller=ctrl, mode=md, budget=b, depth=DEPTH, cold_cache=True
                    )

    def root_auc_vec(md_name: str) -> list[float]:
        per_b = [root_level_summary(cal_roots, fac_budget_eps[md_name][b], budget=b)["_root_c"] for b in BUDGET_CURVE]
        return [sum(per_b[bi][ri] for bi in range(len(BUDGET_CURVE))) / len(BUDGET_CURVE) for ri in range(len(cal_roots))]

    def root_pw128_vec(md_name: str) -> list[float]:
        return root_level_summary(cal_roots, fac_budget_eps[md_name][WORK], budget=WORK)["_root_w"]

    def root_c64_vec(md_name: str) -> list[float]:
        return root_level_summary(cal_roots, fac_budget_eps[md_name][64], budget=64)["_root_c"]

    factorial_summary = {
        "zero_factorial_auc": analyze_2x2_bank_factorial(
            cal_roots, root_auc_vec("normal"), root_auc_vec("env_zero"),
            root_auc_vec("mat_zero"), root_auc_vec("env_mat_zero"), higher_is_better=True,
        ),
        "zero_factorial_closure_b64": analyze_2x2_bank_factorial(
            cal_roots, root_c64_vec("normal"), root_c64_vec("env_zero"),
            root_c64_vec("mat_zero"), root_c64_vec("env_mat_zero"), higher_is_better=True,
        ),
        "zero_factorial_penalized_work_b128": analyze_2x2_bank_factorial(
            cal_roots, root_pw128_vec("normal"), root_pw128_vec("env_zero"),
            root_pw128_vec("mat_zero"), root_pw128_vec("env_mat_zero"), higher_is_better=False,
        ),
        "roll_factorial_auc": analyze_2x2_bank_factorial(
            cal_roots, root_auc_vec("normal"), root_auc_vec("env_roll"),
            root_auc_vec("mat_roll"), root_auc_vec("env_mat_roll"), higher_is_better=True,
        ),
        "roll_factorial_penalized_work_b128": analyze_2x2_bank_factorial(
            cal_roots, root_pw128_vec("normal"), root_pw128_vec("env_roll"),
            root_pw128_vec("mat_roll"), root_pw128_vec("env_mat_roll"), higher_is_better=False,
        ),
    }

    # 9. Multi-axis scaling & adversarial provenance-collision stress suite
    print("[LBF-18 Phase 1] Running 5-track scaling & adversarial provenance-collision stress suite...", flush=True)
    stress_raw = run_stress_test_suite(seed=20261818)
    deep_specs = stress_raw.pop("_deep_search_specs")
    stress_raw["track_s4b_deep_16step_search"] = evaluate_deep_search_stress(rg, deep_specs, rel_models)

    # Strip internal `_root_c`/`_root_w` arrays before writing clean JSON, after computing paired contrasts
    cal_contrasts = {
        "hybrid_minus_sym_prov_readout": paired_root_contrast(
            cal_roots,
            clean_bench_summary["hybrid_sym_rec_inc_rust_cold"]["_root_c"],
            clean_bench_summary["hybrid_sym_rec_inc_rust_cold"]["_root_w"],
            clean_bench_summary["sym_prov_readout_inc_rust_cold"]["_root_c"],
            clean_bench_summary["sym_prov_readout_inc_rust_cold"]["_root_w"],
        ),
        "rel_prov_minus_hybrid": paired_root_contrast(
            cal_roots,
            clean_bench_summary["inc_rust_cold"]["_root_c"],
            clean_bench_summary["inc_rust_cold"]["_root_w"],
            clean_bench_summary["hybrid_sym_rec_inc_rust_cold"]["_root_c"],
            clean_bench_summary["hybrid_sym_rec_inc_rust_cold"]["_root_w"],
        ),
        "rel_prov_adaptive_minus_full": paired_root_contrast(
            cal_roots,
            clean_bench_summary["rel_prov_adaptive_rust_cold"]["_root_c"],
            clean_bench_summary["rel_prov_adaptive_rust_cold"]["_root_w"],
            clean_bench_summary["inc_rust_cold"]["_root_c"],
            clean_bench_summary["inc_rust_cold"]["_root_w"],
        ),
    }
    for v in clean_bench_summary.values():
        v.pop("_root_c", None)
        v.pop("_root_w", None)

    summary = {
        "schema": "lbf18.calibration_and_incremental.v1",
        "sealed_outcomes_opened": False,
        "rust_backend_compiled": RUST_BACKEND.available,
        "lean18_formalization": lean18_cert,
        "equivalence_verification": {
            "calibration_episodes_checked": 144,
            "uncached_vs_inc_py_cold_exact_path_matches": traj_matches_py,
            "uncached_vs_inc_rust_cold_exact_path_matches": traj_matches_rs,
            "uncached_vs_inc_rust_warm_exact_path_matches": traj_matches_warm,
            "fail_closed_invalidation_verified": True,
        },
        "hybrid_fits": hyb_fits,
        "adaptive_pass_calibration": adaptive_cal,
        "clean_inference_benchmark": clean_bench_summary,
        "calibration_root_contrasts": cal_contrasts,
        "calibration_new_lean_certificates": cal_cert_man,
        "factorial_2x2": factorial_summary,
        "stress_suite": stress_raw,
    }
    svg1_sha = render_incremental_and_factorial_svg(summary, out_dir / "fresco_incremental_architecture.svg")
    summary["fresco_incremental_architecture_svg_sha256"] = svg1_sha
    (out_dir / "calibration_phase_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_sealed_confirmation_phase(rg, out_dir: Path, scratch_dir: Path, registration_path: Path) -> dict:
    """Phase 2: Frozen Independent Confirmation on Sealed Ordinary & OOD Cohorts (`20261227` + `20261221`).
    Refuses to run unless `registration_path` exists and `calibration_phase_summary.json` is frozen."""
    if not registration_path.exists():
        raise RuntimeError(f"Refusing to open sealed cohorts: missing registration file {registration_path}")
    cal_sum_path = out_dir / "calibration_phase_summary.json"
    if not cal_sum_path.exists():
        raise RuntimeError("Run --phase calibrate and commit registration before --phase confirm-sealed")

    torch.set_num_threads(1)
    base.certify = certify_single_thread
    reg_sha = hashlib.sha256(registration_path.read_bytes()).hexdigest()
    cal_sum = json.loads(cal_sum_path.read_text(encoding="utf-8"))
    adaptive_margins = (
        float(cal_sum["adaptive_pass_calibration"]["gamma1"]),
        float(cal_sum["adaptive_pass_calibration"]["gamma2"]),
    )

    # Load frozen checkpoints
    rel_m, sym_m, lbf16_m, hyb_m = {}, {}, {}, {}
    for seed in SEEDS:
        rel_m[seed] = load_frozen_lbf17_checkpoint(seed, "rel_prov_opened")[:2]
        sym_m[seed] = load_frozen_lbf17_checkpoint(seed, "sym_prov_readout")[:2]
        lbf16_m[seed] = load_frozen_lbf17_checkpoint(seed, "lbf16_pos1_opened")[:2]

        ck_dir = out_dir / f"{seed}_hybrid_sym_rec_opened"
        fman = json.loads((ck_dir / "manifest.json").read_text(encoding="utf-8"))
        raw = gzip.decompress((ck_dir / "checkpoint.pt.gz").read_bytes())
        if hashlib.sha256(raw).hexdigest() != fman["checkpoint_sha256"]:
            raise ValueError("frozen hybrid checkpoint SHA mismatch")
        hm = HybridSymRecFrescoNet().cpu().float().eval()
        hm.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["weights"])
        for p in hm.parameters():
            p.requires_grad_(False)
        hyb_m[seed] = (hm, fman["checkpoint_sha256"])

    # Load all 4 untouched sealed splits (`20261227` ordinary/ood + `20261221` ordinary/ood)
    suite17_dir = LBF17_RESULTS_DIR / "suite"
    suite16_tmp = scratch_dir / "suite_20261221_sealed"
    if not suite16_tmp.exists():
        rg.freeze(suite16_tmp, seed=LBF16_SEED, counts=m17.COUNTS)
    if rg.file_sha(suite16_tmp / "manifest.json") != LBF16_SUITE_MANIFEST_SHA256:
        raise ValueError("LBF-16 frozen suite manifest SHA mismatch")

    sealed_cohorts = {
        "lbf17_ordinary": _load_gzipped_cases(suite17_dir, "ordinary", rg),
        "lbf17_ood": _load_gzipped_cases(suite17_dir, "ood", rg),
        "lbf16_ordinary": _load_gzipped_cases(suite16_tmp, "ordinary", rg),
        "lbf16_ood": _load_gzipped_cases(suite16_tmp, "ood", rg),
    }

    learned_arms = ("rel_prov_opened", "rel_prov_adaptive", "hybrid_sym_rec_opened", "sym_prov_readout", "lbf16_pos1_opened")
    symbolic_arms = ("bidir", "astar", "geom_mm", "graph_bridge", "exact_lookahead", "uniform")
    all_arms = learned_arms + symbolic_arms

    cohort_results = {}
    pooled_eps: dict[str, dict[int, dict[str, dict]]] = {a: {s: {} for s in SEEDS} for a in learned_arms}
    pooled_sym: dict[str, dict[str, tuple[float, float]]] = {a: {} for a in symbolic_arms}
    pooled_ord_roots, pooled_ood_roots, all_sealed_roots = [], [], []
    sealed_certs_to_compile = []

    for cname, cases in sealed_cohorts.items():
        print(f"[LBF-18 Phase 2] Evaluating frozen protocol on sealed cohort {cname} ({len(cases)} roots)...", flush=True)
        roots = [f"{cname}::{c.root}" for c in cases]
        all_sealed_roots.extend(roots)
        if cname.endswith("_ordinary"):
            pooled_ord_roots.extend(roots)
        else:
            pooled_ood_roots.extend(roots)

        # Verify uncached == incremental on sealed `rel_prov_opened` as an online fail-closed check!
        arm_b_eps: dict[str, dict[int, dict[int, dict[str, dict]]]] = {
            a: {b: {s: {} for s in SEEDS} for b in BUDGET_CURVE} for a in learned_arms
        }
        for seed in SEEDS:
            c_rel = IncrementalFrescoController(rel_m[seed][0], rel_m[seed][1], use_rust=True)
            c_adp = IncrementalFrescoController(rel_m[seed][0], rel_m[seed][1], use_rust=True)
            c_hyb = IncrementalSymController(hyb_m[seed][0], hyb_m[seed][1], use_rust=True)
            c_sym = IncrementalSymController(sym_m[seed][0], sym_m[seed][1], use_rust=True)
            m_16 = lbf16_m[seed][0]

            for b in BUDGET_CURVE:
                for c, rk_id in zip(cases, roots):
                    ep_rel = run_clean_episode(rg, c, controller=c_rel, budget=b, depth=DEPTH, cold_cache=True)
                    if b == WORK:
                        # Cross-verify against uncached PyTorch on every sealed root at B=128!
                        ep_unc = run_clean_episode(rg, c, uncached_model=rel_m[seed][0], budget=b, depth=DEPTH)
                        if ep_rel["path"] != ep_unc["path"] or ep_rel["work"] != ep_unc["work"]:
                            raise RuntimeError(f"sealed uncached vs incremental mismatch on {rk_id} seed {seed}")
                    ep_adp = run_clean_episode(
                        rg, c, controller=c_adp, budget=b, depth=DEPTH, cold_cache=True, adaptive_margins=adaptive_margins
                    )
                    ep_hyb = run_clean_episode(rg, c, controller=c_hyb, budget=b, depth=DEPTH, cold_cache=True)
                    ep_sym = run_clean_episode(rg, c, controller=c_sym, budget=b, depth=DEPTH, cold_cache=True)
                    ep_16 = m17.run_lbf17_episode(rg, c, "learned", model=m_16, arch="lbf16_pos1_opened", budget=b, depth=DEPTH)

                    arm_b_eps["rel_prov_opened"][b][seed][rk_id] = ep_rel
                    arm_b_eps["rel_prov_adaptive"][b][seed][rk_id] = ep_adp
                    arm_b_eps["hybrid_sym_rec_opened"][b][seed][rk_id] = ep_hyb
                    arm_b_eps["sym_prov_readout"][b][seed][rk_id] = ep_sym
                    arm_b_eps["lbf16_pos1_opened"][b][seed][rk_id] = ep_16

                    if b == WORK:
                        for aname, ep_obj in (
                            ("rel_prov_opened", ep_rel),
                            ("rel_prov_adaptive", ep_adp),
                            ("hybrid_sym_rec_opened", ep_hyb),
                            ("sym_prov_readout", ep_sym),
                            ("lbf16_pos1_opened", ep_16),
                        ):
                            pooled_eps[aname][seed][rk_id] = ep_obj
                            if ep_obj["closed"]:
                                sealed_certs_to_compile.append(
                                    (c, ep_obj["path"], f"sealed_{cname}_{aname}_{seed}_{c.root}")
                                )

        sym_b_rates: dict[str, dict[int, tuple[list[float], list[float]]]] = {a: {} for a in symbolic_arms}
        for sa in symbolic_arms:
            for b in BUDGET_CURVE:
                rc_l, rw_l = [], []
                for c, rk_id in zip(cases, roots):
                    if sa == "uniform":
                        u_runs = [m17.run_lbf17_episode(rg, c, "uniform", seed=st, budget=b, depth=DEPTH) for st in STREAMS]
                        c_val = sum(1.0 if x["closed"] else 0.0 for x in u_runs) / len(STREAMS)
                        w_val = sum(float(x["work"] + (0 if x["closed"] else 1)) for x in u_runs) / len(STREAMS)
                        if b == WORK:
                            for st, x in zip(STREAMS, u_runs):
                                if x["closed"]:
                                    sealed_certs_to_compile.append((c, x["path"], f"sealed_{cname}_uniform_{st}_{c.root}"))
                    else:
                        ep_s = m17.run_lbf17_episode(rg, c, sa, budget=b, depth=DEPTH)
                        c_val = 1.0 if ep_s["closed"] else 0.0
                        w_val = float(ep_s["work"] + (0 if ep_s["closed"] else 1))
                        if b == WORK and ep_s["closed"]:
                            sealed_certs_to_compile.append((c, ep_s["path"], f"sealed_{cname}_{sa}_{c.root}"))
                    rc_l.append(c_val)
                    rw_l.append(w_val)
                    if b == WORK:
                        pooled_sym[sa][rk_id] = (c_val, w_val)
                sym_b_rates[sa][b] = (rc_l, rw_l)

        # Build root-level (`R = 16`) summary & contrasts for this cohort
        c_rates_128 = {}
        c_work_128 = {}
        c_auc = {}
        learned_rl = {}
        for la in learned_arms:
            rl = root_level_summary(roots, arm_b_eps[la][WORK], budget=WORK)
            learned_rl[la] = rl
            c_rates_128[la] = rl["mean_closure_rate"]
            c_work_128[la] = rl["mean_penalized_work"]
            c_auc[la] = sum(
                root_level_summary(roots, arm_b_eps[la][b], budget=b)["mean_closure_rate"] for b in BUDGET_CURVE
            ) / len(BUDGET_CURVE)
        for sa in symbolic_arms:
            rc128, rw128 = sym_b_rates[sa][WORK]
            c_rates_128[sa] = sum(rc128) / len(roots)
            c_work_128[sa] = sum(rw128) / len(roots)
            c_auc[sa] = sum(sum(sym_b_rates[sa][b][0]) / len(roots) for b in BUDGET_CURVE) / len(BUDGET_CURVE)

        rel_rc = learned_rl["rel_prov_opened"]["_root_c"]
        rel_rw = learned_rl["rel_prov_opened"]["_root_w"]
        contrasts = {}
        for comp in ("bidir", "exact_lookahead", "geom_mm", "graph_bridge"):
            contrasts[f"rel_prov_minus_{comp}"] = paired_root_contrast(
                roots, rel_rc, rel_rw, sym_b_rates[comp][WORK][0], sym_b_rates[comp][WORK][1]
            )
        for comp_l in ("lbf16_pos1_opened", "sym_prov_readout", "hybrid_sym_rec_opened", "rel_prov_adaptive"):
            contrasts[f"rel_prov_minus_{comp_l}"] = paired_root_contrast(
                roots, rel_rc, rel_rw, learned_rl[comp_l]["_root_c"], learned_rl[comp_l]["_root_w"]
            )
        for v in learned_rl.values():
            v.pop("_root_c", None)
            v.pop("_root_w", None)

        cohort_results[cname] = {
            "independent_roots": len(roots),
            "rates_b128": c_rates_128,
            "mean_penalized_work_b128": c_work_128,
            "budget_curve_auc": c_auc,
            "learned_root_level_details": learned_rl,
            "paired_root_contrasts_b128": contrasts,
        }

    # Compile EVERY closed proof path across all 64 sealed roots standalone in Lean 4.19.0!
    print(f"[LBF-18 Phase 2] Compiling {len(sealed_certs_to_compile)} standalone Lean 4.19.0 sealed proof certificates...", flush=True)
    sealed_cert_man = base.certify(rg, sealed_certs_to_compile, scratch_dir / "lbf18_sealed_certs")
    (out_dir / "sealed_confirmation_certificates").mkdir(parents=True, exist_ok=True)
    (out_dir / "sealed_confirmation_certificates" / "manifest.json").write_text(
        json.dumps(sealed_cert_man, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def build_pooled(subset_roots: list[str]) -> dict:
        out_p = {}
        for la in learned_arms:
            sub_eps = {s: {r: pooled_eps[la][s][r] for r in subset_roots} for s in SEEDS}
            rl = root_level_summary(subset_roots, sub_eps, budget=WORK)
            out_p[la] = rl
        for sa in symbolic_arms:
            rc_s = [pooled_sym[sa][r][0] for r in subset_roots]
            rw_s = [pooled_sym[sa][r][1] for r in subset_roots]
            out_p[sa] = {
                "independent_roots": len(subset_roots),
                "mean_closure_rate": sum(rc_s) / len(subset_roots),
                "mean_penalized_work": sum(rw_s) / len(subset_roots),
                "_root_c": rc_s,
                "_root_w": rw_s,
            }
        rel_c = out_p["rel_prov_opened"]["_root_c"]
        rel_w = out_p["rel_prov_opened"]["_root_w"]
        p_con = {}
        for other in all_arms:
            if other == "rel_prov_opened":
                continue
            p_con[f"rel_prov_minus_{other}"] = paired_root_contrast(
                subset_roots, rel_c, rel_w, out_p[other]["_root_c"], out_p[other]["_root_w"]
            )
        for v in out_p.values():
            v.pop("_root_c", None)
            v.pop("_root_w", None)
        out_p["paired_root_contrasts_b128"] = p_con
        return out_p

    pooled_ord = build_pooled(pooled_ord_roots)
    pooled_ood = build_pooled(pooled_ood_roots)
    pooled_all64 = build_pooled(all_sealed_roots)

    # Evaluate pre-declared confirmation screens (`C1, C2, C3, C4`)
    strongest_heur_64 = max(pooled_all64["geom_mm"]["mean_closure_rate"], pooled_all64["graph_bridge"]["mean_closure_rate"])
    strongest_heur_work = min(pooled_all64["geom_mm"]["mean_penalized_work"], pooled_all64["graph_bridge"]["mean_penalized_work"])
    screens = {
        "screen_C1_sealed_vs_bidir_and_exact_lookahead": bool(
            pooled_all64["rel_prov_opened"]["mean_closure_rate"] >= pooled_all64["bidir"]["mean_closure_rate"] + 0.15
            and pooled_all64["rel_prov_opened"]["mean_penalized_work"] <= 0.70 * pooled_all64["bidir"]["mean_penalized_work"]
            and pooled_all64["rel_prov_opened"]["mean_closure_rate"] > pooled_all64["exact_lookahead"]["mean_closure_rate"]
        ),
        "screen_C2_sealed_vs_strongest_heuristic": bool(
            pooled_all64["rel_prov_opened"]["mean_closure_rate"] >= strongest_heur_64 + 0.05
            and pooled_all64["rel_prov_opened"]["mean_penalized_work"] <= 0.80 * strongest_heur_work
        ),
        "screen_C3_ood_generalization_and_vs_lbf16": bool(
            pooled_ood["rel_prov_opened"]["mean_closure_rate"] >= 0.85
            and pooled_all64["rel_prov_opened"]["mean_closure_rate"] >= pooled_all64["lbf16_pos1_opened"]["mean_closure_rate"]
            and pooled_all64["rel_prov_opened"]["mean_penalized_work"] < pooled_all64["lbf16_pos1_opened"]["mean_penalized_work"]
        ),
        "screen_C4_zero_uncached_divergence_and_lean_certified": bool(
            sealed_cert_man["compiled_axiom_free"] == len(sealed_certs_to_compile)
        ),
    }

    conf = {
        "schema": "lbf18.sealed_confirmation.v1",
        "registration_sha256": reg_sha,
        "lbf17_suite_manifest_sha256": LBF17_SUITE_MANIFEST_SHA256,
        "lbf16_suite_manifest_sha256": LBF16_SUITE_MANIFEST_SHA256,
        "frozen_adaptive_margins": {"gamma1": adaptive_margins[0], "gamma2": adaptive_margins[1]},
        "cohorts": cohort_results,
        "pooled_32_ordinary_roots": pooled_ord,
        "pooled_32_ood_roots": pooled_ood,
        "pooled_64_sealed_roots": pooled_all64,
        "lean_sealed_certificates_compiled": sealed_cert_man["compiled_axiom_free"],
        "lean_sealed_certificates_manifest": sealed_cert_man,
        "confirmation_screens": screens,
    }
    svg2_sha = render_confirmation_and_collisions_svg(
        conf, cal_sum["stress_suite"], out_dir / "fresco_confirmation_and_collisions.svg"
    )
    conf["fresco_confirmation_and_collisions_svg_sha256"] = svg2_sha
    (out_dir / "sealed_confirmation_summary.json").write_text(
        json.dumps(conf, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return conf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LBF-18 Incremental Relational Fresco & Sealed Confirmation")
    ap.add_argument("--phase", choices=("calibrate", "confirm-sealed"), required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "lbf18")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--registration", type=Path, default=HERE / "LBF_18_CONFIRMATION_REGISTRATION.md")
    args = ap.parse_args(argv)

    rg = base.load_upstream(args.upstream)
    if args.phase == "calibrate":
        run_calibration_phase(rg, args.out, args.scratch)
    else:
        run_sealed_confirmation_phase(rg, args.out, args.scratch, args.registration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
