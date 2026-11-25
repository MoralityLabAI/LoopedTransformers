"""LBF-19 harness: expensive-verifier proof search with actual Lean tactic execution in the loop.

Arms run against a `NativeSession` (`lbf19_native_verifier`).  Every novel successor discovered by an arm
(and every novel word touched by an exact lookahead ball) costs one native `conv => ...; rw [...]` tactic
request against Lean's elaborator; duplicates reuse the verified state.  `mode='count'` replays the identical
deduplication without Lean and therefore gives the exact verifier-call counts of the `lean` mode.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import itertools
import json
import math
import pathlib
import random
import sys
import time

import torch

import os

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lbf15_meeting_fresco as m15  # noqa: E402
import lbf18_incremental_fresco as m18  # noqa: E402
import lbf19_native_verifier as nv  # noqa: E402
import lbf19_fast_controller as fc  # noqa: E402
import lbf19_scaled_domain as sd  # noqa: E402

RESULTS = HERE / "results" / "lbf19"
PINNED = pathlib.Path(os.environ.get("LBF19_PINNED_RG", str(HERE.parents[0] / "lean_cgrt_hrm400" / "ppt_gym" / "rewrite_gym.py")))
SEEDS = (107, 109, 113)
DEPTH = 24

LEARNED_ARMS = {
    "sym_prov_readout": ("sym", "sym_prov_readout"),
    "rel_prov_guarded": ("rel", "rel_prov_opened"),
    "hybrid_sym_rec_opened": ("hyb", "hybrid_sym_rec_opened"),
    "lbf16_pos1_opened": ("unc", "lbf16_pos1_opened"),
    "compute_nonrecurrent": ("unc", "compute_nonrecurrent"),
}
SYMBOLIC_ARMS = ("geom_mm", "graph_bridge", "absint_mm", "exact_mm")
CLASSICAL_ARMS = ("bidir_uncapped",)


def load_rg():
    return m18.base.load_upstream(PINNED)


class CapReached(RuntimeError):
    pass


class _CappedSession(nv.NativeSession):
    """Aborts the search as soon as the verifier-call cap is hit (censoring); counts stay exact up to the cap."""

    def discover(self, side, parent, mv, child):
        if self.call_cap is not None and self.calls >= self.call_cap and child not in self.states[side]:
            self.exceeded = True
            raise CapReached()
        super().discover(side, parent, mv, child)


# ------------------------------------------------------------------------------------------------------
class ModelBank:
    def __init__(self):
        self.ctrl = {}
        self.models = {}

    def get(self, arm: str, seed: int):
        key = (arm, seed)
        if key in self.ctrl:
            return self.ctrl[key]
        kind, arch = LEARNED_ARMS[arm]
        if kind == "hyb":
            ck_dir = m18.RESULTS_DIR / f"{seed}_{arch}" if hasattr(m18, "RESULTS_DIR") else HERE / "results" / "lbf18" / f"{seed}_{arch}"
            fman = json.loads((ck_dir / "manifest.json").read_text(encoding="utf-8"))
            raw = gzip.decompress((ck_dir / "checkpoint.pt.gz").read_bytes())
            assert hashlib.sha256(raw).hexdigest() == fman["checkpoint_sha256"]
            m = m18.HybridSymRecFrescoNet().cpu().float().eval()
            m.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["weights"])
            for p in m.parameters():
                p.requires_grad_(False)
            sha = fman["checkpoint_sha256"]
        else:
            m, sha, _ = m18.load_frozen_lbf17_checkpoint(seed, arch)
        if kind == "rel":
            c = m18.IncrementalFrescoController(m, sha, use_rust=True, guard_mode="fable_len_guard")
        elif kind in ("sym", "hyb"):
            c = fc.FastSymController(m, sha, use_rust=True, guard_mode="fable_len_guard")
        else:
            c = ("uncached", m)
        self.ctrl[key] = (c, sha)
        return self.ctrl[key]


# ------------------------------------------------------------------------------------------------------
class LeanBallOracle(m18.ExactBallOracle):
    """Exact 3-hop lookahead whose speculative successor generation is itself verified natively."""

    def __init__(self, rg, case, radius, session):
        super().__init__(rg, case, radius)
        self.session = session
        self.sided = {}
        self.pair = {}

    def ball_side(self, w, side):
        key = (side, w)
        got = self.sided.get(key)
        if got is not None:
            return got
        seen = {w: 0}
        frontier = [w]
        for d in range(1, self.radius + 1):
            nxt = []
            for x in frontier:
                self.ball_nodes_expanded += 1
                self.unique_expanded_words.add(x)
                for mv in self.rg.neighbors(self.case, x):
                    self.generated_successors += 1
                    y = mv.successor
                    if y not in seen:
                        seen[y] = d
                        nxt.append(y)
                        self.session.discover(side, x, mv, y)
            frontier = nxt
        self.sided[key] = seen
        return seen

    def dhat(self, us, vs):
        vb = [self.ball_side(v, 1) for v in vs]
        rows = []
        for u in us:
            bu = self.ball_side(u, 0)
            row = []
            for v, bv in zip(vs, vb):
                k = (u, v)
                val = self.pair.get(k)
                if val is None:
                    small, big = (bu, bv) if len(bu) <= len(bv) else (bv, bu)
                    best = self.no_meet
                    for wm in small.keys() & big.keys():
                        s = bu[wm] + bv[wm]
                        if s < best:
                            best = s
                    val = best
                    self.pair[k] = val
                row.append(val)
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long).reshape(len(us), len(vs))


@contextlib.contextmanager
def verifier_patch(session):
    orig_expand = m15.expand_node
    orig_oracle = m18.ExactBallOracle

    def expand(rg, c, s, side, idx, budget=m15.WORK, depth=m15.DEPTH, nb_fn=None):
        n0 = len(s.dist[side])
        orig_expand(rg, c, s, side, idx, budget, depth, nb_fn)
        if len(s.dist[side]) > n0:
            for child in list(itertools.islice(s.dist[side], n0, None)):
                parent, mv = s.parent[side][child]
                session.discover(side, parent, mv, child)

    m15.expand_node = expand
    class _Patched(LeanBallOracle):
        def __init__(self, rg, case, radius=3):
            super().__init__(rg, case, radius, session)

    m18.ExactBallOracle = _Patched
    try:
        yield
    finally:
        m15.expand_node = orig_expand
        m18.ExactBallOracle = orig_oracle


# ------------------------------------------------------------------------------------------------------
def _path_dicts(path):
    out = []
    for mv in path:
        if isinstance(mv, dict):
            out.append(mv)
        else:
            out.append({"rule": mv.rule, "direction": mv.direction, "position": mv.position, "successor": mv.successor})
    return out


def run_arm(rg, case, arm: str, *, seed: int = 0, budget: int, bank: ModelBank | None, repl, mode: str,
            call_cap: int | None, certify: bool = False, cert_dir: pathlib.Path | None = None) -> dict:
    sess_cls = _CappedSession if call_cap is not None else nv.NativeSession
    sess = sess_cls(case, repl, mode, call_cap=call_cap)
    censored = False
    t0 = time.perf_counter()
    try:
        if arm == "bidir_uncapped":
            r = sd.classical_bidir(case, budget=10 ** 12, depth=DEPTH, on_discover=sess.discover)
            res = {"closed": r["closed"], "work": r["work"], "expanded": r["expanded"], "path": r["path"],
                   "visited": r["visited"]}
        elif arm in SYMBOLIC_ARMS:
            with verifier_patch(sess):
                r = m18.run_unified_symbolic_episode(rg, case, arm, budget=budget, depth=DEPTH)
            res = {k: r[k] for k in ("closed", "work", "expanded")}
            res["path"] = _path_dicts(r["path"]) if r["closed"] else None
            res["fringe_decisions"] = r.get("fringe_decisions", 0)
        elif arm in LEARNED_ARMS:
            c, sha = bank.get(arm, seed)
            with verifier_patch(sess):
                if isinstance(c, tuple):
                    r = m18.run_clean_episode(rg, case, uncached_model=c[1], uncached_guard_mode="legacy",
                                              budget=budget, depth=DEPTH)
                elif isinstance(c, fc.FastSymController):
                    r = fc.run_fast_episode(rg, case, c, budget=budget, depth=DEPTH)
                else:
                    r = m18.run_clean_episode(rg, case, controller=c, budget=budget, depth=DEPTH, cold_cache=True)
            res = {k: r[k] for k in ("closed", "work", "expanded", "fringe_decisions", "neural_flops",
                                     "pair_evaluations", "word_hits", "word_misses", "pair_hits", "pair_misses")}
            res["path"] = _path_dicts(r["path"]) if r["closed"] else None
            res["ckpt_sha256"] = sha
        else:
            raise ValueError(arm)
    except CapReached:
        censored = True
        res = {"closed": False, "work": None, "expanded": None, "path": None}
    wall = time.perf_counter() - t0
    out = {"arm": arm, "seed": seed, "budget": budget, "root": case.root, "L": len(case.source),
           "wall_seconds": wall, "censored_by_call_cap": censored,
           "path_len": (len(res["path"]) if res.get("path") else None), **{k: v for k, v in res.items() if k != "path"},
           **{f"v_{k}": v for k, v in sess.summary().items()}}
    if res.get("path"):
        out["path"] = [(m["rule"], m["direction"], m["position"]) for m in res["path"]]
    if mode == "lean" and certify and res.get("path"):
        cert = sess.certify(res["path"])  # native replay + kernel check (standalone compile batched by caller)
        out["cert"] = cert
    return out


def tape_hash(tape):
    return hashlib.sha256("\n".join(tape).encode()).hexdigest()
