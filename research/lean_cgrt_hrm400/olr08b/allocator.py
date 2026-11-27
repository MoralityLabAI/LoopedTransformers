"""OLR-08B Adaptive Proof Frontier Allocator & Search Engine.

Supports:
  - Fixed single-prior controls: ``s1`` (uniform), ``s2`` (hand), ``s3`` (counted), ``s3c`` (cost-aware)
  - Fixed-budget portfolios: ``portfolio_rr2`` (s2, s3), ``portfolio_rr4`` (s1, s2, s3, s3c),
    ``portfolio_weighted`` (s3, s2, s3, s3c)
  - Online adaptive non-neural allocator: ``adaptive_ucb`` over (s1, s2, s3, s3c)
  - Exact explicit-memory allocator: ``explicit_memory`` backed by ``ProofMemoryCache``
  - Learned adaptive utility allocators: ``recurrent``, ``recurrent_bpd``, ``direct_mlp``,
    ``matched_compute_sham``, ``one_way``, ``linear``

Preserves ALL OLR-08A scientific and infrastructure invariants:
  - Pre-command scope checks (``pre_command_session_v1``, ``env == before``, ``decl_in_env == False``)
  - Fresh pre-command path replay with library-search warm-up (repair 0001)
  - Premise enumerator heartbeat timeout -> empty premise proposals, still charged (repair 0002)
  - Unified ``Meter`` charging ALL native RPCs (premises, step, cert, graph, lemma_graph)
  - Append-only SHA-256-chained ``ImmutableReceiptLedger``
  - Configurable MMS memory budget (0, 64, 256, 1024, 4096 MiB) with build/lookup/maintenance accounting
  - No lossy cross-path PP state deduplication
"""
from __future__ import annotations

from collections import Counter
import heapq
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

OLR08_DIR = Path(__file__).resolve().parent.parent / "olr08"
if str(OLR08_DIR) not in sys.path:
    sys.path.insert(0, str(OLR08_DIR))

import probe  # noqa: E402
import run08  # noqa: E402
from search08 import FREE, RANKED, actions, reprice  # noqa: E402

from bpd_frontier import BPDFrontierDecomposer  # noqa: E402
from mms_cache import EpisodicRecord, ImmutableReceiptLedger, ProofMemoryCache  # noqa: E402
from models import (  # noqa: E402
    ExplicitMemoryPolicy,
    ReceiptTracker,
    UtilityPredictor,
    build_packet,
    predict_costs_and_gates,
)

BASE_PRIORS = ("s1", "s2", "s3", "s3c")
PORTFOLIO_SCHEDULES = {
    "portfolio_rr2": ("s2", "s3"),
    "portfolio_rr4": ("s1", "s2", "s3", "s3c"),
    "portfolio_weighted": ("s3", "s2", "s3", "s3c"),
}


class AuditedMeter(run08.Meter):
    """Meter subclass that appends every native verifier RPC to an ImmutableReceiptLedger."""

    def __init__(self, L: run08.Lean, ledger: ImmutableReceiptLedger, steps=run08.STEP_CAP, rpcs=run08.RPC_CAP, hb=run08.HB_CAP):
        super().__init__(L, steps=steps, rpcs=rpcs, hb=hb)
        self.ledger = ledger

    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        r = super().call(op, **kw)
        # Record immutable receipt snapshot (omit massive graph payloads from ledger digest bloat? No: store exact digest + verdict fields)
        rec = {
            "op": op,
            "ok": r.get("ok"),
            "ngoals": r.get("ngoals"),
            "hb_raw": r.get("request_hb_raw"),
            "server_ms": r.get("request_ms"),
            "error": str(r.get("error", ""))[:240] if r.get("error") is not None else None,
        }
        if op == "step":
            rec["sid"] = kw.get("sid")
            rec["tactic"] = kw.get("tactic")
            rec["child_sid"] = r.get("sid")
        elif op == "cert":
            rec["sid"] = kw.get("sid")
            rec["cert"] = r.get("cert")
        elif op == "premises":
            rec["sid"] = kw.get("sid")
            rec["rw_n"] = len(r.get("rw", [])) if isinstance(r.get("rw"), list) else 0
            rec["apply_n"] = len(r.get("apply", [])) if isinstance(r.get("apply"), list) else 0
        self.ledger.append(op, rec)
        return r


def _all_prior_costs(acts: list[tuple], prior: dict[str, Any] | None, depth: int) -> dict[str, list[float]]:
    s1 = [1.0 for _ in acts]
    s2 = [float(a[0]) for a in acts]
    if prior is not None:
        s3_priced = reprice(acts, prior, depth)
        s3 = [float(a[0]) for a in s3_priced]
        s3c = [
            c + math.log2(max(1.0, prior["hb"].get(tac if k in FREE else k, prior["hb_mean"])))
            for (c, k, _, tac) in s3_priced
        ]
    else:
        s3 = list(s2)
        s3c = list(s2)
    return {"s1": s1, "s2": s2, "s3": s3, "s3c": s3c}


def _state_signature_vec(info: dict[str, Any] | None, depth: int, tracker: ReceiptTracker) -> np.ndarray:
    """Compact deterministic 24-dim proof-state + receipt signature for episodic similarity."""
    vec = np.zeros(24, dtype=np.float32)
    hyps = (info or {}).get("hyps") or []
    defs = (info or {}).get("defs") or []
    vec[0] = depth / 3.0
    vec[1] = min(len(hyps), 20) / 20.0
    vec[2] = sum(1 for h in hyps if h.get("prop")) / max(1, len(hyps))
    vec[3] = sum(1 for h in hyps if h.get("ind")) / max(1, len(hyps))
    vec[4] = min(len(defs), 10) / 10.0
    vec[5] = min((info or {}).get("nconsts", 0), 64) / 64.0
    for d in defs[:4]:
        idx = 6 + (int.from_bytes(d.encode("utf-8"), "little") % 6)
        vec[idx] += 1.0
    for h in hyps[:6]:
        head = str(h.get("head", ""))
        idx = 12 + (int.from_bytes(head.encode("utf-8"), "little") % 6)
        vec[idx] += 1.0
    vec[18] = tracker.ok_count / max(1, tracker.steps)
    vec[19] = min(tracker.fail_streak, 20) / 20.0
    vec[20] = tracker.no_match / max(1, tracker.steps)
    vec[21] = tracker.type_err / max(1, tracker.steps)
    vec[22] = tracker.last_ok
    vec[23] = tracker.last_ngoals_delta / 4.0
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-8 else vec


def _get_packet_cached(
    m: AuditedMeter,
    s: dict[str, Any],
    acts: list[tuple],
    depth: int,
    prior_costs: dict[str, list[float]],
    tracker: ReceiptTracker,
    cache: ProofMemoryCache,
    env_key: str,
    state_sig: np.ndarray,
    parent_ngoals: int,
) -> tuple | None:
    """Fetch graph + lemma graphs (reusing read-only cached lemma subtrees) and build packet."""
    g = m.call("graph", sid=s["sid"])
    if g.get("ok") is not True or not all(k in g for k in ("nodes", "edges", "obligations")):
        return None

    names = sorted({arg for _, k, arg, _ in acts if k.endswith("_lemma") or k == "rw_lemma_rev" or k in ("simp_def", "unfold")})
    lemmas: list[dict[str, Any]] = []
    missing: list[str] = []
    for nm in names:
        cached = cache.get_lemma_graph(env_key, nm)
        if cached is not None:
            lemmas.append(cached)
        else:
            missing.append(nm)
    if missing:
        l_resp = m.call("lemma_graph", sid=s["sid"], names=missing)
        if l_resp.get("ok") is not True or not isinstance(l_resp.get("lemmas"), list):
            return None
        for lem in l_resp["lemmas"]:
            lemmas.append(lem)
            cache.put_lemma_graph(env_key, lem["name"], lem)

    # Query episodic memory for each candidate template (timed under cache.lookup_seconds)
    mem_feats: list[tuple[float, float, float]] = []
    for _, k, _, tac in acts:
        tpl = tac if k in FREE else k
        hit, ok_r, util_e, _ = cache.query_episodic(state_sig, tactic_template=tpl)
        mem_feats.append((1.0 if hit else 0.0, ok_r, util_e))

    g16 = tracker.global_cond16(m.ops["step"], sum(m.ops.values()), m.hb)
    t0 = time.perf_counter()
    p = build_packet(g, lemmas, acts, depth, prior_costs, g16, mem_features=mem_feats, parent_ngoals=parent_ngoals)
    m.feature_seconds += time.perf_counter() - t0
    return p


class FrontierQueue:
    """One priority queue under a specific prior or learned utility scorer."""

    def __init__(
        self,
        name: str,
        meter: AuditedMeter,
        root: dict[str, Any],
        prior: dict[str, Any] | None,
        *,
        max_depth: int = run08.DEPTH,
        tracker: ReceiptTracker,
        cache: ProofMemoryCache,
        env_key: str,
        model: UtilityPredictor | None = None,
        bpd_decomposer: BPDFrontierDecomposer | None = None,
        explicit_policy: ExplicitMemoryPolicy | None = None,
        alpha: float = 0.0,
        arm_idx: int = 0,
        lazy_expand: bool = False,
        bpd_probe_log: list[dict[str, Any]] | None = None,
        haar_null_decomposer: BPDFrontierDecomposer | None = None,
        file_name: str = "",
    ) -> None:
        self.name = name
        self.meter = meter
        self.prior = prior
        self.max_depth = max_depth
        self.tracker = tracker
        self.cache = cache
        self.env_key = env_key
        self.model = model
        self.bpd_decomposer = bpd_decomposer
        self.haar_null_decomposer = haar_null_decomposer
        self.explicit_policy = explicit_policy
        self.alpha = float(alpha)
        self.arm_idx = arm_idx
        self.lazy_expand = lazy_expand
        self.bpd_probe_log = bpd_probe_log
        self.file_name = file_name

        self.heap: list = []
        self.serial = 0
        self.done = False
        self.path: list[str] | None = None
        self.on_path: list[int] = []
        self.cert: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.fallbacks = 0
        self.frontier_state = model.init_frontier_state() if model is not None else None
        self.last_prior_weights = [0.25, 0.25, 0.25, 0.25]
        self.last_state_summary = torch.zeros(40, dtype=torch.float32)

        self.expand(root, 0, 0.0, (), (), parent_ngoals=len(root.get("goals", [1])))

    def expand(
        self,
        s: dict[str, Any],
        depth: int,
        g_cost: float,
        path: tuple[str, ...],
        event_ids: tuple[int, ...],
        *,
        parent_ngoals: int = 1,
    ) -> None:
        acts = run08.enumerate_menu(self.meter, s)
        # Track premise timeout in receipt tracker from last rpc_journal entry
        if self.meter.journal and self.meter.journal[-1]["op"] == "premises":
            last_j = self.meter.journal[-1]
            self.tracker.observe_premise(bool(last_j.get("ok")), int(last_j.get("hb", 0)))

        pcosts = _all_prior_costs(acts, self.prior, depth)
        state_sig = _state_signature_vec(s.get("info"), depth, self.tracker)
        bpd_rows_for_state: np.ndarray | None = None
        haar_rows_for_state: np.ndarray | None = None
        cond_rows_for_state: np.ndarray | None = None

        if self.name in BASE_PRIORS:
            chosen_costs = pcosts[self.name]
            step_g_costs = pcosts["s3"] if self.name == "s3c" else chosen_costs
        elif self.name == "explicit_memory" and self.explicit_policy is not None:
            chosen_costs = self.explicit_policy.score_actions(acts, pcosts["s3"], depth, s.get("info"), cache=self.cache)
            step_g_costs = chosen_costs
        elif self.model is not None and self.alpha:
            p = _get_packet_cached(
                self.meter, s, acts, depth, pcosts, self.tracker, self.cache, self.env_key, state_sig, parent_ngoals
            )
            if p is None:
                self.fallbacks += 1
                chosen_costs = pcosts["s3"]
                step_g_costs = chosen_costs
            else:
                t0 = time.perf_counter()
                out = self.model.forward_full(p, frontier_state=self.frontier_state, bpd_decomposer=self.bpd_decomposer)
                logits = -math.log(2.0) * p[-1] + self.alpha * out["utility"]
                chosen_costs = (-torch.nn.functional.log_softmax(logits, dim=0) / math.log(2.0)).tolist()
                self.last_prior_weights = torch.nn.functional.softmax(out["prior_logits"], dim=0).tolist()
                self.last_state_summary = out["state_summary"].detach()
                self.meter.inference_seconds += time.perf_counter() - t0
                step_g_costs = chosen_costs
                if self.bpd_probe_log is not None and self.bpd_decomposer is not None and out["pass_action_states"]:
                    z_cur = self.frontier_state if self.frontier_state is not None else torch.zeros(40)
                    bpd_rows_for_state = self.bpd_decomposer.extract_action_features(out["pass_action_states"], z_cur).numpy()
                    if self.haar_null_decomposer is not None:
                        haar_rows_for_state = self.haar_null_decomposer.extract_action_features(
                            out["pass_action_states"], z_cur
                        ).numpy()
                    cond_rows_for_state = p[3][:, -24:].detach().numpy()
        else:
            chosen_costs = pcosts["s3"]
            step_g_costs = chosen_costs

        ranks: Counter = Counter()
        for idx, ((_, k, arg, tac), prio_c, g_c) in enumerate(zip(acts, chosen_costs, step_g_costs)):
            rk = ranks[k]
            ranks[k] += 1
            self.serial += 1
            probe_bundle = None
            if bpd_rows_for_state is not None:
                probe_bundle = (
                    bpd_rows_for_state[idx],
                    haar_rows_for_state[idx] if haar_rows_for_state is not None else np.zeros(16, dtype=np.float32),
                    cond_rows_for_state[idx] if cond_rows_for_state is not None else np.zeros(24, dtype=np.float32),
                )
            heapq.heappush(
                self.heap,
                (
                    g_cost + prio_c,
                    0,
                    self.serial,
                    s["sid"],
                    depth,
                    g_cost + g_c,
                    path,
                    event_ids,
                    k,
                    arg,
                    tac,
                    rk,
                    parent_ngoals,
                    s.get("info"),
                    state_sig,
                    probe_bundle,
                ),
            )

    def step(self) -> dict[str, Any] | None:
        if self.done:
            return None
        while self.heap:
            item = heapq.heappop(self.heap)
            if item[1] == 1:
                # Lazy expansion entry
                _, _, _, child_state, d, g_cost, path, ids, p_ngoals = item
                self.expand(child_state, d, g_cost, path, ids, parent_ngoals=p_ngoals)
                continue
            (
                _,
                _,
                _,
                sid,
                d,
                g_cost,
                path,
                ids,
                k,
                arg,
                tac,
                rk,
                parent_ngoals,
                parent_info,
                state_sig,
                probe_bundle,
            ) = item
            fail_before = self.tracker.fail_streak
            a = self.meter.call("step", sid=sid, tactic=tac, hb=500 if tac == "exact?" else 200)
            ok = a.get("ok") is True
            ngoals = a.get("ngoals") if ok else None
            err_cat = "closed" if (ok and ngoals == 0) else probe.category(
                {"ok": ok, "goals_after": ngoals, "error": a.get("error") or ""}
            )
            closed = False
            cert = None
            if ok and ngoals == 0:
                cert = self.meter.call("cert", sid=a["sid"]).get("cert", {})
                closed = run08.close_cert(cert)

            ev_idx = len(self.events)
            ev = {
                "kind": k,
                "argument": arg,
                "tactic": tac,
                "rank": rk,
                "depth": d,
                "ok": ok,
                "closed": closed,
                "ngoals": ngoals,
                "error_category": err_cat,
                "hb_raw": a["request_hb_raw"],
                "fail_streak_before": fail_before,
                "parent_events": list(ids),
            }
            self.events.append(ev)

            step_vec = self.tracker.observe_step(
                ok=ok,
                closed=closed,
                depth=d,
                parent_ngoals=parent_ngoals,
                child_ngoals=ngoals,
                error_category=err_cat,
                hb_raw=int(a["request_hb_raw"]),
                arm_idx=self.arm_idx,
            )
            if self.model is not None and self.frontier_state is not None:
                with torch.no_grad():
                    self.frontier_state = self.model.update_frontier_state(
                        self.frontier_state, self.last_state_summary, step_vec
                    )

            # Update MMS cache with read-only episodic experience (never alters immutable receipts)
            tpl = tac if k in FREE else k
            ghead = ExplicitMemoryPolicy._goal_head(parent_info)
            self.cache.update_exact_stat(
                f"ep|{ghead}|{min(d,1)}|{tpl}",
                ok=ok,
                closed=closed,
                hb_raw=int(a["request_hb_raw"]),
                after_fail_streak=fail_before,
            )
            self.cache.put_episodic(
                EpisodicRecord(
                    key=f"{self.name}:{ev_idx}:{d}:{tpl}",
                    signature=state_sig,
                    tactic_template=tpl,
                    action_kind=k,
                    ok=ok,
                    closed=closed,
                    ngoals_delta=float(parent_ngoals - (0 if closed else (ngoals if ngoals is not None else parent_ngoals))),
                    hb_raw=int(a["request_hb_raw"]),
                    after_fail_streak=fail_before,
                    step_idx=self.tracker.steps,
                )
            )
            if probe_bundle is not None and self.bpd_probe_log is not None:
                bpd_v, haar_v, cond_v = probe_bundle
                self.bpd_probe_log.append(
                    {
                        "event_idx": ev_idx,
                        "file": self.file_name,
                        "ok": int(ok),
                        "closed": int(closed),
                        "on_path": 0,  # backfilled if this queue solves
                        "fail_streak": fail_before,
                        "bpd": bpd_v.tolist(),
                        "haar_null": haar_v.tolist(),
                        "cond": cond_v.tolist(),
                    }
                )

            if closed:
                self.done = True
                self.path = [*path, tac]
                self.on_path = [*ids, ev_idx]
                self.cert = cert
                if self.bpd_probe_log is not None:
                    on_set = set(self.on_path)
                    for rec in self.bpd_probe_log:
                        if rec["event_idx"] in on_set:
                            rec["on_path"] = 1
                return ev

            if ok and ngoals and d + 1 < self.max_depth:
                new_path = (*path, tac)
                new_ids = (*ids, ev_idx)
                if self.lazy_expand:
                    self.serial += 1
                    heapq.heappush(self.heap, (g_cost + 0.5, 1, self.serial, a, d + 1, g_cost, new_path, new_ids, ngoals))
                else:
                    self.expand(a, d + 1, g_cost, new_path, new_ids, parent_ngoals=ngoals)
            return ev
        self.done = True
        return None


def search_olr08b(
    exe: Path,
    source: Path,
    row: dict[str, Any],
    arm: str,
    *,
    prior: dict[str, Any] | None = None,
    model: UtilityPredictor | None = None,
    bpd_decomposer: BPDFrontierDecomposer | None = None,
    explicit_policy: ExplicitMemoryPolicy | None = None,
    alpha: float = 0.0,
    limit: int = run08.STEP_CAP,
    max_depth: int = run08.DEPTH,
    mms_budget_mib: float = 64.0,
    lazy_expand: bool = False,
    collect_bpd_probe: bool = False,
) -> dict[str, Any]:
    """Execute one metered OLR-08B search episode with immutable receipts and MMS accounting."""
    start = time.perf_counter()
    scale = max(1, math.ceil(limit / run08.STEP_CAP))
    ledger = ImmutableReceiptLedger(row["id"])
    cache = ProofMemoryCache(budget_mib=mms_budget_mib)
    if bpd_decomposer is not None:
        bpd_decomposer.store_in_cache(cache)
    haar_null = bpd_decomposer.make_haar_null(seed=101) if (collect_bpd_probe and bpd_decomposer is not None) else None
    bpd_probe_log: list[dict[str, Any]] | None = [] if collect_bpd_probe else None

    env_key = f"{row['file']}:{row['line']}:{row['col']}"
    tracker = ReceiptTracker(step_cap=limit, rpc_cap=run08.RPC_CAP * scale, hb_cap=run08.HB_CAP * scale)

    with run08.Lean(exe) as L:
        s, why, opening = run08.open_root(L, source, row)
        ledger.append("open", {"file": row["file"], "decl": row["decl"], "ok": s is not None, "why": why, **opening})
        if s is None:
            raise ValueError(f"previous scored root lost: {row['id']} {why}")
        m = AuditedMeter(L, ledger, steps=limit, rpcs=run08.RPC_CAP * scale, hb=run08.HB_CAP * scale)
        qs: list[FrontierQueue] = []
        winner: FrontierQueue | None = None
        arm_selections: Counter = Counter()

        try:
            if arm in BASE_PRIORS:
                qs.append(
                    FrontierQueue(
                        arm, m, s, prior if arm in ("s3", "s3c") else None,
                        max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        arm_idx=BASE_PRIORS.index(arm), lazy_expand=lazy_expand,
                    )
                )
            elif arm in PORTFOLIO_SCHEDULES:
                sched = PORTFOLIO_SCHEDULES[arm]
                uniq_names = list(dict.fromkeys(sched))
                q_by_name = {
                    nm: FrontierQueue(
                        nm, m, s, prior if nm in ("s3", "s3c") else None,
                        max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        arm_idx=BASE_PRIORS.index(nm), lazy_expand=lazy_expand,
                    )
                    for nm in uniq_names
                }
                qs = [q_by_name[nm] for nm in uniq_names]
                step_turn = 0
                while not all(q.done for q in qs) and not m.exhausted():
                    target_name = sched[step_turn % len(sched)]
                    step_turn += 1
                    q = q_by_name[target_name]
                    if q.done:
                        continue
                    arm_selections[q.name] += 1
                    q.step()
                    if q.path is not None:
                        winner = q
                        break
            elif arm == "adaptive_ucb":
                # Online non-neural UCB allocator across uniform, hand, counted, cost-aware priors
                qs = [
                    FrontierQueue(
                        nm, m, s, prior if nm in ("s3", "s3c") else None,
                        max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        arm_idx=idx, lazy_expand=True,
                    )
                    for idx, nm in enumerate(BASE_PRIORS)
                ]
                ucb_n = [0, 0, 0, 0]
                ucb_r = [0.0, 0.0, 0.0, 0.0]
                total_t = 0
                while not all(q.done for q in qs) and not m.exhausted():
                    live = [i for i, q in enumerate(qs) if not q.done]
                    if not live:
                        break
                    # Warm-start each prior for 4 steps, then UCB1 with remaining-budget urgency
                    uninit = [i for i in live if ucb_n[i] < 4]
                    if uninit:
                        pick = uninit[0]
                    else:
                        rem_frac = max(0.0, 1.0 - m.ops["step"] / limit)
                        c_exp = 0.65 * (0.4 + 0.6 * rem_frac)
                        pick = max(
                            live,
                            key=lambda i: (ucb_r[i] / max(1, ucb_n[i])) + c_exp * math.sqrt(math.log(total_t + 1.0) / ucb_n[i]),
                        )
                    q = qs[pick]
                    arm_selections[q.name] += 1
                    ev = q.step()
                    total_t += 1
                    ucb_n[pick] += 1
                    if ev is not None:
                        rew = 1.0 if ev["closed"] else (0.55 if ev["ok"] else 0.0)
                        # Slight penalty for very expensive heartbeat steps
                        hb_pen = min(0.25, ev["hb_raw"] / 5_000_000.0)
                        ucb_r[pick] += max(0.0, rew - hb_pen)
                    if q.path is not None:
                        winner = q
                        break
            elif arm == "explicit_memory":
                # Alternating exact counted S3 + explicit-memory re-scored queue under one Meter
                qs.append(
                    FrontierQueue(
                        "s3", m, s, prior, max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        arm_idx=2, lazy_expand=True,
                    )
                )
                qs.append(
                    FrontierQueue(
                        "explicit_memory", m, s, prior, max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        explicit_policy=explicit_policy, arm_idx=3, lazy_expand=True,
                    )
                )
            elif model is not None and alpha:
                # Learned adaptive frontier allocation: 4 base prior queues + learned utility queue
                # Model's prior_gate_head adaptively routes budget between base priors and learned frontier!
                prior_qs = [
                    FrontierQueue(
                        nm, m, s, prior if nm in ("s3", "s3c") else None,
                        max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        arm_idx=idx, lazy_expand=True,
                    )
                    for idx, nm in enumerate(BASE_PRIORS)
                ]
                learned_q = FrontierQueue(
                    arm, m, s, prior, max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                    model=model, bpd_decomposer=bpd_decomposer, alpha=alpha, arm_idx=2, lazy_expand=True,
                    bpd_probe_log=bpd_probe_log, haar_null_decomposer=haar_null, file_name=row["file"],
                )
                qs = [*prior_qs, learned_q]
                turn = 0
                while not all(q.done for q in qs) and not m.exhausted():
                    # Alternate learned frontier queue with adaptively gated base prior queue
                    if turn % 2 == 0 and not learned_q.done:
                        q = learned_q
                    else:
                        live_p = [i for i in range(4) if not prior_qs[i].done]
                        if not live_p:
                            q = learned_q
                        else:
                            weights = learned_q.last_prior_weights
                            # Combine learned prior gate with live empirical progress
                            pick = max(
                                live_p,
                                key=lambda i: weights[i]
                                + 0.15 * (tracker.arm_ok[i] + 0.5) / (tracker.arm_steps[i] + 1.0),
                            )
                            q = prior_qs[pick]
                    turn += 1
                    if q.done:
                        continue
                    arm_selections[q.name] += 1
                    q.step()
                    if q.path is not None:
                        winner = q
                        break
            else:
                qs.append(
                    FrontierQueue(
                        "s3", m, s, prior, max_depth=max_depth, tracker=tracker, cache=cache, env_key=env_key,
                        arm_idx=2, lazy_expand=lazy_expand,
                    )
                )

            # Default loop for single-queue and 2-queue explicit_memory arms
            if arm not in PORTFOLIO_SCHEDULES and arm != "adaptive_ucb" and not (model is not None and alpha):
                turn = 0
                while not all(q.done for q in qs) and not m.exhausted():
                    q = qs[turn % len(qs)]
                    turn += 1
                    if q.done:
                        continue
                    arm_selections[q.name] += 1
                    q.step()
                    if q.path is not None:
                        winner = q
                        break
        except run08.Exhausted:
            pass

        work = m.report()
        fate = "solved" if winner is not None else (
            "censored_steps" if work["steps"] >= work["limits"][0] else (
                "censored_rpcs" if work["rpcs"] >= work["limits"][1] else (
                    "censored_hb" if work["hb_raw"] >= work["limits"][2] else "exhausted_finite_action_space"
                )
            )
        )
        result: dict[str, Any] = {
            "id": row["id"],
            "role": row["role"],
            "family": row["family"],
            "file": row["file"],
            "decl": row.get("decl", ""),
            "arm": arm,
            "mms_budget_mib": mms_budget_mib,
            "solved": winner is not None,
            "fate": fate,
            "path": winner.path if winner else [],
            "on_path": winner.on_path if winner else [],
            "events": winner.events if winner else (qs[0].events if qs else []),
            "queues": [{"arm": q.name, "events": q.events, "fallbacks": q.fallbacks} for q in qs],
            "arm_selections": dict(arm_selections),
            "open": opening,
            "meter": work,
            "mms_cache": cache.receipt(ledger),
            "steps": work["steps"],
            "elapsed": time.perf_counter() - start,
        }
        if collect_bpd_probe and bpd_probe_log is not None:
            result["bpd_probe_records"] = bpd_probe_log

    if winner is not None:
        _, audit = run08.replay(exe, source, row, winner.path)
        result["replay"] = audit
    return result
