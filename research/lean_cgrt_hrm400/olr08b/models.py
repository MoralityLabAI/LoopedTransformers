"""OLR-08B utility predictor and matched architecture controls.

Extends the OLR-08A 122,881-parameter tied graph residual reader (W=40, B=128, VOCAB=2048)
to 128,647 parameters (<146K; well below 500K/5M) by adding:
  - 24-dim conditioning on remaining budget, previously observed verifier receipts,
    multi-prior costs, and MMS memory lookup readouts,
  - Recurrent frontier-state update cell (z_t in R^40) updated from verifier receipts,
  - Multi-head outputs: branch/action utility, tactic applicability logit, expected
    verifier cost, and adaptive 4-prior allocation logits (s1, s2, s3, s3c),
  - Optional BPD component feature projection (parameter-matched across arms).

Arms & Controls (all matched-init/sampler except linear & explicit_memory):
  - ``recurrent``: full 3-visit bidirectional graph + recurrent frontier state (128,647 params)
  - ``recurrent_bpd``: ``recurrent`` augmented with BPD component features (128,647 params)
  - ``direct_mlp``: direct non-recurrent MLP control (0 graph messages, 0 frontier recurrence;
    same 128,647 params and matched compute schedule)
  - ``matched_compute_sham``: exact ``recurrent`` architecture/compute trained on
    within-menu shuffled sham targets (128,647 params)
  - ``one_way``: reverse graph messages masked (OLR-08A directional ablation; 128,647 params)
  - ``linear``: pure linear bag + conditioning control (415 params)
  - ``explicit_memory``: non-neural exact explicit-memory policy (0 learned parameters)
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from bpd_frontier import BPD_FEATURE_DIM, BPDFrontierDecomposer

NEURAL_ARMS = ("recurrent", "recurrent_bpd", "direct_mlp", "matched_compute_sham", "one_way", "linear")
MATCHED_PARAM_ARMS = ("recurrent", "recurrent_bpd", "direct_mlp", "matched_compute_sham", "one_way")
SEEDS = (17, 29, 43)

W = 40
B = 128
VOCAB = 2048
MAX_NODES = 16000
MAX_EDGES = 64000
BASE_SCALAR_DIM = 6
COND_DIM = 24  # 4 budget + 12 verifier receipt history + 8 multi-prior/memory candidate features
EXTRA_DIM = 3 * B + BASE_SCALAR_DIM + COND_DIM  # 390 + 24 = 414
RECEIPT_STEP_DIM = 16  # step receipt vector feeding the recurrent frontier state z_t


def hid(s: Any, n: int) -> int:
    return int.from_bytes(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "little") % n


class ReceiptTracker:
    """Tracks live verifier receipts, remaining budget, and recovery streaks for one root."""

    def __init__(self, step_cap: int, rpc_cap: int, hb_cap: int) -> None:
        self.step_cap = max(1, int(step_cap))
        self.rpc_cap = max(1, int(rpc_cap))
        self.hb_cap = max(1, int(hb_cap))
        self.steps = 0
        self.ok_count = 0
        self.fail_streak = 0
        self.max_fail_streak = 0
        self.no_match = 0
        self.type_err = 0
        self.resource_err = 0
        self.premise_calls = 0
        self.premise_timeouts = 0
        self.total_hb = 0
        self.last_ok = 0.0
        self.last_ngoals_delta = 0.0
        self.depth_counts = [0, 0, 0]
        self.arm_steps = [0, 0, 0, 0]
        self.arm_ok = [0, 0, 0, 0]
        self.arm_progress = [0.0, 0.0, 0.0, 0.0]

    def observe_premise(self, ok: bool, hb_raw: int) -> None:
        self.premise_calls += 1
        self.total_hb += max(0, int(hb_raw))
        if not ok:
            self.premise_timeouts += 1

    def observe_step(
        self,
        *,
        ok: bool,
        closed: bool,
        depth: int,
        parent_ngoals: int,
        child_ngoals: int | None,
        error_category: str,
        hb_raw: int,
        arm_idx: int = 0,
    ) -> Tensor:
        """Record one verifier step receipt and return a [RECEIPT_STEP_DIM] float tensor."""
        self.steps += 1
        self.total_hb += max(0, int(hb_raw))
        d_idx = min(max(0, int(depth)), 2)
        self.depth_counts[d_idx] += 1
        a_idx = min(max(0, int(arm_idx)), 3)
        self.arm_steps[a_idx] += 1
        if ok:
            self.ok_count += 1
            self.fail_streak = 0
            self.arm_ok[a_idx] += 1
            delta = float(parent_ngoals - (0 if closed else (child_ngoals if child_ngoals is not None else parent_ngoals)))
            self.last_ngoals_delta = max(-4.0, min(4.0, delta))
            self.last_ok = 1.0
            prog = 2.0 if closed else (0.6 + 0.2 * max(-1.0, min(2.0, delta)))
            self.arm_progress[a_idx] += prog
        else:
            self.fail_streak += 1
            self.max_fail_streak = max(self.max_fail_streak, self.fail_streak)
            self.last_ok = 0.0
            self.last_ngoals_delta = 0.0
            if error_category == "no_match":
                self.no_match += 1
            elif error_category == "type":
                self.type_err += 1
            elif error_category == "resource":
                self.resource_err += 1

        step_vec = torch.tensor(
            [
                1.0 if ok else 0.0,
                1.0 if closed else 0.0,
                self.last_ngoals_delta / 4.0,
                min(self.fail_streak, 20) / 20.0,
                math.log1p(max(0, hb_raw)) / 20.0,
                1.0 if error_category == "no_match" else 0.0,
                1.0 if error_category == "type" else 0.0,
                1.0 if error_category == "resource" else 0.0,
                d_idx / 2.0,
                a_idx / 3.0,
                self.ok_count / max(1, self.steps),
                self.premise_timeouts / max(1, self.premise_calls),
                max(0.0, 1.0 - self.steps / self.step_cap),
                max(0.0, 1.0 - self.total_hb / self.hb_cap),
                self.arm_ok[a_idx] / max(1, self.arm_steps[a_idx]),
                min(1.0, self.arm_progress[a_idx] / max(1, self.arm_steps[a_idx])),
            ],
            dtype=torch.float32,
        )
        return step_vec

    def global_cond16(self, used_steps: int, used_rpcs: int, used_hb: int) -> list[float]:
        """4 remaining-budget dims + 12 verifier-receipt history dims = 16 dims."""
        rem_steps = max(0.0, 1.0 - used_steps / self.step_cap)
        rem_rpcs = max(0.0, 1.0 - used_rpcs / self.rpc_cap)
        rem_hb = max(0.0, 1.0 - used_hb / self.hb_cap)
        log_rem = math.log1p(max(0, self.step_cap - used_steps)) / math.log1p(self.step_cap)
        n = max(1, self.steps)
        return [
            rem_steps,
            rem_rpcs,
            rem_hb,
            log_rem,
            self.ok_count / n,
            min(self.fail_streak, 25) / 25.0,
            self.no_match / n,
            self.type_err / n,
            self.resource_err / n,
            self.premise_timeouts / max(1, self.premise_calls),
            math.log1p(self.total_hb / n) / 20.0,
            self.last_ok,
            self.last_ngoals_delta / 4.0,
            self.depth_counts[0] / n,
            self.depth_counts[1] / n,
            self.depth_counts[2] / n,
        ]


def build_packet(
    graph: dict[str, Any],
    lemmas: list[dict[str, Any]],
    actions: list[tuple],
    depth: int,
    prior_costs: dict[str, list[float]],
    global_cond16: list[float],
    mem_features: list[tuple[float, float, float]] | None = None,
    parent_ngoals: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    """Build variable-size term/action graph + 414-dim conditioned feature packet.

    Returns ``(ids, edges, act_indices, extra_414, s3_base_costs)`` or ``None`` if oversize.
    """
    nodes = list(graph["nodes"])
    edges = [tuple(e) for e in graph["edges"]]
    gcount = len(nodes)
    obs = graph["obligations"]
    targets = [o["target"] for o in obs]
    if not targets:
        raise ValueError("open obligations required")

    arg_roots: dict[str, list[int]] = defaultdict(list)
    lemroot: dict[str, int] = {}
    lem_nodes: dict[str, list[dict]] = {}
    for i, n in enumerate(nodes):
        if n.get("k") in ("fvar", "const") and "n" in n:
            arg_roots[n["n"]].append(i)
    for lem in lemmas:
        name = lem["name"]
        start = len(nodes)
        nodes.extend(lem["nodes"])
        edges.extend((a + start, b + start, r) for a, b, r in lem["edges"])
        lemroot[name] = start + lem["root"]
        lem_nodes[name] = lem["nodes"]

    def bag(ns: list[dict]) -> Tensor:
        x = torch.zeros(B, dtype=torch.float32)
        for n in ns:
            for k, v in sorted(n.items()):
                x[hid(f"{k}={v}", B)] += 1.0
        return x.log1p() / (max(1, len(ns)) ** 0.5)

    gb = bag(nodes[:gcount])
    eb = bag(
        [{"edge": r, "child": nodes[a].get("k"), "parent": nodes[b].get("k")} for a, b, r in edges if a < gcount and b < gcount]
    )

    s1_c = prior_costs["s1"]
    s2_c = prior_costs["s2"]
    s3_c = prior_costs["s3"]
    s3c_c = prior_costs["s3c"]
    g16 = list(global_cond16)
    if len(g16) != 16:
        raise ValueError(f"expected 16 global conditioning dims, got {len(g16)}")

    an: list[int] = []
    extra: list[Tensor] = []
    kinds: dict[str, int] = defaultdict(int)
    for j, (c, k, arg, tac) in enumerate(actions):
        ix = len(nodes)
        nodes.append({"k": "action", "n": k, "arg": arg, "tactic": tac})
        an.append(ix)
        for target in targets:
            edges.append((target, ix, "obligation"))
        for a in arg_roots.get(arg, []):
            edges.append((a, ix, "argument"))
        if arg in lemroot:
            edges.append((lemroot[arg], ix, "lemma"))
        rank = kinds[k]
        kinds[k] += 1
        sb = bag([nodes[a] for a in arg_roots.get(arg, [])] + lem_nodes.get(arg, []) + [{"kind": k, "arg": arg, "tactic": tac}])
        m_hit, m_ok, m_util = mem_features[j] if mem_features is not None else (0.0, 0.0, 0.0)
        scalars = [
            depth / 3.0,
            min(rank, 100) / 100.0,
            c / 10.0,
            s3_c[j] / 10.0,
            len(targets) / 8.0,
            len(lem_nodes.get(arg, [])) / 256.0,
            *g16,
            s1_c[j] / 10.0,
            s2_c[j] / 10.0,
            s3_c[j] / 10.0,
            s3c_c[j] / 20.0,
            float(m_hit),
            float(m_ok),
            float(m_util),
            min(parent_ngoals, 8) / 8.0,
        ]
        extra.append(torch.cat((gb, eb, sb, torch.tensor(scalars, dtype=torch.float32))))

    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        return None
    if any(a < 0 or b < 0 or a >= len(nodes) or b >= len(nodes) for a, b, _ in edges):
        raise ValueError("invalid native graph edge")

    ids = torch.tensor(
        [
            [
                hid("kind:" + str(n.get("k")), VOCAB),
                hid("name:" + str(n.get("n", n.get("arg", ""))), VOCAB),
                hid(json.dumps(n, sort_keys=True), VOCAB),
            ]
            for n in nodes
        ],
        dtype=torch.long,
    )
    e = torch.tensor([[a, b, hid(r, 64)] for a, b, r in edges], dtype=torch.long).reshape(-1, 3)
    return ids, e, torch.tensor(an, dtype=torch.long), torch.stack(extra), torch.tensor(s3_c, dtype=torch.float32)


class UtilityPredictor(nn.Module):
    """Extended 122,881 -> 128,647 parameter utility predictor and matched controls.

    Outputs:
      - ``utility``: [N_actions] action/branch utility residual over prior
      - ``aux``: [N_actions, 2] -> (applicability_logit, log_hb_pred)
      - ``prior_logits``: [4] -> adaptive allocation logits over (s1, s2, s3, s3c)
    """

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in NEURAL_ARMS:
            raise ValueError(f"unknown neural arm: {arm}")
        self.arm = arm
        if arm == "linear":
            self.direct = nn.Linear(EXTRA_DIM, 1)
            self.prior_lin = nn.Linear(EXTRA_DIM, 4)
            return
        # Preserved OLR-08A 122,881-parameter core backbone (W=40, VOCAB=2048)
        self.emb = nn.Embedding(VOCAB, W)
        self.rel = nn.Embedding(64, W)
        self.up = nn.Linear(W, W)
        self.down = nn.Linear(W, W)
        self.cell = nn.Sequential(nn.Linear(4 * W, 2 * W), nn.GELU(), nn.Linear(2 * W, W))
        self.norm = nn.LayerNorm(W)
        self.extra = nn.Linear(EXTRA_DIM, W)
        self.head = nn.Sequential(nn.Linear(2 * W, W), nn.GELU(), nn.Linear(W, 1))
        # OLR-08B minimal extensions (+5,766 params -> 128,647 total; parameter-matched across all 5 neural arms)
        self.frontier_cell = nn.Linear(2 * W + RECEIPT_STEP_DIM, W)
        self.bpd_proj = nn.Linear(BPD_FEATURE_DIM, W)
        self.aux_head = nn.Linear(W, 2)
        self.prior_gate_head = nn.Linear(W, 4)

    def init_frontier_state(self) -> Tensor:
        return torch.zeros(W, dtype=torch.float32)

    def update_frontier_state(self, z_prev: Tensor, state_summary: Tensor, step_receipt: Tensor) -> Tensor:
        """Recurrent frontier-state update conditioned on verifier receipts."""
        if self.arm in ("linear", "direct_mlp"):
            return torch.zeros(W, dtype=torch.float32)
        inp = torch.cat((z_prev.reshape(W), state_summary.reshape(W), step_receipt.reshape(RECEIPT_STEP_DIM)), dim=0)
        return self.norm(z_prev + torch.tanh(self.frontier_cell(inp)))

    def forward_full(
        self,
        packet: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        frontier_state: Tensor | None = None,
        bpd_decomposer: BPDFrontierDecomposer | None = None,
    ) -> dict[str, Any]:
        ids, e, act, x, _ = packet
        if self.arm == "linear":
            util = self.direct(x).squeeze(-1)
            p_logits = self.prior_lin(x.mean(dim=0))
            aux = torch.zeros((len(act), 2), dtype=torch.float32)
            return {"utility": util, "aux": aux, "prior_logits": p_logits, "pass_action_states": [], "state_summary": torch.zeros(W)}

        z = self.init_frontier_state() if (frontier_state is None or self.arm == "direct_mlp") else frontier_state.reshape(W)
        initial = self.emb(ids).mean(1)
        h = initial
        c, b, r = e.T
        rel = self.rel(r)
        dc = torch.bincount(c, minlength=len(h)).clamp_min(1).unsqueeze(1)
        db = torch.bincount(b, minlength=len(h)).clamp_min(1).unsqueeze(1)

        pass_action_states: list[Tensor] = []
        for _ in range(3):
            up = torch.zeros_like(h).index_add(0, b, self.up(h[c]) + rel) / db
            down = torch.zeros_like(h).index_add(0, c, self.down(h[b]) + rel) / dc
            # direct_mlp masks BOTH message directions (matched-compute feedforward MLP control)
            # one_way masks reverse messages (parent -> child)
            up = up * float(self.arm != "direct_mlp")
            down = down * float(self.arm not in ("direct_mlp", "one_way"))
            h = self.norm(h + self.cell(torch.cat((h, initial, up, down), dim=1)))
            pass_action_states.append(h[act])

        act_h = h[act]
        ex = F.gelu(self.extra(x))
        if self.arm == "recurrent_bpd" and bpd_decomposer is not None:
            bpd_feat = bpd_decomposer.extract_action_features(pass_action_states, z)
            ex = ex + F.gelu(self.bpd_proj(bpd_feat))
        elif self.arm != "direct_mlp":
            ex = ex + z.unsqueeze(0)

        joint = torch.cat((act_h, ex), dim=1)
        util = self.head(joint).squeeze(-1)
        aux = self.aux_head(ex)
        state_summary = act_h.mean(dim=0)
        p_logits = self.prior_gate_head(state_summary + z)
        return {
            "utility": util,
            "aux": aux,
            "prior_logits": p_logits,
            "pass_action_states": pass_action_states,
            "state_summary": state_summary,
        }

    def forward(
        self,
        packet: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        frontier_state: Tensor | None = None,
        bpd_decomposer: BPDFrontierDecomposer | None = None,
    ) -> Tensor:
        return self.forward_full(packet, frontier_state=frontier_state, bpd_decomposer=bpd_decomposer)["utility"]


class ExplicitMemoryPolicy:
    """Exact explicit-memory control (0 learned parameters).

    Stores exact verified empirical outcomes from TRAIN source trajectories and live
    episode receipts inside ``ProofMemoryCache``, scoring candidate tactics via
    empirical-Bayes smoothed closure + applicability utility and failure avoidance.
    """

    def __init__(self) -> None:
        self.global_stats: dict[str, dict[str, int]] = {}
        self.head_stats: dict[str, dict[str, int]] = {}
        self.parameters = 0

    @staticmethod
    def _goal_head(info: dict[str, Any] | None) -> str:
        if not info:
            return "none"
        defs = info.get("defs") or []
        hyps = info.get("hyps") or []
        d0 = defs[0] if defs else "nodef"
        h_ind = sum(1 for h in hyps if h.get("ind"))
        h_prop = sum(1 for h in hyps if h.get("prop"))
        return f"{d0}|ind{min(h_ind,4)}|p{min(h_prop,4)}"

    def fit_from_source_rows(self, rows: list[dict[str, Any]], cache: Any | None = None) -> dict[str, Any]:
        for r in rows:
            if r.get("role") != "train":
                raise ValueError("explicit_memory must fit on TRAIN source rows only")
            on = set(r.get("on_path", []))
            for j, ev in enumerate(r.get("events", [])):
                tpl = ev["tactic"] if ev["kind"] in ("close", "norm", "struct") else ev["kind"]
                key = f"{min(ev['depth'], 1)}|{tpl}"
                st = self.global_stats.setdefault(key, {"n": 0, "ok": 0, "on": 0, "hb": 0})
                st["n"] += 1
                st["ok"] += int(bool(ev.get("ok")))
                st["on"] += int(j in on)
                st["hb"] += int(ev.get("hb_raw", 0))
                if cache is not None:
                    cache.update_exact_stat(
                        "src|" + key,
                        ok=bool(ev.get("ok")),
                        closed=(j in on),
                        hb_raw=int(ev.get("hb_raw", 0)),
                    )
        return {"entries": len(self.global_stats), "parameters": 0}

    def score_actions(
        self,
        acts: list[tuple],
        s3_costs: list[float],
        depth: int,
        info: dict[str, Any] | None,
        cache: Any | None = None,
    ) -> list[float]:
        ghead = self._goal_head(info)
        out: list[float] = []
        for (_, kind, _, tac), base_c in zip(acts, s3_costs):
            tpl = tac if kind in ("close", "norm", "struct") else kind
            key = f"{min(depth, 1)}|{tpl}"
            st = self.global_stats.get(key)
            # Combine TRAIN exact table + live episode cache stat (if available under RAM budget)
            live = cache.get_exact_stat(f"ep|{ghead}|{key}") if cache is not None else None
            bonus = 0.0
            if st is not None:
                p_on = (st["on"] + 0.5) / (st["n"] + 2.0)
                p_ok = (st["ok"] + 1.0) / (st["n"] + 2.0)
                bonus += 0.35 * math.log2(max(1e-4, p_on)) + 0.15 * math.log2(max(1e-4, p_ok))
            if live is not None:
                # Penalize templates repeatedly failing on this exact head signature in-episode
                live_ok = (live["ok"] + 0.5) / (live["n"] + 1.0)
                live_cl = (live["closed"] + 0.25) / (live["n"] + 1.0)
                bonus += 0.6 * math.log2(max(1e-3, live_ok)) + 0.4 * math.log2(max(1e-3, live_cl))
            out.append(max(0.05, float(base_c - bonus)))
        return out


@torch.no_grad()
def predict_costs_and_gates(
    model: UtilityPredictor,
    packet: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    alpha: float = 1.0,
    frontier_state: Tensor | None = None,
    bpd_decomposer: BPDFrontierDecomposer | None = None,
) -> tuple[list[float], list[float], Tensor]:
    """Return ``(reweighted_costs, prior_weights_4, state_summary)``."""
    out = model.forward_full(packet, frontier_state=frontier_state, bpd_decomposer=bpd_decomposer)
    base = packet[-1]
    logits = -math.log(2.0) * base + alpha * out["utility"]
    new_costs = (-F.log_softmax(logits, dim=0) / math.log(2.0)).tolist()
    prior_weights = F.softmax(out["prior_logits"], dim=0).tolist()
    return new_costs, prior_weights, out["state_summary"]


def hash_weights(m: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_utility_model(
    frames: list[dict[str, Any]],
    arm: str,
    seed: int,
    *,
    steps: int = 400,
    lr: float = 0.002,
    wd: float = 0.005,
    bpd_rank: int = 8,
    bpd_learned: bool = False,
) -> tuple[UtilityPredictor, BPDFrontierDecomposer | None, dict[str, Any]]:
    """Train one OLR-08B utility predictor arm on TRAIN-only certified frames.

    Each frame dict has:
      - ``packet``: (ids, edges, act, x_414, s3_costs)
      - ``target_idx``: int (certified path action index in menu)
      - ``ok_mask``: optional float tensor [N_actions] (known applicability from search trace, -1 if unknown)
      - ``best_prior_idx``: optional int in 0..3 (which prior ranked target_idx highest)
    """
    torch.manual_seed(seed)
    m = UtilityPredictor(arm)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    rng = random.Random(seed + 8081)
    sampler = [rng.randrange(len(frames)) for _ in range(steps)]

    # Sham targets: fixed per-frame uniform random menu index + random prior index
    sham_labels = [
        random.Random(seed * 1000003 + i).randrange(len(fr["packet"][-1]))
        if arm == "matched_compute_sham"
        else int(fr["target_idx"])
        for i, fr in enumerate(frames)
    ]
    sham_priors = [
        random.Random(seed * 2000003 + i).randrange(4)
        if arm == "matched_compute_sham"
        else int(fr.get("best_prior_idx", 2))
        for i, fr in enumerate(frames)
    ]

    initial_sha = hash_weights(m)
    bpd_dec: BPDFrontierDecomposer | None = (
        BPDFrontierDecomposer.from_model(m, rank_c=bpd_rank, learned=False, seed=seed)
        if arm == "recurrent_bpd"
        else None
    )

    losses: list[float] = []
    for step_idx, fi in enumerate(sampler):
        fr = frames[fi]
        p = fr["packet"]
        y = sham_labels[fi]
        bp = sham_priors[fi]
        # Periodically refresh SVD seed proxy for recurrent_bpd during training
        if arm == "recurrent_bpd" and step_idx > 0 and step_idx % 80 == 0:
            bpd_dec = BPDFrontierDecomposer.from_model(m, rank_c=bpd_rank, learned=False, seed=seed)

        out = m.forward_full(p, bpd_decomposer=bpd_dec)
        logits = -math.log(2.0) * p[-1] + out["utility"]
        loss_rank = F.cross_entropy(logits[None], torch.tensor([y], dtype=torch.long))
        loss_prior = F.cross_entropy(out["prior_logits"][None], torch.tensor([bp], dtype=torch.long))
        loss = loss_rank + 0.15 * loss_prior

        ok_mask = fr.get("ok_mask")
        if ok_mask is not None and arm != "linear":
            known = ok_mask >= 0.0
            if bool(known.any()):
                target_ok = (
                    torch.rand_like(ok_mask[known]) > 0.5
                    if arm == "matched_compute_sham"
                    else ok_mask[known]
                ).float()
                loss = loss + 0.15 * F.binary_cross_entropy_with_logits(out["aux"][known, 0], target_ok)

        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step_idx} for arm {arm}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))

    m.eval()
    # Final BPD map extraction (optional learned BPD over recorded passes)
    if arm in ("recurrent", "recurrent_bpd"):
        pass_inputs: dict[str, list[Tensor]] = {"up": [], "down": []}
        with torch.no_grad():
            for fr in frames[: min(32, len(frames))]:
                out = m.forward_full(fr["packet"])
                if out["pass_action_states"]:
                    for t_idx, h_act in enumerate(out["pass_action_states"]):
                        if len(pass_inputs["up"]) <= t_idx:
                            pass_inputs["up"].append(h_act)
                            pass_inputs["down"].append(h_act)
                        else:
                            pass_inputs["up"][t_idx] = torch.cat([pass_inputs["up"][t_idx], h_act], dim=0)
                            pass_inputs["down"][t_idx] = torch.cat([pass_inputs["down"][t_idx], h_act], dim=0)
        bpd_dec = BPDFrontierDecomposer.from_model(
            m, rank_c=bpd_rank, learned=bpd_learned, pass_inputs_by_site=pass_inputs, seed=seed
        )
    else:
        bpd_dec = None

    stats = {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "parameters": sum(p.numel() for p in m.parameters()),
        "initial_sha256": initial_sha,
        "sampler_sha256": hashlib.sha256(json.dumps(sampler).encode("utf-8")).hexdigest(),
        "weights_sha256": hash_weights(m),
        "loss_first": round(losses[0], 5),
        "loss_last": round(losses[-1], 5),
        "bpd_summary": bpd_dec.summary() if bpd_dec is not None else None,
    }
    return m, bpd_dec, stats
