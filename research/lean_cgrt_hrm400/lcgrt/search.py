"""Lean-backed proof search with a typed control layer, and the search policies compared.

Control layer (identical for every policy):
  * a node is a Lean state (sid) plus, per open goal, the set of goals above it;
  * expanding a node is one Lean `expand` call and one unit of budget; Lean decides which
    tactics apply and what goals they leave;
  * a child is discarded if one of its new goals repeats a goal on its own path (no
    minimal proof does that) or is a goal this search already proved dead;
  * a goal is dead when Lean offers no tactic for it, or when every Lean-legal tactic
    leads to a dead goal with nothing pruned for cycles;
  * success = a node with no goals; the tactic script is then certified separately.

Policies differ only in which node they expand next and in how they score children:
  greedy      commit to the top child, no backtracking
  dfs         depth-first, children in model order, backtracking
  best        best-first on cumulative child log-probability (one context)
  cgrt        best-first; children scored under C controlled contexts (recurrence depth,
              failure-history visibility) merged into one tree; branch priority from a
              learned scorer over (context scores, agreement, predicted child value,
              verification history)
  cgrt_shuf   cgrt with branch scores permuted among siblings (same budget and contexts)
  random_dfs, bfs   model-free baselines
Searches are generators; `run_searches` advances many at once and batches model calls.
The trajectory does not depend on the budget, so one run at the largest budget gives
success at every smaller budget (expansions used <= budget).
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from .encode import (F_FAILED, F_PREM_FAILED, N_FEATS, Example, build_graph, key_token,
                     state_features, tactic_token)
from .extract import goal_key, parse_context
from .model import N_GLOBAL, flops_per_decision

POLICIES = ("greedy", "dfs", "best", "cgrt", "cgrt_shuf", "random_dfs", "bfs")
BRANCH_FEATURES = ("mean_logp", "max_logp", "min_logp", "agree_top1", "child_value",
                   "n_new_goals", "failed_siblings", "n_siblings", "bias")


@dataclass
class Node:
    sid: int
    goals: list                     # [(key, frozenset(ancestors))]
    script: list
    depth: int = 0
    prio: float = 0.0
    legal_children: dict = field(default_factory=dict)


class Controller:
    def __init__(self, repl, statement: str):
        self.repl = repl
        root = repl.init(statement)
        rg = root["goals"]
        ctx = parse_context(repl.contexts[rg[0]["ctx"]])
        self.rec = {"context": ctx, "root": goal_key(rg[0])}
        self.g = build_graph(self.rec)
        self.root = Node(root["sid"], [(goal_key(rg[0]), frozenset())], [])
        self.sids = [root["sid"]]
        self.expansions = 0
        self.attempts = 0
        self.dead: set = set()

    def expand(self, node: Node):
        """One budget unit. Returns viable children; records Lean-legal children on the node."""
        if not node.goals:
            return []
        self.expansions += 1
        ex = self.repl.expand(node.sid, errors=0)
        main, anc = node.goals[0]
        rest = node.goals[1:]
        legal, viable, pruned = {}, [], False
        for c in ex["candidates"]:
            self.attempts += 1
            if not c["ok"]:
                continue
            st = c["state"]; self.sids.append(st["sid"])
            child = st["goals"]
            new = [goal_key(x) for x in child[: len(child) - len(rest)]]
            legal[c["tactic"]] = new
            if any(k in anc or k == main for k in new):
                pruned = True; continue
            if any(k in self.dead for k in new):
                continue
            up = anc | {main}
            viable.append(Node(st["sid"], [(k, up) for k in new] + rest, node.script + [c["tactic"]],
                               node.depth + 1))
        node.legal_children = legal
        if not viable and not pruned:
            self.dead.add(main)
        return viable

    def example(self, node: Node, viable, hide_history=False):
        keys = [k for k, _ in node.goals]
        x, main = state_features(self.g, keys, node.goals[0][1], node.legal_children,
                                 set() if hide_history else self.dead)
        legal = np.zeros(self.g.n_tok, bool)
        for ch in viable:
            legal[tactic_token(self.g, ch.script[-1])] = True
        n = self.g.n_tok
        lab = np.full(n, -1, np.int8)
        return Example(self.g, x, main, legal, legal.copy(), lab, {})

    def close(self):
        if self.sids:
            self.repl.request(op="drop", sids=self.sids)
            self.sids = []


@dataclass
class Request:
    example: Example
    fast_steps: int | None
    glob: list


class Scorer:
    """Batched model calls. Returns per request (logp over viable tokens, atom log sigma)."""

    def __init__(self, model, device, fast_steps=None, halt_threshold=None):
        self.model, self.device = model, device
        self.fast_steps, self.halt_threshold = fast_steps, halt_threshold
        self.forwards = 0
        self.flops = 0.0
        self.steps_used = []

    @torch.no_grad()
    def __call__(self, reqs):
        from .encode import collate
        out = [None] * len(reqs)
        groups = {}
        for i, r in enumerate(reqs):
            groups.setdefault(r.fast_steps or self.fast_steps, []).append(i)
        for T, idxs in groups.items():
            b = collate([reqs[i].example for i in idxs], self.device)
            glob = torch.tensor([reqs[i].glob for i in idxs], dtype=torch.float32, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                outs = self.model(b, fast_steps=T, glob=glob)
            pick = [len(outs) - 1] * len(idxs)
            if self.halt_threshold is not None and outs[0]["halt"] is not None:
                hp = torch.stack([torch.sigmoid(o["halt"]) for o in outs])          # R,B
                for j in range(len(idxs)):
                    over = (hp[:, j] > self.halt_threshold).nonzero()
                    pick[j] = int(over[0]) if len(over) else len(outs) - 1
            for j, i in enumerate(idxs):
                o = outs[pick[j]]
                lg = o["logits"][j].float()
                logp = torch.log_softmax(lg, -1).cpu().numpy()
                atom = torch.nn.functional.logsigmoid(o["atom"][j].float()).cpu().numpy()
                n = reqs[i].example.graph.n_tok
                steps = o["steps"]
                self.steps_used.append(steps)
                self.flops += flops_per_decision(self.model.cfg, n, steps)
                out[i] = (logp[:n], atom[:n])
            self.forwards += len(idxs)
        return out


def glob_feats(ctl: Controller, node: Node, budget: int):
    return [ctl.expansions / max(1, budget), len(node.goals) / 8, len(node.goals[0][1]) / 16,
            len(ctl.dead) / 8, 1.0]


def child_value(ctl, atom_logsig, ch: Node, parent: Node):
    n_new = len(ch.goals) - (len(parent.goals) - 1)
    return float(sum(atom_logsig[key_token(ctl.g, k)] for k, _ in ch.goals[:n_new])), n_new


def search(ctl: Controller, policy: str, budget: int, rng: random.Random, contexts=None,
           branch_w=None, value_w: float = 0.5, trace: list | None = None, hide_history: bool = False):
    """Generator: yields lists of Requests, receives lists of (logp, atom_logsig).
    Returns dict(success, expansions, script)."""
    root = ctl.root
    if not root.goals:
        return {"success": True, "expansions": 0, "script": []}
    tie = itertools.count()

    def scored(node, viable):
        if policy in ("random_dfs", "bfs"):
            return [(rng.random(), ch, {}) for ch in viable]
        ctxs = contexts if policy in ("cgrt", "cgrt_shuf") else [(None, hide_history)]
        reqs = [Request(ctl.example(node, viable, hide), T, glob_feats(ctl, node, budget)) for T, hide in ctxs]
        res = yield reqs
        toks = [tactic_token(ctl.g, ch.script[-1]) for ch in viable]
        lps = np.array([[r[0][t] for t in toks] for r in res])               # C, children
        top = lps.argmax(1)
        out = []
        for j, ch in enumerate(viable):
            v, n_new = child_value(ctl, res[0][1], ch, node)
            f = {"mean_logp": float(lps[:, j].mean()), "max_logp": float(lps[:, j].max()),
                 "min_logp": float(lps[:, j].min()), "agree_top1": float((top == j).mean()),
                 "child_value": v, "n_new_goals": n_new,
                 "failed_siblings": float(len(node.legal_children) - len(viable)),
                 "n_siblings": float(len(viable)), "bias": 1.0}
            if branch_w is not None:
                z = sum(branch_w[k] * f[k] for k in BRANCH_FEATURES)
                s = -math.log1p(math.exp(-z)) if z > -30 else z            # log sigmoid
            else:
                s = f["mean_logp"] + value_w * v
            out.append((s, ch, f))
        if policy == "cgrt_shuf" and len(out) > 1:
            ss = [s for s, _, _ in out]; rng.shuffle(ss)
            out = [(s2, ch, f) for s2, (_, ch, f) in zip(ss, out)]
        return out

    if policy == "greedy":
        node = root
        while ctl.expansions < budget:
            viable = ctl.expand(node)
            if not viable:
                break
            done = [ch for ch in viable if not ch.goals]
            if done:
                return {"success": True, "expansions": ctl.expansions, "script": done[0].script}
            sc = yield from scored(node, viable)
            node = max(sc, key=lambda t: t[0])[1]
        return {"success": False, "expansions": ctl.expansions, "script": None}

    if policy in ("dfs", "random_dfs"):
        stack = [root]
        while stack and ctl.expansions < budget:
            node = stack.pop()
            if any(k in ctl.dead for k, _ in node.goals):
                continue
            viable = ctl.expand(node)
            if not viable:
                continue
            sc = yield from scored(node, viable)
            for s, ch, f in sorted(sc, key=lambda t: t[0]):
                if not ch.goals:
                    return {"success": True, "expansions": ctl.expansions, "script": ch.script}
                stack.append(ch)
        return {"success": False, "expansions": ctl.expansions, "script": None}

    # best-first family (best, cgrt, cgrt_shuf, bfs)
    heap = [(0.0, next(tie), root)]
    while heap and ctl.expansions < budget:
        neg, _, node = heapq.heappop(heap)
        if any(k in ctl.dead for k, _ in node.goals):
            continue
        viable = ctl.expand(node)
        if not viable:
            continue
        sc = yield from scored(node, viable)
        for s, ch, f in sc:
            if trace is not None and f:
                trace.append((f, ch.script[-1], node.goals[0][0]))
            if not ch.goals:
                return {"success": True, "expansions": ctl.expansions, "script": ch.script}
            prio = (ch.depth if policy == "bfs" else node.prio + s)
            ch.prio = prio
            heapq.heappush(heap, ((prio if policy == "bfs" else -prio), next(tie), ch))
    return {"success": False, "expansions": ctl.expansions, "script": None}


def run_searches(repl, items, policy, scorer, *, budget_fn, contexts=None, branch_w=None,
                 width=64, seed=0, trace=None, hide_history=False):
    """items: [(tid, statement, meta)]. Advances up to `width` searches concurrently,
    batching all their model requests into one call per round."""
    rng = random.Random(seed)
    results = []
    pending = list(items)
    active = []

    def start(item):
        tid, stmt, meta = item
        ctl = Controller(repl, stmt)
        b = budget_fn(meta)
        tr = [] if trace is not None else None
        gen = search(ctl, policy, b, random.Random(rng.random()), contexts, branch_w, trace=tr,
                     hide_history=hide_history)
        return {"tid": tid, "meta": meta, "ctl": ctl, "gen": gen, "budget": b, "req": None, "trace": tr}

    def advance(s, value):
        try:
            s["req"] = s["gen"].send(value)
            return True
        except StopIteration as e:
            r = e.value
            ctl = s["ctl"]
            results.append({"tid": s["tid"], **s["meta"], "success": r["success"], "expansions": r["expansions"],
                            "budget": s["budget"], "script": r["script"], "lean_attempts": ctl.attempts,
                            "dead_goals": len(ctl.dead)})
            if trace is not None:
                trace.append({"tid": s["tid"], "rows": s["trace"]})
            ctl.close()
            return False

    while pending or active:
        while pending and len(active) < width:
            s = start(pending.pop(0))
            if advance(s, None):
                active.append(s)
        reqs, owners = [], []
        for s in active:
            for r in s["req"]:
                reqs.append(r); owners.append(s)
        res = scorer(reqs) if reqs else []
        k = 0
        nxt = []
        for s in active:
            n = len(s["req"])
            if advance(s, res[k:k + n]):
                nxt.append(s)
            k += n
        active = nxt
    return results
