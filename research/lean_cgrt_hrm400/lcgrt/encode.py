"""Turn Lean-reported proof states into graph tensors.

Tokens: one per atom (proposition variable), one per hypothesis (fact or rule; each
is the candidate action `apply <hyp>`), one `constructor` token, and one virtual
conjunction node when the root goal is `A ∧ B`. Edges are typed incidences from the
hypothesis types Lean reported. Nothing derived from the teacher proof, proof depth
or generator metadata is an input; those fields are labels only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

# token types
ATOM, FACT, RULE, CTOR, CONJ = range(5)
# edge types (0 = no edge)
E_SELF, E_HEAD, E_PREM, E_PROD, E_CONS, E_CJ_DOWN, E_CJ_UP, E_CTOR_CJ, E_CJ_CTOR = range(1, 10)
N_EDGE_TYPES = 10

# per-token dynamic feature layout
F_BIAS, F_IS_MAIN, F_IN_GOALS, F_GOAL_POS, F_ON_PATH, F_FAILED, F_LEGAL, F_PRUNED, \
    F_HEAD_MAIN, F_ARITY, F_PREM_ON_PATH, F_PREM_FAILED = range(12)
N_FEATS = 12


@dataclass
class Graph:
    """Static per-theorem structure (from the Lean context)."""
    n_tok: int
    tok_type: np.ndarray          # [n_tok] int8
    edges: np.ndarray             # [E, 3] int16 (src, dst, type)
    atom_index: dict              # "P3" -> token
    hyp_index: dict               # "h0"/"r4" -> token
    ctor: int
    conj: int                     # -1 if none
    conj_parts: list
    hyp_head: dict                # token -> head atom token
    hyp_prem: dict                # token -> [atom tokens]


def build_graph(rec) -> Graph:
    ctx = rec["context"]
    atom_index = {a: i for i, a in enumerate(ctx["atoms"])}
    hyp_index, types, edges = {}, [ATOM] * len(atom_index), []
    hyp_head, hyp_prem = {}, {}
    for name, atom in ctx["facts"]:
        t = len(types); types.append(FACT); hyp_index[name] = t
        hyp_head[t] = atom_index[atom]; hyp_prem[t] = []
    for name, pre, head in ctx["rules"]:
        t = len(types); types.append(RULE); hyp_index[name] = t
        hyp_head[t] = atom_index[head]; hyp_prem[t] = [atom_index[p] for p in pre]
    ctor = len(types); types.append(CTOR)
    conj, parts = -1, []
    if rec["root"].startswith("and("):
        conj = len(types); types.append(CONJ)
        parts = [atom_index[a] for a in rec["root"][4:-1].split(",")]
    for t, h in hyp_head.items():
        edges += [(t, h, E_HEAD), (h, t, E_PROD)]
        for p in hyp_prem[t]:
            edges += [(t, p, E_PREM), (p, t, E_CONS)]
    for p in parts:
        edges += [(conj, p, E_CJ_DOWN), (p, conj, E_CJ_UP)]
    if conj >= 0:
        edges += [(ctor, conj, E_CTOR_CJ), (conj, ctor, E_CJ_CTOR)]
    n = len(types)
    edges += [(i, i, E_SELF) for i in range(n)]
    return Graph(n, np.array(types, np.int8), np.array(edges, np.int16), atom_index, hyp_index,
                 ctor, conj, parts, hyp_head, hyp_prem)


def tactic_token(g: Graph, tac: str) -> int:
    return g.ctor if tac == "constructor" else g.hyp_index[tac.split()[1]]


def key_token(g: Graph, key: str) -> int:
    return g.conj if key.startswith("and(") else g.atom_index[key]


def state_features(g: Graph, goals, ancestors, legal_children: dict, failed: set):
    """goals: goal keys (main first); ancestors: set of keys above the main goal;
    legal_children: Lean-legal tactic -> new goal keys for the main goal;
    failed: goal keys this search has proven unprovable (history)."""
    x = np.zeros((g.n_tok, N_FEATS), np.float32)
    x[:, F_BIAS] = 1.0
    main = key_token(g, goals[0])
    x[main, F_IS_MAIN] = 1.0
    for pos, k in enumerate(goals):
        t = key_token(g, k)
        x[t, F_IN_GOALS] += 0.25
        if x[t, F_GOAL_POS] == 0:
            x[t, F_GOAL_POS] = 1.0 - pos / max(1, len(goals))
    path = {key_token(g, k) for k in ancestors} | {main}
    for t in path:
        x[t, F_ON_PATH] = 1.0
    fail_t = {key_token(g, k) for k in failed if k in g.atom_index}
    for t in fail_t:
        x[t, F_FAILED] = 1.0
    for t, h in g.hyp_head.items():
        x[t, F_ARITY] = len(g.hyp_prem[t]) / 3.0
        x[t, F_HEAD_MAIN] = float(h == main)
        x[t, F_PREM_ON_PATH] = sum(p in path for p in g.hyp_prem[t]) / 3.0
        x[t, F_PREM_FAILED] = sum(p in fail_t for p in g.hyp_prem[t]) / 3.0
    for tac, new in legal_children.items():
        t = tactic_token(g, tac)
        x[t, F_LEGAL] = 1.0
        if any(k in ancestors or k == goals[0] for k in new):
            x[t, F_PRUNED] = 1.0
    return x, main


def path_states(rec):
    """(goals, ancestors) at each step of the Lean replay of the shortest proof."""
    out = []
    stack = [(rec["root"], frozenset())]
    for step in rec["path"]:
        goals = [k for k, _ in stack]
        assert goals == step["goals"], "replay goal list differs from Lean report"
        (k, anc), rest = stack[0], stack[1:]
        out.append((goals, anc, step["tactic"]))
        new = rec["table"][k][step["tactic"]]
        stack = [(n, anc | {k}) for n in new] + rest
    return out


@dataclass
class Example:
    graph: Graph
    x: np.ndarray            # [n_tok, F]
    main: int
    legal: np.ndarray        # [n_tok] bool
    optimal: np.ndarray      # [n_tok] bool
    atom_label: np.ndarray   # [n_tok] int8: 1 provable, 0 unprovable, -1 unknown
    meta: dict


def atom_labels(g: Graph, rec):
    lab = np.full(g.n_tok, -1, np.int8)
    for k, s in rec["size"].items():
        if k in g.atom_index:
            lab[g.atom_index[k]] = 0 if s is None else 1
    return lab


def examples_from_record(rec, rng: random.Random, fail_aug: float = 0.5, only: int | None = None):
    """Policy examples at every state of the shortest proof. With probability
    `fail_aug` a state also gets a random subset of truly unprovable explored goals
    marked as already-failed, imitating what a search would have learned."""
    g = build_graph(rec)
    lab = atom_labels(g, rec)
    dead = [k for k, s in rec["size"].items() if s is None and k in g.atom_index]
    size = rec["size"]
    out = []
    for j, (goals, anc, tac) in enumerate(path_states(rec)):
        if only is not None and j != only:
            continue
        main_key = goals[0]
        children = rec["table"][main_key]
        failed = set()
        if dead and rng.random() < fail_aug:
            failed = {k for k in dead if rng.random() < 0.4}
        x, main = state_features(g, goals, anc, children, failed)
        legal = np.zeros(g.n_tok, bool); opt = np.zeros(g.n_tok, bool)
        best = min(1 + sum(size[n] if size[n] is not None else 10**9 for n in new) for new in children.values())
        for t2, new in children.items():
            tok = tactic_token(g, t2)
            legal[tok] = True
            if 1 + sum(size[n] if size[n] is not None else 10**9 for n in new) == best:
                opt[tok] = True
        out.append(Example(g, x, main, legal, opt, lab, {"tid": rec["tid"], "depth": rec["depth"]}))
    return out


def collate(exs, device):
    """Pad to the largest graph in the batch; build dense typed adjacency on device."""
    B = len(exs); N = max(e.graph.n_tok for e in exs)
    x = torch.zeros(B, N, N_FEATS)
    ttype = torch.full((B, N), 0, dtype=torch.long)
    etype = torch.zeros(B, N, N, dtype=torch.long)
    tok_mask = torch.zeros(B, N, dtype=torch.bool)
    legal = torch.zeros(B, N, dtype=torch.bool)
    opt = torch.zeros(B, N, dtype=torch.bool)
    alab = torch.full((B, N), -1, dtype=torch.long)
    main = torch.zeros(B, dtype=torch.long)
    for i, e in enumerate(exs):
        n = e.graph.n_tok
        x[i, :n] = torch.from_numpy(e.x)
        ttype[i, :n] = torch.from_numpy(e.graph.tok_type.astype(np.int64))
        ed = torch.from_numpy(e.graph.edges.astype(np.int64))
        etype[i, ed[:, 0], ed[:, 1]] = ed[:, 2]
        tok_mask[i, :n] = True
        legal[i, :n] = torch.from_numpy(e.legal)
        opt[i, :n] = torch.from_numpy(e.optimal)
        alab[i, :n] = torch.from_numpy(e.atom_label.astype(np.int64))
        main[i] = e.main
    for i in range(B):                      # padding tokens attend only to themselves
        n = exs[i].graph.n_tok
        idx = torch.arange(n, N)
        etype[i, idx, idx] = E_SELF
    return {k: v.to(device, non_blocking=True) for k, v in dict(
        x=x, ttype=ttype, etype=etype, tok_mask=tok_mask, legal=legal, optimal=opt,
        atom_label=alab, main=main).items()}
