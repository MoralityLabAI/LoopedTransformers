"""Synthetic state-machine tasks.

Every task is a (family of) deterministic finite automata.  A sample is a symbol
string; the label is the automaton state reached after reading it.  Models are
trained on short strings and evaluated on held-out (longer) strings.

Token layout: [context token]? + symbol tokens.  The context token (only present
when a task has several machines, e.g. Storyworld) names which machine is used.
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import numpy as np

DEFAULT_STORYWORLD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "storyworld_worlds")


class Automaton:
    """Bank of DFAs sharing a symbol alphabet.  tables[c, s, a] -> next state."""

    def __init__(self, name, tables, init, n_classes, meta=None):
        self.name = name
        self.tables = np.asarray(tables, dtype=np.int64)
        self.init = np.asarray(init, dtype=np.int64)
        self.n_ctx, self.n_states, self.n_symbols = self.tables.shape
        self.n_classes = int(n_classes)
        self.prefix = 1 if self.n_ctx > 1 else 0
        self.vocab = self.n_symbols + (self.n_ctx if self.n_ctx > 1 else 0)
        self.meta = meta or {}

    # -- generation -------------------------------------------------------
    def fold(self, ctx, state, sym):
        """Advance `state` [B] over symbols `sym` [B, L] under machine `ctx` [B]."""
        for t in range(sym.shape[1]):
            state = self.tables[ctx, state, sym[:, t]]
        return state

    def sample(self, rng, B, N):
        ctx = rng.integers(0, self.n_ctx, size=B) if self.n_ctx > 1 else np.zeros(B, dtype=np.int64)
        sym = rng.integers(0, self.n_symbols, size=(B, N))
        y = self.fold(ctx, self.init[ctx].copy(), sym)
        x = np.concatenate([(self.n_symbols + ctx)[:, None], sym], 1) if self.prefix else sym
        return x.astype(np.int64), y.astype(np.int64), ctx

    def extend(self, ctx, state, suffix):
        """Label of the continuation: state after reading `suffix` [B, L] from `state`."""
        return self.fold(ctx, state.copy(), suffix)

    def sample_suffix(self, rng, B, L):
        return rng.integers(0, self.n_symbols, size=(B, L)).astype(np.int64)

    def majority_acc(self, rng, N, B=4000):
        _, y, _ = self.sample(rng, B, N)
        return float(np.bincount(y, minlength=self.n_classes).max() / B)


# -- concrete machines -----------------------------------------------------
def cyclic(m=7):
    t = np.zeros((1, m, m), dtype=np.int64)
    for s in range(m):
        for a in range(m):
            t[0, s, a] = (s + a) % m
    return Automaton(f"cyclic{m}", t, [0], m)


def permutation_group(n=4):
    """Word problem of S_n: tokens are group elements, state is the running product."""
    perms = list(itertools.permutations(range(n)))
    idx = {p: i for i, p in enumerate(perms)}
    G = len(perms)
    t = np.zeros((1, G, G), dtype=np.int64)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            t[0, i, j] = idx[tuple(q[p[k]] for k in range(n))]
    return Automaton(f"s{n}", t, [idx[tuple(range(n))]], G)


def random_dfa(n_states=5, seed=0):
    """Rotate (a permutation) plus two random non-injective maps.

    The non-injective symbols act as partial resets, so the state depends on the
    suffix since the last reset and on a rotation count: a non-group automaton in which
    earlier input is forgotten at data-dependent times.
    """
    r = np.random.default_rng(1000 + seed)
    t = np.zeros((1, n_states, 3), dtype=np.int64)
    t[0, :, 0] = (np.arange(n_states) + 1) % n_states
    for a in (1, 2):
        while True:
            f = r.integers(0, n_states, size=n_states)
            if 1 < len(set(f.tolist())) < n_states:
                break
        t[0, :, a] = f
    return Automaton(f"dfa{n_states}", t, [0], n_states, meta={"seed": seed})


def storyworld(world_dir=DEFAULT_STORYWORLD_DIR, n_actions=3):
    """Branching Storyworld state machines (storyworld_branching_world_v1).

    state = world `state_id`; symbol a = "take action (a mod #actions)"; a terminal
    state (no actions) restarts the world at `entry_state`, so episodes chain and
    arbitrarily long action strings stay meaningful.  Stochastic reaction selection
    and variable effects of SweepWeave-style worlds are NOT modelled; only the
    discrete state->next_state graph is used.
    """
    files = sorted(glob.glob(os.path.join(world_dir, "*.json")))
    worlds = []
    for f in files:
        try:
            w = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(w, dict) and "states" in w and "entry_state" in w and "overlay" not in os.path.basename(f):
            worlds.append((os.path.basename(f), w))
    if not worlds:
        raise FileNotFoundError(f"no Storyworld files with a `states` graph in {world_dir}")
    S = max(len(w["states"]) for _, w in worlds)
    tables = np.zeros((len(worlds), S, n_actions), dtype=np.int64)
    init = np.zeros(len(worlds), dtype=np.int64)
    for c, (_, w) in enumerate(worlds):
        ids = {s["state_id"]: i for i, s in enumerate(w["states"])}
        entry = ids[w["entry_state"]]
        init[c] = entry
        tables[c] = np.arange(S)[:, None]  # padding states self-loop
        for i, s in enumerate(w["states"]):
            acts = s.get("actions", [])
            for a in range(n_actions):
                if not acts:
                    tables[c, i, a] = entry
                else:
                    tables[c, i, a] = ids.get(acts[a % len(acts)]["next_state"], i)
    return Automaton("storyworld", tables, init, S, meta={"worlds": [n for n, _ in worlds]})


TASKS = {
    "cyclic": lambda **kw: cyclic(5),
    "s3": lambda **kw: permutation_group(3),
    "s4": lambda **kw: permutation_group(4),
    "dfa": lambda **kw: random_dfa(5, seed=kw.get("machine_seed", 0)),
    "storyworld": lambda **kw: storyworld(kw.get("storyworld_dir") or DEFAULT_STORYWORLD_DIR),
}


def make_task(name, **kw):
    return TASKS[name](**kw)
