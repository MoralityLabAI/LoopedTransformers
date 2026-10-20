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

DEFAULT_SWEEPWEAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sweepweave_worlds"))
DEFAULT_STORYWORLD_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "storyworld_worlds"))


class Automaton:
    """Bank of DFAs sharing a symbol alphabet.  tables[c, s, a] -> next state."""

    def __init__(self, name, tables, init, n_classes, meta=None, random_init=False, n_valid=None):
        self.name = name
        self.random_init = random_init   # start state is drawn at random and given to the model as a token
        self.tables = np.asarray(tables, dtype=np.int64)
        self.init = np.asarray(init, dtype=np.int64)
        self.n_ctx, self.n_states, self.n_symbols = self.tables.shape
        self.n_classes = int(n_classes)
        self.n_valid = np.asarray(n_valid if n_valid is not None else [self.n_states] * self.n_ctx)
        self.prefix = (1 if self.n_ctx > 1 else 0) + (1 if random_init else 0)
        self.vocab = self.n_symbols + (self.n_ctx if self.n_ctx > 1 else 0) + (self.n_states if random_init else 0)
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
        if self.random_init:
            s0 = (rng.random(B) * self.n_valid[ctx]).astype(np.int64)
        else:
            s0 = self.init[ctx].copy()
        y = self.fold(ctx, s0.copy(), sym)
        cols = []
        if self.n_ctx > 1:
            cols.append((self.n_symbols + ctx)[:, None])
        if self.random_init:
            cols.append((self.n_symbols + (self.n_ctx if self.n_ctx > 1 else 0) + s0)[:, None])
        x = np.concatenate(cols + [sym], 1) if cols else sym
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


def real_path(d):
    """Map a virtualised AppData/Local/Packages/<pkg>/LocalCache/Roaming path back to AppData/Roaming."""
    marker = os.path.join("AppData", "Local", "Packages")
    d = os.path.abspath(d)
    if marker in d and "LocalCache" + os.sep in d:
        return os.path.join(d.split(marker)[0], "AppData", d.split("LocalCache" + os.sep, 1)[1])
    return d


def _readable_dir(d):
    cands = [d, d.replace(os.sep.join(["AppData", "Local", "Packages"]) + os.sep, "@@").split("@@")[0] if False else d]
    marker = os.path.join("AppData", "Local", "Packages")
    if marker in d and "LocalCache" in d:
        tail = d.split("LocalCache" + os.sep, 1)[1]           # Roaming\Claude\...
        cands.append(os.path.join(d.split(marker)[0], "AppData", tail))
    for c in cands:
        fs = glob.glob(os.path.join(c, "*.json"))
        if fs and all(os.access(f, os.R_OK) and _can_open(f) for f in fs):
            return c
    return d


def _can_open(f):
    try:
        with open(f, "rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def storyworld(world_dir=DEFAULT_STORYWORLD_DIR, n_actions=3):
    """Branching Storyworld state machines (storyworld_branching_world_v1).

    state = world `state_id`; symbol a = "take action (a mod #actions)"; a terminal
    state (no actions) restarts the world at `entry_state`, so episodes chain and
    arbitrarily long action strings stay meaningful.  The start encounter is drawn at random
    and shown to the model as a token (otherwise every world is a fixed clock and the answer
    is a function of the string length alone).  Stochastic reaction selection
    and variable effects of SweepWeave-style worlds are NOT modelled; only the
    discrete state->next_state graph is used.
    """
    # On some Windows installs the process sees a virtualised AppData path whose files cannot all be
    # opened; fall back to the un-virtualised location if the listed files are unreadable.
    world_dir = _readable_dir(world_dir)
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
    return Automaton("storyworld", tables, init, S, meta={"worlds": [n for n, _ in worlds]}, random_init=True,
                     n_valid=[len(w["states"]) for _, w in worlds])


def sweepweave(world_dir=None):
    """SweepWeave-format worlds from the GPTStoryworld repo (encounters -> options -> reactions -> consequence_id).

    state = encounter; symbol a = (option a // R, reaction a % R) with R = max reactions, taken mod the
    number available, i.e. the stochastic reaction draw is treated as an observed input symbol.  An encounter
    without options or a reaction without a valid consequence restarts the world at its first encounter.
    Effects / scripts / desirabilities are not evaluated."""
    world_dir = _readable_dir(world_dir or DEFAULT_SWEEPWEAVE_DIR)
    worlds = []
    for f in sorted(glob.glob(os.path.join(world_dir, "*.json"))):
        try:
            w = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(w, dict) and w.get("encounters"):
            worlds.append((os.path.basename(f), w))
    if not worlds:
        raise FileNotFoundError(f"no SweepWeave worlds in {world_dir}")
    S = max(len(w["encounters"]) for _, w in worlds)
    R = max(len(o.get("reactions", [])) for _, w in worlds for e in w["encounters"] for o in e.get("options", []))
    O = max(len(e.get("options", [])) for _, w in worlds for e in w["encounters"])
    A = O * R
    tables = np.zeros((len(worlds), S, A), dtype=np.int64)
    init = np.zeros(len(worlds), dtype=np.int64)
    for c, (_, w) in enumerate(worlds):
        enc = w["encounters"]
        ids = {e["id"]: i for i, e in enumerate(enc)}
        tables[c] = np.arange(S)[:, None]
        for i, e in enumerate(enc):
            opts = e.get("options", [])
            for a in range(A):
                if not opts:
                    tables[c, i, a] = 0
                    continue
                o = opts[(a // R) % len(opts)]
                rx = o.get("reactions", [])
                tables[c, i, a] = ids.get(rx[(a % R) % len(rx)].get("consequence_id"), 0) if rx else 0
    return Automaton("sweepweave", tables, init, S, meta={"worlds": [n for n, _ in worlds]}, random_init=True,
                     n_valid=[len(w["encounters"]) for _, w in worlds])


TASKS = {
    "cyclic": lambda **kw: cyclic(5),
    "s3": lambda **kw: permutation_group(3),
    "s4": lambda **kw: permutation_group(4),
    "dfa": lambda **kw: random_dfa(5, seed=kw.get("machine_seed", 0)),
    "sweepweave": lambda **kw: sweepweave(kw.get("sweepweave_dir")),
    "storyworld": lambda **kw: storyworld(kw.get("storyworld_dir") or DEFAULT_STORYWORLD_DIR),
}


def make_task(name, **kw):
    return TASKS[name](**kw)
