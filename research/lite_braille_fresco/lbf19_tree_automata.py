"""LBF-19 Track T: tree-automata abstractions for variable-size expression trees.

Terms:    ('L', f) | ('U', f, t) | ('B', f, t1, t2)
Patterns: same plus ('V', i).
Rules: (lhs, rhs) with lhs left-linear, vars(rhs) subset of vars(lhs).
NTA: relational transitions with epsilon edges (mirrors lean/LBF19TreeAutomata.lean).
Completion round = Complete1 (Lean); optional widening = state merge (NTA.map).
Stdlib only.
"""
from __future__ import annotations
import itertools
import json
import os
import random
import sys
import time

# ----------------------------------------------------------------------------
# terms / patterns / rewriting
# ----------------------------------------------------------------------------
LEAVES, UNARY, BINARY = 3, 3, 2
SYMNAMES = {'L': 'abz', 'U': 'sgh', 'B': 'pm'}


def size(t):
    if t[0] == 'L':
        return 1
    if t[0] == 'U':
        return 1 + size(t[2])
    return 1 + size(t[2]) + size(t[3])


def pat_vars(p, acc=None):
    acc = [] if acc is None else acc
    if p[0] == 'V':
        acc.append(p[1])
    elif p[0] == 'U':
        pat_vars(p[2], acc)
    elif p[0] == 'B':
        pat_vars(p[2], acc)
        pat_vars(p[3], acc)
    return acc


def pat_nonvar(p):
    if p[0] == 'V':
        return 0
    if p[0] == 'L':
        return 1
    if p[0] == 'U':
        return 1 + pat_nonvar(p[2])
    return 1 + pat_nonvar(p[2]) + pat_nonvar(p[3])


def rule_ok(rule):
    l, r = rule
    lv = pat_vars(l)
    return len(set(lv)) == len(lv) and set(pat_vars(r)) <= set(lv) and l[0] != 'V'


def match_pat(p, t, sub):
    k = p[0]
    if k == 'V':
        sub[p[1]] = t
        return True
    if k != t[0] or p[1] != t[1]:
        return False
    if k == 'L':
        return True
    if k == 'U':
        return match_pat(p[2], t[2], sub)
    return match_pat(p[2], t[2], sub) and match_pat(p[3], t[3], sub)


def subst(p, sub):
    k = p[0]
    if k == 'V':
        return sub[p[1]]
    if k == 'L':
        return p
    if k == 'U':
        return ('U', p[1], subst(p[2], sub))
    return ('B', p[1], subst(p[2], sub), subst(p[3], sub))


def succs(t, rules):
    """all one-step rewrites of t at any position (set)."""
    out = set()
    _succs(t, rules, out)
    return out


def _succs(t, rules, out):
    for lhs, rhs in rules:
        sub = {}
        if match_pat(lhs, t, sub):
            out.add(subst(rhs, sub))
    k = t[0]
    if k == 'U':
        for c in succs(t[2], rules):
            out.add(('U', t[1], c))
    elif k == 'B':
        for c in succs(t[2], rules):
            out.add(('B', t[1], c, t[3]))
        for c in succs(t[3], rules):
            out.add(('B', t[1], t[2], c))


def ball_layers(seeds, rules, kmax, cap=400000):
    """returns (layers, truncated): layers[k] = set of terms reachable in <=k steps."""
    cur = set(seeds)
    layers = [set(cur)]
    frontier = set(cur)
    trunc = False
    for _ in range(kmax):
        new = set()
        for t in frontier:
            for s in succs(t, rules):
                if s not in cur and s not in new:
                    new.add(s)
            if len(cur) + len(new) > cap:
                trunc = True
                break
        cur = cur | new
        frontier = new
        layers.append(set(cur))
        if trunc:
            for _ in range(len(layers), kmax + 1):
                layers.append(set(cur))
            break
    return layers, trunc


# ----------------------------------------------------------------------------
# random generation
# ----------------------------------------------------------------------------
def rand_term(rng, n):
    if n <= 1:
        return ('L', rng.randrange(LEAVES))
    if n == 2 or rng.random() < 0.4:
        return ('U', rng.randrange(UNARY), rand_term(rng, n - 1))
    a = rng.randint(1, n - 2)
    return ('B', rng.randrange(BINARY), rand_term(rng, a), rand_term(rng, n - 1 - a))


def rand_pat(rng, depth, vars_pool, linear_vars=None, leaf_p=0.15):
    """random pattern of depth<=depth. vars drawn from vars_pool (callable)."""
    if depth == 0:
        return ('V', vars_pool())
    r = rng.random()
    if r < 0.2:
        return ('V', vars_pool()) if depth < 2 else ('L', rng.randrange(LEAVES))
    if r < 0.2 + leaf_p:
        return ('L', rng.randrange(LEAVES))
    if r < 0.6:
        return ('U', rng.randrange(UNARY), rand_pat(rng, depth - 1, vars_pool, leaf_p=leaf_p))
    return ('B', rng.randrange(BINARY), rand_pat(rng, depth - 1, vars_pool, leaf_p=leaf_p),
            rand_pat(rng, depth - 1, vars_pool, leaf_p=leaf_p))


def rand_lhs(rng):
    while True:
        ctr = itertools.count()
        p = ('U', rng.randrange(UNARY), rand_pat(rng, 1, lambda: next(ctr))) if rng.random() < 0.4 else \
            ('B', rng.randrange(BINARY), rand_pat(rng, 1, lambda: next(ctr)), rand_pat(rng, 1, lambda: next(ctr)))
        # deepen occasionally
        if rng.random() < 0.5:
            ctr = itertools.count()
            p = ('U', rng.randrange(UNARY), rand_pat(rng, 2, lambda: next(ctr)))
        if p[0] != 'V' and len(pat_vars(p)) >= 1 and pat_nonvar(p) >= 1:
            return p


def rand_rhs(rng, lhs, mode):
    lv = pat_vars(lhs)
    nl = pat_nonvar(lhs)
    for _ in range(500):
        if mode == 'preserve':
            order = lv[:]
            rng.shuffle(order)
            it = iter(order)
            r = _rhs_exact_vars(rng, order, 2)
        else:
            pool = lambda: rng.choice(lv)
            r = rand_pat(rng, 2, pool)
        if r == lhs or not set(pat_vars(r)) <= set(lv):
            continue
        nr = pat_nonvar(r)
        if mode == 'preserve':
            if nr == nl and sorted(pat_vars(r)) == sorted(lv):
                return r
        elif mode == 'shrink':
            if nr < nl and size_delta(lhs, r) < 0:
                return r
        else:  # mixed, length changing
            if size_delta(lhs, r) != 0:
                return r
    return None


def _rhs_exact_vars(rng, order, depth):
    """random pattern using exactly the variables in `order` once each."""
    if len(order) == 1:
        if depth > 0 and rng.random() < 0.7:
            return ('U', rng.randrange(UNARY), _rhs_exact_vars(rng, order, depth - 1))
        return ('V', order[0])
    if depth > 0 and rng.random() < 0.15:
        return ('U', rng.randrange(UNARY), _rhs_exact_vars(rng, order, depth - 1))
    a = rng.randint(1, len(order) - 1)
    return ('B', rng.randrange(BINARY), _rhs_exact_vars(rng, order[:a], depth - 1 if depth else 0),
            _rhs_exact_vars(rng, order[a:], depth - 1 if depth else 0))


def size_delta(l, r):
    """approx size change per rewrite: nonvar change + variable multiplicity change."""
    lv, rv = len(pat_vars(l)), len(pat_vars(r))
    return (pat_nonvar(r) - pat_nonvar(l)) + (rv - lv)


def random_system(seed, mode, nrules=None):
    rng = random.Random(seed)
    nrules = nrules or rng.randint(2, 4)
    rules = []
    tries = 0
    while len(rules) < nrules and tries < 2000:
        tries += 1
        lhs = rand_lhs(rng)
        m = mode
        if mode == 'mixed':
            m = 'mixed'
        rhs = rand_rhs(rng, lhs, m)
        if rhs is None:
            continue
        if (lhs, rhs) not in rules:
            rules.append((lhs, rhs))
    assert all(rule_ok(r) for r in rules)
    return rules


def named_system():
    V = lambda i: ('V', i)
    p = lambda a, b: ('B', 0, a, b)
    s = lambda a: ('U', 0, a)
    g = lambda a: ('U', 1, a)
    h = lambda a: ('U', 2, a)
    z = ('L', 2)
    return [(p(V(0), z), V(0)),            # x+0 -> x
            (s(g(V(0))), h(V(0))),         # s(g(x)) -> h(x)
            (p(V(0), V(1)), p(V(1), V(0))),  # commutativity
            (g(h(V(0))), V(0))]            # g(h(x)) -> x


def pat_str(p):
    if p[0] == 'V':
        return 'xyw'[p[1]] if p[1] < 3 else 'v%d' % p[1]
    if p[0] == 'L':
        return SYMNAMES['L'][p[1]]
    if p[0] == 'U':
        return '%s(%s)' % (SYMNAMES['U'][p[1]], pat_str(p[2]))
    return '%s(%s,%s)' % (SYMNAMES['B'][p[1]], pat_str(p[2]), pat_str(p[3]))


def rule_str(r):
    return '%s -> %s' % (pat_str(r[0]), pat_str(r[1]))


# ----------------------------------------------------------------------------
# NTA
# ----------------------------------------------------------------------------
class Blowup(Exception):
    pass


class NTA:
    def __init__(self):
        self.leaf = {}      # f -> set(q)
        self.un = {}        # (f, q1) -> set(q0)
        self.bin = {}       # (f, q1, q2) -> set(q0)
        self.eps = {}       # a -> set(b)
        self.final = set()
        self.nstates = 0
        self.memo = {}      # fresh-state memo (key -> state)
        self.pos = {}       # optional positional label of a state (chain mode)
        self.names = {}     # optional name -> state
        self._prep = None

    # construction
    def new_state(self):
        self.nstates += 1
        return self.nstates - 1

    def add_leaf(self, f, q):
        s = self.leaf.setdefault(f, set())
        if q in s:
            return False
        s.add(q)
        return True

    def add_un(self, f, q1, q0):
        s = self.un.setdefault((f, q1), set())
        if q0 in s:
            return False
        s.add(q0)
        return True

    def add_bin(self, f, q1, q2, q0):
        s = self.bin.setdefault((f, q1, q2), set())
        if q0 in s:
            return False
        s.add(q0)
        return True

    def add_eps(self, a, b):
        if a == b:
            return False
        s = self.eps.setdefault(a, set())
        if b in s:
            return False
        s.add(b)
        return True

    def states(self):
        return self.nstates

    def ntrans(self):
        return (sum(len(s) for s in self.leaf.values()) + sum(len(s) for s in self.un.values())
                + sum(len(s) for s in self.bin.values()) + sum(len(s) for s in self.eps.values()))

    def n_active_states(self):
        st = set(self.final)
        for s in self.leaf.values():
            st |= s
        for (f, q1), s in self.un.items():
            st.add(q1); st |= s
        for (f, q1, q2), s in self.bin.items():
            st.add(q1); st.add(q2); st |= s
        for a, s in self.eps.items():
            st.add(a); st |= s
        return len(st)

    def copy(self):
        B = NTA()
        B.leaf = {k: set(v) for k, v in self.leaf.items()}
        B.un = {k: set(v) for k, v in self.un.items()}
        B.bin = {k: set(v) for k, v in self.bin.items()}
        B.eps = {k: set(v) for k, v in self.eps.items()}
        B.final = set(self.final)
        B.nstates = self.nstates
        B.memo = dict(self.memo)
        B.pos = dict(self.pos)
        B.names = dict(self.names)
        return B

    # epsilon closures
    def eps_closure_fwd(self, q):
        seen = {q}
        stack = [q]
        while stack:
            a = stack.pop()
            for b in self.eps.get(a, ()):
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        return seen

    def eps_pred_map(self):
        """pred[q] = {q0 : eps*(q0, q)}"""
        rev = {}
        for a, s in self.eps.items():
            for b in s:
                rev.setdefault(b, set()).add(a)
        cache = {}

        def pred(q):
            if q in cache:
                return cache[q]
            seen = {q}
            stack = [q]
            while stack:
                x = stack.pop()
                for y in rev.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            cache[q] = seen
            return seen
        return pred

    # evaluation (membership)
    def prepare(self):
        clo = {}

        def c(q):
            r = clo.get(q)
            if r is None:
                r = frozenset(self.eps_closure_fwd(q)) if q in self.eps else frozenset((q,))
                clo[q] = r
            return r
        self._clo = c
        self._unf = {}
        self._binf = {}
        self._leaff = {f: frozenset(x for q in s for x in c(q)) for f, s in self.leaf.items()}
        self._memo = {}

    def eval(self, t):
        m = self._memo
        r = m.get(t)
        if r is not None:
            return r
        k = t[0]
        if k == 'L':
            r = self._leaff.get(t[1], frozenset())
        elif k == 'U':
            ch = self.eval(t[2])
            out = set()
            f = t[1]
            for q1 in ch:
                key = (f, q1)
                g = self._unf.get(key)
                if g is None:
                    g = frozenset(x for q0 in self.un.get(key, ()) for x in self._clo(q0))
                    self._unf[key] = g
                out |= g
            r = frozenset(out)
        else:
            a = self.eval(t[2])
            if not a:
                r = frozenset()
            else:
                b = self.eval(t[3])
                out = set()
                f = t[1]
                for q1 in a:
                    for q2 in b:
                        key = (f, q1, q2)
                        g = self._binf.get(key)
                        if g is None:
                            g = frozenset(x for q0 in self.bin.get(key, ()) for x in self._clo(q0))
                            self._binf[key] = g
                        out |= g
                r = frozenset(out)
        m[t] = r
        return r

    def accepts(self, t):
        return bool(self.eval(t) & self.final)

    # widening: state map (NTA.map in Lean)
    def mapped(self, h):
        B = NTA()
        for f, s in self.leaf.items():
            for q in s:
                B.add_leaf(f, h[q])
        for (f, q1), s in self.un.items():
            for q in s:
                B.add_un(f, h[q1], h[q])
        for (f, q1, q2), s in self.bin.items():
            for q in s:
                B.add_bin(f, h[q1], h[q2], h[q])
        for a, s in self.eps.items():
            for b in s:
                B.add_eps(h[a], h[b])
        B.final = {h[q] for q in self.final}
        B.nstates = self.nstates
        B.memo = {k: h[v] for k, v in self.memo.items()}
        B.pos = {h[q]: p for q, p in self.pos.items()}
        B.names = {k: h[v] for k, v in self.names.items()}
        return B


def dag_automaton(terms):
    """hash-consed DAG automaton: one state per distinct subterm; L(A) = terms exactly."""
    A = NTA()
    ids = {}

    def go(t):
        if t in ids:
            return ids[t]
        k = t[0]
        if k == 'L':
            q = A.new_state()
            A.add_leaf(t[1], q)
        elif k == 'U':
            c = go(t[2])
            q = A.new_state()
            A.add_un(t[1], c, q)
        else:
            a = go(t[2])
            b = go(t[3])
            q = A.new_state()
            A.add_bin(t[1], a, b, q)
        ids[t] = q
        return q
    for t in terms:
        A.final.add(go(t))
    return A


# ----------------------------------------------------------------------------
# completion round (Complete1) and widening
# ----------------------------------------------------------------------------
def _matches(A, pred, p, q, memo):
    key = (p, q)
    r = memo.get(key)
    if r is not None:
        return r
    k = p[0]
    out = []
    if k == 'V':
        out = [((p[1], q),)]
    elif k == 'L':
        ls = A.leaf.get(p[1], ())
        if any(q0 in ls for q0 in pred(q)):
            out = [()]
    elif k == 'U':
        for q0 in pred(q):
            for q1 in A._unrev.get((p[1], q0), ()):
                out.extend(_matches(A, pred, p[2], q1, memo))
    else:
        for q0 in pred(q):
            for (q1, q2) in A._binrev.get((p[1], q0), ()):
                m1 = _matches(A, pred, p[2], q1, memo)
                if not m1:
                    continue
                m2 = _matches(A, pred, p[3], q2, memo)
                for a in m1:
                    for b in m2:
                        out.append(a + b)
    memo[key] = out
    return out


def _build_rev(A):
    unrev, binrev = {}, {}
    for (f, q1), s in A.un.items():
        for q0 in s:
            unrev.setdefault((f, q0), []).append(q1)
    for (f, q1, q2), s in A.bin.items():
        for q0 in s:
            binrev.setdefault((f, q0), []).append((q1, q2))
    A._unrev, A._binrev = unrev, binrev


def _top_label(p):
    return ('END',) if p[0] == 'L' else (p[1],)


def _add_rhs(A, ridx, p, rho, q, path, namer, limit):
    k = p[0]
    if k == 'V':
        A.add_eps(rho[p[1]], q)
        return
    if k == 'L':
        A.add_leaf(p[1], q)
        return

    def child_state(sub, sub_path):
        if sub[0] == 'V':
            return rho[sub[1]]
        if namer is not None:
            return namer(A, q, sub, sub_path)
        vs = tuple(sorted((v, rho[v]) for v in set(pat_vars(sub))))
        key = (ridx, sub_path, vs)
        s = A.memo.get(key)
        if s is None:
            s = A.new_state()
            if s > limit:
                raise Blowup('states>%d' % limit)
            A.memo[key] = s
        return s
    if k == 'U':
        c = child_state(p[2], path + (0,))
        A.add_un(p[1], c, q)
        # recurse into child if not a var
        if p[2][0] != 'V':
            _add_rhs(A, ridx, p[2], rho, c, path + (0,), namer, limit)
    else:
        c1 = child_state(p[2], path + (0,))
        c2 = child_state(p[3], path + (1,))
        A.add_bin(p[1], c1, c2, q)
        if p[2][0] != 'V':
            _add_rhs(A, ridx, p[2], rho, c1, path + (0,), namer, limit)
        if p[3][0] != 'V':
            _add_rhs(A, ridx, p[3], rho, c2, path + (1,), namer, limit)


def completion_round(A, rules, limit=30000, deadline=None, namer=None):
    """returns new NTA = one-round completion of A (A unchanged)."""
    _build_rev(A)
    pred = A.eps_pred_map()
    memo = {}
    todo = []
    for ridx, (lhs, rhs) in enumerate(rules):
        for q in range(A.nstates):
            if deadline is not None and time.time() > deadline:
                raise Blowup('timeout')
            for m in _matches(A, pred, lhs, q, memo):
                todo.append((ridx, rhs, dict(m), q))
    B = A.copy()
    for ridx, rhs, rho, q in todo:
        _add_rhs(B, ridx, rhs, rho, q, (), namer, limit)
        if deadline is not None and time.time() > deadline:
            raise Blowup('timeout')
    return B


def signature_classes(A, depth):
    """merge map: states with equal depth-`depth` head signature."""
    pred = A.eps_pred_map()
    unrev, binrev = {}, {}
    for (f, q1), s in A.un.items():
        for q0 in s:
            unrev.setdefault(q0, []).append((f, q1))
    for (f, q1, q2), s in A.bin.items():
        for q0 in s:
            binrev.setdefault(q0, []).append((f, q1, q2))
    leafrev = {}
    for f, s in A.leaf.items():
        for q0 in s:
            leafrev.setdefault(q0, []).append(f)
    n = A.nstates
    sig = [frozenset() for _ in range(n)]
    # depth-0: head symbols
    def heads(q, prev):
        out = set()
        for q0 in pred(q):
            for f in leafrev.get(q0, ()):
                out.add(('L', f))
            for (f, q1) in unrev.get(q0, ()):
                out.add(('U', f) if prev is None else ('U', f, prev[q1]))
            for (f, q1, q2) in binrev.get(q0, ()):
                out.add(('B', f) if prev is None else ('B', f, prev[q1], prev[q2]))
        return frozenset(out)
    cur = [heads(q, None) for q in range(n)]
    for _ in range(depth):
        cur = [heads(q, cur) for q in range(n)]
    rep = {}
    h = {}
    for q in range(n):
        key = (cur[q], q in A.final)
        r = rep.setdefault(key, q)
        h[q] = r
    return h


def run_ta(A0, rules, kmax, widen=None, limit=30000, tlimit=120.0, namer=None, on_round=None):
    """iterate T_R; returns list of (k, automaton) checkpoints; raises Blowup with partial list in .partial"""
    cps = [(0, A0)]
    A = A0
    t0 = time.time()
    deadline = t0 + tlimit
    try:
        for k in range(1, kmax + 1):
            tr = time.time()
            B = completion_round(A, rules, limit=limit, deadline=deadline, namer=namer)
            if widen is not None:
                B = B.mapped(signature_classes(B, widen))
            A = B
            cps.append((k, A, time.time() - t0))
    except Blowup as e:
        e.partial = cps
        raise
    return cps


# ----------------------------------------------------------------------------
# size-agnostic baseline abstraction: root symbols + (sym,depth-bucket) nodes + parent/child-index edges
# ----------------------------------------------------------------------------
def tree_alpha(terms, bucket=3):
    roots, nodes, edges = set(), set(), set()

    def go(t, d):
        b = min(d, bucket)
        sym = (t[0], t[1])
        nodes.add((sym, b))
        if t[0] == 'U':
            c = t[2]
            edges.add((sym, b, 0, (c[0], c[1])))
            go(c, d + 1)
        elif t[0] == 'B':
            for i in (0, 1):
                c = t[2 + i]
                edges.add((sym, b, i, (c[0], c[1])))
                go(c, d + 1)
    for t in terms:
        roots.add((t[0], t[1]))
        go(t, 0)
    return roots, nodes, edges


def alpha_member(al, t, bucket=3):
    roots, nodes, edges = al
    if (t[0], t[1]) not in roots:
        return False

    def go(t, d):
        b = min(d, bucket)
        sym = (t[0], t[1])
        if (sym, b) not in nodes:
            return False
        if t[0] == 'U':
            c = t[2]
            return (sym, b, 0, (c[0], c[1])) in edges and go(c, d + 1)
        if t[0] == 'B':
            for i in (0, 1):
                c = t[2 + i]
                if (sym, b, i, (c[0], c[1])) not in edges or not go(c, d + 1):
                    return False
        return True
    return go(t, 0)


def alpha_size(al):
    return len(al[0]) + len(al[1]) + len(al[2])


# ----------------------------------------------------------------------------
# negative cohorts
# ----------------------------------------------------------------------------
def random_subterm_replace(rng, t):
    """mutate: relabel a node (same arity) or replace subterm by a small random tree."""
    positions = []

    def collect(x, path):
        positions.append(path)
        if x[0] == 'U':
            collect(x[2], path + (2,))
        elif x[0] == 'B':
            collect(x[2], path + (2,))
            collect(x[3], path + (3,))
    collect(t, ())
    path = rng.choice(positions)

    def rebuild(x, path):
        if not path:
            if rng.random() < 0.5:
                if x[0] == 'L':
                    return ('L', rng.randrange(LEAVES))
                if x[0] == 'U':
                    return ('U', rng.randrange(UNARY), x[2])
                return ('B', rng.randrange(BINARY), x[2], x[3])
            return rand_term(rng, rng.randint(1, 5))
        i = path[0]
        if x[0] == 'U':
            return ('U', x[1], rebuild(x[2], path[1:]))
        if i == 2:
            return ('B', x[1], rebuild(x[2], path[1:]), x[3])
        return ('B', x[1], x[2], rebuild(x[3], path[1:]))
    return rebuild(t, path)


def make_negatives(rng, ball_list, ballset, rules, k, nper=600, size_range=(5, 40)):
    n1, n2, n3 = [], [], []
    tries = 0
    while len(n1) < nper and tries < nper * 30 and ball_list:
        tries += 1
        t = rng.choice(ball_list)
        for _ in range(rng.randint(k + 1, k + 2) - k):
            ss = list(succs(t, rules))
            if not ss:
                break
            t = rng.choice(ss)
        # extra steps beyond ball: one or two random further steps from a ball term
        if t not in ballset:
            n1.append(t)
    tries = 0
    while len(n2) < nper and tries < nper * 30 and ball_list:
        tries += 1
        t = random_subterm_replace(rng, rng.choice(ball_list))
        if t not in ballset:
            n2.append(t)
    while len(n3) < nper:
        t = rand_term(rng, rng.randint(*size_range))
        if t not in ballset:
            n3.append(t)
    return {'N1': n1, 'N2': n2, 'N3': n3}


# ----------------------------------------------------------------------------
# experiment on one system
# ----------------------------------------------------------------------------
VARIANTS = {'exact': None, 'sig2': 2, 'sig1': 1}
FN_CHECK_CAP = 50000


def fpr(accept, xs):
    return (sum(1 for x in xs if accept(x)) / len(xs)) if xs else None


def run_system(name, cohort, rules, seeds, ks=(1, 2, 3, 4), variants=VARIANTS, nneg=600,
               seed=0, limit=30000, tlimit=120.0, ball_cap=400000):
    rng = random.Random(seed)
    layers, trunc = ball_layers(seeds, rules, max(ks), cap=ball_cap)
    A0 = dag_automaton(seeds)
    out = {'system': name, 'cohort': cohort, 'rules': [rule_str(r) for r in rules],
           'seed_sizes': [size(t) for t in seeds], 'ball_truncated': trunc,
           'ball_sizes': {str(k): len(layers[k]) for k in ks}, 'cells': []}
    negs = {}
    base = {}
    for k in ks:
        bl = list(layers[k])
        rng_k = random.Random(seed * 100 + k)
        negs[k] = make_negatives(rng_k, bl, layers[k], rules, k, nper=nneg)
        base[k] = tree_alpha(layers[k])
    # FN check sets
    for vname, wdepth in variants.items():
        try:
            cps = run_ta(A0, rules, max(ks), widen=wdepth, limit=limit, tlimit=tlimit)
            status = 'ok'
        except Blowup as e:
            cps = e.partial
            status = 'blowup:%s' % e
        done = {c[0]: c for c in cps}
        for k in ks:
            cell = {'variant': vname, 'k': k}
            if k not in done:
                cell['status'] = 'blowup'
                cell['blowup_reason'] = status
                out['cells'].append(cell)
                continue
            A = done[k][1]
            A.prepare()
            ball = layers[k]
            bl = list(ball)
            if len(bl) > FN_CHECK_CAP:
                chk = random.Random(seed + k).sample(bl, FN_CHECK_CAP)
                cell['fn_checked'] = 'sample'
            else:
                chk = bl
                cell['fn_checked'] = 'all'
            fn = sum(1 for t in chk if not A.accepts(t))
            cell.update({'status': 'ok', 'false_negatives': fn, 'checked': len(chk),
                         'states': A.n_active_states(), 'transitions': A.ntrans(),
                         'seconds': round(done[k][2], 3) if k > 0 else 0.0,
                         'ball_size': len(ball)})
            for c in ('N1', 'N2', 'N3'):
                cell['fpr_' + c] = fpr(A.accepts, negs[k][c])
                cell['baseline_fpr_' + c] = fpr(lambda t: alpha_member(base[k], t), negs[k][c])
            cell['baseline_size'] = alpha_size(base[k])
            cell['n_neg'] = {c: len(negs[k][c]) for c in negs[k]}
            # soundness for the baseline envelope (alpha(ball) contains ball by construction)
            out['cells'].append(cell)
    return out


# ----------------------------------------------------------------------------
# unary-chain (string) case
# ----------------------------------------------------------------------------
ALPH = 4
CHAIN_N = 8


def enc(word):
    t = ('L', 0)
    for c in reversed(word):
        t = ('U', c, t)
    return t


def string_rule(u, v):
    def wp(w):
        p = ('V', 0)
        for c in reversed(w):
            p = ('U', c, p)
        return p
    return (wp(u), wp(v))


def string_succs(w, rules_s):
    out = set()
    for u, v in rules_s:
        L = len(u)
        for i in range(len(w) - L + 1):
            if w[i:i + L] == u:
                out.add(w[:i] + v + w[i + L:])
    return out


def pos_bigram_alpha(words, n=CHAIN_N):
    first = {w[0] for w in words}
    big = {(i, w[i], w[i + 1]) for w in words for i in range(n - 1)}
    return first, big


def pos_bigram_gamma(first, big, n=CHAIN_N):
    nxt = {}
    for (i, c, d) in big:
        nxt.setdefault((i, c), []).append(d)
    res = []

    def go(w):
        i = len(w) - 1
        if len(w) == n:
            res.append(tuple(w))
            return
        for d in nxt.get((i, w[-1]), ()):
            w.append(d)
            go(w)
            w.pop()
    for c in first:
        go([c])
    return set(res)


def chain_automaton(first, big, n=CHAIN_N):
    A = NTA()
    names = A.names
    def st(i, c):
        key = ('P', i, c)
        if key not in names:
            q = A.new_state()
            names[key] = q
            A.pos[q] = i
        return names[key]
    end = st(n, 'END')
    A.add_leaf(0, end)
    for (i, c, d) in big:
        a = st(i, c)
        b = st(i + 1, d) if i + 1 < n else end
        A.add_un(c, b, a)
    for c in first:
        A.final.add(st(0, c))
    # last letter: bigram to END allowed for any letter that appears last; recorded below
    return A, st


def chain_automaton_full(words, n=CHAIN_N):
    first, big = pos_bigram_alpha(words, n)
    A, st = chain_automaton(first, big, n)
    lastl = {w[n - 1] for w in words}
    # (n-1, c) -> END transitions
    for c in lastl:
        A.add_un(c, A.names[('P', n, 'END')], st(n - 1, c))
    return A, first, big, st


def chain_namer(n):
    def namer(A, q, sub, sub_path):
        p = A.pos[q] + 1
        top = 'END' if sub[0] == 'L' else sub[1]
        key = ('P', p, top)
        if key not in A.names:
            s = A.new_state()
            A.names[key] = s
            A.pos[s] = p
        return A.names[key]
    return namer


def all_words(n=CHAIN_N):
    return list(itertools.product(range(ALPH), repeat=n))


def chain_system(seed, ks=(1, 2, 3, 4), n=CHAIN_N):
    rng = random.Random(seed)
    rules_s = []
    while len(rules_s) < 3:
        L = rng.choice([2, 3])
        u = tuple(rng.randrange(ALPH) for _ in range(L))
        v = tuple(rng.randrange(ALPH) for _ in range(L))
        if u != v and (u, v) not in rules_s:
            rules_s.append((u, v))
    words = []
    while len(words) < 6:
        w = tuple(rng.randrange(ALPH) for _ in range(n))
        if w not in words:
            words.append(w)
    rules_t = [string_rule(u, v) for u, v in rules_s]
    # exact ball on strings and via the generic tree engine (embedding check)
    cur = set(words)
    sball = [set(cur)]
    for _ in range(max(ks)):
        nxt = set(cur)
        for w in cur:
            nxt |= string_succs(w, rules_s)
        cur = nxt
        sball.append(set(cur))
    tl, _ = ball_layers([enc(w) for w in words], rules_t, max(ks))
    embed_ok = all({enc(w) for w in sball[k]} == tl[k] for k in ks)
    # LBF-18-style positional-bigram envelope iterate
    first, big = pos_bigram_alpha(words, n)
    env = []
    gam = pos_bigram_gamma(first, big, n)
    envs = {}
    cur_g = gam
    for k in range(1, max(ks) + 1):
        nxt = set(cur_g)
        for w in cur_g:
            nxt |= string_succs(w, rules_s)
        first, big = pos_bigram_alpha(nxt, n)
        cur_g = pos_bigram_gamma(first, big, n)
        envs[k] = cur_g
    # TA with positional canonical naming
    A0, first0, big0, st = chain_automaton_full(words, n)
    cps = run_ta(A0, rules_t, max(ks), namer=chain_namer(n))
    # generic exact DAG-automaton on the same chains
    D0 = dag_automaton([enc(w) for w in words])
    dcps = run_ta(D0, rules_t, max(ks))
    W = all_words(n)
    out = {'seed': seed, 'rules': [[''.join('abcd'[c] for c in u), ''.join('abcd'[c] for c in v)] for u, v in rules_s],
           'seeds': [''.join('abcd'[c] for c in w) for w in words], 'embedding_ok': embed_ok, 'cells': []}
    for k in ks:
        A = cps[k][1]
        A.prepare()
        lang = {w for w in W if A.accepts(enc(w))}
        D = dcps[k][1]
        D.prepare()
        dlang = {w for w in W if D.accepts(enc(w))}
        ball = sball[k]
        out['cells'].append({
            'k': k, 'ball': len(ball), 'ta_positional_lang': len(lang), 'ta_dag_lang': len(dlang),
            'lbf18_env_lang': len(envs[k]),
            'ta_pos_false_negatives': len(ball - lang), 'ta_dag_false_negatives': len(ball - dlang),
            'env_false_negatives': len(ball - envs[k]),
            'ta_pos_subset_env': lang <= envs[k], 'env_subset_ta_pos': envs[k] <= lang,
            'ta_pos_equals_env': lang == envs[k],
            'ta_dag_subset_env': dlang <= envs[k],
            'ta_pos_states': A.n_active_states(), 'ta_pos_transitions': A.ntrans(),
            'ta_dag_states': D.n_active_states(), 'ta_dag_transitions': D.ntrans()})
    return out


# ----------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------
def build_cohorts():
    systems = [('named_arith', 'C1', named_system(), 19400)]
    for i in range(5):
        systems.append(('preserve_%d' % i, 'C2', random_system(19100 + i, 'preserve'), 19410 + i))
    for i in range(5):
        systems.append(('shrink_%d' % i, 'C3', random_system(19200 + i, 'shrink'), 19420 + i))
    for i in range(5):
        systems.append(('mixed_%d' % i, 'C4', random_system(19300 + i, 'mixed'), 19430 + i))
    return systems


def seeds_for(seed):
    rng = random.Random(seed)
    return [rand_term(rng, rng.randint(5, 40)) for _ in range(6)]


def plant_instance(rng, t, rules):
    """AMENDMENT (post-registration, run B): replace a random subterm of t by an instance of a random rule lhs."""
    lhs = rng.choice(rules)[0]
    sub = {v: rand_term(rng, rng.randint(1, 4)) for v in set(pat_vars(lhs))}
    inst = subst(lhs, sub)
    positions = []

    def collect(x, path):
        positions.append(path)
        if x[0] == 'U':
            collect(x[2], path + (2,))
        elif x[0] == 'B':
            collect(x[2], path + (2,))
            collect(x[3], path + (3,))
    collect(t, ())
    path = rng.choice(positions)

    def rebuild(x, path):
        if not path:
            return inst
        if x[0] == 'U':
            return ('U', x[1], rebuild(x[2], path[1:]))
        if path[0] == 2:
            return ('B', x[1], rebuild(x[2], path[1:]), x[3])
        return ('B', x[1], x[2], rebuild(x[3], path[1:]))
    return rebuild(t, path)


def seeds_planted(seed, rules):
    rng = random.Random(seed + 7)
    out = []
    for _ in range(6):
        t = rand_term(rng, rng.randint(5, 30))
        for _ in range(rng.randint(1, 2)):
            t = plant_instance(rng, t, rules)
        out.append(t)
    return out


def summarize(results):
    import statistics as st
    summ = {}
    for vn in VARIANTS:
        for k in (1, 2, 3, 4):
            cells = [c for r in results for c in r['cells'] if c['variant'] == vn and c['k'] == k]
            ok = [c for c in cells if c['status'] == 'ok']
            row = {'cells': len(cells), 'blowups': len(cells) - len(ok),
                   'fn_total': sum(c['false_negatives'] for c in ok)}
            if ok:
                row['mean_states'] = st.mean(c['states'] for c in ok)
                row['mean_transitions'] = st.mean(c['transitions'] for c in ok)
                row['mean_seconds'] = st.mean(c['seconds'] for c in ok)
                for n in ('N1', 'N2', 'N3'):
                    row['mean_fpr_' + n] = st.mean(c['fpr_' + n] for c in ok if c['fpr_' + n] is not None)
                    row['mean_baseline_fpr_' + n] = st.mean(c['baseline_fpr_' + n] for c in ok if c['baseline_fpr_' + n] is not None)
                row['mean_baseline_size'] = st.mean(c['baseline_size'] for c in ok)
            summ['%s|k=%d' % (vn, k)] = row
    return summ


def main(outdir, plant=False):
    os.makedirs(outdir, exist_ok=True)
    results = []
    for name, cohort, rules, sseed in build_cohorts():
        seeds = seeds_planted(sseed, rules) if plant else seeds_for(sseed)
        t0 = time.time()
        r = run_system(name, cohort, rules, seeds, seed=sseed)
        r['wall_seconds'] = round(time.time() - t0, 2)
        results.append(r)
        print(name, 'ball', r['ball_sizes'], 'trunc', r['ball_truncated'], 'wall', r['wall_seconds'], flush=True)
    summ = summarize(results)
    chains = [chain_system(19500 + i) for i in range(5)] if not plant else None
    with open(os.path.join(outdir, 'tree_systems.json'), 'w') as f:
        json.dump({'systems': results, 'summary': summ}, f, indent=1)
    if chains:
        with open(os.path.join(outdir, 'chain_embedding.json'), 'w') as f:
            json.dump({'systems': chains}, f, indent=1)
    write_svgs(outdir, results, summ, chains)
    return results, summ, chains


# ----------------------------------------------------------------------------
# SVG (plain code)
# ----------------------------------------------------------------------------
def _svg_lines(path, title, series, ylabel, logy=False, W=520, H=320):
    import math
    pad = 55
    xs = [1, 2, 3, 4]
    vals = [v for s in series.values() for v in s if v is not None]
    if not vals:
        return
    f = (lambda v: math.log10(max(v, 1e-9))) if logy else (lambda v: v)
    lo, hi = min(f(v) for v in vals), max(f(v) for v in vals)
    if hi == lo:
        hi = lo + 1
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" font-size="11">' % (W, H),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="%d" y="18" font-size="13">%s</text>' % (pad, title)]
    px = lambda x: pad + (x - 1) / 3 * (W - pad - 20)
    py = lambda v: H - pad - (f(v) - lo) / (hi - lo) * (H - pad - 35)
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (pad, H - pad, W - 20, H - pad))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (pad, H - pad, pad, 30))
    for x in xs:
        parts.append('<text x="%.0f" y="%d">k=%d</text>' % (px(x) - 8, H - pad + 16, x))
    parts.append('<text x="5" y="%d" transform="rotate(-90 12 %d)">%s</text>' % (H // 2 + 40, H // 2 + 40, ylabel))
    parts.append('<text x="%d" y="%d">%.3g</text>' % (5, 40, 10 ** hi if logy else hi))
    parts.append('<text x="%d" y="%d">%.3g</text>' % (5, H - pad, 10 ** lo if logy else lo))
    for i, (name, ys) in enumerate(series.items()):
        c = colors[i % len(colors)]
        pts = ' '.join('%.1f,%.1f' % (px(x), py(y)) for x, y in zip(xs, ys) if y is not None)
        parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (c, pts))
        parts.append('<text x="%d" y="%d" fill="%s">%s</text>' % (W - 150, 40 + 14 * i, c, name))
    parts.append('</svg>')
    open(path, 'w').write('\n'.join(parts))


def write_svgs(outdir, results, summ, chains):
    ser_s, ser_f = {}, {}
    for vn in VARIANTS:
        ser_s[vn] = [summ['%s|k=%d' % (vn, k)].get('mean_states') for k in (1, 2, 3, 4)]
        ser_f[vn + ' N1'] = [summ['%s|k=%d' % (vn, k)].get('mean_fpr_N1') for k in (1, 2, 3, 4)]
    ser_f['baseline N1'] = [summ['exact|k=%d' % k].get('mean_baseline_fpr_N1') for k in (1, 2, 3, 4)]
    _svg_lines(os.path.join(outdir, 'fig_states_vs_k.svg'), 'Mean NTA states vs k (non-blowup cells)', ser_s, 'states (log10)', logy=True)
    _svg_lines(os.path.join(outdir, 'fig_fpr_vs_k.svg'), 'Mean false-positive rate on N1 (k+1..k+2 steps)', ser_f, 'FPR')
    if not chains:
        return
    ser_c = {'ball': [], 'TA positional': [], 'LBF18 env': [], 'TA dag': []}
    for k in (1, 2, 3, 4):
        cs = [c for r in chains for c in r['cells'] if c['k'] == k]
        ser_c['ball'].append(sum(c['ball'] for c in cs) / len(cs))
        ser_c['TA positional'].append(sum(c['ta_positional_lang'] for c in cs) / len(cs))
        ser_c['LBF18 env'].append(sum(c['lbf18_env_lang'] for c in cs) / len(cs))
        ser_c['TA dag'].append(sum(c['ta_dag_lang'] for c in cs) / len(cs))
    _svg_lines(os.path.join(outdir, 'fig_chain_language_sizes.svg'), 'Unary-chain case: mean language size (length-8 strings)', ser_c, 'strings (log10)', logy=True)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lbf19', 'tree_automata')
    main(out, plant=(len(sys.argv) > 2 and sys.argv[2] == 'planted'))
