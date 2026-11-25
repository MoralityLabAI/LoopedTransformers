"""LBF-20C witness-carrying guard-set Pareto completion for ground terms."""
from __future__ import annotations

import collections
import functools
import random
import time

from lbf20_ground_witness import subterms, replace, steps

# Each map is guard mask -> concrete path of (rule index, one-hole position).
# Path length is the exact sequential rewrite count. No arbitrary top-k cap.


def insert(dest, mask, path, H):
    k = len(path)
    if k > H:
        return False
    if any(m & mask == m and len(p) <= k for m, p in dest.items()):
        return False
    for m in list(dest):
        if mask & m == mask and k <= len(dest[m]):
            del dest[m]
    dest[mask] = path
    return True


def product(a, b, H, prefix=()):
    out = {}
    for m, p in a.items():
        for n, q in b.items():
            if len(p) + len(q) <= H:
                insert(out, m | n, p + tuple((i, prefix + pos) for i, pos in q), H)
    return out


def at(t, pos):
    for i in pos:
        t = t[i]
    return t


def replay(source, path, rules):
    t = source
    mask = 0
    for i, pos in path:
        l, r, g = rules[i]
        assert at(t, pos) == l, (t, i, pos)
        t = replace(t, pos, r)
        mask |= 1 << g
    return t, mask


class ParetoWitness:
    def __init__(self, source, rules, horizon):
        assert horizon >= 0 and all(type(g) is int and g >= 0 for _, _, g in rules)
        self.rules = tuple(rules)
        self.H = horizon
        self.states = sorted(set(subterms(source)) |
                             {s for l, r, _ in rules for t in (l, r) for s in subterms(t)}, key=repr)
        self.ids = {t: i for i, t in enumerate(self.states)}
        self.n = len(self.states)
        self.E = [[{0: ()} if i == q else {} for q in range(self.n)] for i in range(self.n)]
        self.rounds = self.updates = 0
        G = len({g for _, _, g in rules})
        # Each update adds previously absent coverage in the finite upward
        # closure over guard subsets and lengths <= H. No arbitrary cutoff.
        bound = self.n ** 2 * (1 << G) * (horizon + 1) + 1
        while True:
            self.rounds += 1
            changed = False
            for i, (l, r, g) in enumerate(self.rules):
                for q, vals in enumerate(self.evaluate(l)):
                    for m, p in vals.items():
                        changed |= self._add(self.ids[r], q, m | (1 << g), p + ((i, ()),))
            for z in range(self.n):
                for i in range(self.n):
                    if not self.E[i][z]:
                        continue
                    for q in range(self.n):
                        if not self.E[z][q]:
                            continue
                        for m, p in product(self.E[z][q], self.E[i][z], self.H).items():
                            changed |= self._add(i, q, m, p)
            if not changed:
                break
            if self.rounds > bound:
                raise RuntimeError("finite coverage bound violated")

    def _add(self, i, q, mask, path):
        changed = insert(self.E[i][q], mask, path, self.H)
        self.updates += int(changed)
        return changed

    def evaluate(self, term):
        @functools.lru_cache(None)
        def rec(t):
            children = [rec(x) for x in t[1:]]
            out = [{} for _ in self.states]
            for i, s in enumerate(self.states):
                if (s[0], len(s)) != (t[0], len(t)):
                    continue
                local = {0: ()}
                for j, (x, c) in enumerate(zip(s[1:], children), 1):
                    local = product(local, c[self.ids[x]], self.H, (j,))
                if not local:
                    continue
                for q in range(self.n):
                    for m, p in product(self.E[i][q], local, self.H).items():
                        insert(out[q], m, p, self.H)
            return tuple(out)
        return rec(term)

    def frontier(self, source, target):
        return self.evaluate(target)[self.ids[source]]

    def audit_table(self):
        checked = 0
        def covered(dest, m, k):
            return k > self.H or any(n & m == n and len(p) <= k for n, p in dest.items())
        for i, row in enumerate(self.E):
            assert row[i].get(0) == ()
            for q, labels in enumerate(row):
                for m, p in labels.items():
                    assert replay(self.states[q], p, self.rules) == (self.states[i], m)
                    assert len(p) <= self.H
                    assert not any(n != m and n & m == n and len(v) <= len(p)
                                   for n, v in labels.items())
                    checked += 1
                for z in range(self.n):
                    for m, p in product(self.E[z][q], self.E[i][z], self.H).items():
                        assert covered(labels, m, len(p))
        for l, r, g in self.rules:
            for q, vals in enumerate(self.evaluate(l)):
                for m, p in vals.items():
                    assert covered(self.E[self.ids[r]][q], m | (1 << g), len(p) + 1)
        return checked


def concrete(source, rules, H):
    """Independent exact labeled-state BFS; retain minimal length per (term,mask)."""
    seen = {(source, 0): 0}
    front = {(source, 0)}
    for k in range(1, H + 1):
        nxt = set()
        for t, m in front:
            for i, (l, r, g) in enumerate(rules):
                for _, _, u in steps(t, [(l, r)]):
                    key = (u, m | (1 << g))
                    if key not in seen:
                        seen[key] = k
                        nxt.add(key)
        front = nxt
        if len(seen) > 80000:
            raise RuntimeError("fixture cap exceeded")
    vals = collections.defaultdict(dict)
    for (t, m), k in seen.items():
        vals[t][m] = k
    # Separate direct domination test, not the automaton insert routine.
    return {t: {m: k for m, k in x.items()
                if not any(n != m and n & m == n and j <= k for n, j in x.items())}
            for t, x in vals.items()}


def random_term(rng, d):
    if d == 0 or rng.random() < .55:
        return (rng.choice("abc"),)
    if rng.random() < .5:
        return ("f", random_term(rng, d - 1))
    return ("p", random_term(rng, d - 1), random_term(rng, d - 1))


def audit():
    rng = random.Random(2026112506)
    H = 4
    started = time.perf_counter()
    counts = collections.Counter()
    for i in range(96):
        a, b, c = [(x,) for x in "abc"]
        rules = [(a, b, 0), (b, c, 1), (a, c, 2), (c, a, 3)]
        for _ in range(2):
            l, r = random_term(rng, 2), random_term(rng, 2)
            if len(list(subterms(r))) > len(list(subterms(l))):
                l, r = r, l
            if l != r:
                rules.append((l, r, rng.randrange(5)))
        if i % 8 == 0:
            rules.append((b, ("f", c), 4))
        source = ("p", random_term(rng, 1), random_term(rng, 1))
        expected = concrete(source, rules, H)
        wa = ParetoWitness(source, rules, H)
        counts["epsilon_witnesses_replayed"] += wa.audit_table()
        counts["max_states"] = max(counts["max_states"], wa.n)
        counts["max_rounds"] = max(counts["max_rounds"], wa.rounds)
        targets = set(expected) | {random_term(rng, 2) for _ in range(16)}
        for t in sorted(targets, key=repr):
            v = wa.frontier(source, t)
            assert {m: len(p) for m, p in v.items()} == expected.get(t, {}), (i, t)
            for m, p in v.items():
                assert replay(source, p, rules) == (t, m)
            counts["frontier_comparisons"] += 1
            counts["target_witnesses_replayed"] += len(v)
            counts["max_frontier"] = max(counts["max_frontier"], len(v))
            # Every possible subset of the five guard IDs: filtering the
            # pruned frontier must equal exact labeled-state feasibility.
            for allowed in range(32):
                want = min((k for m, k in expected.get(t, {}).items() if m & ~allowed == 0), default=H+1)
                got = min((len(p) for m, p in v.items() if m & ~allowed == 0), default=H+1)
                assert want == got
                counts["filtered_queries"] += 1
    a, b, c = [(x,) for x in "abc"]
    source = ("p", ("p", a, a), ("p", a, a))
    target = ("p", ("p", c, c), ("p", c, c))
    wa = ParetoWitness(source, [(a,b,0), (b,c,0), (a,c,1)], 8)
    assert {m: len(p) for m,p in wa.frontier(source,target).items()} == {1:8, 2:4}
    # Two independent leaf families: shortest path uses two distinct guards;
    # longer path shares one. A strict mask-cost tradeoff, not a cardinality tie.
    d,e,f = [(x,) for x in "def"]
    s,t = ("p",a,d),("p",c,f)
    wa = ParetoWitness(s, [(a,b,0),(b,c,0),(d,e,0),(e,f,0),(a,c,1),(d,f,2)],4)
    assert {m:len(p) for m,p in wa.frontier(s,t).items()} == {1:4,6:2}
    return {"systems":96,"horizon":H,**dict(counts),"mismatches":0,
            "seconds":time.perf_counter()-started}
