"""Bounded min-plus tree automata for ground rules; independent retrospective fixtures."""
import functools
import json
import random
from pathlib import Path

K = 5
INF = K + 1
A = ('a',)
B = ('b',)
C = ('c',)


def subterms(t):
    yield t
    for x in t[1:]:
        yield from subterms(x)


def succ(t, rules):
    for l, r in rules:
        if t == l:
            yield r
    for i, x in enumerate(t[1:], 1):
        for y in succ(x, rules):
            yield t[:i] + (y,) + t[i+1:]


def ball(root, rules, k):
    d = {root: 0}
    f = {root}
    for i in range(1, k+1):
        f = {u for t in f for u in succ(t, rules)} - d.keys()
        d.update({t: i for t in f})
        if len(d) > 50000:
            raise RuntimeError('fixture too large')
    return d


class WeightedGround:
    def __init__(self, seeds, rules, k=K):
        self.k = k
        self.inf = k+1
        self.states = sorted({s for t in [*seeds, *(x for rule in rules for x in rule)]
                              for s in subterms(t)}, key=repr)
        self.ids = {t: i for i, t in enumerate(self.states)}
        self.n = len(self.states)
        self.e = [[0 if i == j else self.inf for j in range(self.n)] for i in range(self.n)]
        self.rounds = 0
        while True:
            self.rounds += 1
            prev = [row[:] for row in self.e]
            for l, r in rules:
                i = self.ids[r]
                for q, v in enumerate(self.costs(l)):
                    self.e[i][q] = min(self.e[i][q], v+1)
            for h in range(self.n):
                for i in range(self.n):
                    if self.e[i][h] >= self.inf:
                        continue
                    for j in range(self.n):
                        self.e[i][j] = min(self.e[i][j], self.e[i][h] + self.e[h][j])
            if prev == self.e:
                break
            if self.rounds > self.n*self.n*(self.k+1):
                raise RuntimeError('nontermination')

    def costs(self, term):
        @functools.lru_cache(None)
        def rec(t):
            child = [rec(x) for x in t[1:]]
            out = [self.inf]*self.n
            for q, s in enumerate(self.states):
                if s[0] != t[0] or len(s) != len(t):
                    continue
                base = sum(v[self.ids[x]] for v, x in zip(child, s[1:]))
                if base >= self.inf:
                    continue
                for j, w in enumerate(self.e[q]):
                    out[j] = min(out[j], base+w)
            return tuple(out)
        return rec(term)

    def distance(self, seed, t):
        return self.costs(t)[self.ids[seed]]


def random_term(rng, depth):
    if depth <= 0 or rng.random() < .42:
        return rng.choice((A, B, C))
    if rng.random() < .4:
        return (rng.choice(('f', 'g')), random_term(rng, depth-1))
    return ('p', random_term(rng, depth-1), random_term(rng, depth-1))


def main():
    rng = random.Random(20261125)
    rows = []
    s = ('p', A, A)
    t = ('p', B, B)
    assert WeightedGround([s], [(A, B)]).distance(s, t) == 2
    for i in range(100):
        rules = [(A, B), (B, C)]
        for _ in range(3):
            l, r = random_term(rng, 2), random_term(rng, 2)
            if len(tuple(subterms(r))) > len(tuple(subterms(l))):
                l, r = r, l
            if l != r:
                rules.append((l, r))
        if i % 5 == 0:
            rules.append((('f', A), ('p', B, C)))
        seed = ('p', rng.choice(rules)[0], random_term(rng, 2))
        b = ball(seed, rules, K+2)
        wa = WeightedGround([seed], rules)
        probes = set(b) | {random_term(rng, 3) for _ in range(25)}
        fn = fp = 0
        for t in probes:
            expected = min(INF, b.get(t, INF))
            actual = wa.distance(seed, t)
            fn += actual > expected
            fp += actual < expected
            if actual != expected:
                raise AssertionError((i, t, actual, expected))
        rows.append(dict(system=i, probes=len(probes),
                         near_horizon_negatives=sum(K < d <= K+2 for d in b.values()),
                         states=wa.n, rounds=wa.rounds, fn=fn, fp=fp))
    summary = dict(systems=len(rows), probes=sum(r['probes'] for r in rows),
                   near_horizon_negatives=sum(r['near_horizon_negatives'] for r in rows),
                   max_states=max(r['states'] for r in rows), max_rounds=max(r['rounds'] for r in rows),
                   false_negatives=0, false_positives=0, K=K,
                   scope='ground rules only; deterministic retrospective synthetic fixtures', rows=rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
