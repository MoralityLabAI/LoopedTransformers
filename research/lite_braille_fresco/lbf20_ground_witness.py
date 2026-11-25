"""LBF-20A: finite weighted ground completion with independently replayable witnesses.

Development-only, fresh deterministic fixtures; standard-library CPU.
No upstream checkpoints or prior evaluation roots loaded.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import statistics
import time
from pathlib import Path

A, B, C = ("a",), ("b",), ("c",)
SEED = 2026112502
H = 5


def subterms(t):
    yield t
    for x in t[1:]:
        yield from subterms(x)


def replace(t, pos, new):
    if not pos:
        return new
    i, *rest = pos
    return t[:i] + (replace(t[i], rest, new),) + t[i + 1:]


def steps(t, rules, pos=()):
    for i, (l, r) in enumerate(rules):
        if t == l:
            yield i, pos, r
    for j, x in enumerate(t[1:], 1):
        for i, p, y in steps(x, rules, pos + (j,)):
            yield i, p, t[:j] + (y,) + t[j + 1:]


def replay(path, rules):
    assert path
    for x, y in zip(path, path[1:]):
        assert any(y == z for _, _, z in steps(x, rules)), (x, y)
    return len(path) - 1


def ball(root, rules, k, cap=50000):
    d = {root: 0}
    f = {root}
    for i in range(1, k + 1):
        f = {y for t in f for _, _, y in steps(t, rules)} - d.keys()
        d.update({t: i for t in f})
        if len(d) > cap:
            raise RuntimeError("fixture BFS cap exceeded")
    return d


def chain(a, b):
    assert a[-1] == b[0]
    return a + b[1:]


def better(old, new, cap):
    if new is not None and len(new) - 1 <= cap and (old is None or len(new) < len(old)):
        return new
    return old


class WeightedWitness:
    """E[i][q] is a shortest currently known concrete path from state q to i.

    All state labels are ground subterms. Structural transitions cost zero.
    Capped absent paths are infinity, not a claim of unreachability.
    """
    def __init__(self, seeds, rules, horizon):
        assert horizon >= 0 and all(isinstance(t, tuple) and t for t in seeds)
        self.h = horizon
        self.rules = tuple(rules)
        self.states = sorted({s for t in [*seeds, *(x for rule in rules for x in rule)]
                              for s in subterms(t)}, key=repr)
        self.ids = {t: i for i, t in enumerate(self.states)}
        self.n = len(self.states)
        self.e = [[(self.states[i],) if i == q else None for q in range(self.n)] for i in range(self.n)]
        self.rounds = 0
        self.updates = 0
        while True:
            self.rounds += 1
            changed = False
            for l, r in self.rules:
                paths = self.evaluate(l)
                i = self.ids[r]
                for q, p in enumerate(paths):
                    if p is not None:
                        changed |= self._improve(i, q, p + (r,))
            for z in range(self.n):
                for i in range(self.n):
                    if self.e[i][z] is None:
                        continue
                    for q in range(self.n):
                        if self.e[z][q] is not None:
                            changed |= self._improve(i, q, chain(self.e[z][q], self.e[i][z]))
            if not changed:
                break
            if self.rounds > self.n * self.n * (horizon + 1) + 1:
                raise RuntimeError("finite cost descent bound violated")

    def _improve(self, i, q, path):
        old = self.e[i][q]
        best = better(old, path, self.h)
        if best is old:
            return False
        self.e[i][q] = best
        self.updates += 1
        return True

    def evaluate(self, term):
        @functools.lru_cache(None)
        def rec(t):
            children = [rec(x) for x in t[1:]]
            out = [None] * self.n
            for i, s in enumerate(self.states):
                if s[0] != t[0] or len(s) != len(t):
                    continue
                chosen = [v[self.ids[x]] for v, x in zip(children, s[1:])]
                if any(p is None for p in chosen) or sum(len(p) - 1 for p in chosen) > self.h:
                    continue
                # Serialize independent child derivations in left-to-right order.
                cur = s
                local = (s,)
                for j, p in enumerate(chosen, 1):
                    assert cur[j] == p[0]
                    for nxt in p[1:]:
                        cur = cur[:j] + (nxt,) + cur[j + 1:]
                        local += (cur,)
                assert cur == t
                for q, e in enumerate(self.e[i]):
                    if e is not None:
                        out[q] = better(out[q], chain(e, local), self.h)
            return tuple(out)
        return rec(term)

    def witness(self, source, target):
        return self.evaluate(target)[self.ids[source]]

    def distance(self, source, target):
        p = self.witness(source, target)
        return self.h + 1 if p is None else len(p) - 1

    def audit_table(self):
        """Check every finite entry and the fixed-point inequalities independently."""
        count = 0
        cost = lambda p: self.h + 1 if p is None else len(p) - 1
        for i, row in enumerate(self.e):
            assert cost(row[i]) == 0
            for q, p in enumerate(row):
                if p is not None:
                    assert p[0] == self.states[q] and p[-1] == self.states[i]
                    assert replay(p, self.rules) <= self.h
                    count += 1
                for z in range(self.n):
                    assert cost(p) <= cost(self.e[z][q]) + cost(self.e[i][z])
        for l, r in self.rules:
            i = self.ids[r]
            for q, p in enumerate(self.evaluate(l)):
                assert cost(self.e[i][q]) <= cost(p) + 1
        return count


def random_term(rng, depth):
    if depth <= 0 or rng.random() < .45:
        return rng.choice((A, B, C))
    if rng.random() < .45:
        return (rng.choice(("f", "g")), random_term(rng, depth - 1))
    return ("p", random_term(rng, depth - 1), random_term(rng, depth - 1))


def full(t, leaves):
    if leaves == 1:
        return t
    return ("p", full(t, leaves // 2), full(t, leaves // 2))


def expression(t, hole=None):
    if t == ("HOLE",):
        assert hole is not None
        return hole
    if len(t) == 1:
        return "c_" + t[0]
    return "(f_" + t[0] + " " + " ".join(expression(x, hole) for x in t[1:]) + ")"


def lean_proof(name, rules, path):
    # Every concrete step is an explicit equality under a one-hole context.
    terms = [*path, *(t for rule in rules for t in rule)]
    arities = {s[0]: len(s) - 1 for t in terms for s in subterms(t)}
    assert all(len(s) - 1 == arities[s[0]] for t in terms for s in subterms(t))
    binders = ["(T : Type)"]
    for f, arity in sorted(arities.items()):
        symbol = ("c_" if arity == 0 else "f_") + f
        binders.append(f"({symbol} : {' → '.join(['T'] * (arity + 1))})")
    binders += [f"(h{i} : {expression(l)} = {expression(r)})" for i, (l, r) in enumerate(rules)]
    lines = [f"theorem {name} {' '.join(binders)} : {expression(path[0])} = {expression(path[-1])} := by"]
    if len(path) == 1:
        lines += ["  rfl"]
    else:
        lines += ["  calc"]
        for j, (s, t) in enumerate(zip(path, path[1:])):
            i, pos, _ = next((i, p, x) for i, p, x in steps(s, rules) if x == t)
            ctx = expression(replace(s, pos, ("HOLE",)), "z")
            proof = f"congrArg (fun z : T => {ctx}) h{i}"
            lines.append(f"    {expression(s) if j == 0 else '_'} = {expression(t)} := {proof}")
    lines.append(f"#print axioms {name}")
    return "\n".join(lines) + "\n"


def run(out):
    rng = random.Random(SEED)
    rows, proofs = [], []
    for i in range(128):
        rules = [(A, B), (B, C)]
        if i % 4 == 0:
            rules.append((C, A))  # cycle
        for _ in range(3):
            l, r = random_term(rng, 2), random_term(rng, 2)
            if len(tuple(subterms(r))) > len(tuple(subterms(l))):
                l, r = r, l
            if l != r:
                rules.append((l, r))
        if i % 8 == 0:
            rules.append((("f", A), ("p", B, C)))  # bounded growth
        seed = ("p", rng.choice(rules)[0], random_term(rng, 2))
        b = ball(seed, rules, H + 2)
        t0 = time.perf_counter()
        wa = WeightedWitness([seed], rules, H)
        n_table = wa.audit_table()
        finite = probes = 0
        for t in set(b) | {random_term(rng, 3) for _ in range(30)}:
            expected = min(H + 1, b.get(t, H + 1))
            p = wa.witness(seed, t)
            actual = H + 1 if p is None else replay(p, rules)
            assert actual == expected, (i, t, actual, expected)
            probes += 1
            finite += p is not None
        # One independent native certificate from each of the first 16 systems.
        if i < 16:
            target = max((t for t in b if b[t] <= H), key=lambda t: (b[t], repr(t)))
            proofs.append(lean_proof(f"ground_fixture_{i}", rules, wa.witness(seed, target)))
        rows.append({"system": i, "probes": probes, "finite_witnesses": finite,
                     "beyond_horizon": sum(H < d <= H + 2 for d in b.values()),
                     "states": wa.n, "rounds": wa.rounds, "table_witnesses": n_table,
                     "seconds": time.perf_counter() - t0})
    scale = []
    for leaves in (4, 8, 16, 32):
        rules = [(A, B), (B, C), (C, A)]
        start, target = full(A, leaves), full(C, leaves)
        t0 = time.perf_counter()
        wa = WeightedWitness([start], rules, 2 * leaves)
        witness = wa.witness(start, target)
        assert replay(witness, rules) == 2 * leaves
        short = WeightedWitness([start], rules, 2 * leaves - 1)
        assert short.witness(start, target) is None
        if leaves <= 8:
            d = ball(start, rules, 2 * leaves)
            assert len(d) == 3 ** leaves
            assert d[target] == 2 * leaves
        proofs.append(lean_proof(f"ground_scale_{leaves}", rules, witness))
        scale.append({"leaves": leaves, "states": wa.n, "exact_distance": 2 * leaves,
                      "reachable_term_count_analytic": 3 ** leaves, "full_bfs_checked": leaves <= 8,
                      "reject_at_budget_minus_one": True, "seconds": time.perf_counter() - t0})
    # A shrinking and a growing system, deliberately separate from the random fixtures.
    explicit = [
        (("g", ("f", A)), [ (("f", A), B) ], ("g", B), 1),
        (A, [(A, ("f", A))], ("f", ("f", ("f", A))), 3),
    ]
    for i, (s, rules, t, n) in enumerate(explicit):
        p = WeightedWitness([s], rules, n).witness(s, t)
        assert replay(p, rules) == n
        assert WeightedWitness([s], rules, n - 1).witness(s, t) is None
        proofs.append(lean_proof(f"ground_adversarial_{i}", rules, p))
    out.mkdir(parents=True, exist_ok=True)
    (out / "Witnesses.lean").write_text("\n".join(proofs), encoding="utf-8")
    summary = {"schema": "lbf20a.graded-ground.v1", "scope": "development_only",
               "generator_seed": SEED, "horizon": H, "systems": len(rows),
               "term_distance_checks": sum(r["probes"] for r in rows),
               "finite_witnesses_replayed": sum(r["finite_witnesses"] for r in rows),
               "epsilon_witnesses_replayed": sum(r["table_witnesses"] for r in rows),
               "beyond_horizon_negatives": sum(r["beyond_horizon"] for r in rows),
               "distance_mismatches": 0, "max_states": max(r["states"] for r in rows),
               "max_rounds": max(r["rounds"] for r in rows),
               "mean_random_system_seconds": statistics.mean(r["seconds"] for r in rows),
               "scale": scale, "lean_witness_declarations_generated": len(proofs), "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
