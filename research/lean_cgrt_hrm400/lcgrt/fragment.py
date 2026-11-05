"""Synthetic theorem generator for the Lean-CGRT gym (propositional Horn, backward chaining).

A theorem is `∀ (P0 … : Prop) (h… : Pa) (r… : Pb → Pc → Pd) …, Pg` (optionally `Pg ∧ Pk`).
Tactics are `apply <hyp>` and `constructor`, run by Lean on the main goal. Unlike v1,
an atom can have several producers, so proof search has real choices:

  dead   decoy producer whose premise has no derivation at all
  deep   decoy producer whose premise starts a 2-4 step chain ending in a dead atom
  cycle  decoy producer whose premise is derivable only through an ancestor goal
  alt    a valid but longer second derivation (shape `alt`)

Generator metadata (`gen`) records which rules are solution/decoy; it is never a
model input. Lean, not this module, is the authority on legality and goals; the
reference solver here exists to cross-check Lean's answers during extraction.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

SHAPES = ("chain", "tree", "alt")
DECOYS = ("dead", "deep", "cycle")
FAMILIES = tuple(f"{s}-{d}" for s in SHAPES for d in DECOYS)
HELDOUT_FAMILIES = ("alt-cycle", "tree-deep")
TRAIN_FAMILIES = tuple(f for f in FAMILIES if f not in HELDOUT_FAMILIES)
INF = 10 ** 9


@dataclass
class Theorem:
    tid: str
    family: str
    depth: int
    seed: int
    n_atoms: int
    facts: list            # atom index per fact hypothesis h{i}
    rules: list            # [[premises], head] per rule hypothesis r{i}
    goal: list             # [atom] or [atom, atom] (conjunction)
    arity3: bool = False
    gen: dict = field(default_factory=dict)

    def to_json(self):
        return asdict(self)

    @staticmethod
    def from_json(d):
        return Theorem(**d)


def generate(seed: int, depth: int, family: str, *, arity3: bool = False,
             conj_p: float = 0.25, tid: str | None = None,
             deep_len: tuple | None = None, cycle_extra: tuple | None = None,
             mirror_p: float | None = None) -> Theorem:
    """deep_len / cycle_extra (lo, hi) lengthen deep-decoy chains and cycle loops for the
    reasoning-depth holdout; None keeps the training distribution (2-4 steps; 0-1 extra)
    and leaves the random stream, hence every existing theorem, unchanged.

    mirror_p adds, per spine atom with that probability, a *mirror decoy*: a producer whose
    premises copy the real premises' solution derivations rule for rule, with every leaf fact
    replaced by a dead atom and the copies decorated with dead-decoy producers the way spine
    atoms are. Locally the decoy looks like the real derivation; only propagating from the
    leaves (two attention hops per level) separates them."""
    if family not in FAMILIES:
        raise ValueError(family)
    key = f"lcgrt-v1|{seed}|{depth}|{family}|{arity3}"
    if deep_len or cycle_extra:
        key += f"|dl{tuple(deep_len) if deep_len else None}|ce{tuple(cycle_extra) if cycle_extra else None}"
    if mirror_p:
        key += f"|mirror{mirror_p}"
    rng = random.Random(key)
    sol_pre: dict = {}                   # solution atom -> premises of its solution rule
    shape, decoy = family.split("-")
    n = 0
    rules: list = []
    kind: list = []                      # "sol" | "alt" | "decoy" | "distractor"

    def new():
        nonlocal n
        n += 1
        return n - 1

    def add(pre, head, k):
        rules.append([list(pre), head])
        kind.append(k)

    facts = [new() for _ in range(rng.randint(2, 4))]
    height = {f: 0 for f in facts}
    spine_of: dict[int, list] = {}       # skeleton root -> spine atoms, bottom-up

    def side(hmax):
        h = rng.randint(0, hmax)
        if h == 0:
            return rng.choice(facts)
        a = new(); p = side(h - 1); add([p], a, "sol"); height[a] = h; sol_pre[a] = [p]
        return a

    def build(level, spine):
        if level == 0:
            return rng.choice(facts)
        below = build(level - 1, spine)
        a = new(); height[a] = level
        pre = [below]
        if shape == "tree" and rng.random() < 0.5:
            pre.append(side(min(2, level - 1)))
        if arity3 and rng.random() < 0.6:
            while len(pre) < 3:
                pre.append(side(1))
        rng.shuffle(pre)
        add(pre, a, "sol")
        sol_pre[a] = list(pre)
        if shape == "alt" and rng.random() < 0.5:
            x = below
            for _ in range(rng.randint(2, 3)):
                y = new(); add([x], y, "alt"); x = y
            add([x], a, "alt")
        spine.append(a)
        return a

    goal = []
    for k, d in enumerate([depth] + ([rng.randint(1, max(1, depth // 2))] if rng.random() < conj_p else [])):
        spine: list = []
        goal.append(build(d, spine))
        spine_of[goal[-1]] = spine
    solution_atoms = [a for a in range(n) if a not in facts]

    def dead_atom():
        return new()                      # no producer, not a fact

    def provable_below(a):
        cands = [b for b, h in height.items() if h < height.get(a, 0)]
        return rng.choice(cands) if cands else rng.choice(facts)

    for root, spine in spine_of.items():
        for i, a in enumerate(spine):
            if rng.random() > 0.65:
                continue
            for _ in range(rng.randint(1, 2)):
                t = decoy if rng.random() < 0.75 else "dead"
                if t == "cycle" and i == len(spine) - 1:
                    t = "dead"
                if t == "dead":
                    pre = [dead_atom()] + ([provable_below(a)] if rng.random() < 0.5 else [])
                elif t == "deep":
                    x = dead_atom()
                    for _ in range(rng.randint(*(deep_len or (2, 4)))):
                        y = new(); add([x], y, "decoy"); x = y
                    pre = [x]
                else:  # cycle: premise c is derivable only from an ancestor b of a
                    b = rng.choice(spine[i + 1:])
                    c = new(); add([b], c, "decoy")
                    extra_n = (1 if rng.random() < 0.5 else 0) if cycle_extra is None else rng.randint(*cycle_extra)
                    for _ in range(extra_n):  # lengthen the loop
                        c2 = new(); add([c], c2, "decoy"); c = c2
                    pre = [c]
                rng.shuffle(pre)
                add(pre, a, "decoy")
    if mirror_p:
        def mirror(p, memo):
            if p in facts:
                return dead_atom()
            if p not in memo:
                m = new(); memo[p] = m
                add([mirror(x, memo) for x in sol_pre[p]], m, "decoy")
                if rng.random() < 0.65:             # decorate like a spine atom
                    for _ in range(rng.randint(1, 2)):
                        add([dead_atom()], m, "decoy")
            return memo[p]
        for root, spine in spine_of.items():
            for a in spine[1:]:
                if rng.random() < mirror_p:
                    add([mirror(x, {}) for x in sol_pre[a]], a, "decoy")
    # Distractors: rules among extra atoms and facts, never producing a solution atom.
    extra = [new() for _ in range(rng.randint(2, 2 + depth // 2))]
    pool = facts + extra
    for e in extra:
        if rng.random() < 0.7:
            pre = rng.sample([p for p in pool if p != e], k=min(len(pool) - 1, rng.choice((1, 1, 2))))
            add(pre, e, "distractor")
    for _ in range(rng.randint(1, 3)):
        head = rng.choice(extra)
        add([rng.choice(pool)], head, "distractor")
    if not arity3 and any(len(pre) > 2 for pre, _ in rules):
        raise AssertionError("arity bound violated")

    # Alpha-rename atoms and shuffle hypothesis order; names carry no information.
    perm = list(range(n)); rng.shuffle(perm)
    order = list(range(len(rules))); rng.shuffle(order)
    forder = list(range(len(facts))); rng.shuffle(forder)
    rn = lambda a: perm[a]
    thm = Theorem(
        tid=tid or f"{family}_d{depth}_s{seed}{'_a3' if arity3 else ''}",
        family=family, depth=depth, seed=seed, n_atoms=n,
        facts=[rn(facts[i]) for i in forder],
        rules=[[[rn(p) for p in rules[i][0]], rn(rules[i][1])] for i in order],
        goal=[rn(g) for g in goal], arity3=arity3,
        gen={"rule_kind": [kind[i] for i in order], "conj": len(goal) > 1,
             "deep_len": list(deep_len) if deep_len else None,
             "cycle_extra": list(cycle_extra) if cycle_extra else None,
             "mirror_p": mirror_p,
             "solution_atoms": sorted(rn(a) for a in solution_atoms)},
    )
    size = reference_sizes(thm)
    if any(size[g] >= INF for g in thm.goal):
        raise AssertionError("goal not provable")
    return thm


# --- rendering ---------------------------------------------------------------

def binders(t: Theorem) -> str:
    out = ["(" + " ".join(f"P{i}" for i in range(t.n_atoms)) + " : Prop)"]
    out += [f"(h{i} : P{a})" for i, a in enumerate(t.facts)]
    out += [f"(r{i} : " + " → ".join(f"P{a}" for a in pre + [head]) + ")"
            for i, (pre, head) in enumerate(t.rules)]
    return " ".join(out)


def target(t: Theorem) -> str:
    return " ∧ ".join(f"P{g}" for g in t.goal)


def statement(t: Theorem) -> str:
    """Closed proposition for the REPL; `introNP` recovers the binder names."""
    return "∀ " + binders(t) + ", " + target(t)


def stmt_sha(t: Theorem) -> str:
    return hashlib.sha256(statement(t).encode()).hexdigest()


# --- reference semantics (cross-check only) ---------------------------------

def reference_legal(t: Theorem, atom: int) -> set[str]:
    legal = {f"apply h{i}" for i, a in enumerate(t.facts) if a == atom}
    legal |= {f"apply r{i}" for i, (_, h) in enumerate(t.rules) if h == atom}
    return legal


def reference_sizes(t: Theorem) -> dict[int, int]:
    """Minimum number of tactics proving each atom (Bellman-Ford; INF = unprovable)."""
    size = {a: INF for a in range(t.n_atoms)}
    for a in t.facts:
        size[a] = 1
    changed = True
    while changed:
        changed = False
        for pre, h in t.rules:
            s = 1 + sum(size[p] for p in pre)
            if s < size[h]:
                size[h] = s; changed = True
    return size


def canonical_hash(t: Theorem) -> str:
    """WL hash of the hypergraph: invariant to atom names, rule order and fact order."""
    atoms = range(t.n_atoms)
    facts = set(t.facts)
    goal = {g: i for i, g in enumerate(t.goal)}
    color = {a: f"{a in facts}|{goal.get(a, -1)}|{t.facts.count(a)}" for a in atoms}
    for _ in range(t.n_atoms):
        new = {}
        for a in atoms:
            prod = sorted("|".join(sorted(color[x] for x in pre)) for pre, h in t.rules if h == a)
            cons = sorted(color[h] + "<" + "|".join(sorted(color[x] for x in pre))
                          for pre, h in t.rules if a in pre)
            new[a] = hashlib.sha1(json.dumps([color[a], prod, cons]).encode()).hexdigest()[:20]
        if len(set(new.values())) == len(set(color.values())):
            color = new
            break
        color = new
    return hashlib.sha256(json.dumps(sorted(color.values())).encode()).hexdigest()
