"""LBF-19 scaled synthetic domain: lengths 16/20/24 with hardness calibrated by classical search work.

Rule systems follow the pinned LSA-02K distribution exactly (8 length-2 and 4 length-3 equations over
`abcdef`, all sides distinct); only word length and walk length change.  The pinned `Case.validate()`
caps words at 14, so `ScaledCase` is a duck-typed replacement that supports the `rg.neighbors`,
`rg.replay`, `rg._result` and `FringeState` interfaces used by the LBF-15..18 controllers.

Hardness is *measured on the classical comparator only* (uncapped layer-synchronous bidirectional BFS,
the pinned `rg.bidir` semantics): a root is admissible when that search needs between `MIN_EXP` and
`MAX_EXP` expansions.  No learned or abstract arm is consulted during selection.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import random
import time
from collections import deque

ALPHABET = "abcdef"
MIN_EXP = 10_000
MAX_EXP = 1_000_000
DEPTH_CAP = 24


@dataclass(frozen=True)
class ScaledCase:
    root: str
    lineage: str
    split: str
    family: str
    source: str
    target: str
    rules: tuple[tuple[str, str], ...]
    walk_steps: int = 0

    def public(self) -> dict:
        d = asdict(self)
        d["rules"] = [list(x) for x in self.rules]
        return d

    @classmethod
    def parse(cls, o: dict) -> "ScaledCase":
        return cls(o["root"], o["lineage"], o["split"], o["family"], o["source"], o["target"],
                   tuple(tuple(x) for x in o["rules"]), int(o.get("walk_steps", 0)))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


_PERMS = [str.maketrans(ALPHABET, "".join(p)) for p in itertools.permutations(ALPHABET)]


def lineage(rules) -> str:
    """Canonical rule-system identity over alphabet permutations, rule order and orientation."""
    best = min(tuple(sorted(tuple(sorted((a.translate(t), b.translate(t)))) for a, b in rules)) for t in _PERMS)
    return digest(best)


def compile_moves(rules):
    """Oriented (rule, direction, left, right) in the pinned canonical order (rule-major, direction)."""
    out = []
    for i, (a, b) in enumerate(rules):
        out.append((i, 0, a, b))
        out.append((i, 1, b, a))
    return out


def fast_neighbors(moves, word: str):
    """Yield (rule, direction, position, successor) in pinned `rg.neighbors` sorted order."""
    res = []
    n = len(word)
    for i, d, left, right in moves:
        ll = len(left)
        pos = word.find(left)
        while pos != -1:
            res.append((i, d, pos, word[:pos] + right + word[pos + ll:]))
            pos = word.find(left, pos + 1)
    return res


def _fast_sorted(moves, word):
    # Generation is rule-major, direction-major, then ascending position for each (rule, direction): identical
    # to the pinned `sorted(Move)` order because the key is (rule, direction, position, successor).
    return fast_neighbors(moves, word)


def classical_bidir(case, *, budget: int = 10 ** 12, depth: int = DEPTH_CAP, max_expansions: int | None = None,
                    on_discover=None) -> dict:
    """Layer-synchronous smaller-frontier bidirectional BFS with pinned `rg.bidir` accounting.

    One charged work unit per inspected successor (repeats included).  Returns a pinned-style result
    plus `expanded`, `path_len` and a Move-list path.  If `max_expansions` is hit the run is censored.
    """
    moves = compile_moves(case.rules)
    src, tgt = case.source, case.target
    fronts = [deque([src]), deque([tgt])]
    parent = [{src: None}, {tgt: None}]
    dist = [{src: 0}, {tgt: 0}]
    work = expanded = 0
    censored = False
    meet = None
    while fronts[0] and fronts[1] and work < budget and meet is None:
        side = 0 if len(fronts[0]) <= len(fronts[1]) else 1
        layer = fronts[side]
        nxt_layer = deque()
        while layer and work < budget:
            w = layer.popleft()
            g = dist[side][w]
            if g >= depth:
                continue
            expanded += 1
            if max_expansions is not None and expanded > max_expansions:
                censored = True
                break
            for (ri, di, pos, nxt) in fast_neighbors(moves, w):
                if work >= budget:
                    break
                work += 1
                if nxt in dist[side]:
                    continue
                dist[side][nxt] = g + 1
                parent[side][nxt] = (w, (ri, di, pos))
                nxt_layer.append(nxt)
                if on_discover is not None:
                    on_discover(side, w, (ri, di, pos), nxt)
                if nxt in dist[1 - side] and g + 1 + dist[1 - side][nxt] <= depth:
                    meet = nxt
                    break
            if meet is not None or censored:
                break
        if censored:
            break
        fronts[side] = nxt_layer
    path = None
    if meet is not None:
        path = _reconstruct(case, moves, parent, meet)
    return {"closed": path is not None, "work": work, "expanded": expanded, "censored": censored,
            "path": path, "visited": len(dist[0]) + len(dist[1]),
            "path_len": None if path is None else len(path)}


def _reconstruct(case, moves, parent, meet):
    def chain(par, w):
        out = []
        while par[w] is not None:
            prev, mv = par[w]
            out.append((prev, mv))
            w = prev
        return list(reversed(out))

    left = chain(parent[0], meet)
    path = []
    cur = case.source
    for prev, (ri, di, pos) in left:
        a, b = case.rules[ri]
        l, r = (a, b) if di == 0 else (b, a)
        assert cur == prev and cur[pos:pos + len(l)] == l
        cur = cur[:pos] + r + cur[pos + len(l):]
        path.append({"rule": ri, "direction": di, "position": pos, "successor": cur})
    back = chain(parent[1], meet)
    # back chain: from target backwards; entries (prev_word, move) where move applied prev_word -> word_closer_to_meet
    cur = meet
    for prev, (ri, di, pos) in reversed(back):
        # forward-from-meet move is inverse: replace the rewritten right side by left side
        a, b = case.rules[ri]
        l, r = (a, b) if di == 0 else (b, a)
        inv_dir = 1 - di
        assert cur[pos:pos + len(r)] == r
        cur = cur[:pos] + l + cur[pos + len(r):]
        assert cur == prev
        path.append({"rule": ri, "direction": inv_dir, "position": pos, "successor": cur})
    assert cur == case.target
    return path


def replay_path(case, path) -> list[str]:
    word = case.source
    seen = [word]
    for m in path:
        a, b = case.rules[m["rule"]]
        l, r = (a, b) if m["direction"] == 0 else (b, a)
        p = m["position"]
        assert word[p:p + len(l)] == l, "rewrite does not match"
        word = word[:p] + r + word[p + len(l):]
        assert word == m["successor"]
        seen.append(word)
    assert word == case.target
    return seen


def random_rules(rng: random.Random):
    pairs = rng.sample(["".join(x) for x in itertools.product(ALPHABET, repeat=2)], 16)
    triples = rng.sample(["".join(x) for x in itertools.product(ALPHABET, repeat=3)], 8)
    rules = list(zip(pairs[::2], pairs[1::2])) + list(zip(triples[::2], triples[1::2]))
    rng.shuffle(rules)
    return tuple(rules)


def propose(rng: random.Random, L: int, steps: int, split: str, idx: int):
    """Random-walk generator (no policy outcome used).  Returns ScaledCase or None."""
    rules = random_rules(rng)
    moves = compile_moves(rules)
    source = "".join(rng.choice(ALPHABET) for _ in range(L))
    word, visited, path = source, {source}, []
    for _ in range(steps):
        menu = [m for m in fast_neighbors(moves, word) if m[3] not in visited]
        if not menu:
            return None
        m = rng.choice(menu)
        path.append(m)
        word = m[3]
        visited.add(word)
    hd = sum(a != b for a, b in zip(source, word))
    if hd < 4:
        return None
    ident = lineage(rules)
    return ScaledCase(f"{split}_L{L}_{idx:03d}_{ident[:12]}", ident, split, "equality", source, word, rules, steps)


def generate_scaled(seed: int, L: int, n: int, split: str, steps_range: tuple[int, int],
                    exclude_lineages: set[str] | None = None, max_attempts: int = 4000,
                    max_expansions: int = MAX_EXP, min_expansions: int = MIN_EXP,
                    start_idx: int = 0, log=None):
    """Rejection-sample `n` roots whose classical uncapped bidir work lies in [min, max] expansions."""
    rng = random.Random(seed)
    seen = set(exclude_lineages or ())
    out = []
    attempts = 0
    stats = {"attempts": 0, "too_easy": 0, "too_hard": 0, "no_walk": 0, "dup_lineage": 0, "accepted": 0}
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"generator exhausted attempts for L={L} (accepted {len(out)}/{n}) {stats}")
        stats["attempts"] += 1
        steps = rng.randint(*steps_range)
        c = propose(rng, L, steps, split, start_idx + len(out))
        if c is None:
            stats["no_walk"] += 1
            continue
        if c.lineage in seen:
            stats["dup_lineage"] += 1
            continue
        r = classical_bidir(c, max_expansions=max_expansions)
        if r["censored"] or not r["closed"]:
            stats["too_hard"] += 1
            continue
        if r["expanded"] < min_expansions:
            stats["too_easy"] += 1
            continue
        seen.add(c.lineage)
        out.append((c, {"classical_expanded": r["expanded"], "classical_work": r["work"],
                        "classical_path_len": r["path_len"], "classical_visited": r["visited"]}))
        stats["accepted"] += 1
        if log:
            log(f"L={L} accepted {len(out)}/{n} exp={r['expanded']} path={r['path_len']} steps={steps}")
    return out, stats
