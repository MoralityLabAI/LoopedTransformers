"""Reference audit for LBF-17's second-pass provenance suppression.

This module deliberately uses Boolean sets and explicit short-word
concretization. It is an oracle for a subsequent optimized implementation,
not an inference benchmark. Historical LBF-17 code/checkpoints are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

Rules = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Move:
    pos: int
    lhs: str
    rhs: str
    successor: str


@lru_cache(maxsize=512)
def oriented(rules: Rules) -> tuple[tuple[str, str], ...]:
    if any(len(a) not in (2, 3) or len(a) != len(b) or a == b for a, b in rules):
        raise ValueError("expected nonidentity length-preserving 2/3-symbol rules")
    return tuple(sorted(set(rules) | {(b, a) for a, b in rules}))


def moves(word: str, rules: Rules) -> Iterable[Move]:
    for lhs, rhs in oriented(rules):
        for p in range(len(word) - len(lhs) + 1):
            if word[p:p+len(lhs)] == lhs:
                yield Move(p, lhs, rhs, word[:p] + rhs + word[p+len(lhs):])


def change_mask(root: str, word: str) -> int:
    if len(root) != len(word):
        raise ValueError("length mismatch")
    return sum(1 << p for p, (a, b) in enumerate(zip(root, word)) if a != b)


@lru_cache(maxsize=32768)
def first_hop_masks(root: str, rules: Rules, fibers: int = 4) -> tuple[int, ...]:
    """Exact union of changed coordinates of all first-hop words in each fiber."""
    if fibers < 1:
        raise ValueError("positive fiber count required")
    masks = [0] * fibers
    for m in moves(root, rules):
        masks[m.pos % fibers] |= change_mask(root, m.successor)
    return tuple(masks)


def legacy_suppresses(root: str, pos: int, rhs: str, fiber: int, fibers: int = 4) -> bool:
    """LBF-17 symbolic/PyTorch pass-index-1 predicate, for counterexample only."""
    return pos % fibers == fiber and root[pos:pos+len(rhs)] == rhs


def safe_suppresses(root: str, pos: int, rhs: str, fiber: int,
                    masks: tuple[int, ...]) -> bool:
    """Only apply on the SECOND pass, for an uncorrupted first-hop fiber.

    Every concrete member of that fiber differs from root at a subset of
    masks[fiber]. If the candidate window contains that mask and writes
    root's own substring, the concrete result, whenever the rule matches,
    is root. The identity fiber already contains it.
    """
    if not 0 <= fiber < len(masks) or not 0 <= pos <= len(root)-len(rhs):
        raise ValueError("bad fiber/window")
    window = ((1 << len(rhs)) - 1) << pos
    return (legacy_suppresses(root, pos, rhs, fiber, len(masks))
            and (masks[fiber] & ~window) == 0)


def exact_ball(root: str, rules: Rules, horizon: int) -> frozenset[str]:
    if horizon < 0:
        raise ValueError("negative horizon")
    seen = {root}
    for _ in range(horizon):
        seen |= {m.successor for w in seen for m in moves(w, rules)}
    return frozenset(seen)


def alpha2(words: Iterable[str], length: int) -> tuple[frozenset[str], ...]:
    es = [set() for _ in range(length-1)]
    for w in words:
        if len(w) != length:
            raise ValueError("length mismatch")
        for p in range(length-1):
            es[p].add(w[p:p+2])
    return tuple(frozenset(e) for e in es)


def gamma2(edges: tuple[frozenset[str], ...], cap: int = 20000) -> frozenset[str]:
    """Exact adjacent-edge trellis concretization, bounded for audit fixtures."""
    if not edges:
        raise ValueError("length must be at least two")
    words = set(edges[0])
    for edge in edges[1:]:
        words = {w+e[1] for w in words for e in edge if w[-1] == e[0]}
        if len(words) > cap:
            raise ValueError("reference concretization cap exceeded")
    return frozenset(words)


def fiber_ball(root: str, rules: Rules, horizon: int, *, fibers: int = 4,
               mode: str = "safe", abstract: bool = True) -> frozenset[str]:
    """Reference interpreter. Use exact words or best adjacent-2 abstraction.

    All modes retain root in an identity fiber. 'safe' suppresses only on
    pass two with the exact changed-coordinate guard; later passes do not
    suppress. 'legacy' reenacts the suspect pass-two predicate. 'none'
    disables suppression. The abstract version concretizes each bigram
    fiber before rewriting; it is intentionally not a fast implementation.
    """
    if mode not in ("safe", "legacy", "none") or horizon < 0:
        raise ValueError("bad mode/horizon")
    if horizon == 0:
        return frozenset((root,))
    masks = first_hop_masks(root, rules, fibers)
    states = [set() for _ in range(fibers)]
    for m in moves(root, rules):
        states[m.pos % fibers].add(m.successor)

    def reduce(words):
        return set(gamma2(alpha2(words, len(root)))) if abstract else set(words)

    states = [reduce(ws) for ws in states]
    for k in range(2, horizon+1):
        after = []
        for f, ws in enumerate(states):
            dst = set(ws)
            for word in ws:
                for m in moves(word, rules):
                    suppress = False
                    if k == 2 and mode == "legacy":
                        suppress = legacy_suppresses(root, m.pos, m.rhs, f, fibers)
                    elif k == 2 and mode == "safe":
                        suppress = safe_suppresses(root, m.pos, m.rhs, f, masks)
                    if not suppress:
                        dst.add(m.successor)
            after.append(reduce(dst))
        states = after
    return frozenset({root}.union(*states))


REFEREE_ROOT = "abc"
REFEREE_RULES = (("abc", "def"), ("ab", "de"), ("bf", "cc"))


if __name__ == "__main__":
    for k in range(4):
        print(k, "exact", sorted(exact_ball(REFEREE_ROOT, REFEREE_RULES, k)),
              "legacy", sorted(fiber_ball(REFEREE_ROOT, REFEREE_RULES, k, mode="legacy")),
              "safe", sorted(fiber_ball(REFEREE_ROOT, REFEREE_RULES, k)))
