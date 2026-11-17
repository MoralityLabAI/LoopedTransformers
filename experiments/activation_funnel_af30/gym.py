"""AF-30: source-only paired Horn grammars with shifted middle swap locations.

Policy-facing code must consume public_features + physical current state only.
Case metadata and the exact host truth/distance are scorer-only. No Torch.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import random

ATOMS, RULES, GOAL, STOP = 15, 14, 14, 14
ROOTS = frozenset((0, 1, 2, 3))
MARKED = (6, 13, 14)
SPLITS = ("train", "dev", "held_depth5", "held_depth6")
FAMILIES = {
    "train": ("train_left_d5", "train_right_d6"),
    "dev": ("dev_cross_d5", "dev_cross_d6"),
    "held_depth5": ("held5_fork_a", "held5_fork_b"),
    "held_depth6": ("held6_fork_a", "held6_fork_b"),
}
# First productive rule: two known facts -> atom 4.
ROOT_PAIR = {
    "train_left_d5": (0, 1), "train_right_d6": (0, 2),
    "dev_cross_d5": (0, 2), "dev_cross_d6": (1, 2),
    "held5_fork_a": (1, 3), "held5_fork_b": (2, 3),
    "held6_fork_a": (2, 3), "held6_fork_b": (0, 1),
}
LAYOUT_START = {"train": 810, "dev": 820,
                "held_depth5": 830, "held_depth6": 840}
PAIRS_PER_FAMILY = 48
POSITIONS = {
    "train": (1, 2), "dev": (1, 2),
    "held_depth5": (1, 2, 3), "held_depth6": (1, 2, 3),
}


def digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    split: str
    family: str
    pair_index: int
    reachable: bool
    layout: int

    @property
    def depth(self):
        return 6 if self.family.endswith("d6") or self.split == "held_depth6" else 5

    @property
    def swap_position(self):
        return POSITIONS[self.split][self.pair_index % len(POSITIONS[self.split])]

    @property
    def pair_id(self):
        return f"af30-{self.split}-{self.family}-p{self.pair_index:02d}-l{self.layout}"

    @property
    def id(self):
        return f"{self.pair_id}-y{int(self.reachable)}"


@lru_cache(maxsize=4)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("unknown AF30 split")
    base = LAYOUT_START[split]
    return tuple(
        Case(split, family, i, bool(label), base + 2*f_idx + i % 2)
        for f_idx, family in enumerate(FAMILIES[split])
        for i in range(PAIRS_PER_FAMILY)
        for label in (1, 0)
    )


def mates(case):
    return (Case(case.split, case.family, case.pair_index, True, case.layout),
            Case(case.split, case.family, case.pair_index, False, case.layout))


def initial_facts(case):
    return ROOTS


@lru_cache(maxsize=1024)
def logical_rules(case):
    if case.split not in SPLITS or case.family not in FAMILIES[case.split]:
        raise ValueError("invalid AF30 source family")
    d = case.depth
    # All three registered change sites are unary and at distance >= 3
    # from any root or marked-atom anchor. Atom 10 has no producer.
    chain = [
        (ROOT_PAIR[case.family], 4),
        ((4,), 5),
        ((5,), 9),
        ((9,), 11),
    ]
    if d == 5:
        chain.append(((11,), GOAL))
    else:
        chain.extend([((11,), 6), ((6,), GOAL)])
    pos = case.swap_position
    src = chain[pos][0][0]
    if src not in (4, 5, 9):
        raise AssertionError("unregistered unary premise swap")
    chain[pos] = (((src,) if case.reachable else (10,)), chain[pos][1])
    decoy_source = 10 if case.reachable else src
    decoys = [
        ((decoy_source,), 7),
        ((7, 10), 13),     # Short but blocked apparent goal corridor
        ((13,), GOAL),
        ((1, 2), 7),       # Attractive, root-enabled distraction
        ((0, 3), 12),      # Independent root-enabled distraction
        ((7,), 8),
        ((2,), 8),
    ]
    if d == 5:
        decoys.append(((10,), 6))  # Preserve marked 6 without a proof path
    rules = tuple((tuple(p), h) for p, h in chain + decoys)
    if len(rules) > RULES or len(chain) != d:
        raise AssertionError("AF30 physical budget exceeded")
    if sum(h == GOAL for p, h in rules) != 2:
        raise AssertionError("registered two GOAL conclusions changed")
    if any(not p or len(p) > 2 or h in p or h >= ATOMS or
           any(a < 0 or a >= ATOMS for a in p) for p, h in rules):
        raise AssertionError("malformed source Horn rule")
    return rules


@lru_cache(maxsize=1024)
def physical_maps(case):
    rng = random.Random(int(digest([
        "AF30_PHYSICAL_PAIR", case.split, case.family,
        case.pair_index, case.layout
    ])[:16], 16))
    atoms, rules = list(range(ATOMS)), list(range(RULES))
    rng.shuffle(atoms)
    rng.shuffle(rules)
    return tuple(atoms), tuple(rules)


@lru_cache(maxsize=1024)
def physical_rules(case):
    _, slots = physical_maps(case)
    return {slots[i]: (p, h) for i, (p, h) in enumerate(logical_rules(case))}


def to_physical(case, known):
    atoms, _ = physical_maps(case)
    return frozenset(atoms[a] for a in known)


def enabled(case, known):
    known = frozenset(known)
    return tuple(sorted(
        slot for slot, (prem, head) in physical_rules(case).items()
        if head not in known and all(p in known for p in prem)
    ))


def sound_fire(case, known, action):
    known = frozenset(known)
    if action == STOP:
        return known, None
    if action not in enabled(case, known):
        raise ValueError("host refused illegal rule")
    prem, head = physical_rules(case)[action]
    return known | {head}, {"slot": action, "premises": list(prem), "head": head}


def shortest_steps(case, known=None):
    start = frozenset(initial_facts(case) if known is None else known)
    if GOAL in start:
        return 0
    seen, q = {start}, deque([(start, 0)])
    while q:
        state, n = q.popleft()
        for slot in enabled(case, state):
            nxt, _ = sound_fire(case, state, slot)
            if GOAL in nxt:
                return n+1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, n+1))
    return None


def public_features(case):
    atoms, _ = physical_maps(case)
    return {
        "atoms": ATOMS, "rules": RULES,
        "initial_physical": sorted(to_physical(case, ROOTS)),
        "goal_slot": atoms[GOAL],
        "subgoal_slots": [atoms[6], atoms[13]],
        "rules_physical": [
            [slot, sorted(atoms[p] for p in prem), atoms[head]]
            for slot, (prem, head) in sorted(physical_rules(case).items())
        ],
    }


def public_bitset_bfs(public):
    """Independent host: physical IDs and bitset arithmetic only."""
    rules = tuple((sum(1 << p for p in prem), 1 << h)
                  for _, prem, h in public["rules_physical"])
    start = sum(1 << a for a in public["initial_physical"])
    goal = 1 << public["goal_slot"]
    if start & goal:
        return 0
    seen, q = {start}, deque([(start, 0)])
    while q:
        bits, d = q.popleft()
        for prem, head in rules:
            if prem & bits != prem or head & bits:
                continue
            nxt = bits | head
            if nxt & goal:
                return d+1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d+1))
    return None


def public_fixed_point(public, state=None):
    state = set(public["initial_physical"] if state is None else state)
    rules = [(set(p), h) for _, p, h in public["rules_physical"]]
    passes, checks = 0, 0
    while True:
        passes += 1
        future = set(state)
        for prem, head in rules:
            checks += 1
            if prem <= state:
                future.add(head)
        if future == state:
            return frozenset(state), {"closure_passes": passes,
                                       "rule_examinations": checks}
        state = future


def incidence(case):
    public = public_features(case)
    edges, adj = [], defaultdict(set)
    for slot, prem, head in public["rules_physical"]:
        r = ("r", slot)
        for p in prem:
            a = ("a", p)
            edges.append((a, r, "premise"))
            adj[a].add(r)
            adj[r].add(a)
        a = ("a", head)
        edges.append((r, a, "head"))
        adj[r].add(a)
        adj[a].add(r)
    return edges, adj


def degrees(case):
    edges, _ = incidence(case)
    incoming, outgoing = defaultdict(int), defaultdict(int)
    for src, dest, _ in edges:
        outgoing[src] += 1
        incoming[dest] += 1
    return tuple(
        (kind, n, incoming[(kind, n)], outgoing[(kind, n)])
        for kind, size in (("a", ATOMS), ("r", RULES))
        for n in range(size)
    )


def radius2_signature(case):
    edges, adj = incidence(case)
    perm, _ = physical_maps(case)
    anchors = [("a", perm[a]) for a in (*sorted(ROOTS), *MARKED)]
    views = []
    for anchor in anchors:
        dist = {anchor: 0}
        q = deque([anchor])
        while q:
            node = q.popleft()
            if dist[node] >= 2:
                continue
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        visible = set(dist)
        views.append((
            anchor, tuple(sorted(visible)),
            tuple(sorted(e for e in edges if e[0] in visible and e[1] in visible))
        ))
    return tuple(views)


def pair_certificate(positive, negative):
    if not positive.reachable or negative.reachable or positive.pair_id != negative.pair_id:
        raise ValueError("source matched-pair labels/identity invalid")
    if physical_maps(positive) != physical_maps(negative):
        raise AssertionError("pair physical permutations diverge")
    if degrees(positive) != degrees(negative):
        raise AssertionError("physical degree profile changed")
    if radius2_signature(positive) != radius2_signature(negative):
        raise AssertionError("radius-two anchored neighborhood leaks label")
    if enabled(positive, ROOTS) != enabled(negative, ROOTS):
        raise AssertionError("root legal mask differs")
    ap, an = public_features(positive), public_features(negative)
    for key in ("initial_physical", "goal_slot", "subgoal_slots"):
        if ap[key] != an[key]:
            raise AssertionError("source public metadata changed")
    rp, rn = physical_rules(positive), physical_rules(negative)
    changed = [slot for slot in rp if rp[slot] != rn[slot]]
    if len(changed) != 2 or set(rp) != set(rn):
        raise AssertionError("pair must change exactly two rule premises")
    for slot in changed:
        p1, head1 = rp[slot]
        p2, head2 = rn[slot]
        if head1 != head2 or len(p1) != len(p2) or len(p1) != 1:
            raise AssertionError("pair changed a head or arity")
    sp, sn = shortest_steps(positive), shortest_steps(negative)
    bp, bn = public_bitset_bfs(ap), public_bitset_bfs(an)
    if (sp, sn, bp, bn) != (positive.depth, None, positive.depth, None):
        raise AssertionError(f"source/physical proof oracle mismatch: {sp,sn,bp,bn}")
    if len(enabled(positive, ROOTS)) < 2:
        raise AssertionError("source lacks root choice")
    return {
        "pair_id": positive.pair_id,
        "swap_position": positive.swap_position,
        "positive_shortest": sp, "negative_shortest": sn,
        "local_sig": digest(radius2_signature(positive)),
        "degree_sig": digest(degrees(positive)),
        "positive_public_sha256": digest(ap),
        "negative_public_sha256": digest(an),
        "changed_slots": sorted(changed),
        "root_legal_count": len(enabled(positive, ROOTS)),
    }


@lru_cache(maxsize=1)
def manifest():
    old = set()
    from experiments.activation_funnel_af29 import gym as af29
    for split in af29.SPLITS:
        for c in af29.cases_for(split):
            old.add(digest(af29.public_features(c)))
    used, template_signatures, split_info = set(), set(), {}
    for split in SPLITS:
        roster = cases_for(split)
        if len(roster) != 192 or sum(c.reachable for c in roster) != 96:
            raise AssertionError("AF30 source split count changed")
        records = []
        for family in FAMILIES[split]:
            pos = [c for c in roster if c.family == family and c.reachable]
            if len(pos) != PAIRS_PER_FAMILY:
                raise AssertionError("source family pair count changed")
            # Template signatures before permutation, per swap position.
            for slot in sorted({c.swap_position for c in pos}):
                representative = next(c for c in pos if c.swap_position == slot)
                template = digest(logical_rules(representative))
                if template in template_signatures:
                    raise AssertionError("source grammar template reused across families/splits")
                template_signatures.add(template)
            for c in pos:
                p, n = mates(c)
                rec = pair_certificate(p, n)
                if (rec["positive_public_sha256"] in used or
                    rec["negative_public_sha256"] in used or
                    rec["positive_public_sha256"] in old or
                    rec["negative_public_sha256"] in old):
                    raise AssertionError("AF30 duplicate/historical public graph")
                used.update((rec["positive_public_sha256"],
                             rec["negative_public_sha256"]))
                records.append(rec)
        swaps = {str(i): sum(r["swap_position"] == i for r in records)
                 for i in POSITIONS[split]}
        expected = {"1": 48, "2": 48} if split in ("train", "dev") else {
            "1": 32, "2": 32, "3": 32
        }
        if swaps != expected or len(records) != 96:
            raise AssertionError(f"unbalanced AF30 swap-position shift: {swaps}")
        split_info[split] = {
            "pairs": len(records), "cases": len(roster),
            "swap_positions": swaps,
            "mean_root_legal_count": sum(r["root_legal_count"] for r in records)/len(records),
            "pair_manifest_sha256": digest(records),
            "positive_graph_sha256": digest([r["positive_public_sha256"] for r in records]),
            "negative_graph_sha256": digest([r["negative_public_sha256"] for r in records]),
        }
    receipt = {
        "experiment": "AF-30", "source_only": True,
        "opened_original_checkpoint": False,
        "all_pairs_identical_radius2_degrees_and_root_masks": True,
        "all_pair_labels_independently_certified": True,
        "all_public_graphs_unique_and_historically_disjoint": True,
        "splits": split_info,
        "total_pairs": 384, "total_cases": 768,
    }
    receipt["manifest_sha256"] = digest(receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
