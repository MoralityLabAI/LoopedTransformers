"""AF-24B source-only new Horn grammars, frozen before neural scoring.

No PyTorch needed until encode()/observations(). The three goal-marked
hypergraph families differ in goal-rule arity and root dependency topology
from original AF-23. They use all the same 15-atom/14-slot GraphBatch fields.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json
import random

ATOMS, RULES, GOAL, SUBGOALS = 15, 14, 14, (6, 13)
STOP = 14
SEEDS = (17, 29, 43, 61, 73, 89, 103, 131)
DEPTHS = (5, 6)
SPLITS = ("held_depth5", "held_depth6")
FAMILIES = ("shared_overcount", "or_goal_competition", "crossed_conjunction")
POSITIVE_COUNT = {"shared_overcount": 24, "or_goal_competition": 24,
                  "crossed_conjunction": 32}


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    split: str
    depth: int
    family: str
    variant: int
    layout: int

    @property
    def id(self):
        return f"af24b-{self.split}-{self.family}-d{self.depth}-v{self.variant:02d}-p{self.layout}"

    @property
    def designed_positive(self):
        return self.variant < POSITIVE_COUNT[self.family]


@lru_cache(maxsize=None)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("undeclared AF24B split")
    depth = 5 if split == "held_depth5" else 6
    base = 300 if depth == 5 else 310
    return tuple(Case(split, depth, family, variant,
                      base + 2 * index + variant % 2)
                 for index, family in enumerate(FAMILIES)
                 for variant in range(64))


@lru_cache(maxsize=512)
def initial_facts(c):
    if c.family == "shared_overcount":
        roots = frozenset((0, 1))
        if c.designed_positive:
            return roots
        return ({0: frozenset((0,)), 1: frozenset((1,)),
                 2: frozenset()}[(c.variant - POSITIVE_COUNT[c.family]) % 3])
    roots = frozenset((0, 1, 2))
    if c.designed_positive:
        return roots
    # Both graph alternatives explicitly require fact 0 to start.
    return ({0: frozenset((1, 2)), 1: frozenset((1,)),
             2: frozenset((2,)), 3: frozenset()}[
                 (c.variant - POSITIVE_COUNT[c.family]) % 4
             ])


@lru_cache(maxsize=512)
def logical_rules(c):
    """Two goal-concluding rules per case, with at most fourteen slots.

    1. Shared 2/3-firing prefix with two needed subgoals; alternative 7/8-rule
       direct path is chosen by additive closure when shared prefix is counted
       twice (tie is broken toward the alternative at depth 5).
    2. Disjunctive goal: faster subgoal6 path and slower subgoal13 path.
    3. Conjunctive goal with crossed shared prerequisites and direct alternative.
    """
    if c.family not in FAMILIES or c.depth not in DEPTHS:
        raise ValueError("undeclared AF24B grammar")
    if c.family == "shared_overcount":
        if c.depth == 5:
            rs = [
                ((0, 1), 9), ((9,), 10), ((10,), 6), ((10,), 13),
                ((6, 13), 14),
                ((0, 1), 4), ((4,), 5), ((5,), 7), ((7,), 8),
                ((8,), 11), ((11,), 12), ((12,), 14),
                ((0,), 2), ((1,), 3),
            ]
        else:
            rs = [
                ((0, 1), 9), ((9,), 10), ((10,), 11),
                ((11,), 6), ((11,), 13), ((6, 13), 14),
                ((0, 1), 4), ((4,), 5), ((5,), 7), ((7,), 8),
                ((8,), 12), ((12,), 2), ((2,), 3), ((3,), 14),
            ]
    elif c.family == "or_goal_competition":
        if c.depth == 5:
            rs = [
                ((0, 1), 9), ((9, 2), 10), ((10,), 11),
                ((11,), 6), ((6,), 14),
                ((0, 2), 4), ((4,), 5), ((5,), 7),
                ((7,), 8), ((8,), 12), ((12,), 13), ((13,), 14),
            ]
        else:
            rs = [
                ((0, 1), 9), ((9, 2), 10), ((10,), 11),
                ((11,), 12), ((12,), 6), ((6,), 14),
                ((0, 2), 4), ((4,), 5), ((5,), 7), ((7,), 8),
                ((8,), 3), ((3,), 13), ((13,), 14),
            ]
    else:
        if c.depth == 5:
            rs = [
                ((0, 1), 9), ((9,), 6), ((9, 2), 10),
                ((10,), 13), ((6, 13), 14),
                ((0, 2), 4), ((4,), 5), ((5, 1), 7),
                ((7,), 8), ((8,), 12), ((12,), 3), ((3,), 14),
            ]
        else:
            rs = [
                ((0, 1), 9), ((9, 2), 10), ((10,), 6),
                ((9,), 11), ((11,), 13), ((6, 13), 14),
                ((0, 2), 4), ((4,), 5), ((5, 1), 7),
                ((7,), 8), ((8,), 12), ((12,), 3), ((3,), 14),
            ]
    if len(rs) > RULES or sum(head == GOAL for _, head in rs) != 2:
        raise AssertionError("invalid AF24B graph topology")
    if any((head in prem or head >= ATOMS or
            not prem or len(prem) > 2 or
            any(p >= ATOMS for p in prem)) for prem, head in rs):
        raise AssertionError("invalid Horn DAG rule")
    return tuple((tuple(p), h) for p, h in rs)


@lru_cache(maxsize=512)
def physical_maps(c):
    rand = random.Random(int(digest([c.id, "physical"] )[:16], 16))
    atoms = list(range(ATOMS))
    rules = list(range(RULES))
    rand.shuffle(atoms)
    rand.shuffle(rules)
    if c.family == "shared_overcount" and c.depth == 5:
        shared = next(i for i, (prem, h) in enumerate(logical_rules(c))
                      if h == GOAL and len(prem) == 2)
        alternative = next(i for i, (prem, h) in enumerate(logical_rules(c))
                           if h == GOAL and len(prem) == 1)
        # The additive-cost witnesses tie at seven; freeze the source-blind
        # physical slot tiebreak so the long 7-action path is selected.
        if rules[alternative] > rules[shared]:
            rules[alternative], rules[shared] = rules[shared], rules[alternative]
    return tuple(atoms), tuple(rules)


@lru_cache(maxsize=512)
def physical_rules(c):
    _, rpos = physical_maps(c)
    return {rpos[i]: (prem, head)
            for i, (prem, head) in enumerate(logical_rules(c))}


def enabled(c, known):
    known = set(known)
    return tuple(sorted(slot for slot, (prem, head) in physical_rules(c).items()
                        if head not in known and set(prem) <= known))


def sound_fire(c, known, action):
    before = frozenset(known)
    if action == STOP:
        return before, None
    if action not in enabled(c, before):
        raise ValueError("illegal physical Horn rule")
    prem, head = physical_rules(c)[action]
    return before | {head}, {"rule_slot": action,
                              "premises": list(prem), "head": head}


def closure(c, known=None):
    facts = set(initial_facts(c) if known is None else known)
    while True:
        updated = facts | {head for prem, head in physical_rules(c).values()
                           if set(prem) <= facts}
        if updated == facts:
            return frozenset(facts)
        facts = updated


@lru_cache(maxsize=100000)
def shortest_steps(c, known=None):
    start = frozenset(initial_facts(c) if known is None else known)
    if GOAL in start:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        state, depth = q.popleft()
        for action in enabled(c, state):
            after, _ = sound_fire(c, state, action)
            if GOAL in after:
                return depth + 1
            if after not in seen:
                seen.add(after)
                q.append((after, depth + 1))
    return None


def independent_bfs(c, known=None):
    """Independent logical bitset oracle, free of enabled/sound_fire."""
    state = initial_facts(c) if known is None else known
    start = sum(1 << atom for atom in state)
    if start & (1 << GOAL):
        return 0
    masks = [(sum(1 << p for p in prem), 1 << head)
             for prem, head in physical_rules(c).values()]
    q = deque([(start, 0)])
    seen = {start}
    while q:
        bits, depth = q.popleft()
        for pre, head in masks:
            if pre & bits != pre or head & bits:
                continue
            after = bits | head
            if after & (1 << GOAL):
                return depth + 1
            if after not in seen:
                seen.add(after)
                q.append((after, depth + 1))
    return None


def optimal_actions(c, known):
    d = shortest_steps(c, frozenset(known))
    if d is None or d == 0:
        return (STOP,)
    return tuple(slot for slot in enabled(c, known)
                 if shortest_steps(c, sound_fire(c, known, slot)[0]) == d - 1)


def public_features(c):
    apos, _ = physical_maps(c)
    return {
        "atoms": ATOMS, "rules": RULES,
        "initial_physical": sorted(apos[i] for i in initial_facts(c)),
        "goal_slot": apos[GOAL],
        "subgoal_slots": [apos[i] for i in SUBGOALS],
        "rules_physical": [
            [slot, sorted(apos[i] for i in prem), apos[head]]
            for slot, (prem, head) in sorted(physical_rules(c).items())
        ],
    }


@lru_cache(maxsize=512)
def encode(c):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if (base.ATOMS, base.RULES) != (ATOMS, RULES):
        raise AssertionError("AF23 model graph shape changed")
    apos, _ = physical_maps(c)
    af = torch.zeros(ATOMS, 5, dtype=torch.float32)
    rf = torch.zeros(RULES, 3, dtype=torch.float32)
    pu = torch.zeros(RULES, ATOMS)
    hu = pu.clone()
    pd = pu.clone()
    hd = pu.clone()
    for atom in range(ATOMS):
        af[apos[atom]] = torch.tensor([
            float(atom in initial_facts(c)), float(atom == SUBGOALS[0]),
            float(atom == SUBGOALS[1]), float(atom == GOAL), 1.,
        ])
    for slot, (prem, head) in physical_rules(c).items():
        down = head == GOAL
        rf[slot] = torch.tensor([len(prem) / 2., float(down), 1.])
        pp, hh = (pd, hd) if down else (pu, hu)
        for atom in prem:
            pp[slot, apos[atom]] = 1.
        hh[slot, apos[head]] = 1.
    k = closure(c)
    return base.GraphBatch(
        af, rf, pu, hu, pd, hd,
        torch.tensor([apos[i] for i in (*SUBGOALS, GOAL)]),
        torch.tensor([int(i in k) for i in (*SUBGOALS, GOAL)],
                     dtype=torch.long),
    )


def observations(cases, states, with_labels=False):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if not cases or len(cases) != len(states):
        raise ValueError("mismatched AF24B observation roster")
    x = base.GraphBatch(*(
        torch.stack([getattr(encode(c), f.name) for c in cases])
        for f in fields(base.GraphBatch)
    ))
    bits = torch.zeros(len(cases), ATOMS)
    legal = torch.zeros(len(cases), RULES+1, dtype=torch.bool)
    optimal = legal.clone()
    for i, (c, state) in enumerate(zip(cases, states)):
        apos, _ = physical_maps(c)
        for atom in state:
            bits[i, apos[atom]] = 1.
        for action in enabled(c, state):
            legal[i, action] = True
        legal[i, STOP] = True
        if with_labels:
            for action in optimal_actions(c, state):
                optimal[i, action] = True
    return (x, bits, legal, optimal) if with_labels else (x, bits, legal)


def goal_ancestors(c):
    parents = {GOAL}
    for _ in range(ATOMS+1):
        expanded = parents | {
            atom for prem, head in logical_rules(c) if head in parents
            for atom in prem
        }
        if expanded == parents:
            return frozenset(parents)
        parents = expanded
    raise AssertionError("Horn ancestor closure failed to stabilize")


def symbolic_completion(c, arm):
    """Pre-neural structural gate; same first-legal/closure source algorithm."""
    from experiments.activation_funnel_af24 import symbolic
    graph = symbolic.PublicHorn.from_public_features(public_features(c))
    d = shortest_steps(c)
    state = graph.initial
    for t in range(d + 1 if d is not None else 2):
        if graph.goal in state:
            break
        move = symbolic.choose(graph, state, arm, 17, c.id, t)["action"]
        if move is None:
            break
        state = graph.fire(state, move)
    return graph.goal in state


def manifest():
    """Source-only precondition that runs before any neural checkpoint load."""
    seen = set()
    rows = {}
    total_closure = {}
    for split in SPLITS:
        cases = cases_for(split)
        if len(cases) != 192:
            raise AssertionError("wrong AF24B heldout roster")
        positives = [c for c in cases if c.designed_positive]
        if len(positives) != 80:
            raise AssertionError("wrong AF24B provable roster")
        hashes = [digest(public_features(c)) for c in cases]
        if len(set(hashes)) != len(cases) or seen.intersection(hashes):
            raise AssertionError("duplicate public Horn features")
        seen.update(hashes)
        by_family = {}
        for family in FAMILIES:
            cs = [c for c in cases if c.family == family]
            good = [c for c in cs if c.designed_positive]
            if len(cs) != 64 or len(good) != POSITIVE_COUNT[family]:
                raise AssertionError("family template count drift")
            solved_closure = 0
            root_legal_counts = Counter()
            for c in cs:
                oracle = shortest_steps(c)
                independent = independent_bfs(c)
                truthful = GOAL in closure(c)
                if oracle != independent or truthful != (oracle is not None):
                    raise AssertionError("independent BFS and closure disagree")
                if (oracle is not None) != c.designed_positive:
                    raise AssertionError("positive/negative truth roster changed")
                if not c.designed_positive:
                    continue
                if oracle != c.depth or GOAL in initial_facts(c):
                    raise AssertionError("exact positive proof length changed")
                root = initial_facts(c)
                legal = enabled(c, root)
                optimal = optimal_actions(c, root)
                ancestors = goal_ancestors(c)
                nonoptimal_relevant = sum(
                    physical_rules(c)[a][1] in ancestors and a not in optimal
                    for a in legal
                )
                if len(legal) < 2 or len(optimal)/len(legal) > .75 or nonoptimal_relevant < 1:
                    raise AssertionError("uninformative root action ambiguity")
                root_legal_counts[len(legal)] += 1
                solved_closure += symbolic_completion(c, "closure_direct")
            by_family[family] = {
                "cases":len(cs), "provable":len(good),
                "positive_root_enabled_distribution":dict(sorted(root_legal_counts.items())),
                "closure_direct_positive_solved":solved_closure,
                "goal_incoming_rules": 2,
                "goal_rule_arity_signature": sorted(
                    len(prem) for prem,h in logical_rules(good[0]) if h==GOAL
                ),
                "positive_case_ids_sha256":digest([c.id for c in good]),
            }
            if family=="shared_overcount" and solved_closure > len(good)*.5:
                raise AssertionError("shared-overcount adversary does not defeat closure")
        if sum(x["closure_direct_positive_solved"] for x in by_family.values())>60:
            raise AssertionError("pooled closure control remains too easy")
        total_closure[split] = sum(x["closure_direct_positive_solved"]
                                   for x in by_family.values())
        rows[split] = {
            "cases":len(cases),"provable":len(positives),
            "unprovable":len(cases)-len(positives),
            "public_features_sha256":digest(hashes),
            "case_ids_sha256":digest([c.id for c in cases]),
            "physical_layouts":sorted({c.layout for c in cases}),
            "family_structure":by_family,
        }
    signatures = {tuple(rows["held_depth5"]["family_structure"][fam]["goal_rule_arity_signature"])
                  for fam in FAMILIES}
    if len(signatures)<2:
        raise AssertionError("missing distinct OR versus AND goal structures")
    out = {
        "experiment":"AF-24B","source_only":True,"splits":rows,
        "symbolic_pretest_closure_solved":total_closure,
        "original_af23_goal_rule_count":1,
        "new_goal_rule_count":2,
        "model_checkpoints_opened":False,
    }
    out["manifest_sha256"] = digest(out)
    return out


if __name__=="__main__":
    print(json.dumps(manifest(),sort_keys=True,indent=2))
