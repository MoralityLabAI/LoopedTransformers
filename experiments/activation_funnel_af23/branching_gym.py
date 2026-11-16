"""AF-23 preregistered branching Horn gym (source only, no model scoring).

Finite 15-atom/14-rule positive Horn DAGs; public physical incidence and
host-known facts. The oracle is exclusively for training labels, independent
audit, and evaluation budgets, not an input to the inference policy.

The generator is intentionally harder than AF-22's nearly linear start states.
Train/dev and both held-out depths have disjoint physical layouts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import random

ATOMS = 15
RULES = 14
STOP = RULES
GOAL = 14
SUBGOALS = (6, 13)
SEEDS = (17, 29, 43, 61, 73, 89, 103, 131)
SPLITS = ("train", "dev", "held_layout", "held_depth4", "held_depth5")
LAYOUTS = {
    "train": tuple(range(170, 174)),
    "dev": (174, 175),
    "held_layout": (202,),
    "held_depth4": (203, 204),
    "held_depth5": (205, 206),
}


@dataclass(frozen=True)
class Case:
    split: str
    depth: int
    variant: int
    layout: int

    @property
    def id(self):
        return f"af23-{self.split}-d{self.depth}-v{self.variant:02d}-p{self.layout}"

    @property
    def provable_design(self):
        return self.variant < 40


def _sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=None)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("undeclared AF-23 split")
    if split == "held_layout":
        return tuple(Case(split, 3, v, 202)
                     for v in (*range(16), *range(40, 56)))
    depths = (3, 4) if split in ("train", "dev") else (
        (4,) if split == "held_depth4" else (5,)
    )
    return tuple(Case(split, d, v, layout)
                 for layout in LAYOUTS[split]
                 for d in depths
                 for v in range(96))


@lru_cache(maxsize=2048)
def initial_facts(c):
    if c.variant < 40:
        return frozenset((0, 1, 2, 3))
    missing = {
        0: (0,),
        1: (2,),
        2: (0, 2),
        3: (0, 2, 3),
    }[(c.variant - 40) % 4]
    return frozenset(set((0, 1, 2, 3)) - set(missing))


@lru_cache(maxsize=2048)
def logical_rules(c):
    """14 rules; fixed critical path, varied decoy wiring.

    Root positives have six legal choices: four decoys and two productive
    rules. The minimal goal path consumes 3/4/5 firings at designated depths.
    No edge goes backward in the critical chain.
    """
    if c.depth not in (3, 4, 5) or c.variant not in range(96):
        raise ValueError("undeclared depth or variant")
    rng = random.Random(230000 + 9719 * c.variant + 103 * c.depth)
    decoy = [4, 5, 7, 8]
    rng.shuffle(decoy)
    leaves = [9, 10, 9, 10]
    rng.shuffle(leaves)
    rules = [
        ((0,), decoy[0]),
        ((1,), decoy[1]),
        ((2,), decoy[2]),
        ((3,), decoy[3]),
        ((decoy[0],), leaves[0]),
        ((decoy[1],), leaves[1]),
        ((decoy[2],), leaves[2]),
        ((decoy[3],), leaves[3]),
        ((0, 2), 6 if c.depth == 3 else 11),
        ((1, 3), 13),
        ((6, 13), 14),
    ]
    if c.depth == 3:
        rules += [((9, 10), 12), ((12,), 6)]
    elif c.depth == 4:
        rules += [((11,), 6), ((7, 8), 12)]
    else:
        rules += [((11,), 12), ((12,), 6)]
    rules.append(((9, 10), 13))
    assert len(rules) == RULES
    assert all(0 <= h < ATOMS and all(0 <= p < ATOMS for p in ps)
               for ps, h in rules)
    return tuple((tuple(p), h) for p, h in rules)


@lru_cache(maxsize=2048)
def physical_maps(c):
    # Every layout/variant has a deterministic public physical permutation.
    h = int(_sha([c.split, c.depth, c.layout, c.variant])[:16], 16)
    rng = random.Random(h)
    atom = list(range(ATOMS))
    rule = list(range(RULES))
    rng.shuffle(atom)
    rng.shuffle(rule)
    return tuple(atom), tuple(rule)


@lru_cache(maxsize=2048)
def physical_rules(c):
    _, rpos = physical_maps(c)
    return {rpos[i]: v for i, v in enumerate(logical_rules(c))}


def enabled(c, known):
    known = set(known)
    return tuple(sorted(slot for slot, (pre, head) in physical_rules(c).items()
                        if head not in known and set(pre) <= known))


def sound_fire(c, known, action):
    known = frozenset(known)
    if action == STOP:
        return known, None
    if action not in enabled(c, known):
        raise ValueError("host rejected unsupported Horn action")
    prem, head = physical_rules(c)[action]
    return known | {head}, {
        "rule_slot": action, "premises": list(prem), "head": head,
    }


def closure(c, known=None):
    known = set(initial_facts(c) if known is None else known)
    while True:
        next_known = known | {
            h for p, h in physical_rules(c).values() if set(p) <= known
        }
        if next_known == known:
            return frozenset(known)
        known = next_known


@lru_cache(maxsize=65536)
def shortest_steps(c, known=None):
    """Host shortest path in rule firings; None means unreachable."""
    root = frozenset(initial_facts(c) if known is None else known)
    if GOAL in root:
        return 0
    queue = deque([(root, 0)])
    seen = {root}
    while queue:
        state, depth = queue.popleft()
        for action in enabled(c, state):
            next_state, _ = sound_fire(c, state, action)
            if GOAL in next_state:
                return depth + 1
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, depth + 1))
    return None


def independent_bfs(c, known=None):
    """Independent bitset oracle using rule masks, without enabled or sound_fire."""
    states = initial_facts(c) if known is None else known
    start = sum(1 << x for x in states)
    if start & (1 << GOAL):
        return 0
    masks = [
        (sum(1 << p for p in pre), 1 << head)
        for pre, head in physical_rules(c).values()
    ]
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        state, dist = queue.popleft()
        for prem, head in masks:
            if prem & state != prem or head & state:
                continue
            new = state | head
            if new & (1 << GOAL):
                return dist + 1
            if new not in seen:
                seen.add(new)
                queue.append((new, dist + 1))
    return None


def optimal_actions(c, known):
    distance = shortest_steps(c, frozenset(known))
    if distance is None or distance == 0:
        return (STOP,)
    return tuple(
        a for a in enabled(c, known)
        if shortest_steps(c, sound_fire(c, known, a)[0]) == distance - 1
    )


def symbolic_rollout(c, mode, stream=0):
    """No learned model. Correct, fixed and random host-rule baselines."""
    if mode not in ("fixed", "random_legal", "shortest"):
        raise ValueError("unregistered symbolic baseline")
    d = shortest_steps(c)
    budget = (d + 1) if d is not None else 2
    state = initial_facts(c)
    trace = []
    rng = random.Random(int(_sha([c.id, mode, stream])[:16], 16))
    for _ in range(budget):
        if GOAL in state:
            break
        legal = enabled(c, state)
        if mode == "fixed":
            action = legal[0] if legal else STOP
        elif mode == "shortest":
            action = optimal_actions(c, state)[0]
        else:
            action = rng.choice(legal) if legal else STOP
        next_state, proof = sound_fire(c, state, action)
        trace.append({"action": action, "before": sorted(state),
                      "after": sorted(next_state), "proof": proof})
        state = next_state
        if action == STOP or GOAL in state:
            break
    return {"case_id": c.id, "mode": mode, "stream": stream,
            "truth": d is not None, "optimal_steps": d, "budget": budget,
            "solved": GOAL in state, "actions": trace,
            "initial_stop": bool(trace and trace[0]["action"] == STOP),
            "false_accepted_proof": False}


def public_features(c):
    """Hashable public data for disjointness, independent of hidden labels."""
    apos, _ = physical_maps(c)
    facts = initial_facts(c)
    return {
        "atoms": ATOMS, "rules": RULES,
        "initial_physical": sorted(apos[i] for i in facts),
        "goal_slot": apos[GOAL], "subgoal_slots": [apos[i] for i in SUBGOALS],
        "rules_physical": [
            [slot, sorted(apos[i] for i in pre), apos[head]]
            for slot, (pre, head) in sorted(physical_rules(c).items())
        ],
    }


def encode(c):
    """GraphBatch adapter for existing AF-19 / AF-22 6036-weight network."""
    import torch
    from experiments.activation_funnel_af05 import gym as base
    assert base.ATOMS == ATOMS and base.RULES == RULES
    apos, _ = physical_maps(c)
    af = torch.zeros(ATOMS, 5, dtype=torch.float32)
    rf = torch.zeros(RULES, 3, dtype=torch.float32)
    pu = torch.zeros(RULES, ATOMS); hu = pu.clone()
    pd = pu.clone(); hd = pu.clone()
    facts = initial_facts(c)
    for i in range(ATOMS):
        af[apos[i]] = torch.tensor([
            float(i in facts), float(i == SUBGOALS[0]),
            float(i == SUBGOALS[1]), float(i == GOAL), 1.,
        ])
    for slot, (pre, head) in physical_rules(c).items():
        down = (head == GOAL)
        rf[slot] = torch.tensor([len(pre) / 2., float(down), 1.])
        pp, hh = (pd, hd) if down else (pu, hu)
        for p in pre:
            pp[slot, apos[p]] = 1.
        hh[slot, apos[head]] = 1.
    known = closure(c)
    return base.GraphBatch(
        af, rf, pu, hu, pd, hd,
        torch.tensor([apos[i] for i in (*SUBGOALS, GOAL)]),
        torch.tensor([int(i in known) for i in (*SUBGOALS, GOAL)], dtype=torch.long),
    )


def observations(cases, states, with_labels=False):
    """TypedLoop public graph, host-known bit mask, legal and oracle labels."""
    import torch
    from dataclasses import fields
    from experiments.activation_funnel_af05 import gym as base
    if not cases or len(cases) != len(states):
        raise ValueError("empty or mismatched AF23 observation request")
    x = base.GraphBatch(*(
        torch.stack([getattr(encode(c), f.name) for c in cases])
        for f in fields(base.GraphBatch)
    ))
    bits = torch.zeros(len(cases), ATOMS)
    legal = torch.zeros(len(cases), RULES + 1, dtype=torch.bool)
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


def manifest():
    """Pre-score source/data structural certificate; fails closed."""
    expected = {"train": 768, "dev": 384, "held_layout": 32,
                "held_depth4": 192, "held_depth5": 192}
    seen = set()
    rows = {}
    for split in SPLITS:
        cases = cases_for(split)
        assert len(cases) == expected[split]
        hashes = [_sha(public_features(c)) for c in cases]
        if len(set(hashes)) != len(hashes) or bool(seen.intersection(hashes)):
            raise AssertionError(f"AF23 public features overlap at {split}")
        seen.update(hashes)
        rows[split] = {
            "count": len(cases),
            "layouts": list(LAYOUTS[split]),
            "public_features_sha256": _sha(hashes),
            "case_ids_sha256": _sha([c.id for c in cases]),
            "designed_positive": sum(c.provable_design for c in cases),
        }
        if split.startswith("held"):
            for c in cases:
                truth = shortest_steps(c) is not None
                if truth != c.provable_design:
                    raise AssertionError("held-out Horn closure differs from design")
                if not truth:
                    continue
                root = initial_facts(c)
                legal = enabled(c, root)
                opt = optimal_actions(c, root)
                if len(legal) < 5 or len(opt) / len(legal) > .5:
                    raise AssertionError("held-out task lacks meaningful branching")
                if split in ("held_depth4", "held_depth5") and shortest_steps(c) != c.depth:
                    raise AssertionError("held-out shortest proof depth changed")
                if shortest_steps(c) != independent_bfs(c):
                    raise AssertionError("independent BFS disagreement")
            if split in ("held_depth4", "held_depth5"):
                positives = [c for c in cases if c.provable_design]
                fixed = sum(symbolic_rollout(c, "fixed")["solved"] for c in positives)
                if fixed / len(positives) > .75:
                    raise AssertionError("uninformed baseline too successful for search test")
                rows[split]["structural_fixed_solved"] = fixed
                rows[split]["structural_fixed_denominator"] = len(positives)
    out = {
        "experiment": "AF-23",
        "generator": "15 atoms/14 rules, source-owned positive Horn, 4 decoys + 2 productive starts",
        "splits": rows,
        "independent_physical_layouts": True,
        "no_cross_split_public_feature_overlap": True,
    }
    out["manifest_sha256"] = _sha(out)
    return out


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
