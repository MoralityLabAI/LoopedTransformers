"""AF-04 proof-state intervention gym: fully observed facts/rules, hidden closure.

All status labels come from an independent least-fixed-point Horn oracle. The
model sees initial facts, public role marks, rule arities and incidence; it
never sees rule applicability, derived subgoals, future answers, proof traces,
or the donor's correctness labels. Stage-2 incidence does not expose stage-1
premises, requiring the looper to carry inferred intermediate proof state.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import random

import torch

ATOMS, RULES = 11, 10
SUBGOALS = (4, 9)
GOAL = 10
SPLITS = ("train", "dev", "held_composition", "held_layout", "held_depth")
STRESS_COMPOSITION = ("or", "or", "and")
KINDS = ("and", "or")
GATES = ("and", "or")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    split: str
    depth: int
    kinds: tuple[str, str]
    gate: str
    facts: tuple[int, int, int, int]
    permutation: int

    @property
    def id(self):
        return f"{self.split}_d{self.depth}_{''.join(k[0] for k in self.kinds)}_{self.gate[0]}_{''.join(map(str,self.facts))}_p{self.permutation}"

    @property
    def skeleton(self):
        return self.depth, self.kinds, self.gate, self.permutation


def logical_rules(case):
    """Two independent Horn subproofs followed by an AND or OR goal."""
    upstream = []
    for branch, kind in enumerate(case.kinds):
        start = 5 * branch
        a, b, relay1, relay2, subgoal = range(start, start + 5)
        latest = a
        if case.depth >= 2:
            upstream.append(((latest,), relay1))
            latest = relay1
        if case.depth >= 3:
            upstream.append(((latest,), relay2))
            latest = relay2
        if kind == "and":
            upstream.append(((latest, b), subgoal))
        else:
            upstream.extend([((latest,), subgoal), ((b,), subgoal)])
    if case.gate == "and":
        downstream = [((SUBGOALS[0], SUBGOALS[1]), GOAL)]
    else:
        downstream = [((SUBGOALS[0],), GOAL), ((SUBGOALS[1],), GOAL)]
    assert len(upstream) + len(downstream) <= RULES
    return tuple(upstream), tuple(downstream)


def initial_facts(case):
    return {p for p, v in zip((0, 1, 5, 6), case.facts) if v}


def closure(known, rules):
    """Independent finite least-fixed-point reference semantics."""
    known = set(known)
    history = []
    for _ in range(ATOMS + 1):
        candidates = [(i, h) for i, (p, h) in enumerate(rules)
                      if h not in known and all(x in known for x in p)]
        if not candidates:
            return frozenset(known), tuple(history)
        for i, head in candidates:
            if head not in known:
                known.add(head)
                history.append((i, head))
    raise ValueError("non-convergent Horn closure")


def oracle(case):
    up, down = logical_rules(case)
    k, trace = closure(initial_facts(case), up)
    result, goal_trace = closure(k, down)
    status = (int(SUBGOALS[0] in k), int(SUBGOALS[1] in k), int(GOAL in result))
    direct = tuple(int(case.facts[2 * i] and case.facts[2 * i + 1]) if kind == 'and'
                   else int(case.facts[2 * i] or case.facts[2 * i + 1])
                   for i, kind in enumerate(case.kinds))
    direct_goal = int(all(direct)) if case.gate == "and" else int(any(direct))
    if status != direct + (direct_goal,):
        raise AssertionError('independent closure disagrees with Boolean evaluation')
    return status, trace, goal_trace


@lru_cache(maxsize=64)
def permutation_maps(index):
    """Public deterministic physical permutations; no hidden role-order shortcut."""
    if index not in range(8):
        raise ValueError('undeclared graph permutation')
    rng = random.Random(1031 + index * 7919)
    nodes = list(range(ATOMS)); edges = list(range(RULES))
    rng.shuffle(nodes); rng.shuffle(edges)
    return tuple(nodes), tuple(edges)


@dataclass
class GraphBatch:
    atom_features: torch.Tensor
    rule_features: torch.Tensor
    prem_up: torch.Tensor
    head_up: torch.Tensor
    prem_down: torch.Tensor
    head_down: torch.Tensor
    read_slots: torch.Tensor
    targets: torch.Tensor


def encode_case(case):
    up, down = logical_rules(case)
    atom_pos, rule_pos = permutation_maps(case.permutation)
    af = torch.zeros(ATOMS, 5, dtype=torch.float32)
    for logical in range(ATOMS):
        p = atom_pos[logical]
        af[p, 0] = float(logical in initial_facts(case))
        af[p, 1] = float(logical == SUBGOALS[0])
        af[p, 2] = float(logical == SUBGOALS[1])
        af[p, 3] = float(logical == GOAL)
        af[p, 4] = 1.
    rf = torch.zeros(RULES, 3, dtype=torch.float32)
    pu = torch.zeros(RULES, ATOMS)
    hu = torch.zeros_like(pu)
    pd = torch.zeros_like(pu)
    hd = torch.zeros_like(pu)
    for logical_rule, (premises, head) in enumerate(up + down):
        rp = rule_pos[logical_rule]
        rf[rp, 0] = len(premises) / 2.
        rf[rp, 1] = float(logical_rule >= len(up))
        rf[rp, 2] = 1.
        prem = pu if logical_rule < len(up) else pd
        heads = hu if logical_rule < len(up) else hd
        for atom in premises:
            prem[rp, atom_pos[atom]] = 1.
        heads[rp, atom_pos[head]] = 1.
    return GraphBatch(af, rf, pu, hu, pd, hd,
                      torch.tensor([atom_pos[n] for n in (*SUBGOALS, GOAL)]),
                      torch.tensor(oracle(case)[0], dtype=torch.long))


def stack_cases(cases):
    rows = [encode_case(c) for c in cases]
    if not rows:
        raise ValueError('nonempty batch required')
    return GraphBatch(*(torch.stack([getattr(x, field) for x in rows])
                        for field in GraphBatch.__dataclass_fields__))


@lru_cache(maxsize=12)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError(f'unknown split: {split}')
    cases = []
    if split == 'train':
        depths, permutations, special = (2,), (0, 1, 2), False
    elif split == 'dev':
        depths, permutations, special = (2,), (3,), False
    elif split == 'held_composition':
        depths, permutations, special = (2,), (4,), True
    elif split == 'held_layout':
        depths, permutations, special = (2,), (5,), False
    else:
        depths, permutations, special = (3,), (6,), None
    for depth, k0, k1, gate, facts, perm in itertools.product(
            depths, KINDS, KINDS, GATES,
            itertools.product((0, 1), repeat=4), permutations):
        is_special = (k0, k1, gate) == STRESS_COMPOSITION
        if special is True and not is_special:
            continue
        if special is False and is_special:
            continue
        cases.append(Case(split, depth, (k0, k1), gate, tuple(facts), perm))
    return tuple(cases)


def oracle_interchange(recipient, donor, selected):
    if selected not in (0, 1) or recipient.skeleton != donor.skeleton:
        raise ValueError('invalid mismatched proof-state donor')
    if tuple(recipient.facts[2 * (1 - selected):2 * (2 - selected)]) != \
       tuple(donor.facts[2 * (1 - selected):2 * (2 - selected)]):
        raise ValueError('donor modifies the untouched independent subproof')
    old = oracle(recipient)[0]
    incoming = oracle(donor)[0]
    if old[selected] == incoming[selected]:
        raise ValueError('donor must change inferred subgoal status')
    q = list(old[:2]); q[selected] = incoming[selected]
    goal = int(all(q)) if recipient.gate == 'and' else int(any(q))
    return (q[0], q[1], goal)


@lru_cache(maxsize=12)
def donors_for(split):
    cases = cases_for(split)
    groups = {}
    for i, case in enumerate(cases):
        groups.setdefault(case.skeleton, []).append(i)
    roster = {}
    for i, recipient in enumerate(cases):
        for selected in (0, 1):
            candidates = [j for j in groups[recipient.skeleton]
                          if all(cases[j].facts[k] == recipient.facts[k]
                                 for k in (2 * (1-selected), 2 * (1-selected) + 1))
                          and oracle(cases[j])[0][selected] != oracle(recipient)[0][selected]]
            if not candidates:
                raise ValueError('donor-pair coverage incomplete')
            j = min(candidates, key=lambda index:
                    (sum(a != b for a, b in zip(cases[index].facts, recipient.facts)),
                     cases[index].facts))
            roster[i, selected] = (j, oracle_interchange(recipient, cases[j], selected))
    return roster


def manifest():
    receipt = {'experiment': 'AF-04', 'input_contract': 'public-facts-and-rule-incidence-no-derived-truth',
               'graph_nodes': ATOMS, 'graph_rules': RULES, 'splits': {}}
    seen_ids, seen_payloads = set(), set()
    for split in SPLITS:
        cases = cases_for(split)
        samples = []
        for case in cases:
            datum = encode_case(case)
            payload = [getattr(datum, name).tolist() for name in
                       ('atom_features', 'rule_features', 'prem_up', 'head_up',
                        'prem_down', 'head_down', 'read_slots')]
            casehash = digest(payload)
            if case.id in seen_ids or casehash in seen_payloads:
                raise AssertionError('data split identity leakage')
            seen_ids.add(case.id); seen_payloads.add(casehash)
            samples.append({'id': case.id, 'input_sha256': casehash,
                            'oracle_sha256': digest(oracle(case)[0])})
        receipt['splits'][split] = {'count': len(cases), 'inputs_sha256': digest(samples),
                                    'labels_sha256': digest([oracle(c)[0] for c in cases]),
                                    'pairs': len(donors_for(split))}
    receipt['manifest_sha256'] = digest(receipt)
    return receipt


def lean_expression(case, target):
    """Expand a DAG's declared rules to a closed Boolean term for Lean4's kernel."""
    all_rules = logical_rules(case)[0] + logical_rules(case)[1]
    facts = initial_facts(case)
    @lru_cache(maxsize=None)
    def expression(node):
        inputs = ['true'] if node in facts else []
        for prem, head in all_rules:
            if head == node:
                parts = [expression(p) for p in prem]
                inputs.append('(' + ' && '.join(parts) + ')')
        return '(' + ' || '.join(inputs) + ')' if inputs else 'false'
    return expression(target)


def lean_certificate(split):
    """Closed Boolean-Horn certificates, separate from model success metrics."""
    if split not in SPLITS:
        raise ValueError(split)
    lines = [f'-- AF-04 exact generated Horn-DAG oracle, {split}',
             '-- Each proposition expands declared rule antecedents; Lean checks the expected Boolean outcome.']
    for i, case in enumerate(cases_for(split)):
        truth = oracle(case)[0]
        for j, node in enumerate((*SUBGOALS, GOAL)):
            term = lean_expression(case, node)
            expected = 'true' if truth[j] else 'false'
            lines.append(f'theorem af04_{split}_{i}_{j} : ({term}) = {expected} := by decide')
    return '\n'.join(lines) + '\n'
