"""Lean proof pairs whose first obligation fixes a shared existential witness."""

from itertools import combinations, product
import random
import re

from .data import canonical, digest
from .hybrid_rl.corpus import SearchCase
from .rewrite import HYPOTHESIS, NAME


FAMILY = "bpd-dependent-witness"
DEEP_FAMILY = "bpd-deep-dependent-witness"
PREFIX = ("apply Exists.intro", "constructor")
FORK = ("exact h0", "exact h1")
CHECKS = 8
BUDGETS = {FAMILY: CHECKS, DEEP_FAMILY: 10}
FORK_INDEX = len(PREFIX)
PREDICATE = re.compile(rf"({NAME}) (\?{NAME}|{NAME}|[0-9]+)")


def examples():
    cluster = digest(FAMILY)
    for winner in range(2):
        binders = ("(P Q : Nat → Prop) (h0 : P 0) (h1 : P 1) "
                   f"(h2 : Q {winner}) (h3 : Q {winner} → Q {1 - winner})")
        case = SearchCase(FAMILY, "train", FAMILY, digest([cluster, winner]), cluster,
                          f"bpd_shared_witness_{winner}", binders, "∃ n, P n ∧ Q n")
        yield case, winner


def script(winner, branch):
    close = ("exact h2",) if branch == winner else ("apply h3", "exact h2")
    return (*PREFIX, FORK[branch], *close)


def extra_tactic(action):
    return action == PREFIX[0] or (action.startswith("exact ") and HYPOTHESIS.fullmatch(action[6:]) is not None)


def cohorts(split):
    """Paired native routes combine witnessed instantiation and Horn continuations."""
    ranges = {"train": slice(0, 12), "dev": slice(12, 36)}
    if split not in ranges:
        raise ValueError("expected a dependent witness training or development split")
    shapes = list(product(range(3), range(3), range(2), range(2), range(2)))
    random.Random(947).shuffle(shapes)
    for shape in shapes[ranges[split]]:
        left_kind, right_kind, delayed_side, extra_fork, numbering = shape
        cluster = digest([FAMILY, "horn-continuation", shape])
        for winner in range(2):
            rng = random.Random(cluster)
            letters = iter(rng.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 26))
            p, q = next(letters), next(letters)
            numbers = (0, 1) if numbering == 0 else (2, 7)
            facts = []
            scripts = []
            predicates = {p, q}

            def new_atom(n):
                predicate = next(letters)
                predicates.add(predicate)
                return f"{predicate} {n}"

            def fact(ty, method):
                key = len(facts)
                facts.append(ty)
                return method, key

            for side, kind in enumerate((left_kind, right_kind)):
                n = numbers[side]
                head = f"{q} {n}"
                plan = []
                # The extra leading implication makes the other witness arrive one check late.
                if side != winner:
                    premise = new_atom(n)
                    plan.append(fact(f"{premise} → {head}", "apply"))
                    head = premise
                dead = new_atom(n)
                fact(f"{dead} → {q} {n}", "apply")
                if kind == 0:
                    for step in range(4):
                        premise = new_atom(n)
                        if extra_fork and step == 1:
                            dead = new_atom(n)
                            fact(f"{dead} → {head}", "apply")
                        plan.append(fact(f"{premise} → {head}", "apply"))
                        head = premise
                    plan.append(fact(head, "exact"))
                else:
                    if kind == 2:
                        premise = new_atom(n)
                        plan.append(fact(f"{premise} → {head}", "apply"))
                        head = premise
                    left, right = new_atom(n), new_atom(n)
                    plan.extend((fact(f"({left} ∧ {right}) → {head}", "apply"), ("constructor", None)))
                    for index, atom in enumerate((left, right)):
                        if kind == 1 and index == delayed_side:
                            premise = new_atom(n)
                            if extra_fork:
                                dead = new_atom(n)
                                fact(f"{dead} → {atom}", "apply")
                            plan.append(fact(f"{premise} → {atom}", "apply"))
                            atom = premise
                        plan.append(fact(atom, "exact"))
                scripts.append(plan)
            order = list(range(len(facts)))
            rng.shuffle(order)
            names = {key: f"h{i}" for i, key in enumerate(order, 2)}
            binders = [f"({' '.join(sorted(predicates))} : Nat → Prop)", f"(h0 : {p} {numbers[0]})",
                       f"(h1 : {p} {numbers[1]})", *(f"({names[key]} : {facts[key]})" for key in order)]
            case = SearchCase(FAMILY, split, FAMILY, digest([cluster, winner]), cluster,
                              f"bpd_witness_horn_{cluster[:12]}_{winner}", " ".join(binders), f"∃ n, {p} n ∧ {q} n")
            paths = [(*PREFIX, FORK[side], *(method if key is None else f"{method} {names[key]}" for method, key in plan))
                     for side, plan in enumerate(scripts)]
            yield case, winner, paths


def deep_cohorts(split):
    """Swap remote support while preserving the number of rules and facts."""
    ranges = {"train": slice(0, 12), "dev": slice(12, 36)}
    if split not in ranges:
        raise ValueError("expected a deep witness training or development split")
    # Split pairs of distinct proof structures before drawing names and numerals.
    branches = [(0, False, False), (0, False, True), *product(range(1, 5), (False, True), (False, True))]
    shapes = list(combinations(branches, 2))
    random.Random(2201).shuffle(shapes)
    for shape in shapes[ranges[split]]:
        cluster = digest([DEEP_FAMILY, shape])
        for winner in range(2):
            rng = random.Random(cluster)
            numbers = rng.choice(((0, 1), (3, 8)))
            sides = shape if rng.getrandbits(1) else shape[::-1]
            alphabet = rng.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 26)
            p, q = alphabet[:2]
            hypotheses, scripts = [], []
            predicates = {p, q}

            def fact(ty, method="apply"):
                hypotheses.append(ty)
                return method, len(hypotheses) - 1

            for side, (kind, main_first, deep_decoy) in enumerate(sides):
                n = numbers[side]
                letters = iter(alphabet[2:])

                def atom():
                    name = next(letters)
                    predicates.add(name)
                    return f"{name} {n}"

                fast, slow, base, dead, spare = (atom() for _ in range(5))
                head = f"{q} {n}"
                fast_end = fact(fast, "exact")
                slow_step = fact(f"{base} → {slow}")
                slow_end = fact(base, "exact")
                if not deep_decoy:
                    fact(f"{dead} → {head}")
                viable = side == winner
                endpoint, unused = (fast, slow) if viable else (slow, fast)
                fact(f"{unused} → {spare}")
                plan = []
                lead = 5 if kind == 0 else kind - 1
                for i in range(lead):
                    premise = atom()
                    plan.append(fact(f"{premise} → {head}"))
                    head = premise
                    if deep_decoy and i == 0:
                        fact(f"{dead} → {head}")
                if kind == 0:
                    plan.append(fact(f"{endpoint} → {head}"))
                    tail = []
                else:
                    sibling = atom()
                    sibling_end = fact(sibling, "exact")
                    tail, main = [], endpoint
                    for _ in range(3 - lead):
                        above = atom()
                        tail.append(fact(f"{main} → {above}"))
                        main = above
                    if deep_decoy and lead == 0:
                        fact(f"{dead} → {main}")
                    left, right = (main, sibling) if main_first else (sibling, main)
                    plan.extend((fact(f"({left} ∧ {right}) → {head}"), ("constructor", None)))
                    if not main_first:
                        plan.append(sibling_end)
                plan.extend(reversed(tail))
                plan.extend((fast_end,) if viable else (slow_step, slow_end))
                if kind and main_first:
                    plan.append(sibling_end)
                scripts.append(plan)
            order = list(range(len(hypotheses)))
            rng.shuffle(order)
            names = {key: f"h{i}" for i, key in enumerate(order, 2)}
            binders = [f"({' '.join(sorted(predicates))} : Nat → Prop)", f"(h0 : {p} {numbers[0]})",
                       f"(h1 : {p} {numbers[1]})", *(f"({names[key]} : {hypotheses[key]})" for key in order)]
            case = SearchCase(DEEP_FAMILY, split, DEEP_FAMILY, digest([cluster, winner]), cluster,
                              f"bpd_deep_witness_{cluster[:12]}_{winner}", " ".join(binders), f"∃ n, {p} n ∧ {q} n")
            paths = [(*PREFIX, FORK[side], *(method if key is None else f"{method} {names[key]}" for method, key in plan))
                     for side, plan in enumerate(scripts)]
            yield case, winner, paths


def proof_view(goals):
    """The two proof slots see the shared Nat hole through their metavariable."""
    if len(goals) == 2 and goals[1]["target"] == {"raw": "Nat"} and "and" in goals[0]["target"]:
        atoms = [PREDICATE.fullmatch(t.get("raw", "")) for t in goals[0]["target"]["and"]]
        if len(atoms) == 2 and all(atoms) and atoms[0][2].startswith("?") and atoms[0][2] == atoms[1][2]:
            return goals[:1]
    if len(goals) <= 2:
        return goals
    targets = [PREDICATE.fullmatch(g["target"].get("raw", "")) for g in goals[:2]]
    if (len(goals) != 3 or goals[2]["target"] != {"raw": "Nat"} or not all(targets)
            or not targets[0][2].startswith("?") or targets[0][2] != targets[1][2]):
        raise ValueError("the witness hole is not the registered shared Nat metavariable")
    return goals[:2]


def proof_hypotheses(goal):
    return (hyp for hyp in goal["hypotheses"] if HYPOTHESIS.fullmatch(hyp["name"]))


def dependent_menu(goals, index):
    if index < FORK_INDEX:
        return (PREFIX[index],)
    if index == FORK_INDEX:
        if len(goals) != 3:
            raise ValueError("expected two pending proofs and the shared witness hole")
        proof_view(goals)
        return FORK
    focus = goals[0]
    target = focus["target"]
    if "and" in target:
        return ("constructor",)
    sources = list(proof_hypotheses(focus))
    exact = tuple(f"exact {h['name']}" for h in sources if h["type"] == target)
    return exact or tuple(f"apply {h['name']}" for h in sources
                          if h["type"].get("imp", [None, None])[1] == target)


def action_projection(goals, action):
    """Project only simple public combinations whose native effect is registered."""
    targets = [g["target"] for g in goals]
    context = {h["name"]: h["type"] for h in goals[0]["hypotheses"]}
    if action.startswith("apply "):
        rule = context.get(action[6:], {}).get("imp", ())
        if len(rule) == 2 and rule[1] == targets[0]:
            targets[0] = rule[0]
    if not action.startswith("exact "):
        return targets
    assigned = context.get(action[6:], {})
    head = PREDICATE.fullmatch(targets[0].get("raw", ""))
    evidence = PREDICATE.fullmatch(assigned.get("raw", ""))
    if head and evidence and head[1] == evidence[1] and head[2].startswith("?") and not evidence[2].startswith("?"):
        targets[0] = None
        for index in range(1, len(targets)):
            pending = PREDICATE.fullmatch(targets[index].get("raw", ""))
            if pending and pending[2] == head[2]:
                targets[index] = {"raw": f"{pending[1]} {evidence[2]}"}
    elif assigned == targets[0]:
        targets[0] = None
    return targets


def horn_graph(goal, targets):
    """Describe the visible facts and rules shared by projected goals."""
    terms, facts, edges = set(), set(), set()

    def register(prop):
        key = canonical(prop)
        if key not in terms:
            terms.add(key)
            if "and" in prop:
                lhs, rhs = (register(item) for item in prop["and"])
                edges.add((key, lhs, rhs))
        return key

    for hyp in proof_hypotheses(goal):
        ty = hyp["type"]
        if "imp" in ty:
            body, head = (register(item) for item in ty["imp"])
            edges.add((head, body, ""))
        else:
            facts.add(register(ty))
    active = [register(target or {"closed": True}) for target in targets]
    return terms, facts, edges, active


def horn_problem(goal, target):
    terms, facts, edges, (active,) = horn_graph(goal, (target,))
    return terms, facts, edges, active
