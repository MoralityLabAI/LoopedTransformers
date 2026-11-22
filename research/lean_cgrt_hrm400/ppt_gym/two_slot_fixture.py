"""Registered two-slot proof pairs and their public tactic grammars."""

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
import random

from .data import digest
from .hybrid_rl.corpus import SearchCase
from .rewrite import HYPOTHESIS, rewrite_hypothesis, rewrite_menu, rewrite_problem


FAMILY = "bpd-two-slot-rewrite"
GUARD = "bpd-two-slot-guard"
SPLIT = "bpd-two-slot-split"
ROUTE = "bpd-two-slot-continuation"
CHECKS = 4
BUNDLES = ("all_goals rw [h0]", "all_goals rw [h1]")


@dataclass(frozen=True)
class Probe:
    prefix: str
    fork: tuple[str, str]
    checks: int
    bridge: str | None


PROBES = {
    FAMILY: Probe("constructor", BUNDLES, CHECKS, "h2"),
    GUARD: Probe("constructor", ("apply h0", "apply h1"), 4, "h2"),
    SPLIT: Probe("rw [h7]", ("rw [h0]", "constructor"), 5, "h0"),
    ROUTE: Probe("constructor", BUNDLES, 7, None),
}


def fixtures(split):
    if split not in ("train", "dev"):
        raise ValueError("expected train or dev two-slot fixtures")
    for layout in range(6 if split == "train" else 1):
        if split == "train":
            rng = random.Random(404 + layout)
            names = rng.sample([f"n{i}" for i in range(64)], 6)
        else:
            names = ["a", "b", "c", "u", "v", "w"]
        a, b, c, u, v, w = names
        cluster = digest([FAMILY, split, layout])
        order = list(range(7))
        if split == "train":
            rng.shuffle(order)
        for winner, (finish, decoy) in enumerate(((b, c), (c, b))):
            edges = [(a, b), (a, c), (b, c), (b, u), (c, u), (finish, v), (decoy, w)]
            binders = [f"({' '.join(names)} : Nat)"]
            binders.extend(f"(h{i} : {edges[i][0]} = {edges[i][1]})" for i in order)
            yield SearchCase(FAMILY, split, FAMILY, digest([cluster, winner]), cluster,
                             f"two_slot_{split}_{layout}_{winner}", " ".join(binders),
                             f"{a} = {u} ∧ {a} = {v}"), BUNDLES[winner]
    if split == "dev":
        yield from structural_fixtures()


def structural_fixtures():
    for winner, (finish, decoy) in enumerate((("b", "c"), ("c", "b"))):
        cluster = digest([GUARD, "dev"])
        binders = ("(a b c u v z w : Nat) (h0 : b = z → a = u) (h1 : c = z → a = u) "
                   f"(h2 : b = c) (h3 : {finish} = z) (h4 : a = v) (h5 : {decoy} = w)")
        yield SearchCase(FAMILY, "dev", GUARD, digest([cluster, winner]), cluster,
                         f"two_slot_guard_{winner}", binders, "a = u ∧ a = v"), PROBES[GUARD].fork[winner]
    for winner, focus in enumerate(("b", "a")):
        cluster = digest([SPLIT, "dev"])
        binders = f"(s a b u v : Nat) (h7 : s = a) (h0 : a = b) (h3 : {focus} = u) (h4 : b = v)"
        yield SearchCase(FAMILY, "dev", SPLIT, digest([cluster, winner]), cluster,
                         f"two_slot_split_{winner}", binders, "s = u ∧ s = v"), PROBES[SPLIT].fork[winner]


def continuation_fixtures(split):
    """Each root reaches both goals, with independently placed dead-end rewrites."""
    ranges = {"train": slice(0, 12), "dev": slice(12, 36), "confirm": slice(36, 60)}
    if split not in ranges:
        raise ValueError("expected train, dev or confirm continuation fixtures")
    shapes = list(product(range(2), range(3), range(2), range(2), range(2), range(2)))
    random.Random(831).shuffle(shapes)
    for shape in shapes[ranges[split]]:
        cheap_focus, slow_layout, left_share, right_share, left_spur, right_spur = shape
        cheap = (2, 3) if cheap_focus else (3, 2)
        slow = ((3, 3), (2, 4), (4, 2))[slow_layout]
        cluster = digest([ROUTE, shape])
        for winner in range(2):
            rng = random.Random(cluster)
            names = iter(rng.sample(list("abcdefghijklmnopqrstuvwxyz"), 26))
            a, b, c, u, v = (next(names) for _ in range(5))
            edges = []
            for branch, start, shared, interior in ((0, b, left_share, left_spur), (1, c, right_share, right_spur)):
                lengths = cheap if branch == winner else slow
                hub = next(names) if shared else start
                if shared:
                    edges.append((start, hub))
                for i, (target, length) in enumerate(zip((u, v), lengths, strict=True)):
                    path = [hub, *(next(names) for _ in range(length - shared - 1)), target]
                    edges.extend(zip(path, path[1:]))
                    if i == 0:
                        edges.append(((hub if shared else path[1]) if interior else start, next(names)))
            rng.shuffle(edges)
            edges = [(a, b), (a, c), *edges]
            atoms = " ".join(sorted({atom for edge in edges for atom in edge}))
            binders = [f"({atoms} : Nat)", *(f"(h{i} : {x} = {y})" for i, (x, y) in enumerate(edges))]
            case = SearchCase(FAMILY, split, ROUTE, digest([cluster, winner]), cluster,
                              f"two_slot_continuation_{cluster[:12]}_{winner}",
                              " ".join(binders), f"{a} = {u} ∧ {a} = {v}")
            yield case, BUNDLES[winner]


def registered_tactic(tactic, family=FAMILY):
    probe = PROBES.get(family)
    if probe is None:
        return False
    exact = tactic.removeprefix("exact ")
    rewrite = rewrite_hypothesis(tactic)
    return (tactic in (probe.prefix, *probe.fork) or (rewrite is not None and rewrite == probe.bridge)
            or (family == ROUTE and rewrite is not None and not tactic.startswith("rw [← "))
            or (exact != tactic and HYPOTHESIS.fullmatch(exact) is not None))


def public_action(goals, action):
    focus = goals[0]
    target = focus["target"]
    if action == "constructor":
        return len(goals) == 1 and "and" in target
    if action in BUNDLES:
        return len(goals) == 2 and all(action.removeprefix("all_goals ") in rewrite_menu(goal) for goal in goals)
    if action.startswith("apply "):
        return len(goals) == 2 and any(hyp["name"] == action.removeprefix("apply ")
                                      and hyp["type"].get("imp", [None, None])[1] == target
                                      for hyp in focus["hypotheses"])
    return any(action in rewrite_menu({**focus, "target": conjunct}) for conjunct in target.get("and", [target]))


def menu(goals, step, family=FAMILY):
    if not goals:
        return ()
    probe = PROBES[family]
    if step <= 1:
        candidates = (probe.prefix,) if step == 0 else probe.fork
        return tuple(action for action in candidates if public_action(goals, action))
    focus = goals[0]
    if family == SPLIT and public_action(goals, "constructor"):
        return ("constructor",)
    exact = tuple(f"exact {hyp['name']}" for hyp in focus["hypotheses"]
                  if HYPOTHESIS.fullmatch(hyp["name"]) and hyp["type"] == focus["target"])
    if family == ROUTE:
        return exact or tuple(tactic for tactic in rewrite_menu(focus) if not tactic.startswith("rw [← "))
    return exact or tuple(tactic for tactic in rewrite_menu(focus)
                          if rewrite_hypothesis(tactic) == probe.bridge)


def continuation_witness_move(goal, actions):
    """The finite shortest-path reference for the directed equality grammar."""
    if len(actions) == 1:
        return actions[0]
    _, target, edges = rewrite_problem(goal)
    predecessors, following = defaultdict(list), {}
    for action, source, destination in edges:
        if not action.startswith("rw [← "):
            predecessors[destination].append(source)
            following[action] = destination
    distances, queue = {target: 0}, deque([target])
    while queue:
        current = queue.popleft()
        for prior in predecessors[current]:
            if prior not in distances:
                distances[prior] = distances[current] + 1
                queue.append(prior)
    return min(actions, key=lambda action: distances.get(following[action], float("inf")))
