"""Registered two-slot proof pairs and their public tactic grammars."""

from dataclasses import dataclass
import random

from .data import digest
from .hybrid_rl.corpus import SearchCase
from .rewrite import HYPOTHESIS, rewrite_hypothesis, rewrite_menu


FAMILY = "bpd-two-slot-rewrite"
GUARD = "bpd-two-slot-guard"
SPLIT = "bpd-two-slot-split"
CHECKS = 4
BUNDLES = ("all_goals rw [h0]", "all_goals rw [h1]")


@dataclass(frozen=True)
class Probe:
    prefix: str
    fork: tuple[str, str]
    checks: int
    bridge: str


PROBES = {
    FAMILY: Probe("constructor", BUNDLES, CHECKS, "h2"),
    GUARD: Probe("constructor", ("apply h0", "apply h1"), 4, "h2"),
    SPLIT: Probe("rw [h7]", ("rw [h0]", "constructor"), 5, "h0"),
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


def registered_tactic(tactic, family=FAMILY):
    probe = PROBES.get(family)
    if probe is None:
        return False
    exact = tactic.removeprefix("exact ")
    return (tactic in (probe.prefix, *probe.fork) or rewrite_hypothesis(tactic) == probe.bridge
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
    return exact or tuple(tactic for tactic in rewrite_menu(focus)
                          if rewrite_hypothesis(tactic) == probe.bridge)
