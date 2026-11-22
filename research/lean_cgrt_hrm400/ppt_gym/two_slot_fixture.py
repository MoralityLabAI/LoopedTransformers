"""The registered two-goal rewrite fork and its public tactic grammar."""

import random

from .data import digest
from .hybrid_rl.corpus import SearchCase
from .rewrite import HYPOTHESIS, rewrite_hypothesis, rewrite_menu


FAMILY = "bpd-two-slot-rewrite"
CHECKS = 4
BUNDLES = ("all_goals rw [h0]", "all_goals rw [h1]")


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


def registered_tactic(tactic):
    exact = tactic.removeprefix("exact ")
    return (tactic in BUNDLES or rewrite_hypothesis(tactic) == "h2"
            or (exact != tactic and HYPOTHESIS.fullmatch(exact) is not None))


def menu(goals, step):
    if step == 0 and len(goals) == 1 and "and" in goals[0]["target"]:
        return ("constructor",)
    if step == 1 and len(goals) == 2:
        legal = [rewrite_menu(goal) for goal in goals]
        return tuple(bundle for bundle in BUNDLES
                     if all(bundle.removeprefix("all_goals ") in options for options in legal))
    if step >= 2 and goals:
        focus = goals[0]
        exact = tuple(f"exact {hyp['name']}" for hyp in focus["hypotheses"]
                      if HYPOTHESIS.fullmatch(hyp["name"]) and hyp["type"] == focus["target"])
        return exact or tuple(tactic for tactic in rewrite_menu(focus) if rewrite_hypothesis(tactic) == "h2")
    return ()
