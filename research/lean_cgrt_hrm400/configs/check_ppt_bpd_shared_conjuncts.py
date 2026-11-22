"""Verify when a single public rewrite can discharge work in two conjuncts."""

import argparse
from collections import deque
from dataclasses import asdict
import gzip
from itertools import product
import json
from pathlib import Path
import random

from configs.check_ppt_bpd_dependent_rewrites import BUDGET, SEED
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


SCHEMA = "ppt.dependent-rewrite-shared-conjunct-witness.v1"
STRESS_SCHEMA = "ppt.dependent-rewrite-shared-conjunct-decoy-witness.v1"
CONFIRMATION_SCHEMA = "ppt.dependent-rewrite-shared-conjunct-decoy-confirmation.v1"
TERM_CHECKS = 4


def layouts():
    rng = random.Random(SEED + 5)
    masks = [mask for mask in range(1, 1 << TERM_CHECKS) if mask.bit_count() % 2 == 0]
    choices = [(step, left, right) for step, left, right in product(range(1, TERM_CHECKS), masks, masks) if left != right]
    sampled = rng.sample(choices, 36)
    return {"train": sampled[:24], "dev": sampled[24:]}


def stress_layouts(split="dev", *, confirmation=False):
    masks = [mask for mask in range(1, 1 << TERM_CHECKS) if mask.bit_count() % 2 == 0]
    used = {shape for split in layouts().values() for shape in split}
    choices = [shape for shape in product(range(1, TERM_CHECKS), masks, masks) if shape[1] != shape[2] and shape not in used]
    ordinary = random.Random(SEED + 6).sample(choices, 12)
    if split == "dev" and not confirmation:
        return ordinary
    training = random.Random(SEED + 7).sample([shape for shape in choices if shape not in ordinary], 12)
    if split == "train" and not confirmation:
        return training
    if split != "dev" or not confirmation:
        raise ValueError("unexpected decoy cohort role")
    return random.Random(SEED + 8).sample([shape for shape in choices if shape not in ordinary and shape not in training], 12)


def world(split, shape, winner, *, stress=False, confirmation=False):
    schema = CONFIRMATION_SCHEMA if confirmation else STRESS_SCHEMA if stress else SCHEMA
    rng = random.Random(digest([schema, shape]))
    nats = [f"n{rng.randrange(1 << 48):012x}" for _ in range(11 if stress else 7)]
    terms = [f"f{rng.randrange(1 << 48):012x}" for _ in range(8)]
    if len(set(nats + terms)) != len(nats + terms):
        raise ValueError("the generated proof identifiers collided")
    a, b, c, d, shared, first, second = nats[:7]
    start, stop = terms[:2]
    lanes = [(start, *terms[2:5], stop), (start, *terms[5:], stop)]
    guarded, *masks = shape
    permutations = [[a, b], [a, b]]
    term_scripts, premises = [[], []], []
    for step, side in product(range(TERM_CHECKS), range(2)):
        lane, permutation = lanes[side], permutations[side]
        variable = ("x", "y")[permutation.index((a, b)[side])]
        guard = f"({variable} = {c} ∧ {variable} = {d}) → " if step == guarded else ""
        swap = masks[side] & (1 << step)
        output = "y x" if swap else "x y"
        term_scripts[side].append(f"rw [h{len(premises)}]")
        premises.append(f"∀ (x y : Nat), {guard}{lane[step]} x y = {lane[step + 1]} {output}")
        if swap:
            permutation.reverse()
    common, distinct = (a, b) if winner == 0 else (b, a)
    facts = [(common, shared), (shared, c), (shared, d), (distinct, first), (first, c), (distinct, second), (second, d)]
    if stress:
        decoy, tail, spur_a, spur_b = nats[7:]
        facts.extend([(distinct, decoy), (decoy, c), (decoy, tail), (tail, d), (common, spur_a), (common, spur_b)])
    rng.shuffle(facts)
    rewrites = {f"h{i}": edge for i, edge in enumerate(facts, len(premises))}
    premises.extend(f"{left} = {right}" for left, right in facts)
    binders = [f"({' '.join(sorted(nats))} : Nat)", f"({' '.join(sorted(terms))} : Nat → Nat → Nat)",
               *(f"(h{i} : {formula})" for i, formula in enumerate(premises))]
    cluster = digest([schema, shape])
    case = SearchCase("loopt_dependent_shared_conjunct", split, BPD_REWRITE_FAMILY, digest([cluster, winner]), cluster,
                      f"shared_conjunct_{cluster[:12]}_{winner}", " ".join(binders), f"{start} {a} {b} = {stop} {a} {b}")
    goals = [((a, c), (a, d)), ((b, c), (b, d))]
    return case, term_scripts, rewrites, goals


def shortest_guard(conjuncts, rewrites, *, joined=True):
    """Enumerate the finite conjunction and equality states for the allowed single-step tactics."""
    queue = deque([(joined, conjuncts, [])])
    visited = {(joined, conjuncts)}
    while queue:
        joined, goals, script = queue.popleft()
        if not goals:
            return script
        successors = []
        if joined:
            successors.append(("constructor", False, goals))
        elif goals[0][0] == goals[0][1]:
            successors.append(("rfl", False, goals[1:]))
        considered = goals if joined else goals[:1]
        for name, pair in rewrites.items():
            for reverse, (source, dest) in enumerate((pair, pair[::-1])):
                changed = tuple(tuple(dest if atom == source else atom for atom in goal) for goal in considered)
                if changed == considered:
                    continue
                if not joined:
                    changed = (() if changed[0][0] == changed[0][1] else changed) + goals[1:]
                tactic = f"rw [{'← ' if reverse else ''}{name}]"
                successors.append((tactic, joined, changed))
        for tactic, coupled, successor in successors:
            key = coupled, successor
            if key not in visited:
                queue.append((coupled, successor, [*script, tactic]))
                visited.add(key)
    raise ValueError("the registered public guard has no finite proof plan")


def run_script(repl, case, script, *, capture):
    from ppt_gym.conjunctive_rewrite import proposal_menu
    from ppt_gym.live_lean import public_state, state_input

    repl.reset()
    state = repl.init(case.statement, pp=True)
    events = []
    for index, tactic in enumerate(script):
        packet, before = state_input(repl, state, (), max(0, BUDGET - index))
        menu = proposal_menu(json.loads(packet)["state"])
        if tactic not in menu:
            raise ValueError(f"the reference proof used a tactic outside the public menu: {case.tid}, {tactic}")
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError(f"Lean rejected the guard plan for {case.tid}: {tactic}")
        state = reply["state"]
        after = public_state(repl, state)
        event = {"tactic": tactic, "menu": menu, "before": before, "after": digest(after),
                 "targets": [goal["target"] for goal in after["goals"]]}
        if capture or index == 0:
            event["packet"] = packet
        events.append(event)
    kernel = repl.kernel(state["sid"])
    if events[-1]["targets"] or not kernel["ok"] or kernel["axioms"]:
        raise ValueError("the shared-conjunct proof failed native verification")
    return events


def audit(split, tree, *, stress=False, confirmation=False):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 103)
    schema = CONFIRMATION_SCHEMA if confirmation else STRESS_SCHEMA if stress else SCHEMA
    shapes = stress_layouts(split, confirmation=confirmation) if stress else layouts()[split]
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the shared-conjunct cohort uses Lean 4.19.0")
        for shape, winner in product(shapes, (0, 1)):
            case, term_scripts, rewrites, guards = world(split, shape, winner, stress=stress, confirmation=confirmation)
            branches = []
            for choice, term in enumerate(term_scripts):
                # Splitting immediately is the same length on both branches. Only one can reuse a rewrite.
                solo = [shortest_guard((guard,), rewrites, joined=False) for guard in guards[choice]]
                if any(len(path) != 2 for path in solo):
                    raise ValueError("the task leaked the winner through an independent conjunct")
                suffix = shortest_guard(guards[choice], rewrites)
                script = term + suffix
                if len(script) != BUDGET + int(choice != winner):
                    raise ValueError("the two branches must differ by exactly one shared tactic")
                start = tree.cpu_seconds()
                capture = split == "train"
                events = run_script(repl, case, script, capture=capture)
                within = tree.cpu_seconds() - start <= 6
                targets = events[TERM_CHECKS - 1]["targets"]
                expected = [{"and": [{"raw": " = ".join(pair)} for pair in guards[choice]]}]
                if targets != expected or bool(events[BUDGET - 1]["targets"]) != (choice != winner):
                    raise ValueError("Lean changed a registered conjunct or its eight-check outcome")
                if not within:
                    raise ValueError(f"the shared-conjunct proof exceeded its CPU budget: {case.tid}, branch {choice}")
                start = tree.cpu_seconds()
                replay = run_script(repl, case, script, capture=capture)
                if tree.cpu_seconds() - start > 6 or events != replay:
                    raise ValueError("the shared-conjunct proof changed on replay")
                certificate = certify(case, {"live_kernel": True, "accepted_script": script})
                if not certificate or not certificate["passed"] or certificate["axioms"]:
                    raise ValueError("the shared-conjunct proof failed independent compilation")
                branches.append({"choice": choice, "checks": len(script), "events": events,
                                 "sum_if_split_first": TERM_CHECKS + 1 + sum(map(len, solo)), "certificate": certificate})
            rows[case.root] = {"case": asdict(case), "shape": shape, "winner": winner, "branches": branches}
    return {"schema": schema, "split": split, "lean": "4.19.0", "tactic_checks": BUDGET, "rows": rows,
            "source_sha256": sha256_file(Path(__file__)), "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("split", choices=("train", "dev"))
    cli.add_argument("output", type=Path)
    variants = cli.add_mutually_exclusive_group()
    variants.add_argument("--decoy", action="store_true")
    variants.add_argument("--decoy-confirmation", action="store_true")
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        result = audit(args.split, tree, stress=args.decoy or args.decoy_confirmation, confirmation=args.decoy_confirmation)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(f"Independently certified both shared-conjunct paths in {len(result['rows'])} {args.split} worlds.", flush=True)


if __name__ == "__main__":
    main()
