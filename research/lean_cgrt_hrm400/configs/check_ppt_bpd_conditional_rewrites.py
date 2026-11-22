"""Verify delayed proposition goals created by conditional Lean rewrites."""

import argparse
from dataclasses import asdict
import gzip
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import BPD_REWRITE_FAMILY, EQUALITY, rewrite_menu, rewrite_problem


SEED = 39827145
SCHEMA = "ppt.conditional-rewrite-witness.v1"
BUDGET = 8
LAYOUTS = ((3, 1, 2, 1), (3, 1, 2, 2), (4, 1, 3, 1), (4, 2, 3, 2),
           (5, 1, 3, 1), (5, 2, 4, 1), (6, 1, 3, 1), (6, 2, 5, 1))


def paired_world(layout, shape, winner):
    length, first, second, first_cost = shape
    second_cost = BUDGET - length - first_cost
    rng = random.Random(digest([SEED, layout, shape]))
    atoms = {"n": [], "p": []}
    rules, horn = [], {}

    def fresh(kind):
        name = f"{kind}{rng.randrange(1 << 40):010x}"
        if name in atoms[kind]:
            raise ValueError("proof variable names collided")
        atoms[kind].append(name)
        return name

    def premise(formula):
        name = f"h{len(rules)}"
        rules.append(formula)
        return name

    def chain(target, cost):
        path = [*[fresh("p") for _ in range(cost - 1)], target]
        horn[path[0]] = f"apply {premise(path[0])}"
        for prior, following in zip(path, path[1:]):
            name = premise(f"{prior} → {following}")
            horn[following] = f"apply {name}"

    start, stop = fresh("n"), fresh("n")
    branches = [[start, *(fresh("n") for _ in range(length - 1)), stop] for _ in range(2)]
    rewrites = [[f"rw [{premise(f'{path[0]} = {path[1]}')}]"] for path in branches]
    for side, path in enumerate(branches):
        early, late, decoy = fresh("p"), fresh("p"), fresh("p")
        for step in range(1, length):
            equation = f"{path[step]} = {path[step + 1]}"
            if step == first:
                equation = f"{early} → {equation}"
            if step == second:
                equation = f"{late} → {equation}"
            name = premise(equation)
            rewrites[side].append(f"rw [{name}]")
        chain(early, first_cost)
        chain(late if side == winner else decoy, second_cost)
        chain(decoy if side == winner else late, second_cost + 1)

    bindings = [f"({' '.join(sorted(atoms['n']))} : Nat)", f"({' '.join(sorted(atoms['p']))} : Prop)"]
    bindings.extend(f"(h{index} : {formula})" for index, formula in enumerate(rules))
    cluster = digest([SCHEMA, shape])
    case = SearchCase("loopt_conditional_rewrite", "dev", BPD_REWRITE_FAMILY, digest([cluster, winner]), cluster,
                      f"conditional_rewrite_{layout}_{winner}", " ".join(bindings), f"{start} = {stop}")
    return case, rewrites, horn


def play(repl, case, rewrites, horn):
    from ppt_gym.conditional_rewrite import project_equalities
    from ppt_gym.live_lean import candidate_menu, public_state

    repl.reset()
    state = repl.init(case.statement, pp=True)
    events, position = [], 0
    initial, = public_state(repl, state)["goals"]
    projected, _ = project_equalities(initial)
    public_equalities = rewrite_menu(projected)
    if public_equalities != ("rw [h0]", "rw [h1]"):
        raise ValueError("the conditional fixture needs two indistinguishable root rewrite lengths")
    for _ in range(BUDGET):
        before = public_state(repl, state)
        if not before["goals"]:
            break
        active = before["goals"][0]
        if EQUALITY.fullmatch(active["target"].get("raw", "")):
            visible, _ = project_equalities(active)
            menu = rewrite_menu(visible)
            tactic = rewrites[position]
            position += 1
        else:
            menu = candidate_menu(repl, state)
            tactic = horn[active["target"]["a"]]
        if tactic not in menu:
            raise ValueError("the conditional path used an unavailable public tactic")
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError("Lean rejected a proposed conditional rewrite or its obligation")
        state = reply["state"]
        after = public_state(repl, state)
        events.append({"tactic": tactic, "menu": menu, "before": digest(before), "after": digest(after),
                       "goal_types": [goal["target"] for goal in after["goals"]]})
    closed = not events[-1]["goal_types"]
    if closed:
        kernel = repl.kernel(state["sid"])
        if not kernel["ok"] or kernel["axioms"] != []:
            raise ValueError("the conditional proof failed the live kernel check")
    return {"events": events, "closed": closed, "equality_signature": digest(rewrite_problem(projected))}


def audit(tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify

    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the conditional witness requires Lean 4.19.0")
        for layout, shape in enumerate(LAYOUTS):
            signatures = set()
            for winner in (0, 1):
                case, branches, horn = paired_world(layout, shape, winner)
                receipts = []
                for choice, rewrites in enumerate(branches):
                    record = play(repl, case, rewrites, horn)
                    if (record != play(repl, case, rewrites, horn) or len(record["events"]) != BUDGET
                            or record["closed"] != (choice == winner)):
                        raise ValueError("the conditional branch must replay with the expected eight-check outcome")
                    script = [step["tactic"] for step in record["events"]]
                    cert = certify(case, {"live_kernel": record["closed"], "accepted_script": script})
                    if record["closed"] and (not cert or not cert["passed"] or cert["axioms"]):
                        raise ValueError("the winning conditional branch lacks an independent proof")
                    signatures.add(record["equality_signature"])
                    receipts.append({**record, "replay_passed": True, "certificate": cert})
                rows[case.root] = {"case": asdict(case), "winner": winner, "equality_checks": shape[0], "branches": receipts}
            if len(signatures) != 1:
                raise ValueError("a paired conditional layout changed its public equality graph")
    source = Path(__file__).resolve()
    return {"schema": SCHEMA, "source_sha256": sha256_file(source),
            "public_view_sha256": sha256_file(source.parents[1] / "ppt_gym/conditional_rewrite.py"), "lean": "4.19.0",
            "tactic_checks": BUDGET, "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(f"Lean verified and replayed both conditional openings in {len(report['rows'])} worlds.", flush=True)


if __name__ == "__main__":
    main()
