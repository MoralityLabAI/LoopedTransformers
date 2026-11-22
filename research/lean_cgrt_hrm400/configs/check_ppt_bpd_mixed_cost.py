"""Verify eight-check choices between Lean equality and proposition obligations."""

import argparse
from dataclasses import asdict
import gzip
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import BPD_REWRITE_FAMILY, rewrite_problem


SEED = 84619751
SHAPES = ((2, 4), (2, 5), (3, 5), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6))
SCHEMA = "ppt.bpd-mixed-cost-witness.v1"
BUDGET = 8


def make_world(layout, costs, winner):
    rng = random.Random(digest([SEED, layout, costs]))
    atoms = {"n": [], "p": []}

    def fresh(kind):
        name = f"{kind}{rng.randrange(1 << 40):010x}"
        if name in atoms[kind]:
            raise ValueError("proof variables collided")
        atoms[kind].append(name)
        return name

    def premise(formula):
        name = f"h{len(hypotheses)}"
        hypotheses.append(formula)
        return name

    goal = fresh("p")
    pending = [fresh("p") for _ in costs]
    chains = [[fresh("n") for _ in range(cost + 1)] for cost in costs]
    hypotheses = [f"({chain[0]} = {chain[-1]}) → {prop} → {goal}"
                  for chain, prop in zip(chains, pending, strict=True)]
    scripts, obligations = [], []
    for side, (chain, prop, eq_cost) in enumerate(zip(chains, pending, costs, strict=True)):
        equality = []
        for left, right in zip(chain, chain[1:]):
            name = premise(f"{left} = {right}")
            equality.append(f"rw [{name}]")
        decoy = fresh("p")
        viable_chain = []
        for length in (7 - eq_cost, 8 - eq_cost):
            endpoint = prop if (length == 7 - eq_cost) == (side == winner) else decoy
            path = [*[fresh("p") for _ in range(length - 1)], endpoint]
            rule_names = [premise(path[0])]
            rule_names.extend(premise(f"{a} → {b}") for a, b in zip(path, path[1:]))
            if endpoint == prop:
                viable_chain = [f"apply {name}" for name in reversed(rule_names)]
        scripts.append([f"apply h{side}", *equality, *viable_chain])
        obligations.append({"equality": len(equality), "pending_proposition": len(viable_chain)})

    declarations = [f"({' '.join(sorted(atoms['n']))} : Nat)", f"({' '.join(sorted(atoms['p']))} : Prop)"]
    declarations.extend(f"(h{index} : {type_})" for index, type_ in enumerate(hypotheses))
    cluster = digest([SCHEMA, costs])
    # The rewrite family enables independent compilation for both rw and apply.
    case = SearchCase("loopt_mixed_eq_horn", "dev", BPD_REWRITE_FAMILY, digest([cluster, winner]), cluster,
                      f"mixed_cost_{layout}_{winner}", " ".join(declarations), goal)
    return case, scripts, obligations


def native_prefix(repl, case, script):
    from ppt_gym.live_lean import candidate_menu, public_state, state_input

    repl.reset()
    state = repl.init(case.statement, pp=True)
    checked = []
    for attempt, tactic in enumerate(script[:BUDGET], 1):
        visible = public_state(repl, state)
        if not visible["goals"]:
            raise ValueError("the independent path closed before its recorded proof")
        menu = candidate_menu(repl, state, rewrite=rewrite_problem(visible["goals"][0]) is not None)
        if (attempt == 1 and menu != ("apply h0", "apply h1")) or tactic not in menu:
            raise ValueError("a prespecified tactic was absent from the public grammar")
        packet, before = state_input(repl, state, (), BUDGET - attempt + 1)
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError("Lean rejected a prespecified mixed proof tactic")
        state = reply["state"]
        public_after = public_state(repl, state)
        checked.append({"before": before, "packet": digest(packet), "after": digest(public_after),
                        "menu": list(menu), "tactic": tactic, "pending_goals": len(public_after["goals"])})
    complete = not checked[-1]["pending_goals"]
    if complete:
        kernel = repl.kernel(state["sid"])
        if kernel.get("ok") is not True or kernel.get("axioms") != []:
            raise ValueError("the live mixed proof failed kernel checking")
    return checked, complete


def audit(tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify

    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the mixed-cost witness requires the pinned Lean verifier")
        for layout, lengths in enumerate(SHAPES):
            costs = lengths if layout % 2 == 0 else lengths[::-1]
            for winner in (0, 1):
                case, scripts, obligations = make_world(layout, costs, winner)
                trials = []
                for side, script in enumerate(scripts):
                    events, closed = native_prefix(repl, case, script)
                    replay = native_prefix(repl, case, script)
                    if (len(events) != BUDGET or len(script) != BUDGET + int(side != winner)
                            or closed != (side == winner) or replay != (events, closed)):
                        raise ValueError("mixed proof did not reproduce the intended eight-check decision")
                    certificate = certify(case, {"live_kernel": closed, "accepted_script": script[:BUDGET]})
                    if closed and (not certificate or not certificate["passed"] or certificate["axioms"]):
                        raise ValueError("the verified branch lacks a standalone certificate")
                    trials.append({"first": side, "events": events, "replay_passed": True, "certificate": certificate})
                rows[case.root] = {"case": asdict(case), "winner": winner, "costs": obligations, "trials": trials}
    return {"schema": SCHEMA, "seed": SEED, "lean": "4.19.0", "tactic_checks": BUDGET,
            "source_sha256": sha256_file(Path(__file__)), "rows": rows, "memory": tree.stats()}


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
    print(f"Lean replayed both choices in {len(report['rows'])} mixed-obligation worlds.", flush=True)


if __name__ == "__main__":
    main()
