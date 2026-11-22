"""Verify an eight-check proof gate that requires opening the right bundle."""

import argparse
import gzip
import json
from pathlib import Path
import random
import subprocess
import tempfile

from configs.check_ppt_bpd_rewrite import EqualityReference
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


SHAPES = ((1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (4, 6))
SEED = 123510831
GOAL = "target"
START = "start"


def layout(shape, number, rng):
    names = {role: f"v{rng.randrange(1 << 44):011x}" for role in (START, GOAL, *(
        f"{side}_{node}" for side in (0, 1)
        for node in (*range(1, 7), "dead", "spur0", "spur1")
    ))}
    if len(set(names.values())) != len(names):
        raise ValueError("bundle witness atom names collided")
    variables = sorted(names.values())
    parent_menu = []
    graphs = []
    for side in (0, 1):
        spine = [START, *(f"{side}_{step}" for step in range(1, 7))]
        arcs = [*zip(spine, spine[1:]), (spine[-1], GOAL)]
        arcs.extend((spine[index], f"{side}_spur{spur}") for spur, index in enumerate(shape))
        order = rng.sample(range(len(arcs)), len(arcs))
        binders = ", ".join(f"r{100 + 100 * side + i}" for i in range(len(arcs)))
        parent_menu.append(f"rcases h{side} with ⟨{binders}⟩")
        graphs.append((arcs, order))
    for winner in (0, 1):
        bundles = []
        for side, (arcs, order) in enumerate(graphs):
            premises = [f"{names[left]} = {names[right if side == winner or right != GOAL else f'{side}_dead']}"
                        for left, right in (arcs[i] for i in order)]
            nested = premises[-1]
            for premise in reversed(premises[:-1]):
                nested = f"({premise}) ∧ ({nested})"
            bundles.append(f"(h{side} : {nested})")
        tid = f"bpd_bundled_opening_{number}_{winner}"
        statement = f"∀ ({' '.join(variables)} : Nat) {' '.join(bundles)}, {names[START]} = {names[GOAL]}"
        yield {"name": tid, "statement": statement, "winner": winner, "menu": parent_menu, "variables": variables}


def replay(repl, statement, script):
    repl.reset()
    state = repl.init(statement, pp=True)
    for tactic in script:
        step = repl.run(state["sid"], tactic)
        if not step["ok"]:
            raise ValueError("an original bundle operation did not replay")
        state = step["state"]
    if state["closed"]:
        kernel = repl.kernel(state["sid"])
        if not kernel["ok"] or kernel["axioms"]:
            raise ValueError("the bundle proof did not pass the kernel")
    return state


def proof(repl, world, first):
    from ppt_gym.live_lean import candidate_menu, public_state, state_input

    statement, winner = world["statement"], world["winner"]
    openings = [world["menu"][first]]
    if first != winner:
        openings.append(world["menu"][winner])
    state = replay(repl, statement, openings)
    if state["closed"] or len(public_state(repl, state)["goals"]) != 1:
        raise ValueError("the bundle openings must expose facts while retaining the goal")
    script = list(openings)
    reference = EqualityReference()
    while len(script) < 8 and not state["closed"]:
        menu = candidate_menu(repl, state, rewrite=True)
        packet, _ = state_input(repl, state, (), 8 - len(script))
        scores, _ = reference(packet, menu, 4)
        tactic = menu[int(scores.argmax())]
        step = repl.run(state["sid"], tactic)
        if not step["ok"]:
            raise ValueError("the known rewrite route was rejected")
        script.append(tactic)
        state = step["state"]
    repeated = replay(repl, statement, script)
    if len(script) != 8 or repeated["closed"] != state["closed"] or state["closed"] != (first == winner):
        raise ValueError("the bundle pair did not distinguish the only viable first opening")
    return script, state["closed"]


def compile_proof(world, script, file):
    from lcgrt import config

    name = world["name"]
    variables = " ".join(world["variables"])
    source = (f"theorem {name} : {world['statement']} := by\n  intro {variables} h0 h1\n"
              + "".join(f"  {step}\n" for step in script) + f"#print axioms {name}\n")
    file.write_text(source, encoding="utf-8")
    proc = subprocess.run([str(config.LEAN_EXE), str(file)], env=config.lean_env(),
                          capture_output=True, text=True, encoding="utf-8", timeout=15)
    if proc.returncode != 0 or f"'{name}' does not depend on any axioms" not in proc.stdout.splitlines():
        raise ValueError("the successful bundle opening lacks an independent axiom-free proof")
    return {"source": source, "source_sha256": digest(source), "axioms": []}


def audit(tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    rng = random.Random(SEED)
    pairs = {}
    with LeanRepl(timeout=15) as repl, tempfile.TemporaryDirectory() as scratch:
        if repl.version != "4.19.0":
            raise ValueError("the bundle witness requires pinned native Lean 4.19")
        path = Path(scratch) / "Witness.lean"
        for index, shape in enumerate(SHAPES):
            rows = []
            for world in layout(shape, index, rng):
                outcomes = []
                for first in (0, 1):
                    start = tree.cpu_seconds()
                    script, certified = proof(repl, world, first)
                    if tree.cpu_seconds() - start > 6:
                        raise ValueError("bundle witness exceeded the search CPU budget")
                    outcomes.append({"first": first, "checks": len(script), "script": script,
                                     "replay_passed": True, "certificate": compile_proof(world, script, path) if certified else None})
                rows.append({**world, "outcomes": outcomes})
            pairs[digest(["bpd-bundled-opening-witness", shape])] = rows
    return {"scope": "synthetic BPD-style paired local bundle opening feasibility", "lean": "4.19.0",
            "generator_sha256": sha256_file(Path(__file__)), "seed": SEED, "pairs": pairs, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(tree)
        payload = gzip.compress(canonical(report).encode("utf-8"), mtime=0)
        with args.output.open("xb") as out:
            out.write(payload)
    finally:
        tree.close()
    print("Native Lean verified one viable opening in every world of", len(report["pairs"]), "bundle pairs.")


if __name__ == "__main__":
    main()
