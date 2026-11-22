"""Verify a witness pair where a local proof can be reused across obligations."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.dependent_witness import FORK, PREFIX, proof_view
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget

from configs.check_ppt_dependent_witness import action_costs


CHECKS = 14
CACHE_NAME = "h30"


def fixture(shared):
    hypotheses = ["P 0", "P 1"]
    paths = {}
    for witness in range(2):
        atoms = {name: f"{name} {witness}" for name in "ABCDEFGHQUV"}

        def fact(proposition, tactic="apply"):
            hypotheses.append(proposition)
            return f"{tactic} h{len(hypotheses) - 1}"

        def chain(nodes):
            seed = fact(atoms[nodes[0]], "exact")
            steps = [fact(f"{atoms[parent]} → {atoms[child]}") for parent, child in zip(nodes, nodes[1:])]
            return (*reversed(steps), seed)

        left, alternate = chain("ABCD"), chain("EFGH")
        to_u = fact(f"{atoms['D']} → {atoms['U']}")
        to_v = fact(f"{atoms['D' if witness == shared else 'H']} → {atoms['V']}")
        split = fact(f"({atoms['U']} ∧ {atoms['V']}) → {atoms['Q']}")
        start = (*PREFIX, FORK[witness], split)
        cached = (f"exact {CACHE_NAME}",)
        right = left if witness == shared else alternate

        def with_cache(atom, proof, lhs, rhs):
            have = f"have {CACHE_NAME} : {atoms[atom]} := ?_"
            return (*start, have, "constructor", to_u, *lhs, to_v, *rhs, *proof)

        paths[witness] = {
            "direct": (*start, "constructor", to_u, *left, to_v, *right),
            "cache_d": with_cache("D", left, cached, cached if witness == shared else right),
            "cache_h": with_cache("H", alternate, left, right if witness == shared else cached),
        }
    names = tuple(f"h{i}" for i in range(len(hypotheses)))
    predicates = "P Q U V A B C D E F G H"
    locals_ = " ".join(f"({name} : {ty})" for name, ty in zip(names, hypotheses, strict=True))
    statement = f"∀ ({predicates} : Nat → Prop) {locals_}, ∃ n, P n ∧ Q n"
    return statement, (*predicates.split(), *names), paths


def replay(repl, statement, tactics):
    repl.reset()
    state = repl.init(statement)
    initial = public_state(repl, state)["goals"]
    events = []
    for tactic in tactics:
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError(f"Lean rejected {tactic}: {reply.get('error')}")
        state = reply["state"]
        public = public_state(repl, state)
        events.append({"action": tactic, **public})
    kernel = repl.kernel(state["sid"])
    if events[-1]["goals"] or not kernel["ok"] or kernel["axioms"]:
        raise ValueError("the reuse path did not close without axioms")
    return initial, events


def check(repl):
    rows, declarations = [], []
    for shared in range(2):
        statement, names, paths = fixture(shared)
        for witness, variants in paths.items():
            for mode, tactics in variants.items():
                tid = f"reuse_{shared}_{witness}_{mode}"
                initial, events = replay(repl, statement, tactics)
                front = proof_view(events[len(PREFIX) - 1]["goals"])
                tree_costs = action_costs(front, FORK, len(PREFIX))
                if tree_costs != [CHECKS + 1, CHECKS + 1]:
                    raise ValueError("the independent-obligation cost exposed the shared witness")
                expected = {
                    "direct": CHECKS + 1,
                    "cache_d": CHECKS if witness == shared else CHECKS + 3,
                    "cache_h": CHECKS + 6 if witness == shared else CHECKS + 3,
                }[mode]
                if len(events) != expected:
                    raise ValueError(f"{tid} took {len(events)} tactics instead of {expected}")
                declarations.append(f"theorem {tid} : {statement} := by\n  intro {' '.join(names)}\n"
                                    + "".join(f"  {tactic}\n" for tactic in tactics)
                                    + f"#print axioms {tid}\n")
                rows.append({"tid": tid, "shared_witness": shared, "witness": witness, "mode": mode,
                             "checks": len(events), "independent_costs": tree_costs,
                             "initial_goals": initial, "events": events})

    certificate = "\n".join(declarations)
    with TemporaryDirectory() as scratch:
        source = Path(scratch) / "WitnessReuse.lean"
        source.write_text(certificate, encoding="utf-8")
        lean = subprocess.run([str(LEAN_EXE), str(source)], check=True, capture_output=True, text=True,
                              encoding="utf-8", timeout=60, env=lean_env())
    axiom_report = [line for line in lean.stdout.splitlines() if line.endswith(" does not depend on any axioms")]
    if set(axiom_report) != {f"'{row['tid']}' does not depend on any axioms" for row in rows}:
        raise ValueError("the standalone verifier did not certify every witness path")
    return {"schema": "ppt.witness-reuse-feasibility.v1", "checks": CHECKS, "lean": repl.version,
            "server_sha256": sha256_file(REPL_EXE), "verifier_sha256": sha256_file(LEAN_EXE),
            "certificate": certificate, "certificate_digest": digest(certificate),
            "axiom_report": axiom_report, "rows": rows}


def opening_scores(checkpoint, report):
    import torch

    from configs.probe_ppt_witness_workspace import ARMS, observe_fork, prefix_carry
    from configs.run_ppt_slot_continuations import read_continuation_fit, restore_policy
    from ppt_gym.ram_campaign import configure

    configure(1, 2201)
    fit = read_continuation_fit(checkpoint, report["lean"])
    if not fit.get("anonymous_witness"):
        raise ValueError("the opening probe requires a binder-invariant checkpoint")
    cases = [row for row in report["rows"] if row["mode"] == "direct" and row["witness"] == 0]
    results = {}
    with torch.no_grad():
        for arm in ARMS:
            policy = restore_policy(fit, arm, fit["weights"][arm])
            rows = []
            for case in cases:
                states = [case["initial_goals"], *(event["goals"] for event in case["events"][:len(PREFIX)])]
                prefix = [{"goals": proof_view(goals), "menu": [action], "action": action}
                          for goals, action in zip(states[:-1], PREFIX, strict=True)]
                fork = {"goals": proof_view(states[-1]), "menu": FORK}
                carry = prefix_carry(policy, [*prefix, fork], CHECKS)
                logits, signals = observe_fork(policy, fork, CHECKS, carry)
                rows.append({"shared_witness": case["shared_witness"], "logits": logits.tolist(),
                             "pending_readout": [signal.tolist() for signal in signals]})
            results[arm] = rows
    return {"checkpoint_sha256": sha256_file(checkpoint), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="record frozen opening scores on both reuse worlds")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the reuse fixture requires the pinned Lean verifier")
            report = check(repl)
        if args.checkpoint:
            report["opening_scores"] = opening_scores(args.checkpoint, report)
        with args.output.open("x", encoding="utf-8") as out:
            out.write(canonical(report) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
