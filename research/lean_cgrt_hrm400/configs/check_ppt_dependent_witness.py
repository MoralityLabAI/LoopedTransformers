"""Replay and independently certify Lean proofs with a shared witness."""

import argparse
from dataclasses import asdict
from pathlib import Path

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.dependent_witness import examples, script
from ppt_gym.hybrid_rl.__main__ import certify
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget


def verify_pair(repl):
    rows = []
    for case, winner in examples():
        branches = []
        for branch in range(2):
            repl.reset()
            state = repl.init(case.statement)
            events = []
            tactics = script(winner, branch)
            for tactic in tactics:
                reply = repl.run(state["sid"], tactic)
                if not reply["ok"]:
                    raise ValueError(f"Lean rejected a shared-witness tactic: {reply.get('error')}")
                state = reply["state"]
                observation = public_state(repl, state)
                events.append({"tactic": tactic, "state": observation, "state_sha256": digest(observation)})
            before = events[1]["state"]["goals"]
            after = events[2]["state"]["goals"]
            if (len(before) != 3 or before[2]["target"] != {"raw": "Nat"}
                    or before[0]["target"].get("raw", "")[:2] != "P "
                    or before[1]["target"].get("raw", "")[:2] != "Q "
                    or len(after) != 1 or after[0]["target"] != {"raw": f"Q {branch}"}):
                raise ValueError("the first proof did not assign the witness and specialize its sibling")
            metaterm = before[0]["target"]["raw"][2:]
            if not metaterm.startswith("?") or before[1]["target"]["raw"][2:] != metaterm:
                raise ValueError("the pending goals did not share one unresolved Lean metavariable")
            if bool(events[3]["state"]["goals"]) == (branch == winner):
                raise ValueError("the dependent branch disagreed with its four-check deadline")
            kernel = repl.kernel(state["sid"])
            if not kernel["ok"] or kernel["axioms"]:
                raise ValueError("the dependent witness proof failed its live kernel check")
            certificate = certify(case, {"live_kernel": True, "accepted_script": tactics}, timeout=60)
            if not certificate or not certificate["passed"] or certificate["axioms"]:
                raise ValueError("the dependent witness proof failed standalone Lean certification")
            branches.append({"choice": branch, "checks": len(tactics), "events": events, "certificate": certificate})
        if branches[0]["events"][:2] != branches[1]["events"][:2]:
            raise ValueError("dependent branches lost their shared native prefix")
        rows.append({"case": asdict(case), "winner": winner, "branches": branches})
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the dependent witness requires the pinned Lean verifier")
            result = {"schema": "ppt.shared-existential-witness.v1", "lean": repl.version, "rows": verify_pair(repl),
                      "server_sha256": sha256_file(REPL_EXE), "verifier_sha256": sha256_file(LEAN_EXE)}
        with args.output.open("x", encoding="utf-8") as target:
            target.write(canonical(result) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
