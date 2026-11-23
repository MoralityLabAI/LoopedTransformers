"""Certify native guard reordering, probing, and recovery transitions."""

import argparse
from pathlib import Path

from configs.freeze_ppt_computational_guards import start
from configs.freeze_ppt_native_progress import advance, native_sources, observe
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.computational_guards import GUARD_CHECK, OPEN, PREFIX
from ppt_gym.data import digest, sha256_file
from ppt_gym.guard_recovery import SCHEMA, cases
from ppt_gym.ram_limits import TreeBudget


def verify_case(repl, case):
    root, public = start(repl, case)
    visible_sha = digest(public)
    introductions = [hyp["name"] for hyp in public["goals"][0]["hypotheses"]]
    for tactic in PREFIX:
        status, root, public = advance(repl, root, public, tactic)
        if status != "progress":
            raise ValueError("native guard recovery lost its witness fork")
    truth = []
    for group in case["guards"]:
        for claim in group:
            status, _, _ = advance(repl, root, public, f"have pptProbe : {claim} := by {GUARD_CHECK}")
            if status not in ("progress", "rejected"):
                raise ValueError("a native probe was resource-limited")
            truth.append(status == "progress")
    if [i for i, good in enumerate(truth) if not good] != [case["false_guard"]]:
        raise ValueError("the native probe disagreed with the independent computational fold")

    routes, prefixes = {}, {}
    for branch, opening in enumerate(OPEN):
        status, state, after = advance(repl, root, public, opening)
        if status != "progress" or len(after["goals"]) != 3:
            raise ValueError("a native branch did not open its two guards and pending witness goal")
        prefixes[branch] = after
        for ordering in ((0, 1), (1, 0)):
            current, visible = state, after
            remaining = [0, 1, 2]
            script, trace = [*PREFIX, opening], []
            for slot in ordering:
                position = remaining.index(slot)
                if position:
                    focused = repl.focus(current["sid"], position)
                    if not focused["ok"]:
                        raise ValueError("Lean refused to focus a registered side condition")
                    current = focused["state"]
                    selected = visible["goals"][position]["target"]
                    visible = observe(repl, current)
                    if visible["goals"][0]["target"] != selected:
                        raise ValueError("Lean focused the wrong side condition")
                status, current, visible = advance(repl, current, visible, GUARD_CHECK)
                expected = "progress" if truth[2 * branch + slot] else "rejected"
                if status != expected:
                    raise ValueError(f"native focus changed the selected guard: {status} expected {expected}")
                trace.append({"slot": slot, "focus": position, "action": GUARD_CHECK, "status": status})
                if status == "rejected":
                    break
                # REPL focus moves one goal to the front, while the standalone
                # script rotates past the pending witness goal after the first check.
                if ordering[0] == 1:
                    script.append("rotate_left 1")
                script.append(GUARD_CHECK)
                remaining.pop(position)
            closed = len(trace) == 2 and trace[-1]["status"] == "progress"
            if closed:
                finish = f"exact h{branch + 2}"
                status, _, _ = advance(repl, current, visible, finish)
                if status != "closed":
                    raise ValueError("changing guard order lost the instantiated pending witness")
                script.append(finish)
            routes[f"{branch}_{ordering[0]}"] = {"trace": trace, "proof": script if closed else []}
    result = {key: val for key, val in case.items() if key != "false_guard"}
    return {**result, "visible_sha256": visible_sha, "intro_names": introductions,
            "fork": public, "branch_states": prefixes, "truth": truth, "routes": routes}


def certify_routes(directory, env, imports, certificates):
    batches = [verify_independent(directory, env, imports, certificates[offset:offset + 64])
               for offset in range(0, len(certificates), 64)]
    return {"certificates": sum(batch["certificates"] for batch in batches), "batches": batches}


def run(args, tree):
    paths = args.bpd, args.mathcan
    pools, certificates = {}, []
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean version")
            for partition, count in (("source", 48), ("development", 40)):
                pool = [verify_case(repl, case) for case in cases(partition, count)]
                pools[partition] = pool
                certificates.extend((case, route["proof"]) for case in pool for route in case["routes"].values() if route["proof"])
                print(partition, "native worlds", len(pool), flush=True)
        independent = certify_routes(directory, env, imports, certificates)
    write_receipt(args.bank, {"schema": SCHEMA, **pools, "independent": independent, "memory": tree.stats(),
                             "lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE),
                             "imports": {path.name: sha256_file(path) for path in paths}})
    print("independently certified", independent["certificates"], "guard-order scripts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bpd", "mathcan", "bank"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
