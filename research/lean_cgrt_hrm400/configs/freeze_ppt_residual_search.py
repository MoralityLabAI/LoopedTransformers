"""Collect short native tactic trees at residual states of original Lean proofs."""

import argparse
from pathlib import Path

from configs.freeze_ppt_native_progress import advance, candidate_menu, observe
from configs.freeze_ppt_scope_recovery import PREFIX_MODULE, native_scope, payloads, prefix_text
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import candidate_menu as horn_menu
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.residual-search-native.v1"
FIXED = ("assumption", "rfl", "simp", "omega", "decide", "constructor", "left", "right")
MAX_LOCAL = 5
DEPTH = 2


def menu(repl, state, public):
    try:
        proposed = horn_menu(repl, state)
    except ValueError:
        proposed = ()
    if not proposed:
        locals_ = [action for action in candidate_menu(public) if action.startswith(("apply ", "cases "))]
        proposed = sorted(locals_, key=lambda action: digest([SCHEMA, action]))[:MAX_LOCAL]
    return tuple(dict.fromkeys((*proposed, *FIXED)))


def task_clusters(repl, payload):
    target = payload.target
    proof = payload.proof
    if len(proof) == 1:
        # Keep a residual and a distinct sibling in the one-step MathCan cases.
        target = f"({target}) ∧ True"
        proof = ("constructor", *proof, "trivial")
    statement = f"∀ {payload.binders}, {target}"
    repl.reset()
    state = repl.init(statement, pp=True)
    public = observe(repl, state)
    theorem = {"module": payload.root, "theorem": f"residual_{payload.root[:12]}",
               "source": payload.source, "split": payload.split, "statement": statement,
               "intro_names": [hyp["name"] for hyp in public["goals"][0]["hypotheses"]],
               "reference": list(proof)}
    snapshots, logs = [], []
    for step, action in enumerate(proof):
        snapshots.append((state, public))
        status, state, public = advance(repl, state, public, action)
        logs.append({"action": action, "status": status, "after": digest(public)})
        if status != ("closed" if step == len(proof) - 1 else "progress"):
            raise ValueError("an original proof changed before collecting the residual states")
    if len(snapshots) < 2:
        return theorem, []
    cache = {}

    def collect(state, visible, depth):
        key = state["sid"], depth
        if key in cache:
            return cache[key]
        actions = {}
        result = {"public": visible, "key": digest(visible), "actions": actions}
        if depth:
            for tactic in menu(repl, state, visible):
                status, child, after = advance(repl, state, visible, tactic)
                actions[tactic] = {"status": status, "after": digest(after)}
                if status == "progress" and depth > 1:
                    actions[tactic]["child"] = collect(child, after, depth - 1)
        cache[key] = result
        return result

    tasks = []
    # The final native obligation remains untouched for the first target case.
    # Adjacent but distinct earlier and later residuals supply movable evidence.
    for target in sorted({0, max(0, len(snapshots) - 3), len(snapshots) - 2}):
        support = target + 1
        a, b = snapshots[target], snapshots[support]
        if digest(a[1]) == digest(b[1]):
            continue
        tasks.append({"case": digest([payload.root, target, support]), "target": collect(*a, DEPTH),
                      "sibling": collect(*b, 1), "target_prefix": list(proof[:target]),
                      "sibling_prefix": list(proof[:support])})
    return {**theorem, "reference_trace": logs}, tasks


def freeze(args, tree):
    archive = read_receipt(args.archive)
    prelude = prefix_text(args.mathcan)
    if (archive["prelude_sha256"] != digest(prelude) or archive["verifier_sha256"] != sha256_file(LEAN_EXE)
            or archive["server_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the archived native payload or verifier pins changed")

    def collect(directory, env):
        rows = []
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native verifier")
            for payload in payloads(archive):
                theorem, tasks = task_clusters(repl, payload)
                rows.append({**theorem, "tasks": tasks})
                print(payload.split, payload.source, len(tasks), flush=True)
        certificates = verify_independent(directory, env, (PREFIX_MODULE,),
                                          [(row, row["reference"]) for row in rows])
        return rows, certificates

    rows, verified = native_scope(prelude, collect)
    write_receipt(args.bank, {"schema": SCHEMA, "archive": sha256_file(args.archive), "prelude": digest(prelude),
                             "lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE),
                             "rows": rows, "independent": verified, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "mathcan", "bank"):
        parser.add_argument(name, type=Path)
    tree = TreeBudget(2, 1)
    try:
        freeze(parser.parse_args(), tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
