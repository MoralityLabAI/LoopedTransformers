"""Freeze paired witness forks validated by original Lean computation definitions."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from configs.freeze_ppt_live_quote_fragments import collect_native
from configs.freeze_ppt_native_progress import advance, native_sources, observe
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.computational_guards import COMPLETION, OPEN, PREFIX, SCHEMA, cases, menu
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_quote_fragments import NativeFragments, NativeQuoteBackup
from ppt_gym.live_quote_targets import suffix_returns, witness_prefixes
from ppt_gym.private_quotes import QUOTES
from ppt_gym.ram_limits import TreeBudget


def start(repl, case):
    repl.reset()
    state = repl.init(case["statement"], pp=True)
    if "sid" not in state:
        raise ValueError(f"a computational guard proposition failed to elaborate: {state}")
    return state, observe(repl, state)


def type_menu(repl, state):
    return menu(observe(repl, state))


def fork_witness(repl, case):
    state, public = start(repl, case)
    root = digest(public)
    names = [hyp["name"] for hyp in public["goals"][0]["hypotheses"]]
    for move in PREFIX:
        status, state, public = advance(repl, state, public, move)
        if status != "progress":
            raise ValueError("the dependent existential setup changed")
    if menu(public) != OPEN:
        raise ValueError("the public witness menu changed")
    arms = []
    for index, move in enumerate(OPEN):
        status, branch, child = advance(repl, state, public, move)
        if status != "progress":
            raise ValueError("a branch no longer makes immediate native progress")
        goals = [goal["target"] for goal in child["goals"]]
        receipts = [{"action": move, "status": status}]
        script = [*PREFIX, move]
        for guard in range(case["guards"]):
            if menu(child) != ("decide",):
                raise ValueError("an opening did not create its declared computational conditions")
            status, branch, child = advance(repl, branch, child, "decide")
            receipts.append({"action": "decide", "status": status, "guard": guard})
            expected = "rejected" if guard == case["guards"] - 1 and index != case["kernel_winner"] else "progress"
            if status != expected:
                raise ValueError(f"Lean disagreed with the folded guard oracle: {case['family']} {receipts}")
            if status == "rejected":
                break
            script.append("decide")
        if index == case["kernel_winner"]:
            for j, move in enumerate(COMPLETION[index]):
                status, branch, child = advance(repl, branch, child, move)
                receipts.append({"action": move, "status": status})
                if status != ("progress" if j == 0 else "closed"):
                    raise ValueError("the shared witness did not discharge its second dependent slot")
                script.append(move)
            if tuple(script) != case["reference"]:
                raise ValueError("the native trace changed its reference")
        arms.append({"branch": index, "successor_goals": goals, "trace": receipts,
                     "proof": list(script) if index == case["kernel_winner"] else []})
    return {**case, "intro_names": names, "visible_sha256": root}, arms


def paid_control(repl, case, probe):
    state, public = start(repl, case)
    transcript, script = [], []

    def attempt(current, visible, move):
        outcome, following, after = advance(repl, current, visible, move)
        transcript.append({"action": move, "status": outcome})
        return outcome, following, after

    for tactic in PREFIX:
        status, state, public = attempt(state, public, tactic)
        if status != "progress":
            raise ValueError("the control witness prefix changed")
        script.append(tactic)
    selected = 0
    if probe and case["guards"] == 2:
        tactic = f"have pptGuard : {case['test_claims'][0]} := by decide"
        status, state, public = attempt(state, public, tactic)
        if status == "progress":
            script.append(tactic)
        elif status == "rejected":
            selected = 1
        else:
            raise ValueError("a paid guard normalizer yielded an unsupported result")
    for move in (OPEN[selected], *(("decide",) * case["guards"]), *COMPLETION[selected]):
        if len(transcript) == 8:
            break
        status, state, public = attempt(state, public, move)
        if status not in ("closed", "progress"):
            break
        script.append(move)
        if status == "closed":
            break
    proved = transcript[-1]["status"] == "closed"
    return {"branch": selected, "checks": len(transcript), "proved": proved,
            "script": script, "trace": transcript}


def source_targets(log):
    graph = NativeFragments(log)
    paths, suffixes = witness_prefixes(log)
    actions = {move for edges in graph.edges.values() for move in edges}
    billing = [{"prices": dict.fromkeys(actions, 1), "bits": dict.fromkeys(QUOTES, 0)}]
    backup = NativeQuoteBackup(graph, dict.fromkeys(QUOTES, 1), billing)
    examples = []
    for key, path in paths.items():
        remaining = 8 - len(path)
        if remaining <= 0 or not graph.states[key]["menu"]:
            continue
        row = {"state": key, "path": path, "remaining": remaining, "labels": {}}
        for move in graph.states[key]["menu"]:
            row["labels"][move] = {"bellman": backup.action(key, remaining, remaining, (0,), 0, 0, move),
                                   "suffix": suffix_returns(backup, suffixes[key], (0,), remaining, remaining, move)}
        examples.append(row)
    initial = backup.solve(0, 8, 0)
    backup.action.cache_clear()
    backup.value.cache_clear()
    return examples, initial


def run(args, tree):
    paths = (args.bpd, args.mathcan)
    source, held, witnesses, controls, certificates, refutations = [], [], [], defaultdict(list), {}, {}
    coverage = Counter()
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean version")
            for partition, count in (("source", 48), ("development", 40)):
                for raw in cases(partition, count):
                    case, arms = fork_witness(repl, raw)
                    certificates[case["module"], raw["reference"]] = case
                    false_guard = case["test_claims"][1 - case["kernel_winner"]]
                    refutations.setdefault(false_guard, {"statement": f"¬ ({false_guard})", "intro_names": []})
                    if partition == "source":
                        log = collect_native(repl, case, 28, menu_for=type_menu)
                        labels, joined = source_targets(log)
                        safe = {key: value for key, value in case.items() if key not in ("kernel_winner", "reference", "test_claims")}
                        source.append({**safe, "native": log, "targets": labels})
                        for row in labels:
                            for target in row["labels"].values():
                                coverage.update(actions=1, exact=target["bellman"][0] == target["bellman"][1],
                                                suffix_known=target["suffix"]["mean"] is not None)
                        coverage.update(stitched=joined["proof"])
                        if joined["proof"]:
                            certificates[case["module"], tuple(joined["script"])] = case
                    else:
                        held.append(case)
                        witnesses.append({"module": case["module"], "fork": arms})
                        for name, probe in (("immediate_progress", False), ("paid_native_probe", True)):
                            row = paid_control(repl, case, probe)
                            controls[name].append({"module": case["module"], "pair": case["pair"],
                                                   "family": case["family"], "guards": case["guards"], **row})
                            if row["proved"]:
                                certificates[case["module"], tuple(row["script"])] = case
        independent = verify_independent(directory, env, imports, [(case, key[1]) for key, case in certificates.items()])
        counterproofs = verify_independent(directory, env, imports, [(case, ("decide",)) for case in refutations.values()])
    write_receipt(args.bank, {"schema": SCHEMA, "source": source, "held": held, "source_coverage": coverage,
                             "lean_sha256": sha256_file(LEAN_EXE), "repl_sha256": sha256_file(REPL_EXE),
                             "imports": {path.name: sha256_file(path) for path in paths}, "memory": tree.stats()})
    write_receipt(args.audit, {"schema": SCHEMA, "bank_sha256": sha256_file(args.bank), "witnesses": witnesses,
                               "controls": controls, "independent": independent, "guard_refutations": counterproofs,
                               "memory": tree.stats()})
    print("source", dict(coverage), "compiled", independent["certificates"], flush=True)
    for arm, rows in controls.items():
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["pair"]].append(row["proved"])
        print(arm, sum(row["proved"] for row in rows), "/", len(rows),
              "complete pairs", sum(all(values) for values in pairs.values()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bpd", "mathcan", "bank", "audit"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
