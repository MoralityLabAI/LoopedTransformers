"""Verify short-history training and longer-history native proof layouts."""

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
from itertools import product
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget

from configs.check_ppt_mathcan_native_costs import PREFIX_MODULE, invoke_lean
from configs.freeze_ppt_rollback_diagnostics import REVEAL, certify_batches, mathcan_payloads, native_payloads
from configs.run_ppt_two_slot import request_state


SCHEMA = "ppt.rollback-length-transfer.v1"
OPERATORS = ("xor", "xnor", "and", "or")
ALIASES = ("diagA", "diagB", "diagC", "diagD", "diagE", "diagF")


def expression(arity, operator):
    names = ALIASES[:arity]
    if operator in ("xor", "xnor"):
        parity = f"({' + '.join(names)}) % 2"
        return parity if operator == "xor" else f"1 - ({parity})"
    if operator == "and":
        return " * ".join(names)
    term = names[-1]
    for name in reversed(names[:-1]):
        term = f"max {name} {term}" if term == names[-1] else f"max {name} ({term})"
    return term


def value(bits, operator):
    if operator in ("xor", "xnor"):
        return sum(bits) % 2 ^ int(operator == "xnor")
    return int(all(bits)) if operator == "and" else int(any(bits))


def layout_catalog(args):
    base = (args.cohort, args.audit, args.ledger)
    train = list(native_payloads(*base, source_counts={"bpd_lg02": (4, 0), "loopt_horn": (4, 0),
                                                       "evalmath_lg01": (2, 0)}))
    math, prelude = mathcan_payloads(args.mathcan_archive, args.mathcan_source)
    train.extend(replace(payload, split="train") for payload in math)
    dev = list(native_payloads(*base, source_counts={"bpd_lg02": (0, 8), "loopt_horn": (0, 8)},
                               skip={"bpd_lg02": 14, "loopt_horn": 14}))
    if any(len(payload.proof) < 3 for payload in dev):
        raise ValueError("a fresh diagnostic root would allow an early witness retry")
    if {row.root for row in train} & {row.root for row in dev}:
        raise ValueError("the long-history payloads overlap training")
    layouts = []
    for i, payload in enumerate(train):
        for operator in (OPERATORS[i % 2], OPERATORS[2 + (i // 2) % 2]):
            layouts.extend((payload, length, operator) for length in (2, 3))
    for i, payload in enumerate(dev):
        # Cross each operator with both horizons within each native source.
        layouts.append((payload, 4 + (i % 2), OPERATORS[(i % 8) // 2]))
    return layouts, prelude


def acquisition_catalog(args):
    base = (args.cohort, args.audit, args.ledger)
    payloads = list(native_payloads(*base, source_counts={"bpd_lg02": (0, 5), "loopt_horn": (0, 1)},
                                    skip={"bpd_lg02": 22, "loopt_horn": 22}))
    _, prelude = mathcan_payloads(args.mathcan_archive, args.mathcan_source)
    layouts = []
    for index, payload in enumerate(payloads):
        arity = 5 + (index % 2)
        if len(payload.proof) + 3 <= arity:
            raise ValueError("a new acquisition layout would permit a witness retry")
        for operator in (OPERATORS[(index // 2) % 2], OPERATORS[2 + (index + index // 2) % 2]):
            layouts.append((payload, arity, operator))
    return layouts, prelude


def diagnostic(repl, root, alias):
    action = f"have hProbe : {alias} = 2 := by decide"
    response = repl.run(root["sid"], action)
    match = REVEAL.search(response.get("error", ""))
    if response["ok"] or "state" in response or not match:
        raise ValueError("the native rejection did not reveal one diagnostic value")
    restored = request_state(repl.focus(root["sid"], 0))
    if public_state(repl, root) != public_state(repl, restored):
        raise ValueError("the restored proof observation changed")
    return restored, {"action": action, "error": response["error"], "value": int(match[1])}


def native_commit(repl, root, payload_proof, witness, expected_axioms):
    actions = (f"refine ⟨{witness}, ?_⟩", "constructor", *payload_proof, "decide")
    state = root
    for index, action in enumerate(actions, 1):
        reply = repl.run(state["sid"], action)
        if not reply["ok"]:
            if index != len(actions):
                raise ValueError(f"the original source tactic {action} failed: {reply['error']}")
            return {"passed": False, "checks": index, "error": reply["error"]}
        state = reply["state"]
    proof = repl.kernel(state["sid"])
    if state["goals"] or not proof["ok"] or sorted(proof["axioms"]) != sorted(expected_axioms):
        raise ValueError("the witness proof altered the certified source obligations or axioms")
    return {"passed": True, "checks": len(actions), "axioms": sorted(proof["axioms"]), "actions": actions}


def layout(repl, payload, arity, operator):
    cluster = digest([SCHEMA, payload.root, arity, operator])
    names = ALIASES[:arity]
    formula = expression(arity, operator)
    rows, starting_view = [], None
    for bits in product((0, 1), repeat=arity):
        lets = "".join(f"let {name} : Nat := {bit}; " for name, bit in zip(names, bits, strict=True))
        statement = f"∀ {payload.binders}, {lets}∃ n : Nat, ({payload.target}) ∧ n = {formula}"
        repl.reset()
        root = repl.init(statement)
        visible = public_state(repl, root)
        if starting_view is None:
            starting_view = visible
        elif visible != starting_view:
            raise ValueError("a hidden valuation changed the initial public proof state")
        observations = []
        for name, bit in zip(names, bits, strict=True):
            root, observation = diagnostic(repl, root, name)
            if observation["value"] != bit:
                raise ValueError("Lean's own diagnostic disagreed with the generated valuation")
            observations.append(observation)
        trials = [native_commit(repl, root, payload.proof, witness, payload.axioms) for witness in (0, 1)]
        correct = value(bits, operator)
        if [row["passed"] for row in trials] != [correct == witness for witness in (0, 1)]:
            raise ValueError("the independent Boolean oracle disagreed with the Lean witness")
        if trials[1 - correct]["checks"] <= arity:
            raise ValueError("a wrong guess could fall back within the probe allowance")
        tid = f"length_{cluster[:12]}_{''.join(map(str, bits))}"
        proof = trials[correct]
        intro = " ".join((*payload.names, *names))
        source = (f"theorem {tid} : {statement} := by\n  intro {intro}\n"
                  + "".join(f"  {action}\n" for action in proof["actions"]) + f"#print axioms {tid}\n")
        rows.append({"tid": tid, "cluster": cluster, "payload_root": payload.root, "source": payload.source,
                     "split": payload.split, "arity": arity, "operator": operator, "public": visible,
                     "observations": observations, "witness_trials": trials, "winning_witness": correct,
                     "budget": proof["checks"] + arity, "certificate_source": source})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "audit", "ledger", "mathcan_archive", "mathcan_source", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--acquisition-transfer", action="store_true")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        layouts, prelude = acquisition_catalog(args) if args.acquisition_transfer else layout_catalog(args)
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            prefix = directory / f"{PREFIX_MODULE}.lean"
            prefix.write_text(prelude, encoding="utf-8")
            env = lean_env()
            invoke_lean(directory, env, "-o", f"{PREFIX_MODULE}.olean", prefix.name)
            env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
            with LeanRepl(imports=(PREFIX_MODULE,), cwd=directory, env=env) as repl:
                if repl.version != LEAN_VERSION:
                    raise ValueError("the length transfer requires the pinned native Lean")
                rows = [row for payload, arity, op in layouts for row in layout(repl, payload, arity, op)]
            certify_batches(rows, directory, env)
        inputs = {name: sha256_file(getattr(args, name))
                  for name in ("cohort", "audit", "ledger", "mathcan_archive", "mathcan_source")}
        totals = defaultdict(int)
        for row in rows:
            totals[f"{row['split']}/{row['arity']}/{row['operator']}"] += 1
        report = {"schema": SCHEMA, "lean": LEAN_VERSION, "server_sha256": sha256_file(REPL_EXE),
                  "verifier_sha256": sha256_file(LEAN_EXE), "source_hashes": inputs,
                  "prelude_sha256": digest(prelude), "counts": dict(totals), "rows": rows, "memory": tree.stats()}
        if args.acquisition_transfer:
            report["study"] = "acquisition-transfer"
        with args.output.open("xb") as handle:
            handle.write(gzip.compress((canonical(report) + "\n").encode(), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
