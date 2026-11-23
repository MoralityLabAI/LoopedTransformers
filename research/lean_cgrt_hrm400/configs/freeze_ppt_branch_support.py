"""Witness a session's paid scope probes in Lean before selecting a proof branch."""

import argparse
from itertools import product
from pathlib import Path

from configs.freeze_ppt_scope_recovery import PREFIX_MODULE, native_scope, prefix_text
from configs.run_ppt_rollback_diagnostics import write_receipt
from configs.run_ppt_two_slot import request_state
from lcgrt.config import LEAN_EXE, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.branch-support-native.v1"
ALIASES = ("scopeRouteA", "scopeRouteB", "scopeRouteC")
KEYS = tuple(range(1, 8))


def statement(bits):
    bindings = "".join(f"let {name} : Nat := {bit}; " for name, bit in zip(ALIASES, bits, strict=True))
    return f"∀ (_ready : True), {bindings}True"


def expression(key):
    terms = [name for index, name in enumerate(ALIASES) if key & (1 << index)]
    if not terms or key not in KEYS:
        raise ValueError("a native support key must select at least one registered address")
    return f"(({' + '.join(terms)}) % 2)"


def witness(repl, state, key, asserted=0):
    public = public_state(repl, state)
    action = f"have hScopeRoute : {expression(key)} = {asserted} := by decide"
    response = repl.run(state["sid"], action)
    restored = request_state(repl.focus(state["sid"], 0))
    if public_state(repl, restored) != public:
        raise ValueError("a support observation changed the restored proof state")
    return restored, {"action": action, "accepted": response["ok"],
                      "value": asserted if response["ok"] else 1 - asserted}


def freeze(args, tree):
    prelude = prefix_text(args.mathcan)

    def collect(directory, env):
        records, shared = [], None
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            for bits in product((0, 1), repeat=len(ALIASES)):
                repl.reset()
                source = statement(bits)
                state = repl.init(source)
                public = public_state(repl, state)
                if shared is None:
                    shared = public
                elif shared != public:
                    raise ValueError("native support exposed a private route bit in the initial observation")
                evidence = []
                for key in KEYS:
                    state, result = witness(repl, state, key)
                    state, opposite = witness(repl, state, key, asserted=1)
                    expected = sum(bit for index, bit in enumerate(bits) if key & (1 << index)) % 2
                    if result["value"] != expected or opposite["value"] != expected:
                        raise ValueError("Lean contradicted the independent branch-scope oracle")
                    evidence.append(result)
                records.append({"source": source, "observations": evidence})
        return shared, records

    public, worlds = native_scope(prelude, collect)
    write_receipt(args.bank, {"schema": SCHEMA, "lean": sha256_file(LEAN_EXE),
                              "repl": sha256_file(REPL_EXE), "prelude": digest(prelude),
                              "public": public, "worlds": worlds, "memory": tree.stats()})
    print("witnessed", sum(len(row["observations"]) for row in worlds), "native scope routing observations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mathcan", type=Path)
    parser.add_argument("bank", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        freeze(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
