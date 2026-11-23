"""Kernel-check both native proof menus and the underlying residue labels."""

import argparse
from pathlib import Path

from configs.freeze_ppt_failed_prefix import setup, verify_case
from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import advance, native_sources
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import GUARD_CHECK
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.residue_recovery import SCHEMA, cases


COUNTS = {"source": 48, "development": 40}


def certify_pair(repl, pair):
    source, target = verify_case(repl, pair["first"]), verify_case(repl, pair["second"])
    fork, _, _ = setup(repl, pair["first"])
    check = f"have hResidue : {pair['certificate']} := by {GUARD_CHECK}"
    state, _, _ = advance(repl, *fork, check)
    if state != "progress":
        raise ValueError("the independent source residue certificate failed")
    return {**pair, "first": source, "second": target, "certified_residue": True}


def freeze(args, tree):
    paths = (args.bpd, args.mathcan)
    bank, successful = {}, []
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected Lean residue-bank toolchain")
            for name, count in COUNTS.items():
                bank[name] = [certify_pair(repl, pair) for pair in cases(name, count)]
                for pair in bank[name]:
                    for trial in (pair["first"], pair["second"]):
                        successful.extend((trial, route["proof"]) for route in trial["routes"] if route["proof"])
                print("certified", len(bank[name]), "native residue pairs for", name, flush=True)
        independent = certify_routes(directory, env, imports, successful)
    write_receipt(args.bank, {"schema": SCHEMA, **bank, "independent": independent, "memory": tree.stats(),
                              "imports": {path.name: sha256_file(path) for path in paths},
                              "lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bpd", "mathcan", "bank"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        freeze(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
