"""Witness every paid query in the sealed residue-recovery theorems."""

import argparse
from pathlib import Path

from configs.freeze_ppt_failed_prefix import setup
from configs.freeze_ppt_native_progress import advance, native_sources
from configs.run_ppt_residue_recovery import locked
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.residue_acquisition import queries


SCHEMA = "ppt.residue-acquisition-native.v1"


def witness(repl, pair, menu):
    _, saved, _ = setup(repl, pair["second"])
    if saved[1] != pair["second"]["saved"]:
        raise ValueError("the saved proof state changed")
    results = []
    for query in menu:
        status, _, _ = advance(repl, *saved, query["tactic"])
        if status not in ("progress", "rejected"):
            raise ValueError("the diagnostic did not return a check result")
        bit = status == "progress"
        if bit != (pair["residue"] in query["members"]):
            raise ValueError("the paid check disagrees with the independent residue certificate")
        results.append(bit)
    return results


def freeze(args, tree):
    bank, pins = locked(args)
    partitions = {}
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the native query toolchain changed")
            for name in ("source", "development"):
                rows = []
                for pair in bank[name]:
                    menu = queries(pair)
                    rows.append({"pair": pair["pair"], "menu": menu, "outcomes": witness(repl, pair, menu)})
                partitions[name] = rows
                print("Lean checked", sum(len(row["menu"]) for row in rows), name, "observations", flush=True)
    write_receipt(args.native, {"schema": SCHEMA, "bank": pins["bank"], **partitions, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "native"):
        parser.add_argument(name, type=Path)
    tree = TreeBudget(2, 1)
    try:
        freeze(parser.parse_args(), tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
