"""Seal native deep-source guard roots with an independent generator namespace."""

import argparse
from collections import Counter
from pathlib import Path

from configs.freeze_ppt_guard_recovery import certify_routes, verify_case
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.guard_recovery import SCHEMA as BANK_SCHEMA
from ppt_gym.guard_recovery import cases
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-deep-source.v1"
NAMESPACE = "warm-probe-source-1126"
PAIRS = 48


def guard_expressions(rows):
    return {guard.split(" = ")[0] for row in rows for witness in row["guards"] for guard in witness}


def audit_cohort(new, original):
    if len(new) != 2 * PAIRS or len({case["module"] for case in new}) != len(new):
        raise ValueError("expected exactly two unique worlds for each deep-source root")
    if Counter(Counter(case["pair"] for case in new).values()) != {2: PAIRS}:
        raise ValueError("the deep source lost its paired worlds")
    for name in ("source", "development"):
        for key in ("pair", "module", "statement"):
            if {case[key] for case in new} & {case[key] for case in original[name]}:
                raise ValueError(f"deep source overlaps the original {name} on {key}")
    return {"deep_expressions": len(guard_expressions(new)),
            "source_overlap": len(guard_expressions(new) & guard_expressions(original["source"])),
            "development_overlap": len(guard_expressions(new) & guard_expressions(original["development"]))}


def freeze(args, tree):
    original = read_receipt(args.bank)
    paths = (args.bpd, args.mathcan)
    pins = {"bank": sha256_file(args.bank), "lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE),
            "imports": {p.name: sha256_file(p) for p in paths}}
    if original["schema"] != BANK_SCHEMA or any(original[k] != pins[k] for k in ("lean", "repl", "imports")):
        raise ValueError("the original native source files or Lean runtime changed")
    planned = list(cases("development", PAIRS, namespace=NAMESPACE))
    overlap = audit_cohort(planned, original)
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean version")
            source = [verify_case(repl, case) for case in planned]
        certificates = [(case, route["proof"]) for case in source for route in case["routes"].values() if route["proof"]]
        independent = certify_routes(directory, env, imports, certificates)
    write_receipt(args.results, {"schema": SCHEMA, "namespace": NAMESPACE, "pins": pins, "source": source,
                                 "overlap": overlap, "independent": independent, "memory": tree.stats()})
    print("Deep source roots", len(source) // 2, "native certificates", independent["certificates"], overlap, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "bpd", "mathcan", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        freeze(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
