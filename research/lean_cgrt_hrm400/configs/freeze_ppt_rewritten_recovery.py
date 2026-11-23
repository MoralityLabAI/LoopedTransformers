"""Native verification of arithmetic rewrites in the second Lean proof."""

import argparse
from pathlib import Path

from configs.freeze_ppt_failed_prefix import verify_case
from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_paired_recovery import pairs
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import SCHEMA as BANK_SCHEMA
from ppt_gym.paired_recovery import order
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewritten_recovery import SCHEMA, rewrite


def freeze(args, tree):
    bank = read_receipt(args.bank)
    paths = (args.bpd, args.mathcan)
    identity = {"lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE),
                "imports": {path.name: sha256_file(path) for path in paths}}
    if bank["schema"] != BANK_SCHEMA or any(bank[name] != value for name, value in identity.items()):
        raise ValueError("the native failed-prefix bank or imported definitions changed")
    rewritten, certificates = {}, []
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected Lean rewrite toolchain")
            partitions = ("development",) if args.unfold else ("source", "development")
            for partition in partitions:
                rows = bank[partition]
                candidates = []
                for first, second in pairs(rows):
                    canonical = rows[second]
                    new = verify_case(repl, rewrite(canonical, partition, unfold=args.unfold))
                    if (new["truth"] != canonical["truth"]
                            or [route["trace"] for route in new["routes"]] != [route["trace"] for route in canonical["routes"]]
                            or {route["guard"] for route in new["routes"]} & {route["guard"] for route in canonical["routes"]}):
                        raise ValueError("a rewritten guard changed its Lean result, cost or surface identity")
                    candidates.append({"source": rows[first]["module"], "canonical": canonical["module"],
                                       "alignment": order(rows[first], canonical), "target": new})
                    certificates.extend((new, route["proof"]) for route in new["routes"] if route["proof"])
                rewritten[partition] = candidates
                print("native", partition, "rewrites", len(candidates), flush=True)
        certified = certify_routes(directory, env, imports, certificates)
    write_receipt(args.rewritten, {"schema": SCHEMA, "bank_sha256": sha256_file(args.bank), **identity, **rewritten,
                                    "form": "unfolded-development" if args.unfold else "arithmetic",
                                    "independent": certified, "memory": tree.stats()})
    print("independently compiled rewritten proofs", certified["certificates"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "bpd", "mathcan", "rewritten"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--unfold", action="store_true", help="generate the post-hoc recurrence-unfolding stress set")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        freeze(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
