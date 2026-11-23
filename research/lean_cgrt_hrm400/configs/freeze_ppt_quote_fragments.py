"""Register source-only fragmented price feedback after native plan certification."""

import argparse
from collections import Counter
from pathlib import Path

from configs.freeze_ppt_private_quotes import SCHEMA as BANK_SCHEMA
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, REPL_EXE
from ppt_gym.data import sha256_file
from ppt_gym.private_quotes import COMMITS, REGIMES, public_view
from ppt_gym.quote_fragments import QuoteFragmentGraph, source_fragments
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.fragmented-price-quotes.v1"


def run(args, tree):
    bank = read_receipt(args.bank)
    source = bank["source"]
    source_modules = {case["module"] for case in source}
    held_modules = {case["module"] for case in bank["held"]}
    if (bank["schema"] != BANK_SCHEMA or source_modules & held_modules
            or bank["independent"]["certificates"] != 2 * (len(source_modules) + len(held_modules))
            or bank["lean_sha256"] != sha256_file(LEAN_EXE) or bank["repl_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the frozen native proof bank changed before source collection")
    rows, summary = [], Counter()
    for case in source:
        for regime in REGIMES:
            view = public_view(case, regime)
            fragments = source_fragments(case, regime)
            graph = QuoteFragmentGraph(view, fragments)
            for _, action, (bellman, empirical) in graph.rows():
                summary.update(decisions=1, backed_up=bellman is not None, empirical=empirical is not None)
                if action not in COMMITS:
                    summary.update(quotes=1, additional_quote_labels=bellman is not None and empirical is None,
                                   larger_quote_return=bellman is not None and empirical is not None and bellman > empirical)
            summary.update(fragments=len(fragments), recorded_events=sum(len(fragment["steps"]) for fragment in fragments),
                           paid_feedback_events=sum(step["action"] not in COMMITS
                                                    for fragment in fragments for step in fragment["steps"]))
            rows.append({"module": case["module"], "regime": regime, "view": view, "fragments": fragments})
    write_receipt(args.output, {"schema": SCHEMA, "source": rows, "source_summary": summary,
                               "bank_sha256": sha256_file(args.bank), "memory": tree.stats()})
    print(dict(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
