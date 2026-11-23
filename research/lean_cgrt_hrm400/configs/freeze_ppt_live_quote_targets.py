"""Freeze matched Bellman-interval and observed-suffix targets for source only."""

import argparse
from collections import Counter
from pathlib import Path

from configs.freeze_ppt_live_quote_fragments import DETOUR_SCHEMA, SCHEMA as NATIVE_SCHEMA
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.live_quote_targets import build_targets
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.live-quote-source-targets.v1"


def run(args, tree):
    bank = read_receipt(args.bank)
    source = {case["module"] for case in bank["source"]}
    development = {case["module"] for case in bank["held"]}
    if bank["schema"] not in (NATIVE_SCHEMA, DETOUR_SCHEMA) or source & development:
        raise ValueError("the frozen source split changed")
    rows, counts = [], Counter()
    for case in bank["source"]:
        decisions = build_targets(case)
        for row in decisions:
            counts.update(decisions=1, root_decisions=row["state"] == case["native"]["root"])
            for targets in row["targets"].values():
                low, high = targets["bellman"]
                empirical = targets["suffix"]["mean"]
                direct_positive = empirical is not None and empirical > 0
                at_root = row["state"] == case["native"]["root"]
                counts.update(actions=1, exact=low == high, interval=low != high,
                              empirical=empirical is not None, additional_positive=low > 0 and not direct_positive,
                              positive_root=at_root and low > 0, suffix_positive_root=at_root and direct_positive)
        rows.append({"root": case["module"], "rows": decisions})
    write_receipt(args.output, {"schema": SCHEMA, "bank_sha256": sha256_file(args.bank),
                                "source": rows, "summary": counts, "memory": tree.stats()})
    print(dict(counts), flush=True)


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
