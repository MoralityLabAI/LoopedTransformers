"""Allow native continuation while restricting all paid quotes to the opening."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from configs.freeze_ppt_live_quote_fragments import meter
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import lean_env
from ppt_gym.bound_quotes import REGIMES, public_view
from ppt_gym.data import sha256_file
from ppt_gym.live_quote_fragments import NativeFragments, NativeQuoteBackup
from ppt_gym.ram_limits import TreeBudget


def run(args, tree):
    bank, audit = read_receipt(args.bank), read_receipt(args.audit)
    bank_sha = sha256_file(args.bank)
    if audit["bank_sha256"] != bank_sha:
        raise ValueError("the structural witness no longer matches the native proof bank")
    graphs = {log["root"]: NativeFragments(log) for log in audit["held_graphs"]}
    upfront, proofs = [], {}
    for case in bank["held"]:
        graph = graphs[case["visible_sha256"]]
        billing = meter(case, graph)
        for regime in REGIMES:
            view = public_view(case, regime)
            policy = NativeQuoteBackup(graph, view["fees"], billing, quotes_before_tactics=True)
            for world in policy.initial_belief:
                row = policy.solve(world, view["budget"], 3)
                if row["proof"]:
                    proofs[case["module"], tuple(row["script"])] = case
                upfront.append({"root": case["module"], "regime": regime, "world": world, **row})
            policy.action.cache_clear()
            policy.value.cache_clear()
    with TemporaryDirectory(prefix="ppt-upfront-check-") as scratch:
        certificates = verify_independent(Path(scratch), lean_env(), (),
                                          [(case, key[1]) for key, case in proofs.items()])
    write_receipt(args.output, {"schema": "ppt.late-detour-upfront-quote-audit.v1", "bank_sha256": bank_sha,
                                "structural_audit_sha256": sha256_file(args.audit), "results": upfront,
                                "independent": certificates, "memory": tree.stats()})
    print("upfront", sum(row["proof"] for row in upfront), sum(row["quote_effort"] for row in upfront),
          sum(row["utility"] for row in upfront) / len(upfront), "compiled", certificates["certificates"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
