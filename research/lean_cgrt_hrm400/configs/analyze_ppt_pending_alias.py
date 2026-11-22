"""Verify the routed and unrouted controls on the pending-rule development cohort."""

import argparse
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_branch_recovery import audit
from configs.run_ppt_branch_recovery import UNROUTED
from ppt_gym.data import canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "initialization", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed-study", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if json.loads(args.cohort.read_bytes())["registration"].get("pending_arrival_ablation") != UNROUTED:
        parser.error("expected the registered pending-arrival comparison")
    args.probe = None
    summary, evidence = audit(args)
    with args.summary.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.evidence.open("xb") as out:
        out.write(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for arm, values in summary["arms"].items():
        print(arm, values["final"]["solved_both_worlds"], values["final"]["proofs"])


if __name__ == "__main__":
    main()
