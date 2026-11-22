"""Verify both bundle choices through the standard Lean search and replay."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from ppt_gym.bundled_rewrite import BPD_BUNDLE_FAMILY, bundle_hypothesis
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


class OpeningWitness:
    architecture = "hybrid"

    def macs(self, _actions, _passes):
        return 0

    def __call__(self, _packet, actions, _passes, **_options):
        import torch

        return torch.zeros(len(actions)), None


def audit(cohort, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    cases = load(cohort)["dev"]
    reg = json.loads(cohort.read_bytes())["registration"]
    if len(cases) != reg["reference_certificates"] or any(case.family != BPD_BUNDLE_FAMILY for case in cases):
        raise ValueError("expected the original bundled development schedule")
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000, bundle_continuation=True)
    model = OpeningWitness()
    rows, groups = {}, defaultdict(list)
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the bundle comparison requires pinned Lean 4.19.0")
        for case in cases:
            winner = int(case.tid.rsplit("_", 1)[1])
            proposals = []
            for first in (0, 1):
                ep = rollout(repl, tree, case, model, options, first_choice=first)
                replay = replay_search(repl, case, ep)
                opening = ep["events"][0]
                if (not ep["within_cpu_budget"] or not replay["replay_passed"] or ep["tactic_checks"] != 8
                        or ep["verified_closed"] != (first == winner) or opening["selected_index"] != first
                        or tuple(map(bundle_hypothesis, opening["candidate_menu"])) != ("h0", "h1")):
                    raise ValueError(f"the bundled gym witness failed for {case.tid}")
                proposals.append({"first": first, "menu": opening["candidate_menu"], "checks": ep["tactic_checks"],
                                  "script": ep["accepted_script"], "certificate": ep["certificate"], "replay": replay})
            successful = proposals[winner]
            rows[case.root] = {"cluster": case.cluster, "winner": winner, "certified": True,
                               "checks": successful["checks"], "certificate": successful["certificate"],
                               "replay": successful["replay"], "proposals": proposals}
            groups[case.cluster].append(case.root)
    for roots in groups.values():
        if (len(roots) != 2 or {rows[root]["winner"] for root in roots} != {0, 1}
                or rows[roots[0]]["proposals"][0]["menu"] != rows[roots[1]]["proposals"][0]["menu"]):
            raise ValueError("the bundled worlds must exchange the viable opening with the same menu")
    return {"kind": "bundled-equality", "cohort_sha256": sha256_file(cohort), "lean": "4.19.0",
            "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args.cohort, tree)
        payload = canonical(report) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
    finally:
        tree.close()
    print("The standard gym verified both opening actions in", len(report["rows"]), "Lean worlds.")


if __name__ == "__main__":
    main()
