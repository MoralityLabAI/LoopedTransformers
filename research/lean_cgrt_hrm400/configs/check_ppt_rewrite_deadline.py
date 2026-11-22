"""Verify that each frozen equality world has exactly one in-budget root rewrite."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.check_ppt_bpd_rewrite import EqualityReference
from configs.freeze_ppt_rewrite_deadline import GENERATOR, MINIMUM_MARGIN, diffusion_margin, edges
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def audit(cohort, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    reg = json.loads(cohort.read_bytes())["registration"]
    layouts = {digest([GENERATOR, layout]): (label, layout)
               for label, group in reg["layouts"]["dev"].items() for layout in group}
    cases = load(cohort)["dev"]
    if (reg["layout_namespace"] != GENERATOR or len(cases) != 2 * len(layouts)
            or set(layouts) != {case.cluster for case in cases}):
        raise ValueError("the equality deadline layouts differ from the frozen cohort")
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    routes = defaultdict(list)
    model = EqualityReference()
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the equality deadline witness requires pinned Lean 4.19")
        for case in cases:
            stratum, layout = layouts[case.cluster]
            expected = int(case.tid.rsplit("_", 1)[1])
            margin = diffusion_margin(edges(layout, expected), expected)
            if (abs(margin) < MINIMUM_MARGIN or (margin > 0) != (stratum == "aligned")):
                raise ValueError("the registered root diffusion stratum changed")
            proposals = []
            for action in (0, 1):
                episode = rollout(repl, tree, case, model, options, first_choice=action)
                replay = replay_search(repl, case, episode)
                first = episode["events"][0]
                success = episode["verified_closed"]
                if (not episode["within_cpu_budget"] or not replay["replay_passed"]
                        or first["candidate_menu"] != ["rw [h0]", "rw [h1]"]
                        or first["selected_index"] != action or episode["tactic_checks"] != 8
                        or success != (action == expected)):
                    raise ValueError(f"the paired deadline witness failed for {case.tid}")
                proposals.append({"action": action, "script": episode["accepted_script"],
                                  "checks": episode["tactic_checks"], "certified": success,
                                  "certificate": episode["certificate"], "replay": replay})
            routes[case.cluster].append({"root": case.root, "winner": expected,
                                         "diffusion_margin": margin, "stratum": stratum, "proposals": proposals})
    if any(len(pair) != 2 or {world["winner"] for world in pair} != {0, 1} for pair in routes.values()):
        raise ValueError("each equality pair must switch the unique in-budget root choice")
    return {"cohort_sha256": sha256_file(cohort), "native_lean": "4.19.0",
            "pairs": routes, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args.cohort, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print("Native Lean verified both root actions on", len(report["pairs"]), "deadline pairs.")


if __name__ == "__main__":
    main()
