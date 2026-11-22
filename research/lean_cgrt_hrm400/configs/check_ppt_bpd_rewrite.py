"""Certify original BPD rewrite proofs through the gym's native search and replay."""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from types import SimpleNamespace

from configs.freeze_ppt_bpd_rewrite import rewrites
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import rewrite_problem


class EqualityReference:
    architecture = "hybrid"

    def macs(self, _candidates, _passes):
        return 0

    def __call__(self, packet, menu, _passes, **_options):
        import torch

        visible = json.loads(packet)["state"]["goals"][0]
        _, target, edges = rewrite_problem(visible)
        reverse = defaultdict(list)
        for _, left, right in edges:
            reverse[right].append(left)
        distance = {target: 0}
        queue = deque([target])
        while queue:
            successor = queue.popleft()
            for predecessor in reverse[successor]:
                if predecessor not in distance:
                    distance[predecessor] = distance[successor] + 1
                    queue.append(predecessor)
        endpoints = {action: right for action, _, right in edges}
        return torch.tensor([-distance.get(endpoints[action], 1000) for action in menu]), None


def native_check(cohort, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    dev = load(cohort)["dev"]
    expected = {case.root: (case, valid) for case, valid in rewrites() if case.split == "dev"}
    if len(dev) != len(expected) or {case.root for case in dev} != set(expected):
        raise ValueError("the native witness requires the original BPD development statements")
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    reference = EqualityReference()
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("expected pinned native Lean 4.19.0")
        for case in dev:
            original, expected_valid = expected[case.root]
            episode = rollout(repl, tree, case, reference, options)
            replay = replay_search(repl, case, episode)
            if (case != original or episode["verified_closed"] != expected_valid
                    or not episode["within_cpu_budget"] or not replay["replay_passed"]):
                raise ValueError(f"native bounded equality reference disagrees for {case.tid}")
            rows[case.root] = {"case": case.tid, "cluster": case.cluster, "certified": expected_valid,
                               "checks": episode["tactic_checks"], "script": episode["accepted_script"],
                               "certificate": episode["certificate"], "replay": replay}
    return {"cohort_sha256": sha256_file(cohort), "rows": rows, "lean": "4.19.0", "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = native_check(args.cohort, tree)
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(canonical(report) + "\n")
    finally:
        tree.close()
    print("native rewrite development:", sum(row["certified"] for row in report["rows"].values()),
          "certificates on", len(report["rows"]), "original BPD declarations")


if __name__ == "__main__":
    main()
