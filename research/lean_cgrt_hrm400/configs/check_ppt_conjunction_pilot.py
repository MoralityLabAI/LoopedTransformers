"""Audit eight-check Lean witnesses and the equal-mean property before interpretation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.__main__ import rollout
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.hybrid_rl.graph_model import ResidualFactTransport, visible_graph
from ppt_gym.live_search import replay_search
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_limits import TreeBudget


class ProofLengthWitness:
    """Derive shortest public Horn proofs for this auditor, outside all model trials."""

    architecture = "hybrid"

    def macs(self, _candidates, _passes):
        return 0

    def __call__(self, packet, menu, _passes, **_kwargs):
        graph = visible_graph(packet, menu)
        if graph is None or not all(graph.actions):
            raise ValueError("witness requires an in-range Horn graph and premise tactics")
        too_long = 1_000_000
        cost = [too_long] * len(graph.features)
        sources = [[i - 1 for i in row if i] for row in graph.premises]
        for _ in graph.features:
            for row, head in zip(sources, graph.heads, strict=True):
                cost[head] = min(cost[head], 1 + sum(cost[i] for i in row))
        return torch.tensor([-1 - sum(cost[i] for i in sources[a - 1]) for a in graph.actions]), None


def root_information(event, transport):
    graph = visible_graph(canonical(event["input"]), event["candidate_menu"])
    if graph is None or len(graph.actions) != 2 or not all(graph.actions):
        raise ValueError("witness did not have two common root actions")
    seed = torch.tensor(graph.features)[:, :1]
    premises, heads = torch.tensor(graph.premises), torch.tensor(graph.heads)
    with torch.no_grad():
        scores = transport(seed, premises, heads, 8)[torch.tensor(graph.actions) - 1]
    difference = float(scores[0] - scores[1])
    if abs(difference) > 2e-6:
        raise ValueError("linear transport distinguishes the claimed equal-mean root")
    return {"menu": event["candidate_menu"], "linear_margin": difference}


def audit(path, tree):
    configure(tree.threads, 0)
    linear = ResidualFactTransport()
    with torch.no_grad():
        linear.action.weight.fill_(1)
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    pairs = defaultdict(list)
    witness = ProofLengthWitness()
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("expected pinned native Lean")
        for case in load(path)["dev"]:
            trials = [rollout(repl, tree, case, witness, options, first_choice=i) for i in range(2)]
            success = [ep["verified_closed"] and ep["within_cpu_budget"] for ep in trials]
            if sum(success) != 1 or any(ep["tactic_checks"] > 8 for ep in trials):
                raise ValueError("a world must resolve exactly one of the two proposals")
            winner = trials[success.index(True)]
            if winner["tactic_checks"] != 8:
                raise ValueError("winning proof must spend the entire eight-check budget")
            result = root_information(winner["events"][0], linear)
            result.update(root=case.root, winner=success.index(True), certificate=winner["certificate"],
                          checks=[ep["tactic_checks"] for ep in trials],
                          replay=[replay_search(repl, case, ep) for ep in trials])
            if not all(item["replay_passed"] for item in result["replay"]):
                raise ValueError("native witness replay failed")
            pairs[case.cluster].append(result)
    for pair in pairs.values():
        a, b = pair
        if a["menu"] != b["menu"] or a["winner"] == b["winner"]:
            raise ValueError("paired worlds must share actions and swap the winning first tactic")
    return {"cohort_sha256": sha256_file(path), "native_lean": "4.19.0", "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        report = audit(args.cohort, tree)
    finally:
        tree.close()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"verified both root proposals in {len(report['pairs'])} native Lean pairs")


if __name__ == "__main__":
    main()
