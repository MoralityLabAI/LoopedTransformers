"""Score fixed transport and exact public costs on the dependent rewrite witness."""

import argparse
from collections import defaultdict, deque
import gzip
import heapq
import json
from pathlib import Path

from configs.check_ppt_bpd_rewrite import EqualityReference
from configs.probe_ppt_bpd_dependent_policy import read_witness, trajectory
from configs.probe_ppt_bpd_rewrite_revisits import audited_policies
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import rewrite_problem


class GuardCostReference(EqualityReference):
    def __call__(self, packet, menu, passes):
        import torch

        from ppt_gym.dependent_rewrite import term_problem

        goal = json.loads(packet)["state"]["goals"][0]
        problem = term_problem(goal)
        if problem is None:
            return super().__call__(packet, menu, passes)
        left, right, rules = problem
        _, _, plain = rewrite_problem({**goal, "target": {"raw": f"{left[1]} = {left[2]}"}})
        neighbors = defaultdict(list)
        for _, start, end in plain:
            neighbors[start].append(end)
        distances = {}
        for anchor in {rule.guard[1] for rule in rules if rule.guard}:
            known, queue = {anchor: 0}, deque([anchor])
            while queue:
                vertex = queue.popleft()
                for other in neighbors[vertex]:
                    if other not in known:
                        known[other] = known[vertex] + 1
                        queue.append(other)
            distances[anchor] = known

        def outgoing(term):
            for rule in rules:
                for reverse in (False, True):
                    step = rule.apply(term, reverse=reverse)
                    if step is None:
                        continue
                    action, following, guard = step
                    cost = distances[guard[1]].get(guard[0], 1000) if guard else 0
                    yield action, following, 1 + cost

        def finish(term):
            queue, visited = [(0, term)], set()
            while queue:
                cost, current = heapq.heappop(queue)
                if current == right:
                    return cost
                if current in visited:
                    continue
                visited.add(current)
                for _, following, checks in outgoing(current):
                    if following not in visited:
                        heapq.heappush(queue, (cost + checks, following))
            return 1000

        estimates = {action: checks + finish(following) for action, following, checks in outgoing(left)}
        return torch.tensor([-estimates[action] for action in menu]), None


class FixedTransport:
    architecture = "hybrid"

    def __init__(self, model):
        self.graph = model.graph

    def macs(self, _candidates, passes):
        return self.graph.upper_macs(passes)

    def __call__(self, packet, actions, passes):
        import torch

        from ppt_gym.dependent_rewrite import dependent_graph
        from ppt_gym.hybrid_rl.graph_model import visible_graph

        graph = dependent_graph(packet, actions) or visible_graph(packet, actions)
        if graph is None or not all(graph.actions):
            raise ValueError("the fixed transport control lacks its public proof graph")
        features = self.graph.node.weight.new_tensor(graph.features)
        premises = torch.tensor(graph.premises, dtype=torch.long)
        heads = torch.tensor(graph.heads, dtype=torch.long)
        scores = self.graph.transport(features[:, :1], premises, heads, 2 * passes)
        return scores[torch.tensor(graph.actions, dtype=torch.long) - 1], None


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure

    fixture = read_witness(args.witness, "dev")
    source = json.loads(args.audit.read_bytes())
    configure(tree.threads, source["seed"])
    _, calibrated, checkpoint, _ = next(audited_policies(args.sessions, args.audit, source["cohort_sha256"],
                                                        source["reference_sha256"], selected=("hybrid_calibration",)))
    results = {}
    with LeanRepl(timeout=15) as repl, torch.no_grad():
        if repl.version != fixture["lean"]:
            raise ValueError("the fixed controls require the registered native verifier")
        for label, policy in (("fixed_transport", FixedTransport(calibrated)), ("exact_guard_cost", GuardCostReference())):
            outcomes, groups = {}, defaultdict(list)
            for root, row in fixture["rows"].items():
                case = SearchCase(**row["case"])
                choice = int(policy(row["packet"], row["menu"], 4)[0].argmax())
                events, closed, within, macs = trajectory(repl, case, policy, tree)
                script = [event["tactic"] for event in events]
                replay, replay_closed, _, _ = trajectory(repl, case, policy, tree, script)
                if replay != events or replay_closed != closed:
                    raise ValueError("the fixed dependent control failed native replay")
                cert = certify(case, {"live_kernel": closed, "accepted_script": script})
                if closed and (not cert or not cert["passed"] or cert["axioms"]):
                    raise ValueError("the fixed control lacks an independent proof")
                opening = row["branches"][choice]["closed"]
                groups[case.cluster].append((opening, closed and within))
                outcomes[root] = {"choice": choice, "events": events, "certificate": cert,
                                  "opening_certified": opening, "search_certified": closed and within, "macs": macs}
            counts = {name: {"proofs": sum(sum(value[index] for value in group) for group in groups.values()),
                             "pairs": sum(all(value[index] for value in group) for group in groups.values())}
                      for index, name in enumerate(("opening", "search"))}
            results[label] = {"counts": counts, "rows": outcomes}
            print(label, counts, flush=True)
    return {"witness_sha256": sha256_file(args.witness), "calibration_checkpoint": checkpoint,
            "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "audit", "sessions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
