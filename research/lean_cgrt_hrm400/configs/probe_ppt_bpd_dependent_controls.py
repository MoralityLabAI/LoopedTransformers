"""Score fixed transport and exact public costs on the dependent rewrite witness."""

import argparse
from collections import defaultdict, deque
import copy
import gzip
import heapq
import json
from pathlib import Path

from configs.check_ppt_bpd_rewrite import EqualityReference
from configs.check_ppt_bpd_dependent_rewrites import SPUR_SCHEMA
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


def fit_calibrated_margin(policy, training):
    """Fit frozen root features with layout-held-out ridge regularization."""
    import torch

    policy.requires_grad_(False)
    heads = (policy.actor.weight, policy.graph.action.weight, policy.graph.transport.action.weight)
    for weight in heads:
        weight.requires_grad_(True)
    features, targets, layouts = [], [], []
    for row in training["rows"].values():
        logits, _ = policy(row["packet"], row["menu"], 4)
        margin = logits[0] - logits[1]
        jacobian = torch.autograd.grad(margin, heads)
        features.append(torch.cat([weight.flatten() for weight in jacobian]))
        target = 2. if row["winner"] == 0 else -2.
        targets.append(target - float(margin.detach()))
        layouts.append(row["case"]["cluster"])
    design = torch.stack(features)
    target = design.new_tensor(targets)
    scale = float(design.square().sum(1).median())
    if not scale > 0.:
        raise ValueError("the original hybrid has no root features available for calibration")

    def ridge(rows, bound):
        x = design[rows]
        gram = x @ x.T
        regularizer = torch.eye(len(x), dtype=x.dtype, device=x.device) * bound
        return x.T @ torch.linalg.solve(gram + regularizer, target[rows])

    losses = {}
    for multiplier in (0.01, 0.1, 1., 10.):
        error = 0.
        for layout in sorted(set(layouts)):
            held = torch.tensor([value == layout for value in layouts])
            delta = ridge(~held, multiplier * scale)
            error += float((design[held] @ delta - target[held]).square().sum())
        losses[multiplier] = error / len(layouts)
    selected = min(losses, key=lambda value: (losses[value], value))
    delta = ridge(slice(None), selected * scale)
    with torch.no_grad():
        for weight, update in zip(heads, torch.split(delta, tuple(weight.numel() for weight in heads)), strict=True):
            weight.add_(update.reshape_as(weight))
    policy.requires_grad_(False)
    return {"training_layouts": len(set(layouts)), "regularizer_scale": scale, "selected_multiplier": selected,
            "layout_held_out_mean_square_error": {str(key): value for key, value in losses.items()}}


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.dependent_rewrite import DependentRewritePolicy
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure

    fixture = read_witness(args.witness, "dev")
    source = json.loads(args.audit.read_bytes())
    configure(tree.threads, source["seed"])
    _, calibrated, checkpoint, _ = next(audited_policies(args.sessions, args.audit, source["cohort_sha256"],
                                                        source["reference_sha256"], selected=("hybrid_calibration",)))
    controls = [("fixed_transport", FixedTransport(calibrated)), ("exact_guard_cost", GuardCostReference())]
    if fixture["schema"] == SPUR_SCHEMA:
        weighted = copy.deepcopy(calibrated)
        gate = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            gate.weight.fill_(4.)
        weighted.graph.transport.evidence_gate = gate
        controls.insert(1, ("support_weighted_transport", FixedTransport(weighted)))
    ridge_result = None
    if args.training:
        training = read_witness(args.training, "train")
        if training["schema"] != fixture["schema"] or training["source_sha256"] != fixture["source_sha256"]:
            raise ValueError("the ridge calibrator requires the matched training layouts")
        _, warm, warm_identity, _ = next(audited_policies(args.sessions, args.audit, source["cohort_sha256"],
                                                        source["reference_sha256"], selected=("hybrid_rl",)))
        model = DependentRewritePolicy("hybrid", warm.width, residual_transport=True, premise_interaction=True).eval()
        model.load_state_dict(warm.state_dict())
        ridge_result = {**fit_calibrated_margin(model, training), "source_checkpoint": warm_identity,
                        "training_sha256": sha256_file(args.training)}
        controls.append(("ridge_readout_calibration", model))
    results = {}
    with LeanRepl(timeout=15) as repl, torch.no_grad():
        if repl.version != fixture["lean"]:
            raise ValueError("the fixed controls require the registered native verifier")
        for label, policy in controls:
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
            "ridge_calibration": ridge_result, "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "audit", "sessions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--training", type=Path)
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
