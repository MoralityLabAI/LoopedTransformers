"""Fit a small recurrent evidence router on verified proof returns and matched imitation."""

import argparse
from collections import defaultdict
import gzip
from itertools import product
import json
import math
from pathlib import Path

from configs.check_ppt_bpd_dependent_rewrites import BUDGET, ROUTING_SCHEMA, SPUR_SCHEMA
from configs.probe_ppt_bpd_dependent_controls import GuardCostReference, fit_calibrated_margin
from configs.probe_ppt_bpd_dependent_policy import read_witness, training_batches, trajectory
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


GATES = ((0., 0.), (.5, .5), (1., 1.), (2., 2.), (4., 4.), (0., 4.), (4., 0.), (4., -4.), (-4., 4.))
GAINS = (1., .25, .0625, .015625)
FROZEN = {"uniform_frozen": (0., 0., 1.), "support_weighted_frozen": (4., 4., .0625)}


def verified_returns(path, training_path, training):
    result = json.loads(gzip.decompress(path.read_bytes()))
    rows = result["rows"]
    if (result["schema"] != "ppt.dependent-rewrite-return-training.v1" or result["eight_check_budget"] != BUDGET
            or result["prefix_sha256"] != sha256_file(training_path) or rows.keys() != training["rows"].keys()):
        raise ValueError("the proof returns differ from the registered training receipts")
    rewards = {}
    for root, row in training["rows"].items():
        values = []
        for trace, branch in zip(rows[root], row["branches"], strict=True):
            checks, cert = trace["checks_to_certify"], trace["certificate"]
            expected = [{key: event[key] for key in ("tactic", "before", "after")} for event in branch["events"]]
            if (trace["choice"] != branch["choice"] or trace["within_eight"] != branch["closed"]
                    or len(trace["events"]) != checks or checks != BUDGET + int(not branch["closed"])
                    or trace["events"][:BUDGET] != expected or not cert["passed"] or cert["axioms"]
                    or digest(cert["source"]) != cert["source_digest"]):
                raise ValueError("a routed training path lost its compiled receipt or deadline outcome")
            values.append(float(trace["within_eight"]) - .025 * BUDGET)
        rewards[root] = tuple(values)
    return rewards


def warm_policy(path):
    import torch

    from ppt_gym.gated_dependent_rewrite import RoutedDependentRewritePolicy
    from ppt_gym.ram_learning import tensor_hash

    package = torch.load(path, map_location="cpu", weights_only=True)
    state = package["state_dict"]
    if package["schema"] != "ppt.bpd-rewrite.checkpoint.v1" or tensor_hash(state) != package["weights_sha256"]:
        raise ValueError("the routed trial requires the exact audited warm CPU checkpoint")
    model = RoutedDependentRewritePolicy(**package["architecture"]).eval().requires_grad_(False)
    missing, extra = model.load_state_dict(state, strict=False)
    if set(missing) != {"recurrent_log_gain", "graph.transport.evidence_gate.weight"} or extra:
        raise ValueError("the routed trial changed the inherited warm architecture")
    return model, package


def apply_route(policy, candidate):
    import torch

    mean, weakest, gain = candidate
    gate = policy.graph.transport.evidence_gate.weight
    with torch.no_grad():
        gate.copy_(gate.new_tensor([[mean, weakest]]))
        policy.recurrent_log_gain.fill_(math.log(gain))


def training_observations(training):
    observations, decisions, roots = {}, [], {}
    for batch in training_batches(training, match_search_menu=True):
        for episode in batch:
            for event in episode["events"]:
                choices = event["rewrite_choices"]
                if len(choices) == 1:
                    continue
                packet, menu = canonical(event["input"]), tuple(event["candidate_menu"])
                key = digest([packet, menu])
                observations[key] = packet, menu
                decisions.append((key, choices, event["selected_index"]))
                if event["attempt"] == 1:
                    roots[episode["root"]] = key
    if roots.keys() != training["rows"].keys():
        raise ValueError("both return and imitation fitting require every registered training opening")
    return observations, decisions, roots


def cached_workspace(policy, observations):
    import torch

    cache = {}
    features, premises, heads = [], [], []
    node_offset = edge_offset = 0
    with torch.no_grad():
        for key, (packet, menu) in observations.items():
            graph = policy.observation_graph(packet, menu)
            if graph is None:
                raise ValueError("a routed training decision lacks a public proof graph")
            latent, _ = policy.workspace_scores(packet, menu, 4, graph)
            features.append(latent.new_tensor(graph.features)[:, :1])
            source = torch.as_tensor(graph.premises, dtype=torch.long)
            premises.append(torch.where(source > 0, source + node_offset, 0))
            heads.append(torch.as_tensor(graph.heads, dtype=torch.long) + node_offset)
            actions = torch.as_tensor(graph.actions, dtype=torch.long)
            cache[key] = latent, torch.where(actions > 0, actions + edge_offset, 0)
            node_offset += len(features[-1])
            edge_offset += len(source)
    return cache, (torch.cat(features), torch.cat(premises), torch.cat(heads))


def score_routes(policy, training, rewards):
    import torch

    observations, imitation, roots = training_observations(training)
    cached, graph = cached_workspace(policy, observations)
    groups = {root: row["case"]["cluster"] for root, row in training["rows"].items()}
    scores = []
    with torch.no_grad():
        for (mean, weakest), gain in product(GATES, GAINS):
            route = (mean, weakest, gain)
            apply_route(policy, route)
            flow = policy.graph.transport(*graph, 8)
            flow = torch.cat((flow.new_zeros(1), flow))
            gain_tensor = policy.recurrent_log_gain.exp()
            logits = {key: gain_tensor * latent + flow[selected] for key, (latent, selected) in cached.items()}
            returns_by_layout = defaultdict(list)
            for root, key in roots.items():
                returns_by_layout[groups[root]].append(rewards[root][int(logits[key].argmax())])
            returns = [reward for layout in returns_by_layout.values() for reward in layout]
            losses = [-logits[key][choices].log_softmax(0)[choices.index(chosen)] for key, choices, chosen in imitation]
            scores.append({"route": route, "pairs": sum(all(reward > 0 for reward in r) for r in returns_by_layout.values()),
                           "mean_return": math.fsum(returns) / len(returns), "imitation_loss": float(torch.stack(losses).mean())})
    rl = max(scores, key=lambda row: (row["mean_return"], row["pairs"]))
    sft = min(scores, key=lambda row: row["imitation_loss"])
    return {"routed_rl": rl, "routed_imitation": sft}, scores, len(imitation)


def fit(args, tree):
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    configure(tree.threads, 103)
    training = read_witness(args.training, "train")
    if training["schema"] != SPUR_SCHEMA:
        raise ValueError("the routing fit expects the registered dead-end training layouts")
    rewards = verified_returns(args.returns, args.training, training)
    policy, package = warm_policy(args.warm)
    selected, scores, decisions = score_routes(policy, training, rewards)
    print({arm: row for arm, row in selected.items()}, flush=True)
    return {"schema": "ppt.dependent-rewrite-route-fit.v1", "scope": "offline policy search with both verified opening returns",
            "training_sha256": sha256_file(args.training),
            "returns_sha256": sha256_file(args.returns), "warm_sha256": sha256_file(args.warm),
            "warm_weights_sha256": package["weights_sha256"], "weights_before": tensor_hash(package["state_dict"]),
            "training_worlds": len(rewards), "imitation_decisions": decisions, "selected": selected,
            "candidate_scores": scores, "memory": tree.stats()}


def evaluate(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.dependent_rewrite import DependentRewritePolicy
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    configure(tree.threads, 103)
    registered = json.loads(gzip.decompress(args.fit.read_bytes()))
    witness = read_witness(args.witness, "dev")
    training = read_witness(args.training, "train")
    if (registered["schema"] != "ppt.dependent-rewrite-route-fit.v1" or witness["schema"] != ROUTING_SCHEMA
            or training["schema"] != SPUR_SCHEMA or registered["training_sha256"] != sha256_file(args.training)
            or registered["warm_sha256"] != sha256_file(args.warm) or set(witness["rows"]) & set(training["rows"])):
        raise ValueError("the held-out routing evaluation changed its frozen fit or disjoint cohort")
    routed, package = warm_policy(args.warm)
    calibrated = DependentRewritePolicy(**package["architecture"]).eval()
    calibrated.load_state_dict(package["state_dict"])
    ridge = fit_calibrated_margin(calibrated, training)
    trm_report = json.loads(gzip.decompress(args.trm_audit.read_bytes()))
    trm_receipt = trm_report["results"]["trm_rl"]
    if trm_report["training_sha256"] != registered["training_sha256"] or trm_receipt["checkpoint_sha256"] != sha256_file(args.trm):
        raise ValueError("the comparison TRM changed from the matched audited return training")
    warm_trm = torch.load(args.trm, map_location="cpu", weights_only=True)
    trm = DependentRewritePolicy("trm", warm_trm["roles"].shape[-1]).eval()
    trm.load_state_dict(warm_trm)
    routes = {**{arm: row["route"] for arm, row in registered["selected"].items()}, **FROZEN}
    controls = {"ridge_readout_calibration": calibrated, "graph_trm_rl": trm, "exact_guard_cost": GuardCostReference()}
    results = {}
    args.checkpoints.mkdir()
    with LeanRepl(timeout=15) as repl, torch.no_grad():
        if repl.version != witness["lean"]:
            raise ValueError("the registered routing verifier changed")
        for arm in (*routes, *controls):
            policy = controls.get(arm, routed)
            if arm in routes:
                apply_route(policy, routes[arm])
                path = args.checkpoints / (arm + ".pt")
                state = policy.state_dict()
                torch.save({"architecture": package["architecture"], "state_dict": state,
                            "weights_sha256": tensor_hash(state), "fit_sha256": sha256_file(args.fit)}, path)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                policy.load_state_dict(saved["state_dict"])
                if tensor_hash(policy.state_dict()) != saved["weights_sha256"]:
                    raise ValueError("the routed proof checkpoint changed on CPU reload")
            depth = 4 if policy.architecture == "hybrid" else 1
            pairs = defaultdict(list)
            rows = {}
            for root, row in witness["rows"].items():
                case = SearchCase(**row["case"])
                first = int(policy(row["packet"], row["menu"], depth)[0].argmax())
                events, closed, within, macs = trajectory(repl, case, policy, tree)
                script = [event["tactic"] for event in events]
                replay, repeated, _, _ = trajectory(repl, case, policy, tree, script)
                cert = certify(case, {"live_kernel": closed, "accepted_script": script})
                if replay != events or repeated != closed or (closed and (not cert or not cert["passed"] or cert["axioms"]
                                                                        or digest(cert["source"]) != cert["source_digest"])):
                    raise ValueError("a routed development search failed replay or independent certification")
                outcome = row["branches"][first]["closed"], closed and within
                pairs[case.cluster].append(outcome)
                rows[root] = {"opening": first, "opening_certified": outcome[0], "search_certified": outcome[1],
                              "events": events, "certificate": cert, "macs": macs}
            counts = {stage: {"proofs": sum(sum(row[index] for row in pair) for pair in pairs.values()),
                              "pairs": sum(all(row[index] for row in pair) for pair in pairs.values())}
                      for index, stage in enumerate(("opening", "search"))}
            print(arm, counts, flush=True)
            results[arm] = {"counts": counts, "rows": rows}
            if arm in routes:
                results[arm].update(route=routes[arm], checkpoint_sha256=sha256_file(path),
                                    parameter_count=policy.parameter_count())
    return {"fit_sha256": sha256_file(args.fit), "witness_sha256": sha256_file(args.witness), "ridge": ridge,
            "trm_weights_sha256": tensor_hash(warm_trm), "results": results, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="stage", required=True)
    train = subs.add_parser("fit")
    score = subs.add_parser("evaluate")
    for parser in (train, score):
        for name in ("training", "warm", "output"):
            parser.add_argument(name, type=Path)
    train.add_argument("returns", type=Path)
    for name in ("witness", "fit", "trm", "trm_audit", "checkpoints"):
        score.add_argument(name, type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = (fit if args.stage == "fit" else evaluate)(args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
