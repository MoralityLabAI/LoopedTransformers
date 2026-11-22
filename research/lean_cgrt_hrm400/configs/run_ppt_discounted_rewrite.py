"""Update recurrent proof-return transport from live Lean feedback and score the frozen fit."""

import argparse
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_dependent_rewrites import BUDGET, EFFORT_SCHEMA
from configs.probe_ppt_bpd_dependent_controls import FixedTransport, GuardCostReference, fit_calibrated_margin
from configs.probe_ppt_bpd_dependent_policy import read_witness, training_batches, trajectory
from configs.probe_ppt_bpd_dependent_routing import load_routed_checkpoint, score_native, verified_returns
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def build_policy(source):
    from ppt_gym.discounted_rewrite import DiscountedRewritePolicy

    _, checkpoint = load_routed_checkpoint(source)
    policy = DiscountedRewritePolicy(**checkpoint["architecture"]).eval().requires_grad_(False)
    missing, unexpected = policy.load_state_dict(checkpoint["state_dict"], strict=False)
    if set(missing) != {"proof_transport." + name for name, _ in policy.proof_transport.named_parameters()} or unexpected:
        raise ValueError("discounted rewriting needs the exact routed warm-start model")
    return policy, checkpoint["architecture"]


def frozen_hash(policy):
    from ppt_gym.ram_learning import tensor_hash

    return tensor_hash({name: value for name, value in policy.state_dict().items() if not name.startswith("proof_transport.")})


def probe_returns(repl, case, row, policy, tree):
    from ppt_gym.hybrid_rl.__main__ import certify, proof_return

    runs = []
    before = digest(json.loads(row["packet"])["state"])
    for opening in row["menu"]:
        events, closed, within, macs = trajectory(repl, case, policy, tree, opening=opening)
        if not within or not events or events[0]["before"] != before or events[0]["tactic"] != opening:
            raise ValueError("live return training lost a registered opening or its CPU budget")
        script = [event["tactic"] for event in events]
        replay, repeated, _, _ = trajectory(repl, case, policy, tree, script)
        cert = certify(case, {"live_kernel": closed, "accepted_script": script})
        if replay != events or repeated != closed or (closed and (not cert or not cert["passed"] or cert["axioms"]
                                                                or digest(cert["source"]) != cert["source_digest"])):
            raise ValueError("a training rollout failed replay or independent certification")
        reward = proof_return({"verified_closed": closed, "within_cpu_budget": within, "tactic_checks": len(events)})
        runs.append({"events": events, "reward": reward, "closed": closed, "certificate": cert, "macs": macs})
    return runs


def record(policy, packet, menu):
    import torch

    graph = policy.observation_graph(packet, menu)
    if graph is None:
        raise ValueError("the registered decision lacks a public proof graph")
    with torch.no_grad():
        base, _ = policy.routed_scores(packet, menu, 4, graph)
    return (base, base.new_tensor(graph.features)[:, 0], torch.tensor(graph.premises, dtype=torch.long),
            torch.tensor(graph.heads, dtype=torch.long), torch.tensor(graph.actions, dtype=torch.long))


def cached_logits(policy, observation):
    import torch

    base, facts, premises, heads, actions = observation
    effort = policy.proof_transport(facts, premises, heads, 8)
    return base + torch.cat((base.new_zeros(1), effort))[actions]


def fit_live(repl, policy, training, batches, optimizer, tree):
    import torch

    traces = {}
    for index, batch in enumerate(batches, 1):
        terms = []
        for root in dict.fromkeys(item["root"] for item in batch):
            row = training["rows"][root]
            case = SearchCase(**row["case"])
            with torch.no_grad():
                outcomes = probe_returns(repl, case, row, policy, tree)
            scores = cached_logits(policy, record(policy, row["packet"], row["menu"]))
            returns = scores.new_tensor([branch["reward"] for branch in outcomes])
            probabilities = (scores / .15).softmax(-1)
            terms.append(-(probabilities * (returns - returns.mean())).sum())
            traces[root] = {"returns": outcomes, "probabilities": probabilities.detach().tolist()}
        update(policy, optimizer, terms)
        if index % 8 == 0:
            print(f"Updated the return channel on {index} native layout pairs.", flush=True)
    return traces


def fit_imitation(policy, batches, optimizer):
    from torch.nn import functional as F

    cache, decisions = {}, 0
    for batch in batches:
        terms = []
        for episode in batch:
            for step in episode["events"]:
                choices = step["rewrite_choices"]
                if len(choices) == 1:
                    continue
                packet, menu = canonical(step["input"]), step["candidate_menu"]
                key = (packet, tuple(menu))
                if key not in cache:
                    cache[key] = record(policy, packet, menu)
                scores = cached_logits(policy, cache[key])[choices] / .15
                target = choices.index(step["selected_index"])
                terms.append(-F.log_softmax(scores, dim=0)[target])
        decisions += len(terms)
        update(policy, optimizer, terms)
    return decisions


def update(policy, optimizer, terms):
    import torch

    loss = torch.stack(terms).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("discounted rewriting produced a non-finite training objective")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.proof_transport.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()


def train(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.ram_learning import tensor_hash

    training = read_witness(args.training, "train")
    if training["schema"] != EFFORT_SCHEMA:
        raise ValueError("discounted rewriting requires the registered equal-depth training layouts")
    verified_returns(args.returns, args.training, training)
    batches = training_batches(training, match_search_menu=True)
    args.checkpoints.mkdir()
    results = {}
    for arm in ("live_rl", "imitation"):
        policy, architecture = build_policy(args.warm)
        policy.proof_transport.requires_grad_(True)
        inherited = frozen_hash(policy)
        before = tensor_hash(policy.proof_transport.state_dict())
        optimizer = torch.optim.Adam(policy.proof_transport.parameters(), lr=.08)
        if arm == "live_rl":
            with LeanRepl(timeout=15) as repl:
                if repl.version != training["lean"]:
                    raise ValueError("live proof-return fitting requires the registered Lean verifier")
                history = fit_live(repl, policy, training, batches, optimizer, tree)
        else:
            history = fit_imitation(policy, batches, optimizer)
        if before == tensor_hash(policy.proof_transport.state_dict()) or frozen_hash(policy) != inherited:
            raise ValueError("the proof-return update did not isolate the three new parameters")
        state = policy.state_dict()
        path = args.checkpoints / (arm + ".pt")
        torch.save({"architecture": architecture, "state_dict": state, "weights_sha256": tensor_hash(state)}, path)
        weights = {name: float(value.detach()) for name, value in policy.proof_transport.named_parameters()}
        print(arm, weights, flush=True)
        results[arm] = {"checkpoint_sha256": sha256_file(path), "weights": weights, "history": history}
    return {"training_sha256": sha256_file(args.training), "returns_sha256": sha256_file(args.returns),
            "warm_sha256": sha256_file(args.warm), "results": results, "memory": tree.stats()}


def evaluate(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.dependent_rewrite import DependentRewritePolicy
    from ppt_gym.ram_learning import tensor_hash

    witness = read_witness(args.witness, "dev")
    training = read_witness(args.training, "train")
    fit = json.loads(gzip.decompress(args.fit.read_bytes()))
    if (training["schema"] != EFFORT_SCHEMA or witness["schema"] != EFFORT_SCHEMA
            or set(witness["rows"]) & set(training["rows"]) or fit["training_sha256"] != sha256_file(args.training)
            or fit["warm_sha256"] != sha256_file(args.warm)):
        raise ValueError("the serial effort evaluation changed its registered training or held-out layouts")
    trm_audit = json.loads(gzip.decompress(args.trm_audit.read_bytes()))
    if (trm_audit["training_sha256"] != fit["training_sha256"]
            or trm_audit["results"]["trm_rl"]["checkpoint_sha256"] != sha256_file(args.trm)):
        raise ValueError("the matched graph TRM checkpoint changed before evaluation")
    state = torch.load(args.trm, map_location="cpu", weights_only=True)
    trm = DependentRewritePolicy("trm", state["roles"].shape[-1]).eval()
    trm.load_state_dict(state)
    routed, _ = load_routed_checkpoint(args.warm)
    weighted, _ = load_routed_checkpoint(args.warm)
    with torch.no_grad():
        weighted.graph.transport.evidence_gate.weight.fill_(4.)
    neutral, _ = build_policy(args.warm)
    serial, _ = build_policy(args.warm)
    serial.seriality_override = 1.
    calibrated, _ = build_policy(args.warm)
    calibrated.seriality_override = 1.
    ridge = fit_calibrated_margin(calibrated, training, additional_readouts=(calibrated.proof_transport.readout_weight,))
    controls = {"previous_return_router": routed, "isolated_support_weighting": FixedTransport(weighted),
                "frozen_initial": neutral, "frozen_serial": serial, "serial_ridge_calibration": calibrated,
                "graph_trm_rl": trm, "exact_guard_cost": GuardCostReference()}
    for arm in ("live_rl", "imitation"):
        path = args.checkpoints / (arm + ".pt")
        if sha256_file(path) != fit["results"][arm]["checkpoint_sha256"]:
            raise ValueError("a fitted discounted rewrite checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policy, _ = build_policy(args.warm)
        policy.load_state_dict(saved["state_dict"])
        if tensor_hash(policy.state_dict()) != saved["weights_sha256"]:
            raise ValueError("a fitted discounted rewrite checkpoint changed on CPU reload")
        controls[arm] = policy
    results = {}
    order = ("live_rl", "imitation", "serial_ridge_calibration", "frozen_serial", "frozen_initial",
             "previous_return_router", "isolated_support_weighting", "graph_trm_rl", "exact_guard_cost")
    with LeanRepl(timeout=15) as repl, torch.no_grad():
        if repl.version != witness["lean"]:
            raise ValueError("the serial effort evaluation needs the registered Lean verifier")
        for arm in order:
            results[arm] = score_native(repl, controls[arm], witness, tree)
            print(arm, results[arm]["counts"], flush=True)
    return {"witness_sha256": sha256_file(args.witness), "fit_sha256": sha256_file(args.fit),
            "ridge": ridge, "results": results, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="stage", required=True)
    fit, score = stages.add_parser("train"), stages.add_parser("evaluate")
    for parser in (fit, score):
        for name in ("training", "warm", "checkpoints", "output"):
            parser.add_argument(name, type=Path)
    fit.add_argument("returns", type=Path)
    for name in ("witness", "fit", "trm", "trm_audit"):
        score.add_argument(name, type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        from ppt_gym.ram_campaign import configure

        configure(tree.threads, 103)
        result = (train if args.stage == "train" else evaluate)(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
