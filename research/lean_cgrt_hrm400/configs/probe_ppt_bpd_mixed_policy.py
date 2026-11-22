"""Fit a shared recurrent readout over equality and proposition obligations."""

import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_mixed_cost import BUDGET, SCHEMA, make_world, native_prefix
from configs.probe_ppt_bpd_rewrite_revisits import audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def root_packet(repl, case):
    from ppt_gym.live_lean import candidate_menu, state_input

    repl.reset()
    initial = repl.init(case.statement, pp=True)
    menu = list(candidate_menu(repl, initial))
    if menu != ["apply h0", "apply h1"]:
        raise ValueError("the mixed root opening menu changed")
    packet, _ = state_input(repl, initial, (), BUDGET)
    return packet, menu


def collect(repl, tree):
    from ppt_gym.hybrid_rl.__main__ import certify

    batches = []
    for alias in range(3):
        for offset in range(1, 6):
            costs = (offset, offset + 1) if (alias + offset) % 2 else (offset + 1, offset)
            opportunities = []
            for winner in (0, 1):
                case, scripts, _ = make_world(100 + 5 * alias + offset, costs, winner)
                case = replace(case, split="train")
                packet, menu = root_packet(repl, case)
                for opening, script in enumerate(scripts):
                    start = tree.cpu_seconds()
                    events, closed = native_prefix(repl, case, script)
                    within = tree.cpu_seconds() - start <= 6
                    if (not within or closed != (opening == winner) or len(events) != BUDGET
                            or events[0]["packet"] != digest(packet)
                            or native_prefix(repl, case, script) != (events, closed)):
                        raise ValueError("the mixed training branch failed native replay or its budget")
                    certificate = certify(case, {"live_kernel": closed, "accepted_script": script[:BUDGET]})
                    if closed and (not certificate or not certificate["passed"] or certificate["axioms"]):
                        raise ValueError("a winning mixed training branch lacks a standalone proof")
                    decision = {"input": json.loads(packet), "candidate_menu": menu, "selected_index": opening,
                                "kind": "tactic", "attempt": 1, "status": "accepted_nonterminal"}
                    opportunities.append({"root": case.root, "family": case.family, "verified_closed": closed,
                                          "within_cpu_budget": within, "tactic_checks": BUDGET, "events": [decision],
                                          "path": events, "native_replay": True, "certificate": certificate,
                                          "case": asdict(case)})
            batches.append(opportunities)
        print(f"Verified {len(batches)} of 15 adjacent-cost training pairs.", flush=True)
    return batches


def phase_update(policy, episodes, optimizer):
    import torch
    from torch.nn import functional as F

    losses = []
    censored = 0
    for episode in episodes:
        decision, = episode["events"]
        costs = policy.phase_costs(canonical(decision["input"]), decision["candidate_menu"], episode["depth"])
        estimate = costs[decision["selected_index"]]
        # The first check opens both goals. The first one-goal state ends the equality phase.
        equality = next(i for i, event in enumerate(episode["path"]) if event["pending_goals"] == 1)
        proposition = episode["tactic_checks"] - 1 - equality
        equality_loss = F.smooth_l1_loss(estimate[0], estimate.new_tensor(equality))
        if episode["verified_closed"]:
            proposition_loss = F.smooth_l1_loss(estimate[1], estimate.new_tensor(proposition))
        else:
            censored += 1
            shortage = (estimate.new_tensor(proposition + 1) - estimate[1]).clamp_min(0)
            proposition_loss = F.smooth_l1_loss(shortage, shortage.new_zeros(()))
        losses.append(equality_loss + proposition_loss)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"decisions": len(losses), "loss": float(loss.detach()), "censored_phases": censored}


def fit(policy, arm, batches, dev, phase_credit, epochs):
    import torch

    from ppt_gym.hybrid_rl.__main__ import imitate, paired_action_value
    from ppt_gym.ram_learning import tensor_hash

    calibrated = arm.endswith("calibration")
    if calibrated:
        policy.graph.requires_grad_(False)
    passes = 4 if policy.architecture == "hybrid" else 1
    root_macs = policy.macs(2, passes)
    if root_macs > 1_200_000_000:
        raise ValueError("the mixed obligation readout exceeds the registered arithmetic ceiling")

    def readout():
        with torch.no_grad():
            predictions = {}
            for root, (packet, menu, _) in dev.items():
                scores, _ = policy(packet, menu, passes)
                predictions[root] = int(scores.argmax())
            return predictions

    start = readout()
    before = tensor_hash(policy.state_dict())
    graph_before = tensor_hash(policy.graph.state_dict())
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=0.001)
    updates = []
    for _ in range(epochs):
        for batch in batches:
            episodes = [{**episode, "depth": passes} for episode in batch]
            if arm.endswith("imitation"):
                update = imitate(policy, episodes, optimizer)
                if update["decisions"] != 2:
                    raise ValueError("imitation missed a certified root choice")
            elif phase_credit:
                update = phase_update(policy, episodes, optimizer)
            else:
                update = paired_action_value(policy, episodes, optimizer, deadline=None)
                if update["paired_values"] != 2:
                    raise ValueError("matched return fit missed a world with both native outcomes")
            updates.append(update)
    after = tensor_hash(policy.state_dict())
    if before == after:
        raise ValueError("the mixed proof policy did not update")
    if calibrated and graph_before != tensor_hash(policy.graph.state_dict()):
        raise ValueError("calibration changed the frozen obligation encoder")
    return {"initial": start, "final": readout(), "weights_before": before, "weights_after": after,
            "root_macs": root_macs, "updates": updates}


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.mixed_obligation import MixedArrivalCostPolicy, MixedObligationPolicy, MixedPhaseCostPolicy, obligation_views
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    native = json.loads(gzip.decompress(args.witness.read_bytes()))
    source = json.loads(args.source_audit.read_bytes())
    if native["schema"] != SCHEMA or len(native["rows"]) != 16:
        raise ValueError("expected the frozen mixed-obligation development witness")
    configure(tree.threads, 151)
    with LeanRepl(timeout=15) as repl:
        if repl.version != native["lean"]:
            raise ValueError("the native mixed verifier changed")
        dev = {}
        for root, row in native["rows"].items():
            packet, menu = root_packet(repl, SearchCase(**row["case"]))
            if digest(packet) != row["trials"][0]["events"][0]["packet"]:
                raise ValueError("the native development observation changed")
            obligation_views(packet, menu)
            dev[root] = packet, menu, row
        if args.training_evidence:
            recorded = json.loads(gzip.decompress(args.training_evidence.read_bytes()))
            if (recorded["witness_sha256"] != sha256_file(args.witness)
                    or recorded["source_audit_sha256"] != sha256_file(args.source_audit)
                    or len(recorded["batches"]) != 15
                    or any(len(batch) != 4 or not all(ep["native_replay"] and ep["within_cpu_budget"]
                                and ep["tactic_checks"] == BUDGET for ep in batch) for batch in recorded["batches"])):
                raise ValueError("the reused training evidence differs from the verified mixed pilot")
            batches = recorded["batches"]
        else:
            batches = collect(repl, tree)
    if set(dev) & {episode["root"] for batch in batches for episode in batch}:
        raise ValueError("mixed training overlapped the held-out proof layouts")
    controls = {arm: (model, checkpoint) for arm, model, checkpoint, _ in audited_policies(
        args.sessions, args.source_audit, source["cohort_sha256"], source["reference_sha256"],
        selected=("hybrid_rl", "trm_rl"))}
    results = {}
    if args.arrival_history:
        policy_type = MixedArrivalCostPolicy
    else:
        policy_type = MixedPhaseCostPolicy if args.phase_credit else MixedObligationPolicy
    args.checkpoints.mkdir(exist_ok=False)
    with LeanRepl(timeout=15) as repl:
        for arm in ("hybrid_rl", "hybrid_calibration", "hybrid_imitation", "trm_rl", "trm_calibration"):
            model, checkpoint = controls[arm.split("_", 1)[0] + "_rl"]
            policy = policy_type(model)
            result = fit(policy, arm, batches, dev, args.phase_credit, args.epochs)
            path = args.checkpoints / f"{arm}.pt"
            torch.save(policy.state_dict(), path)
            restored = policy_type(model)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if tensor_hash(restored.state_dict()) != result["weights_after"]:
                raise ValueError("the mixed architecture weights did not restore")
            by_pair = {}
            for root, (_, _, row) in dev.items():
                choice = result["final"][root]
                by_pair.setdefault(row["case"]["cluster"], []).append(row["trials"][choice]["certificate"] is not None)
            complete = sorted(cluster for cluster, worlds in by_pair.items() if all(worlds))
            selected = complete[0] if complete else min(by_pair)
            replays = {}
            with torch.no_grad():
                for root, (_, _, row) in dev.items():
                    case = SearchCase(**row["case"])
                    if case.cluster != selected:
                        continue
                    packet, menu = root_packet(repl, case)
                    passes = 4 if model.architecture == "hybrid" else 1
                    scores, _ = restored(packet, menu, passes)
                    choice = int(scores.argmax())
                    witness = row["trials"][choice]
                    script = [event["tactic"] for event in witness["events"]]
                    events, closed = native_prefix(repl, case, script)
                    certificate = certify(case, {"live_kernel": closed, "accepted_script": script})
                    if (choice != result["final"][root] or events != witness["events"]
                            or closed != (witness["certificate"] is not None)
                            or certificate != witness["certificate"]):
                        raise ValueError("the restored mixed policy did not reproduce its native branch")
                    replays[root] = {"choice": choice, "events": events, "certificate": certificate}
            counts = {"proofs": sum(sum(pair) for pair in by_pair.values()), "pairs": len(complete)}
            results[arm] = {**result, "initialization": checkpoint, "counts": counts, "native_reload": replays,
                            "checkpoint_sha256": sha256_file(path)}
            print(arm, counts, flush=True)
    return {"scope": "mixed-obligation development on held-out equality cost gaps", "witness_sha256": sha256_file(args.witness),
            "source_audit_sha256": sha256_file(args.source_audit), "batches": batches, "results": results,
            "training_evidence_sha256": sha256_file(args.training_evidence) if args.training_evidence else None,
            "phase_credit": args.phase_credit, "arrival_history": args.arrival_history, "epochs": args.epochs,
            "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "source_audit", "sessions", "output", "checkpoints"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--training-evidence", type=Path)
    parser.add_argument("--phase-credit", action="store_true")
    parser.add_argument("--arrival-history", action="store_true")
    parser.add_argument("--epochs", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.arrival_history and not args.phase_credit:
        parser.error("--arrival-history requires --phase-credit")
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
