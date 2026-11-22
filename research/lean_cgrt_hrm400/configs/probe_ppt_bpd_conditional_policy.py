"""Train and compare the conditional rewrite graph on the same Lean returns."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_conditional_rewrites import BUDGET, LAYOUTS, paired_world, play
from configs.freeze_ppt_bpd_rewrite import ARMS
from configs.probe_ppt_bpd_mixed_policy import fit
from configs.probe_ppt_bpd_rewrite_revisits import audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def readout_only(policy):
    policy.requires_grad_(False)
    for module in (policy.actor, policy.critic, policy.graph.action, policy.graph.value):
        module.requires_grad_(True)
    if policy.graph.transport is not None:
        policy.graph.transport.action.requires_grad_(True)


def training_batches(raw, witness, witness_path):
    if raw["witness_sha256"] != sha256_file(witness_path) or raw["tactic_checks"] != BUDGET:
        raise ValueError("conditional training uses a different native witness")
    groups = defaultdict(list)
    forbidden = {row["case"]["cluster"] for row in witness["rows"].values()}
    for root, row in raw["rows"].items():
        case = SearchCase(**row["case"])
        if case.root != root or case.split != "train" or case.cluster in forbidden:
            raise ValueError("an invalid or held-out conditional proof entered training")
        if {branch["choice"] for branch in row["branches"]} != {0, 1} or sum(b["closed"] for b in row["branches"]) != 1:
            raise ValueError("the verifier did not give both conditional root outcomes")
        packet = json.loads(row["packet"])
        for branch in row["branches"]:
            cert = branch["certificate"]
            if (not branch["replay_passed"] or len(branch["events"]) != BUDGET
                    or digest(packet["state"]) != branch["events"][0]["before"]
                    or (branch["closed"] and (not cert["passed"] or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]))):
                raise ValueError("a conditional update lacks authoritative native evidence")
            event = {"input": packet, "candidate_menu": row["menu"], "selected_index": branch["choice"], "kind": "tactic",
                     "attempt": 1, "status": "accepted_nonterminal"}
            groups[case.cluster].append({"root": root, "family": case.family, "tactic_checks": BUDGET,
                                         "verified_closed": branch["closed"], "within_cpu_budget": True, "events": [event]})
    if len(groups) != 27 or any(len(group) != 4 or len({row["root"] for row in group}) != 2 for group in groups.values()):
        raise ValueError("the disjoint conditional training schedule is incomplete")
    return [groups[cluster] for cluster in sorted(groups)]


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.conditional_rewrite import ConditionalRewritePolicy, project_equalities
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.live_lean import public_state, state_input
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash
    from ppt_gym.rewrite import rewrite_menu

    witness = json.loads(gzip.decompress(args.witness.read_bytes()))
    evidence = json.loads(gzip.decompress(args.training.read_bytes()))
    batches = training_batches(evidence, witness, args.witness)
    source = json.loads(args.audit.read_bytes())
    configure(tree.threads, source["seed"])
    controls = {arm.split("_", 1)[0]: (model, checkpoint) for arm, model, checkpoint, _ in audited_policies(
        args.sessions, args.audit, source["cohort_sha256"], source["reference_sha256"], selected=("hybrid_rl", "trm_rl"),
    )}
    args.checkpoints.mkdir(exist_ok=False)
    results = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != witness["lean"]:
            raise ValueError("the conditional proof verifier changed")
        dev = {}
        for root, row in witness["rows"].items():
            repl.reset()
            state = repl.init(SearchCase(**row["case"]).statement, pp=True)
            packet, identity = state_input(repl, state, (), BUDGET)
            projected, _ = project_equalities(public_state(repl, state)["goals"][0])
            menu = rewrite_menu(projected)
            if identity != row["branches"][0]["events"][0]["before"]:
                raise ValueError("the conditional development packet changed")
            dev[root] = packet, menu, row
        for arm in ARMS:
            old, source_checkpoint = controls[arm.split("_", 1)[0]]
            hybrid = old.architecture == "hybrid"
            model = ConditionalRewritePolicy(old.architecture, old.width, residual_transport=hybrid, premise_interaction=hybrid).eval()
            model.load_state_dict(old.state_dict())
            result = fit(model, arm, batches, dev, False, 4, calibrate=readout_only)
            path = args.checkpoints / f"{arm}.pt"
            torch.save(model.state_dict(), path)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if tensor_hash(model.state_dict()) != result["weights_after"]:
                raise ValueError("the trained conditional policy changed on CPU reload")
            pairs = defaultdict(list)
            for root, (_, _, row) in dev.items():
                choices = row["branches"]
                pairs[row["case"]["cluster"]].append(choices[result["final"][root]]["closed"])
            trials = sorted(key for key, values in pairs.items() if all(values))
            sample = trials[0] if trials else min(pairs)
            replays = {}
            with torch.no_grad():
                for root, (packet, menu, row) in dev.items():
                    if row["case"]["cluster"] != sample:
                        continue
                    choice = int(model(packet, menu, 4 if hybrid else 1)[0].argmax())
                    layout = int(row["case"]["tid"].split("_")[-2])
                    case, scripts, horn = paired_world(layout, LAYOUTS[layout], row["winner"])
                    native = play(repl, case, scripts[choice], horn)
                    expected = row["branches"][choice]
                    certificate = certify(case, {"live_kernel": native["closed"],
                                                 "accepted_script": [item["tactic"] for item in native["events"]]})
                    if (choice != result["final"][root] or case.root != root
                            or canonical(native["events"]) != canonical(expected["events"])
                            or certificate != expected["certificate"]):
                        raise ValueError("the reloaded policy differs from its native conditional path")
                    replays[root] = {"choice": choice, "events": native["events"], "certificate": certificate}
            counts = {"proofs": sum(sum(values) for values in pairs.values()), "pairs": len(trials)}
            results[arm] = {**result, "counts": counts, "source_checkpoint": source_checkpoint,
                            "checkpoint_sha256": sha256_file(path), "native_reload": replays}
            print(arm, counts, flush=True)
    return {"scope": "conditional rewrite development with held-out layouts and matched native branch replay",
            "witness_sha256": sha256_file(args.witness), "training_sha256": sha256_file(args.training),
            "source_audit_sha256": sha256_file(args.audit), "seed": source["seed"], "training_pairs": len(batches),
            "epochs": 4, "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "training", "audit", "sessions", "output", "checkpoints"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
