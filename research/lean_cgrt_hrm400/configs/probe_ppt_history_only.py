"""Check a stronger recovery control that retains only the history margin."""

import argparse
from collections import Counter
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.freeze_ppt_pending_recovery import ARMS, SEEDS
from configs.probe_ppt_recovery_components import single_goal_witness
from configs.run_ppt_pending_recovery import frozen_policy, verified
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def suppress_readout(_module, _inputs, scores):
    return scores * 0


def saved_evaluation(session, manifest, roots):
    from ppt_gym import ram_store

    with ram_store.open_store(session) as db:
        if ram_store.read_manifest(db) != manifest:
            raise ValueError("the original evaluation checkpoint manifest changed")
        episodes = [ep for row in ram_store.ledger(db) if row["phase"] == "final" for ep in row["episodes"]]
    if len(episodes) != len(roots) or {ep["root"] for ep in episodes} != set(roots):
        raise ValueError("the original evaluation ledger is incomplete")
    return {ep["root"]: ep for ep in episodes}


def same_continuation(policy, episode):
    for event in episode["events"]:
        if event["kind"] != "tactic" or event["attempt"] == 1 or len(event["candidate_menu"]) == 1:
            continue
        logits, _ = policy(canonical(event["input"]), event["candidate_menu"], episode["depth"])
        if int(logits.argmax()) != event["selected_index"]:
            return False
    return True


def evaluate_policies(tree, registration, cases, witness, sessions, arms, *, native, reuse_identical):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.search_control import DirectFactContinuation
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 0)
    if set(cases) != set(witness):
        raise ValueError("history-only control and native branch witness differ")
    results, counts, hashes = {}, {}, {}
    repl_context = LeanRepl(timeout=15) if native else nullcontext()
    with torch.no_grad(), repl_context as repl:
        for seed, directory in sessions:
            label = str(seed)
            results[label], counts[label], hashes[label] = {}, {}, {}
            for arm in arms:
                session = directory / f"{arm}.sqlite"
                model, manifest, _, weights = frozen_policy(session, registration, seed, arm)
                hashes[label][arm] = weights
                if repl is not None and fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != manifest["verifier"]:
                    raise ValueError("native history-only control requires the original Lean verifier")
                rows, choices, proofs = {}, Counter(), Counter()
                original = saved_evaluation(session, manifest, cases) if reuse_identical else {}
                reused = 0
                options = SimpleNamespace(max_macs=manifest["max_macs"], cpu_seconds=manifest["cpu_seconds"])
                if model.macs(2, 4) + model.recovery_macs(4) > options.max_macs:
                    raise ValueError("history-only recovery exceeds the registered budget")
                policy = DirectFactContinuation(model)
                with model.recovery_head.readout.register_forward_hook(suppress_readout):
                    for root, (cluster, observation, winner) in witness.items():
                        logits, _ = model.recover(canonical(observation), 4)
                        if not torch.isfinite(logits).all():
                            raise ValueError("history-only control produced a nonfinite margin")
                        choice = int(logits.argmax())
                        choices[cluster] += choice == winner
                        receipt = {"cluster": cluster, "choice": choice, "winner": winner}
                        if repl is not None:
                            saved = original.get(root)
                            saved_choice = (next(event["selected_index"] for event in saved["events"]
                                                 if event["kind"] == "recovery") if saved is not None else None)
                            same_choice = (saved_choice == choice and saved["within_cpu_budget"]
                                           and same_continuation(policy, saved))
                            ep = saved if same_choice else rollout(repl, tree, cases[root], policy, options)
                            record, = (event for event in ep["events"] if event["kind"] == "recovery")
                            replay = ep["native_replay"] if same_choice else replay_search(repl, cases[root], ep)
                            if (record["input"] != observation or record["selected_index"] != choice
                                    or not ep["within_cpu_budget"] or not replay["replay_passed"]):
                                raise ValueError("native history-only evaluation deviated from the frozen choice")
                            ep["native_replay"] = replay
                            certified = verified(ep)
                            if certified and choice != winner:
                                raise ValueError("native proof contradicted the recovery witness")
                            reused += same_choice
                            proofs[cluster] += certified
                            receipt.update(certified=certified, script=ep["accepted_script"],
                                           checks=ep["tactic_checks"], certificate=ep["certificate"],
                                           native_replay=replay, macs=ep["model_macs"], reused_identical_choice=same_choice)
                        rows[root] = receipt
                summary = {"correct_choices": sum(choices.values()),
                           "complete_viable_choice_pairs": sum(value == 2 for value in choices.values())}
                if repl is not None:
                    summary.update(certified_proofs=sum(proofs.values()),
                                   complete_certified_pairs=sum(value == 2 for value in proofs.values()),
                                   reused_identical_choices=reused)
                results[label][arm], counts[label][arm] = rows, summary
                print(seed, arm, summary, flush=True)
    return {"native_end_to_end": native, "weights_sha256": hashes,
            "counts": counts, "roots": results, "memory": tree.stats()}


def evaluate(args, tree):
    registration = json.loads(args.pending_cohort.read_bytes())["registration"]
    witness = single_goal_witness(args.cohort, args.witness, registration["previous_cohort_sha256"])
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    hybrid = [name for name in ARMS if name.startswith("hybrid_")]
    result = evaluate_policies(tree, registration, cases, witness, zip(SEEDS, args.sessions, strict=True), hybrid,
                               native=args.native, reuse_identical=args.reuse_identical)
    return {"scope": "single-goal development control with parent/current readout contribution zeroed",
            "cohort_sha256": sha256_file(args.cohort), "witness_sha256": sha256_file(args.witness), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "witness", "pending_cohort", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("sessions", type=Path, nargs=len(SEEDS))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--reuse-identical", action="store_true")
    args = parser.parse_args()
    if args.reuse_identical and not args.native:
        parser.error("--reuse-identical requires --native")
    tree = TreeBudget(2, 2)
    try:
        report = evaluate(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
