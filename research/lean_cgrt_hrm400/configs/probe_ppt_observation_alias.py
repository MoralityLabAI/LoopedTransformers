"""Certify a pair of Lean tasks aliased by the current proof-policy observation."""
import argparse
import json
from pathlib import Path

from ppt_gym.data import digest
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def witness(side):
    branches = [[0, *range(3, 9)], [1, *range(9, 15)]]
    rules = [(0, 2), (1, 2)]
    for chain in branches:
        rules.extend((source, target) for target, source in zip(chain, chain[1:]))
    binders = ["(" + " ".join(f"P{i}" for i in range(15)) + " : Prop)"]
    binders.append(f"(h0 : P{branches[side][-1]})")
    binders.extend(f"(r{i} : P{source} → P{target})" for i, (source, target) in enumerate(rules))
    binders = " ".join(binders)
    return SearchCase("loopt_horn", "dev", "observation-alias-witness", digest(binders),
                      "alias-pair", f"alias_world_{side}", binders, "P2")


def evaluate(tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.hybrid_rl.model import ProofPolicy
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search, search_episode
    from ppt_gym.ram_campaign import configure

    configure(2, 0)
    models = [(ProofPolicy("hybrid", 256), 4), (ProofPolicy("trm", 256), 1)]
    trials = []
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("expected the pinned Lean verifier")
        for side in (0, 1):
            case = witness(side)
            for action in ("apply r0", "apply r1"):
                inputs, predictions = {}, {}
                first = None

                def rank(packet, menu, remaining):
                    nonlocal first
                    if remaining != 8:
                        return 0, False
                    first = {"packet": json.loads(packet), "candidates": list(menu)}
                    if set(menu) != {"apply r0", "apply r1"}:
                        raise AssertionError("the public initial choice changed")
                    for model, passes in models:
                        def capture(_module, args):
                            inputs[model.architecture] = list(args[0])
                        with model.text.register_forward_pre_hook(capture):
                            logits, value = model(packet, menu, passes)
                        predictions[model.architecture] = {
                            "logits": logits.tolist(), "value": float(value)}
                    return menu.index(action), True

                with torch.no_grad():
                    ep = search_episode(repl, case, None, policy="forced-root-witness", depth=4,
                                        check_budget=8, rollback_budget=8, ranker=rank)
                certificate = certify(case, ep)
                replay = replay_search(repl, case, ep)
                certified = bool(certificate and certificate["passed"])
                if certified != (action == f"apply r{side}") or ep["tactic_checks"] != 8:
                    raise AssertionError("the forced root action did not give the expected native result")
                trials.append({"source_root": case.root, "world": side, "root_action": action,
                               "first_observation": first, "encoded_inputs": inputs,
                               "native_transcript": ep, "predictions": predictions,
                               "certificate": certificate, "replay": replay, "certified": certified})
        verifier = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)

    for trial in trials[1:]:
        if trial["encoded_inputs"] != trials[0]["encoded_inputs"] or trial["predictions"] != trials[0]["predictions"]:
            raise AssertionError("expected an exact collision in the two policy inputs and scores")
    if trials[0]["first_observation"]["packet"] == trials[2]["first_observation"]["packet"]:
        raise AssertionError("the counterexamples must have different public premise graphs")
    return {"scope": "exact finite witness on two equally weighted axiom-free Lean worlds",
            "maximum_first_action_success_from_current_observation": "1/2",
            "proof_checks": 8, "verifier": verifier, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        result = evaluate(tree)
        result["memory"] = tree.stats()
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
