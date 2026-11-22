"""Lock trained checkpoints and score the independent branched Lean readout."""
import argparse
from collections import Counter
import json
from pathlib import Path

from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.analyze_ppt_hybrid_replication import read_arm
from configs.freeze_ppt_transport_confirmation import ARMS, SCHEMA
from ppt_gym.data import sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase, load
from ppt_gym.ram_limits import TreeBudget

KINDS = {"transport": ("hybrid", False, True), "transport-head": ("hybrid", True, True),
         "trm": ("trm", False, False), "head-only": ("trm", True, False)}


def read_seal(path):
    seal = json.loads(path.read_bytes())
    if seal["schema"] != SCHEMA or tuple(seal["registration"]["arms"]) != ARMS:
        raise ValueError("unexpected confirmation registry")
    cases = [SearchCase(**case) for case in seal["cases"]]
    groups = Counter(case.cluster for case in cases)
    if (len(groups) != seal["registration"]["independent_pairs"] or set(groups.values()) != {2}
            or len({case.root for case in cases}) != len(cases)):
        raise ValueError("each registered structural pair must contain both distinct worlds")
    return seal, cases


def lock_models(args):
    seal, _ = read_seal(args.seal)
    development_sha = sha256_file(args.development)
    if development_sha != seal["development_sha256"]:
        raise ValueError("development cohort differs from the seal")
    cases = {case.root: case for case in load(args.development)["dev"]}
    clusters = {}
    for root, case in cases.items():
        clusters.setdefault(case.cluster, []).append(root)
    if len(clusters) != 12 or any(len(roots) != 2 for roots in clusters.values()):
        raise ValueError("development activation requires 12 complete world pairs")
    seed = seal["registration"]["checkpoint_seed"]
    receipt = {"seal_sha256": sha256_file(args.seal), "arms": {}, "implementation": implementation()}
    pairs_solved = {}
    for arm in ARMS:
        path = args.pilot / f"{seed}-{arm}.sqlite"
        trial = read_arm(path, cases, development_sha)
        pilot = trial["manifest"]
        verify_kind(pilot, arm, seal)
        pairs_solved[arm] = pair_counts(trial["stages"]["final"], clusters)["solved_both_worlds"]
        receipt["arms"][arm] = {"session_sha256": trial["session_sha256"],
                                "weights_sha256": trial["final_weights_sha256"],
                                "verifier": pilot["verifier"]}
    if pairs_solved["transport"] <= pairs_solved["trm"]:
        raise ValueError("registered development activation gate did not pass")
    receipt["pilot_both_worlds_solved"] = pairs_solved
    with args.output.open("x", encoding="utf-8") as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def implementation():
    from ppt_gym import hybrid_rl

    folder = Path(hybrid_rl.__file__).parent
    return {p.name: sha256_file(p) for p in [Path(__file__), *sorted(folder.glob("*.py"))]}


def verify_kind(pilot, arm, seal):
    fields = pilot["architecture"], pilot["head_only"], pilot["residual_fact_transport"]
    reg = seal["registration"]
    if (fields != KINDS[arm] or not pilot["graph_context"] or pilot["seed"] != reg["checkpoint_seed"]
            or pilot["cohort"] != seal["development_sha256"] or pilot["cpu_seconds"] != reg["proof_cpu_seconds"]
            or pilot["max_macs"] != reg["max_macs"] or pilot["objective"] != "rl"
            or pilot.get("alias_training", "standard") != reg.get("alias_training", "standard")):
        raise ValueError("pilot model differs from the registered matched arm")


def evaluate(args, budget):
    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym import ram_store
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    seal, cases = read_seal(args.seal)
    lock = json.loads(args.lock.read_bytes())
    if lock["seal_sha256"] != sha256_file(args.seal) or lock["implementation"] != implementation():
        raise ValueError("confirmation protocol changed after locking the models")
    entry = lock["arms"][args.arm]
    if sha256_file(args.session) != entry["session_sha256"]:
        raise ValueError("the checkpoint session differs from the pre-readout lock")
    with ram_store.open_store(args.session) as source:
        pilot = ram_store.read_manifest(source)
        weights = ram_store.checkpoint(source)["model"]
    verify_kind(pilot, args.arm, seal)
    if tensor_hash(weights) != entry["weights_sha256"]:
        raise ValueError("locked weights changed")
    configure(args.threads, pilot["seed"])
    model = GraphProofPolicy(pilot["architecture"], pilot["width"], residual_transport=pilot["residual_fact_transport"])
    model.load_state_dict(weights)
    args.cpu_seconds = pilot["cpu_seconds"]
    args.max_macs = pilot["max_macs"]
    with LeanRepl(timeout=15) as repl, ram_store.open_store(args.output, write=True) as db:
        verifier = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        if verifier != entry["verifier"]:
            raise ValueError("confirmation requires the locked native Lean toolchain")
        ram_store.initialize(db, {"arm": args.arm, "lock_sha256": sha256_file(args.lock), **entry,
                                  "threads": args.threads, "cases": len(cases)})
        logged = [row["episode"]["root"] for row in ram_store.ledger(db)]
        if logged != [case.root for case in cases[:len(logged)]]:
            raise ValueError("confirmation ledger does not match the sealed case order")
        for case in cases[len(logged):]:
            ep = rollout(repl, budget, case, model, args)
            ep["native_replay"] = replay_search(repl, case, ep)
            with db:
                ram_store.append(db, {"episode": ep, "weights_sha256": entry["weights_sha256"]})
    print(f"{args.arm} finished {len(cases)} locked cases without emitting the readout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("seal", type=Path)
    lock.add_argument("pilot", type=Path)
    lock.add_argument("development", type=Path)
    lock.add_argument("output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("seal", type=Path)
    run.add_argument("lock", type=Path)
    run.add_argument("arm", choices=ARMS)
    run.add_argument("session", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--ram", type=int, choices=(2, 4, 6), default=2)
    args = parser.parse_args()
    if args.command == "lock":
        lock_models(args)
        return
    budget = TreeBudget(args.ram, args.threads)
    try:
        evaluate(args, budget)
    finally:
        budget.close()


if __name__ == "__main__":
    main()
