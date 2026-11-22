"""Compare bundle-opening update rules on identical native Lean branches."""

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.check_ppt_bpd_bundled_gym import OpeningWitness, opening_degrees, verified_distance_schedule
from configs.probe_ppt_bpd_rewrite_revisits import warm_policies
from ppt_gym.bundled_rewrite import BPD_BUNDLE_FAMILY
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def prepare(cohort, reference):
    from ppt_gym.hybrid_rl.__main__ import _paired_training_batches

    registration = json.loads(cohort.read_bytes())["registration"]
    witness = json.loads(reference.read_bytes())
    data = load(cohort)
    if (not registration.get("balanced_endpoint_degrees")
            or witness["cohort_sha256"] != sha256_file(cohort)
            or witness["matched_endpoint_degree_worlds"] != len(data["dev"])
            or not verified_distance_schedule(registration, witness)
            or set(witness["rows"]) != {case.root for case in data["dev"]}):
        raise ValueError("expected the verified endpoint-matched bundle cohort")
    seed, = registration["model_seeds"]
    training = [group for group in _paired_training_batches(data["train"], seed)
                if group[0].family == BPD_BUNDLE_FAMILY]
    identity = {"cohort_sha256": sha256_file(cohort), "reference_sha256": sha256_file(reference),
                "policy_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "ppt_gym/hybrid_rl/__main__.py"),
                "seed": seed}
    return registration, training, data["dev"], witness, identity


def check_training(groups, batches):
    for group, episodes in zip(groups, batches, strict=True):
        roots = {case.root: int(case.tid.rsplit("_", 1)[1]) for case in group}
        counts = Counter()
        for episode in episodes:
            event = episode["events"][0]
            first = event["selected_index"]
            winner = roots[episode["root"]]
            passed = episode["verified_closed"]
            certificate = episode["certificate"]
            if (not episode["bundle_continuation"] or not episode["within_cpu_budget"]
                    or episode.get("reward_censored") or episode["tactic_checks"] != 8
                    or not episode["native_replay"]["replay_passed"] or passed != (first == winner)
                    or (passed and (not certificate["passed"] or certificate["axioms"]
                                    or digest(certificate["source"]) != certificate["source_digest"]))):
                raise ValueError("a training branch differs from its verified opening outcome")
            opening_degrees(event["input"]["state"]["goals"][0])
            counts[(episode["root"], first)] += 1
        if counts != {(root, first): 1 for root in roots for first in (0, 1)}:
            raise ValueError("both opening proposals are required in every training world")


def collect(cohort, reference, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import paired_alias_rollouts
    from ppt_gym.live_lean import candidate_menu, state_input
    from ppt_gym.ram_campaign import configure

    registration, groups, dev, witness, identity = prepare(cohort, reference)
    seed = identity["seed"]
    configure(tree.threads, seed)
    rng = torch.Generator().manual_seed(seed)
    options = SimpleNamespace(seed=seed, alias_training="contrast-search", rollouts=registration["rollouts"],
                              cpu_seconds=registration["cpu_seconds"], max_macs=1_200_000_000,
                              bundle_continuation=True)
    model = OpeningWitness()
    batches, observations = [], {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != witness["lean"]:
            raise ValueError("the Lean verifier differs from the native reference")
        for index, group in enumerate(groups, 1):
            episodes = paired_alias_rollouts(repl, tree, group, model, options, rng)
            check_training([group], [episodes])
            batches.append(episodes)
            if index % 8 == 0:
                print(f"Verified {index} of {len(groups)} training layout pairs.", flush=True)
        for case in dev:
            repl.reset()
            initial = repl.init(case.statement, pp=True)
            menu = list(candidate_menu(repl, initial, rewrite=True, bundles=True))
            if menu != witness["rows"][case.root]["proposals"][0]["menu"]:
                raise ValueError("the development menu differs from the verified native witness")
            packet, _ = state_input(repl, initial, (), registration["tactic_checks"])
            observations[case.root] = {"packet": packet, "menu": menu}
    return {**identity, "lean": witness["lean"], "batches": batches, "observations": observations,
            "memory": tree.stats()}


def score(model, observations, witness):
    import torch

    passes = 4 if model.architecture == "hybrid" else 1
    rows = {}
    with torch.no_grad():
        for root, observation in observations.items():
            logits, _ = model(observation["packet"], observation["menu"], passes)
            if not torch.isfinite(logits).all():
                raise ValueError("the opening score is nonfinite")
            opening = int(logits.argmax())
            choice = witness[root]["proposals"][opening]
            certified = choice["certificate"] is not None
            if (not choice["replay"]["replay_passed"] or certified != (opening == witness[root]["winner"])
                    or (certified and (not choice["certificate"]["passed"] or choice["certificate"]["axioms"]))):
                raise ValueError("the selected opening lacks a matching native outcome")
            rows[root] = {"opening": opening, "certified": certified, "margin": float(logits[0] - logits[1])}
    return rows


def verify_reloaded(repl, tree, model, cases, rows, witness, registration):
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search

    options = SimpleNamespace(cpu_seconds=registration["cpu_seconds"], max_macs=1_200_000_000,
                              bundle_continuation=True)
    verified = {}
    for case in cases:
        episode = rollout(repl, tree, case, model, options)
        replay = replay_search(repl, case, episode)
        expected = rows[case.root]
        proposal = witness[case.root]["proposals"][expected["opening"]]
        if (not episode["within_cpu_budget"] or not replay["replay_passed"]
                or episode["verified_closed"] != expected["certified"]
                or episode["events"][0]["selected_index"] != expected["opening"]
                or episode["accepted_script"] != proposal["script"]):
            raise ValueError("a fresh native rollout disagrees with the shared branch evidence")
        verified[case.root] = {"checks": episode["tactic_checks"], "script": episode["accepted_script"],
                               "certificate": episode["certificate"], "replay": replay}
    return verified


def fit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import imitate, paired_credit
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    reg, groups, dev, reference, identity = prepare(args.cohort, args.reference)
    configure(tree.threads, identity["seed"])
    evidence = json.loads(gzip.decompress(args.evidence.read_bytes()))
    observations, witnesses = evidence["observations"], reference["rows"]
    if (any(evidence.get(key) != value for key, value in identity.items()) or evidence["lean"] != reference["lean"]
            or set(observations) != set(witnesses)
            or any(observation["menu"] != witnesses[root]["proposals"][0]["menu"]
                   for root, observation in observations.items())):
        raise ValueError("the shared training branches do not belong to this frozen experiment")
    batches = evidence["batches"]
    check_training(groups, batches)

    by_layout = defaultdict(list)
    for case in dev:
        by_layout[case.cluster].append(case)
    warm = {arch: (model, source) for arch, model, source, _ in warm_policies(args.initialization, reg, identity["seed"])}
    readout_prefixes = ("actor.", "critic.", "graph.action.", "graph.value.", "graph.transport.action.")
    results = {}
    args.checkpoints.mkdir(exist_ok=False)
    with LeanRepl(timeout=15) as repl:
        if repl.version != reference["lean"]:
            raise ValueError("the verifier toolchain changed")
        for arm in reg["arms"]:
            architecture, objective = arm.split("_", 1)
            initial, source = warm[architecture]
            model = copy.deepcopy(initial)
            passes = 4 if architecture == "hybrid" else 1
            if objective == "calibration":
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(name.startswith(readout_prefixes))
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=reg["lr"])
            stages = {"initial": score(model, observations, witnesses)}
            weights_before = tensor_hash(model.state_dict())
            updates = []
            for batch in batches:
                episodes = [{**episode, "depth": passes} for episode in batch]
                update = (imitate(model, episodes, optimizer) if objective == "imitation" else
                          paired_credit(model, episodes, optimizer, continue_search=True))
                expected = 2 if objective == "imitation" else 4
                if update["decisions"] != expected:
                    raise ValueError("the update skipped registered native decisions")
                updates.append(update)
            state = model.state_dict()
            weights_after = tensor_hash(state)
            if weights_before == weights_after:
                raise ValueError("the policy weights did not change")
            if objective == "calibration":
                original = initial.state_dict()
                if any(not torch.equal(value, original[name]) for name, value in state.items()
                       if not name.startswith(readout_prefixes)):
                    raise ValueError("calibration changed a frozen feature weight")
            destination = args.checkpoints / f"{arm}.pt"
            torch.save(state, destination)
            restored = copy.deepcopy(initial)
            restored.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True))
            if tensor_hash(restored.state_dict()) != weights_after:
                raise ValueError("the persisted run did not restore the updated CPU weights")
            stages["final"] = score(restored, observations, witnesses)
            pairs = {phase: sorted(layout for layout, cases in by_layout.items()
                                   if all(rows[case.root]["certified"] for case in cases))
                     for phase, rows in stages.items()}
            improved = sorted(set(pairs["final"]) - set(pairs["initial"]))
            verification = by_layout[(improved or pairs["final"] or sorted(by_layout))[0]]
            with torch.no_grad():
                verified = verify_reloaded(repl, tree, restored, verification, stages["final"], witnesses, reg)
            results[arm] = {"initialization": source, "weights_before": weights_before, "weights_after": weights_after,
                            "native_reload": verified, "checkpoint_sha256": sha256_file(destination),
                            "training_decisions": sum(update["decisions"] for update in updates), "updates": updates,
                            "stages": stages, "pairs": pairs, "trainable_parameters": sum(p.numel() for p in trainable)}
            print(arm, {phase: (sum(row["certified"] for row in rows.values()), len(pairs[phase]))
                        for phase, rows in stages.items()}, flush=True)
    return {**identity, "scope": "shared native branch replay on new bundle training layouts",
            "training_evidence_sha256": sha256_file(args.evidence), "training_layout_pairs": len(groups),
            "development_layout_pairs": len(by_layout), "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "output"):
        parser.add_argument(name, type=Path)
    for name in ("evidence", "initialization", "checkpoints"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if any((args.evidence, args.initialization, args.checkpoints)) and not all(
            (args.evidence, args.initialization, args.checkpoints)):
        parser.error("supply all of --evidence, --initialization and --checkpoints to train the matched policies")
    tree = TreeBudget(2, 1)
    try:
        report = fit(args, tree) if args.evidence else collect(args.cohort, args.reference, tree)
        payload = gzip.compress(canonical(report).encode("utf-8"), mtime=0)
        with args.output.open("xb") as output:
            output.write(payload)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
