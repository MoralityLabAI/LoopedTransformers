"""Score frozen keep-or-abandon decisions against audited native branches."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_pending_recovery import audit_edits
from configs.freeze_ppt_pending_recovery import ARMS, EDITS, SEEDS
from configs.run_ppt_pending_recovery import frozen_cohort, frozen_policy, history_edit, recovery_event, verified
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


def checked_branches(cohort, witness):
    registration, groups = frozen_cohort(cohort)
    source = json.loads(gzip.decompress(witness.read_bytes()))
    if source["cohort_sha256"] != sha256_file(cohort) or set(source["rows"]) != set(groups):
        raise ValueError("recorded branches do not cover the frozen cohort")
    branches = {}
    for cluster, cases in groups.items():
        entries = source["rows"][cluster]
        if [e["root"] for e in entries] != [case.root for case in cases]:
            raise ValueError("witness changed the paired theorem roots")
        for entry in entries:
            by_choice = []
            observation = None
            for choice, episode in enumerate(entry["forced"]):
                event = recovery_event(episode)
                first, *later = [step for step in episode["events"] if step["kind"] == "tactic"]
                if (episode["root"] != entry["root"] or first["candidate_menu"] != event["input"]["parent_menu"]
                        or first["selected_index"] != 0 or event["selected_index"] != choice
                        or any(len(step["candidate_menu"]) != 1 or step["selected_index"] != 0 for step in later)):
                    raise ValueError("a recorded branch contains a model-dependent tactic after recovery")
                if observation is not None and observation != event["input"]:
                    raise ValueError("the two recorded actions start at different checkpoints")
                observation = event["input"]
                success = verified(episode)
                if success != (choice == entry["winner"]):
                    raise ValueError("inconsistent certified branch label")
                by_choice.append({
                    "certified": success, "checks": episode["tactic_checks"],
                    "script_sha256": digest(episode["accepted_script"]),
                    "certificate_sha256": episode["certificate"]["source_digest"] if success else None,
                })
            if len(by_choice) != 2:
                raise ValueError("a keep-or-abandon decision requires both native branches")
            branches[entry["root"]] = (cluster, observation, by_choice)
    return registration, branches


def summarize(roots, branches):
    by_pair = Counter()
    for root, outcome in roots.items():
        if outcome["certified"]:
            by_pair[branches[root][0]] += 1
    return {"complete_pairs": sum(n == 2 for n in by_pair.values()), "proofs": sum(by_pair.values())}


def evaluate(cohort, witness, directories, tree):
    import torch

    from ppt_gym.ram_campaign import configure, horn_choice

    configure(tree.threads, 0)
    registration, branches = checked_branches(cohort, witness)
    solver = {}
    for root, (_, observation, outcomes) in branches.items():
        best_root = horn_choice(canonical(observation["parent"]), observation["parent_menu"])
        choice = int(best_root != observation["explored"])
        solver[root] = {"choice": choice, **outcomes[choice]}

    counts, actions, checkpoint_hashes = {}, {}, {}
    with torch.no_grad():
        for seed, directory in zip(SEEDS, directories, strict=True):
            label = str(seed)
            counts[label], actions[label], checkpoint_hashes[label] = {}, {}, {}
            for arm in ARMS:
                model, manifest, _, weights_sha = frozen_policy(directory / f"{arm}.sqlite", registration, seed, arm)
                checkpoint_hashes[label][arm] = weights_sha
                passes = 4 if model.architecture == "hybrid" else 1
                macs = model.macs(2, passes) + model.recovery_macs(passes)
                if macs > manifest["max_macs"]:
                    raise ValueError("the forced first tactic and recovery exceed the model budget")
                modes = EDITS if arm == ARMS[0] else (EDITS[0],)
                rows = {}
                for root, (cluster, observation, outcomes) in branches.items():
                    rows[root] = {}
                    for mode in modes:
                        edits = []
                        with history_edit(model, mode, cluster, seed, edits):
                            logits, _ = model.recover(canonical(observation), passes)
                        if not torch.isfinite(logits).all():
                            raise ValueError("nonfinite recovery score")
                        choice = int(logits.argmax())
                        rows[root][mode] = {"choice": choice, **outcomes[choice], "edit": edits, "macs": macs}
                    if arm == ARMS[0]:
                        audit_edits(rows[root], cluster, seed)
                actions[label][arm] = rows
                for mode in modes:
                    name = arm if mode == "pristine" else mode
                    counts[label][name] = summarize({root: row[mode] for root, row in rows.items()}, branches)
                print(label, arm, counts[label][arm], flush=True)
    return {
        "scope": "frozen recovery decisions over native-checked branches with singleton continuations",
        "cohort_sha256": sha256_file(cohort), "witness_sha256": sha256_file(witness),
        "model_weights_sha256": checkpoint_hashes, "counts": counts, "actions": actions,
        "public_horn_solver": {"counts": summarize(solver, branches), "actions": solver},
        "memory": tree.stats(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "witness", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("sessions", type=Path, nargs=len(SEEDS))
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        result = evaluate(args.cohort, args.witness, args.sessions, tree)
        args.output.write_bytes(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
        print("public Horn solver", result["public_horn_solver"]["counts"])
    finally:
        tree.close()


if __name__ == "__main__":
    main()
