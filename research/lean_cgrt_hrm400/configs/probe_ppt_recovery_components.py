"""Locate which recovery-history readouts help or hurt a verified branch choice."""

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import json
import math
from pathlib import Path
from statistics import median

from configs.freeze_ppt_pending_recovery import ARMS, SEEDS
from configs.replay_ppt_pending_choices import checked_branches
from configs.run_ppt_pending_recovery import frozen_policy
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


COMPONENTS = ("linear", "interaction", "arrival", "history")


def sham_direction(cluster, seed, name):
    key = digest(["recovery-component-sham-v1", cluster, seed, name])
    return 1 if int(key[0], 16) % 2 == 0 else -1


def single_goal_witness(cohort, witness, expected):
    reference = json.loads(witness.read_bytes())
    if reference["cohort_sha256"] != expected or sha256_file(cohort) != expected:
        raise ValueError("the single-goal witness differs from the model's training cohort")
    registered = json.loads(cohort.read_bytes())["registration"]["structural_pairs"]["dev"]
    if set(reference["pairs"]) != {digest(["early-branch-recovery-v1", shape]) for shape in registered}:
        raise ValueError("the witness differs from the registered development layouts")
    rows = {}
    for cluster, pair in reference["pairs"].items():
        if len(pair) != 2 or sorted(row["winner"] for row in pair) != [0, 1]:
            raise ValueError("the single-goal worlds lack contrasting viable decisions")
        for row in pair:
            forced = row["outcomes"]
            winner = row["winner"]
            observation = forced[0]["observation"]
            certificate = forced[winner]["certificate"]
            if (observation != forced[1]["observation"] or observation["explored"] != 0
                    or len(observation["current"]["state"]["goals"]) != 1
                    or [trial["success"] for trial in forced].count(True) != 1
                    or not forced[winner]["success"] or not certificate["passed"] or certificate["axioms"] != []
                    or digest(certificate["source"]) != certificate["source_digest"]
                    or not all(trial["native"]["replay_passed"] for trial in forced)):
                raise ValueError("the single-goal branch lacks a verified unique recovery action")
            rows[row["root"]] = (cluster, observation, winner)
    return rows


def observed_components(model, observation, passes):
    import torch

    from ppt_gym.hybrid_rl.graph_model import visible_graph

    head = model.recovery_head
    transport = head.transport
    captured = {}

    def capture(name):
        def retain(_module, _inputs, scores):
            if name in captured:
                raise ValueError("the recovery component was evaluated twice")
            captured[name] = scores
        return retain

    with ExitStack() as stack:
        stack.enter_context(head.history.register_forward_hook(capture("history")))
        stack.enter_context(transport.interaction_readout.register_forward_hook(capture("interaction")))
        stack.enter_context(transport.arrival_readout.register_forward_hook(capture("arrival")))
        logits, _ = model.recover(canonical(observation), passes)
    graph = visible_graph(canonical(observation["parent"]), observation["parent_menu"])
    if graph is None or not all(graph.actions) or set(captured) != {"history", "interaction", "arrival"}:
        raise ValueError("incomplete paired recovery transport readout")
    vectors = {"history": captured["history"]}
    for name in ("interaction", "arrival"):
        scores = captured[name]
        vectors[name] = torch.cat((scores.new_zeros(1), scores))[graph.actions]
    vectors["linear"] = vectors["history"] - vectors["interaction"] - vectors["arrival"]
    return logits, vectors


def perturbations(model, observation, logits, vectors, *, cluster, seed, check, passes):
    import torch

    explored = observation["explored"]
    other = 1 - explored
    margin = logits[0] - logits[1]
    result = {"pristine": {"choice": int(logits.argmax()), "margin": float(margin)}}
    for name in COMPONENTS:
        vector = vectors[name]
        direction = sham_direction(cluster, seed, name)
        signed = direction * vector.norm() / (vector.numel() ** 0.5) * vector.new_tensor([1, -1])
        for kind, delta in (("remove", -vector), ("sham", signed)):
            edited_margin = margin + (delta[explored] - delta[other])
            if check or abs(float(edited_margin)) < 2e-6:
                with model.recovery_head.history.register_forward_hook(
                    lambda _module, _inputs, scores: scores + delta
                ):
                    live, _ = model.recover(canonical(observation), passes)
                actual_margin = live[0] - live[1]
                if not torch.isclose(actual_margin, edited_margin, rtol=2e-5, atol=2e-6):
                    raise ValueError("component arithmetic disagrees with a live history edit")
                edited_margin = actual_margin
            result[f"{kind}-{name}"] = {"choice": int(float(edited_margin) < 0), "margin": float(edited_margin),
                                        "norm": float(delta.norm())}
    return result


def count(rows):
    results = {}
    conditions = next(iter(rows.values()))["edits"]
    for mode in conditions:
        outcomes = Counter()
        for row in rows.values():
            outcomes[row["cluster"]] += row["edits"][mode]["choice"] == row["winner"]
        results[mode] = {"correct_worlds": sum(outcomes.values()),
                         "complete_choice_pairs": sum(value == 2 for value in outcomes.values())}
    return results


def summarize_matched(result):
    """Evaluate a randomized sign at the same action-margin dose as removal."""
    comparisons = {}
    for label, families in result["roots"].items():
        comparisons[label] = {}
        for family, rows in families.items():
            centered = {root: {**row, "edits": {}} for root, row in rows.items()}
            doses = {}
            for name in COMPONENTS:
                ratios, zeros = [], 0
                for root, row in rows.items():
                    a, b = row["components"][name]
                    pristine = row["edits"]["pristine"]["margin"]
                    removed = row["edits"][f"remove-{name}"]["margin"]
                    contrast = abs(a - b)
                    if not math.isclose(removed, pristine - (a - b), rel_tol=2e-5, abs_tol=2e-6):
                        raise ValueError("recorded removal shifted the wrong decision margin")
                    margin = pristine + sham_direction(row["cluster"], int(label), name) * contrast
                    centered[root]["edits"][f"contrast-sham-{name}"] = {"choice": int(margin < 0)}
                    if contrast:
                        ratios.append(math.sqrt(2) * math.hypot(a, b) / contrast)
                    else:
                        zeros += 1
                doses[name] = {"zero_removed_margin_worlds": zeros,
                               "median_raw_sham_to_removed_margin": median(ratios) if ratios else None}
            comparisons[label][family] = {
                "counts": {**result["counts"][label][family], **count(centered)},
                "doses": doses,
            }
    return comparisons


def evaluate(args, tree):
    import torch

    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 0)
    reg, branches = checked_branches(args.pending_cohort, args.pending_witness)
    old = single_goal_witness(args.branch_cohort, args.branch_witness, reg["previous_cohort_sha256"])
    pending = {}
    for root, (cluster, observation, outcomes) in branches.items():
        winner, = (choice for choice, outcome in enumerate(outcomes) if outcome["certified"])
        pending[root] = (cluster, observation, winner)
    hashes, data, counts = {}, {}, {}
    with torch.no_grad():
        for seed, directory in zip(SEEDS, args.sessions, strict=True):
            model, _, _, weight_sha = frozen_policy(directory / f"{ARMS[0]}.sqlite", reg, seed, ARMS[0])
            label = str(seed)
            hashes[label], data[label], counts[label] = weight_sha, {}, {}
            for family, source in (("single-goal", old), ("pending-goal", pending)):
                rows = {}
                for index, (root, (cluster, observation, winner)) in enumerate(source.items()):
                    logits, vectors = observed_components(model, observation, 4)
                    edits = perturbations(model, observation, logits, vectors, cluster=cluster, seed=seed,
                                          check=index == 0, passes=4)
                    rows[root] = {"cluster": cluster, "winner": winner, "edits": edits,
                                  "components": {name: vector.tolist() for name, vector in vectors.items()}}
                data[label][family] = rows
                counts[label][family] = count(rows)
                print(seed, family, {mode: values["complete_choice_pairs"] for mode, values in counts[label][family].items()},
                      flush=True)
    result = {"scope": "development recovery-choice diagnostic against native viability witnesses",
            "component_scope": "remove only the contribution read by the recovery head",
            "cohort_sha256": {"single-goal": sha256_file(args.branch_cohort),
                              "pending-goal": sha256_file(args.pending_cohort)},
            "witness_sha256": {"single-goal": sha256_file(args.branch_witness),
                               "pending-goal": sha256_file(args.pending_witness)},
            "weights_sha256": hashes, "counts": counts, "roots": data, "resources": tree.stats()}
    result["margin_matched"] = summarize_matched(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("branch_cohort", "branch_witness", "pending_cohort", "pending_witness", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("sessions", type=Path, nargs=len(SEEDS))
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        result = evaluate(args, tree)
        with args.output.open("xb") as path:
            path.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
