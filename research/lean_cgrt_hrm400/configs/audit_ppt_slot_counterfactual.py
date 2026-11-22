"""Compare stored Lean decisions and fixed path-bank evidence on the opened cohort."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.slot_path_bank import HOPS_PER_UPDATE, SlotPathBank


def read_archive(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def certified(trial):
    proof = trial["certificate"]
    return bool(trial["success"] and trial["within"] and proof and proof["passed"] and not proof["axioms"])


def paired_trajectories(learned, control, fork_index):
    compared = []
    for live, other in zip(learned, control, strict=True):
        if live["case"] != other["case"]:
            raise ValueError("the score report did not preserve paired Lean roots across arms")
        for trial in (live, other):
            if trial["success"] != certified(trial):
                raise ValueError("the recorded success and axiom-free Lean proof receipt disagree")
        left = [e["action"] for e in live["events"]]
        right = [e["action"] for e in other["events"]]
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
        if left == right:
            stage = "identical"
            first = None
        elif first == min(len(left), len(right)):
            stage = "different_length"
        else:
            if any(live["events"][first][key] != other["events"][first][key] for key in ("goals", "menu")):
                raise ValueError("the first different tactic did not start from the same public decision")
            stage = "opening" if first == fork_index else "continuation"
        case = SearchCase(**live["case"])
        compared.append({"root": case.root, "cluster": case.cluster, "stage": stage, "first_difference": first,
                         "learned_proof": live["success"], "control_proof": other["success"]})
    return {"stages": dict(Counter(row["stage"] for row in compared)),
            "learned_only": sum(row["learned_proof"] and not row["control_proof"] for row in compared),
            "control_only": sum(row["control_proof"] and not row["learned_proof"] for row in compared),
            "rows": compared}


def fixed_bank(bank, trial, fork_index):
    if not certified(trial):
        raise ValueError("the fixed readout requires an independently certified reference path")
    decisions = []
    for index, event in enumerate(trial["events"]):
        if len(event["menu"]) <= 1:
            continue
        view, values, gates = bank.start(event["goals"], event["menu"])
        rounds = []
        for _ in range(2):
            values, _ = bank.advance(view, values, gates)
            support = values[view.read_indices].reshape(view.shape)
            sums = support.sum(1)
            peak = sums.max()
            choices = [action for action, score in zip(event["menu"], sums, strict=True) if score == peak]
            rounds.append({"support": support.tolist(), "best": choices})
        decisions.append({"index": index, "reference": event["action"], "rounds": rounds})
    fork = next(event for event in decisions if event["index"] == fork_index)
    return {"decisions": decisions, "fork_matched": fork["rounds"][-1]["best"] == [fork["reference"]],
            "script_matched": all(event["rounds"][-1]["best"] == [event["reference"]] for event in decisions)}


def audit(evaluation, witness):
    import torch

    scores, native = read_archive(evaluation), read_archive(witness)
    if (scores["split"] != "counterfactual" or scores["witness_sha256"] != sha256_file(witness)
            or scores["cohort"] != digest([row["case"] for row in native["rows"]])):
        raise ValueError("the stored score report and native reference cohort differ")
    learned = scores["results"]["live_rl"]["rows"]
    if [row["case"] for row in learned] != [row["case"] for row in native["rows"]]:
        raise ValueError("the stored native reference has different Lean roots")
    fork_index = 1
    comparisons = {name: paired_trajectories(learned, arm["rows"], fork_index)
                   for name, arm in scores["results"].items() if name != "live_rl"}
    bank = SlotPathBank(4)
    references = []
    with torch.no_grad():
        for row in native["rows"]:
            menu = row["trials"][0]["events"][fork_index]["menu"]
            viable = row["trials"][menu.index(row["winner"])]
            case = SearchCase(**row["case"])
            references.append({"root": case.root, "cluster": case.cluster, **fixed_bank(bank, viable, fork_index)})
    return {"schema": "ppt.counterfactual-path-audit.v1", "evaluation_sha256": sha256_file(evaluation),
            "witness_sha256": scores["witness_sha256"], "checkpoint_sha256": scores["checkpoint_sha256"],
            "hops": 2 * HOPS_PER_UPDATE, "comparisons": comparisons,
            "fixed_readout": {"openings": sum(row["fork_matched"] for row in references),
                              "scripts": sum(row["script_matched"] for row in references), "rows": references}}


def main():
    from ppt_gym.ram_campaign import configure

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "witness", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    configure(1, 831)
    raw = (canonical(audit(args.evaluation, args.witness)) + "\n").encode("utf-8")
    with args.output.open("xb") as output:
        output.write(gzip.compress(raw, mtime=0) if args.output.suffix == ".gz" else raw)


if __name__ == "__main__":
    main()
