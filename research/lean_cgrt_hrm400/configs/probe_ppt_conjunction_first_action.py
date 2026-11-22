"""Disentangle an insufficient interaction margin from an incorrect interaction ranking."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from configs.check_ppt_conjunction_pilot import ProofLengthWitness
from configs.eval_ppt_conjunction_calibration import read_calibration
from configs.probe_ppt_hybrid_bank import EditedTransportPolicy
from ppt_gym import ram_store
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash


DOSES = (0, 0.5, 1, 2, 4, "interaction-only")


def probe(args):
    _, cases = read_calibration(args.registry, args.calibration_input)
    with ram_store.open_store(args.training_session) as db:
        trained = ram_store.read_manifest(db)
        if not (trained["architecture"] == "hybrid" and trained["premise_interaction"]):
            raise ValueError("expected a trained premise-interaction hybrid")
        weights = ram_store.checkpoint(db)["model"]
    weight_id = tensor_hash(weights)
    configure(2, trained["seed"])
    policy = GraphProofPolicy("hybrid", trained["width"], residual_transport=True, premise_interaction=True)
    policy.load_state_dict(weights)
    edited = EditedTransportPolicy(policy, interaction_only=True)
    witness = ProofLengthWitness()
    by_pair = defaultdict(list)
    with ram_store.open_store(args.calibration_session) as db, torch.no_grad():
        session = ram_store.read_manifest(db)
        if (session["arm"] != "hybrid" or session["checkpoint_sha256"] != weight_id
                or session["registry_sha256"] != sha256_file(args.registry)
                or session["calibration_sha256"] != sha256_file(args.calibration_input)):
            raise ValueError("calibration checkpoint or registry changed")
        for row, case in zip(ram_store.ledger(db), cases, strict=True):
            ep = row["episode"]
            if ep["root"] != case.root or not ep["native_replay"]["replay_passed"]:
                raise ValueError("calibration data no longer match the recorded native replay")
            event = ep["events"][0]
            packet, actions = canonical(event["input"]), event["candidate_menu"]
            target, _ = witness(packet, actions, ep["depth"])
            winner = int(target.argmax())
            logits, _ = policy(packet, actions, ep["depth"])
            with edited.edit("ablate-interaction"):
                base, _ = edited(packet, actions, ep["depth"])
            if len(actions) != 2 or int(logits.argmax()) != event["selected_index"] or edited.hits != 1:
                raise ValueError("probe must reproduce the recorded two-action root and one intervention")
            interaction = logits - base
            result = {
                "root": case.root, "winner": winner, "recorded_action": event["selected_index"],
                "base_margin_toward_winner": float(base[winner] - base[1 - winner]),
                "interaction_margin_toward_winner": float(interaction[winner] - interaction[1 - winner]),
                "correct": {
                    str(dose): int((interaction if dose == "interaction-only" else base + dose * interaction).argmax())
                    == winner for dose in DOSES
                },
            }
            by_pair[case.cluster].append(result)
    if not all(len(pair) == 2 and pair[0]["winner"] != pair[1]["winner"] for pair in by_pair.values()):
        raise ValueError("probe does not cover both opposite first choices")
    counts = {}
    for dose in DOSES:
        named = str(dose)
        counts[named] = {"correct_first_worlds": sum(row["correct"][named] for rows in by_pair.values() for row in rows),
                         "correct_first_in_both": sum(all(row["correct"][named] for row in rows)
                                                      for rows in by_pair.values())}
    errors = Counter()
    for pair in by_pair.values():
        for row in pair:
            if not row["correct"]["1"]:
                margin = row["interaction_margin_toward_winner"]
                errors["correct_interaction_overridden" if margin > 1e-7 else
                       "interaction_incorrect" if margin < -1e-7 else "interaction_tied"] += 1
    return {"scope": "offline first-tactic ranks on reused calibration worlds, no new Lean proof results",
            "checkpoint_sha256": weight_id, "calibration_session_sha256": sha256_file(args.calibration_session),
            "counts": counts, "original_first_choice_errors": dict(errors), "pairs": by_pair}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "calibration_input", "training_session", "calibration_session", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = probe(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["counts"])
    print(report["original_first_choice_errors"])


if __name__ == "__main__":
    main()
