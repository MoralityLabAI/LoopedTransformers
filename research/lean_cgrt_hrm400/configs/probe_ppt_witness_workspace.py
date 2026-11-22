"""Interchange pending proof evidence at the two-slot witness decision."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path

from lcgrt.config import LEAN_VERSION
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.dependent_witness import BUDGETS, FORK, FORK_INDEX
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.two_slot import candidate_state

from configs.check_ppt_dependent_witness import read_receipt
from configs.run_ppt_slot_continuations import SCHEMA, WITNESS, heldout_registry, read_continuation_fit, restore_policy


ARMS = ("live_rl", "calibrated", "trm_workspace")
WITNESS_NAMES = {"h0": "h1", "h1": "h0"}


def prefix_carry(policy, events, checks):
    carry = None
    for i, event in enumerate(events[:FORK_INDEX]):
        if tuple(event["menu"]) != (event["action"],):
            raise ValueError("the replay prefix contains an unforced tactic")
        _, predicted = policy(event["goals"], event["menu"], checks - i, carry)
        carry = policy.advance(candidate_state(predicted, 0), events[i + 1]["goals"])
    return carry


def observe_fork(policy, event, checks, carry, donor=None):
    """Keep the focused slot and patch both pending-slot channels on each pass."""
    if tuple(event["menu"]) != FORK or len(event["goals"]) != 2:
        raise ValueError("the pending readout patch requires two registered witness actions and slots")
    if donor is not None and (len(donor) != 2 or any(signal.shape != (2, 2) for signal in donor)):
        raise ValueError("the donor must supply two pending-slot readout channels per action and pass")
    captured = []

    def patch(_readout, args):
        signals, = args
        if signals.shape != (4, 2) or len(captured) >= 2:
            raise ValueError("unexpected witness readout shape or pass count")
        pending = signals[1::2]
        if donor is not None:
            restored = signals.clone()
            restored[1::2] = donor[len(captured)]
        captured.append(pending.detach().clone())
        return (restored,) if donor is not None else None

    with policy.path_bank.read.register_forward_pre_hook(patch):
        logits, _ = policy(event["goals"], event["menu"], checks - FORK_INDEX, carry)
    if len(captured) != 2 or not logits.isfinite().all():
        raise ValueError("the registered fork readout is incomplete")
    return logits, tuple(captured)


def relabel_witness_prefix(events):
    """Exchange the two proof binder names while preserving types and positions."""
    return [
        {**event, "goals": [
            {**goal, "hypotheses": [
                {**hyp, "name": WITNESS_NAMES.get(hyp["name"], hyp["name"])}
                for hyp in goal["hypotheses"]
            ]}
            for goal in event["goals"]
        ]}
        for event in events[:FORK_INDEX + 1]
    ]


def binder_intervention(policy, events, checks, carry, logits, signals):
    """Keep both proofs fixed and check whether name changes alter their ranking."""
    import torch

    event = events[FORK_INDEX]
    reversed_logits, _ = policy(event["goals"], FORK[::-1], checks - FORK_INDEX, carry)
    torch.testing.assert_close(reversed_logits.flip(0), logits)

    renamed = relabel_witness_prefix(events)
    renamed_carry = prefix_carry(policy, renamed, checks)
    renamed_logits, renamed_signals = observe_fork(policy, renamed[-1], checks, renamed_carry)
    for original, renamed_signal in zip(signals, renamed_signals, strict=True):
        torch.testing.assert_close(renamed_signal.flip(0), original, atol=0., rtol=0.)
    return {"menu_reversed": reversed_logits.flip(0).tolist(), "renamed": renamed_logits.flip(0).tolist(),
            "semantic_flip": int(renamed_logits.flip(0).argmax()) != int(logits.argmax())}


def perturb_pair(policy, pair, observed, checks, *, opening_diagnostics=False):
    natural, carries, forks = [], [], []
    if any(len(observed[row["case"]["tid"]]["events"]) <= FORK_INDEX for row in pair):
        return [{"case": row["case"], "censored": True} for row in pair]
    for row in pair:
        events = row["trials"][0]["events"]
        fork = events[FORK_INDEX]
        trajectory = observed[row["case"]["tid"]]["events"]
        native = trajectory[FORK_INDEX]
        if (trajectory[:FORK_INDEX] != events[:FORK_INDEX]
                or any(native[key] != fork[key] for key in ("menu", "goals"))):
            raise ValueError("the scored proof trace differs from the registered fork")
        carry = prefix_carry(policy, events, checks)
        forks.append(fork)
        carries.append(carry)
        outcome = observe_fork(policy, fork, checks, carry)
        if FORK[int(outcome[0].argmax())] != native["action"]:
            raise ValueError("the natural fork failed to reproduce the scored native action")
        natural.append(outcome)

    rows = []
    for i, row in enumerate(pair):
        logits, signals = natural[i]
        donor = 1 - i
        identity, _ = observe_fork(policy, forks[i], checks, carries[i], signals)
        patched, _ = observe_fork(policy, forks[i], checks, carries[i], natural[donor][1])
        if not identity.equal(logits):
            raise ValueError("the identity patch changed the fork scores")
        target = FORK.index(pair[donor]["winner"])
        initial = logits[target] - logits[1 - target]
        changed = patched[target] - patched[1 - target]
        viable = int(logits.argmax()) != target
        receipt = {"case": row["case"], "censored": False, "donor_action": FORK[target],
                   "natural": logits.tolist(), "identity": identity.tolist(), "patched": patched.tolist(),
                   "pending_readout": [signal.tolist() for signal in signals],
                   "donor_margin_delta": float(changed - initial),
                   "natural_viable": viable, "donor_flip": viable and int(patched.argmax()) == target}
        if opening_diagnostics:
            receipt["binder_intervention"] = binder_intervention(policy, row["trials"][0]["events"], checks,
                                                                  carries[i], logits, signals)
        rows.append(receipt)
    return rows


def audit(checkpoint, evaluation, *, opening_diagnostics=False):
    import torch

    fit = read_continuation_fit(checkpoint, LEAN_VERSION)
    split = fit.get("split", "dev")
    if not fit.get("witness_workspace", False):
        raise ValueError("the checkpoint requires a persistent witness workspace")
    raw = evaluation.read_bytes()
    score = json.loads(gzip.decompress(raw) if evaluation.suffix == ".gz" else raw)
    cohort = heldout_registry(split)
    checkpoint_hash = sha256_file(checkpoint)
    checks = BUDGETS[cohort[0][0].family]
    if (score["schema"] != SCHEMA or score["lean"] != LEAN_VERSION or score["tactic_checks"] != checks
            or score["checkpoint_sha256"] != checkpoint_hash
            or score["witness_sha256"] != fit["witness_sha256"] or score["split"] != split
            or score["cohort"] != digest([asdict(case) for case, _ in cohort])):
        raise ValueError("the BPD reference requires the matching native model evaluation")
    reference = read_receipt(WITNESS[split])["rows"]
    expected = {case.root for case, _ in cohort}
    results = {}
    arms = ARMS + ("unweighted_rl",) if opening_diagnostics and "unweighted_rl" in fit["weights"] else ARMS
    with torch.no_grad():
        for arm in arms:
            policy = restore_policy(fit, arm, fit["weights"][arm])
            observed = {row["case"]["tid"]: row for row in score["results"][arm]["rows"]}
            if len(observed) != len(expected) or {SearchCase(**r["case"]).root for r in observed.values()} != expected:
                raise ValueError(f"the scored cohort is incomplete for {arm}")
            rows = []
            for pair in zip(reference[::2], reference[1::2], strict=True):
                if pair[0]["case"]["cluster"] != pair[1]["case"]["cluster"] or {r["winner"] for r in pair} != set(FORK):
                    raise ValueError("the BPD donors must be opposite worlds from one layout")
                rows.extend(perturb_pair(policy, pair, observed, checks, opening_diagnostics=opening_diagnostics))
            flips = Counter(row["case"]["cluster"] for row in rows if row.get("donor_flip"))
            viable = Counter(row["case"]["cluster"] for row in rows if row.get("natural_viable"))
            results[arm] = {"donor_flips": flips.total(), "pairs_with_two_flips": sum(n == 2 for n in flips.values()),
                            "natural_viable_openings": viable.total(), "eligible_pairs": sum(n == 2 for n in viable.values()),
                            "censored": sum(row["censored"] for row in rows), "rows": rows}
    report = {"schema": "ppt.witness-workspace-bpd.v1", "checkpoint_sha256": checkpoint_hash,
              "evaluation_sha256": sha256_file(evaluation), "witness_sha256": fit["witness_sha256"],
              "cohort": score["cohort"], "results": results}
    if opening_diagnostics:
        report["opening_diagnostics"] = True
    return report


def main():
    from ppt_gym.ram_campaign import configure

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opening-diagnostics", action="store_true",
                        help="also intervene on witness names and include the unweighted fit if present")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    configure(1, 2201)
    result = audit(args.checkpoint, args.evaluation, opening_diagnostics=args.opening_diagnostics)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(canonical(result) + "\n")


if __name__ == "__main__":
    main()
