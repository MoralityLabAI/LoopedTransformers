"""Verify, fit, or evaluate the persistent two-slot Lean proof probe."""

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.__main__ import certify, proof_return
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.two_slot_fixture import BUNDLES, CHECKS, PROBES, fixtures, menu


SCHEMA = "ppt.two-slot-rewrite.v1"
TRM_SCHEMA = "ppt.two-slot-trm.v1"
RESULT_SCHEMA = "ppt.two-slot-structural-probe.v2"
SEED = 404
MAC_LIMIT = 150_000_000
CPU_LIMIT = 6


def request_state(reply):
    if not reply["ok"]:
        raise ValueError(f"Lean rejected a registered two-slot move: {reply['error']}")
    return reply["state"]


def step(repl, state, action):
    if action not in BUNDLES:
        return request_state(repl.run(state["sid"], action))
    # The existing server runs a tactic on the focused goal. Apply the same
    # rewrite to both goals and restore their order before exposing the state.
    rewrite = action.removeprefix("all_goals ")
    for _ in range(2):
        state = request_state(repl.run(state["sid"], rewrite))
        if len(state["goals"]) != 2:
            raise ValueError("the registered bundle must retain both goals")
        state = request_state(repl.focus(state["sid"], 1))
    return state


def trajectory(repl, tree, case, *, policy=None, fork=None, reset_slots=False, extra_checks=0):
    import torch

    from ppt_gym.live_lean import public_state

    repl.reset()
    state = repl.init(case.statement)
    probe = PROBES[case.family]
    goals = public_state(repl, state)["goals"]
    carry, events, macs = None, [], 0
    start = tree.cpu_seconds()
    for index in range(probe.checks + extra_checks):
        if not goals or tree.cpu_seconds() - start > CPU_LIMIT:
            break
        actions = menu(goals, index, case.family)
        if (index == 1 and actions != probe.fork) or not actions:
            raise ValueError("the registered two-slot tactic menu changed")
        left = max(0, probe.checks - index)
        if policy is None:
            if index == 1:
                action = fork
            else:
                action, = actions
        else:
            macs += policy.macs(len(actions))
            if macs > MAC_LIMIT:
                raise ValueError("the two-slot trajectory exceeded its arithmetic envelope")
            with torch.no_grad():
                scores, predictions = policy(goals, actions, left, carry)
            action = actions[int(scores.argmax())]
        if action not in actions:
            raise ValueError("the selected tactic is outside the public menu")
        state = step(repl, state, action)
        following = public_state(repl, state)["goals"]
        events.append({"goals": goals, "menu": actions, "action": action, "after": digest(following)})
        if policy is not None and following:
            with torch.no_grad():
                updated = policy.advance(predictions[actions.index(action)], following)
            carry = None if reset_slots else updated
        goals = following
    within = tree.cpu_seconds() - start <= CPU_LIMIT
    closed = not goals
    certificate = None
    if closed:
        kernel = repl.kernel(state["sid"])
        if not kernel["ok"] or kernel["axioms"]:
            raise ValueError("the live two-slot proof failed the kernel check")
        certificate = certify(case, {"live_kernel": True, "accepted_script": [row["action"] for row in events]},
                              timeout=60)
        if not certificate or not certificate["passed"] or certificate["axioms"]:
            raise ValueError("the two-slot proof failed independent certification")
    reward = proof_return({"verified_closed": closed, "within_cpu_budget": within, "tactic_checks": len(events)})
    return {"events": events, "certificate": certificate, "success": closed and within,
            "within": within, "reward": reward, "macs": macs}


def reference(repl, tree, case, winner, *, verify_late=False):
    probe = PROBES[case.family]
    trials = [trajectory(repl, tree, case, fork=action) for action in probe.fork]
    if [row["success"] for row in trials] != [action == winner for action in probe.fork]:
        raise ValueError(f"Lean disagreed with the {probe.checks}-check witness for {case.tid}")
    for row in trials:
        if not row["within"] or len(row["events"]) != probe.checks:
            raise ValueError("the two-slot witness changed its tactic or CPU cost")
    if trials[0]["events"][:1] != trials[1]["events"][:1]:
        raise ValueError("the two forced branches received different prefixes")
    result = {"case": asdict(case), "checks": probe.checks, "winner": winner, "trials": trials}
    if verify_late:
        losing = 1 - probe.fork.index(winner)
        late = trajectory(repl, tree, case, fork=probe.fork[losing], extra_checks=1)
        if (not late["success"] or len(late["events"]) != probe.checks + 1
                or late["events"][:probe.checks] != trials[losing]["events"]):
            raise ValueError(f"Lean disagreed with the late branch continuation for {case.tid}")
        result["late"] = late
    return result


def fork_features(policy, record):
    prefix, fork = record["trials"][0]["events"][:2]
    _, prediction = policy(prefix["goals"], prefix["menu"], CHECKS)
    carry = policy.advance(prediction[0], fork["goals"])
    _, proposals = policy(fork["goals"], fork["menu"], CHECKS - 1, carry)
    return proposals[:, 0]


def calibrate(policy, records):
    import torch

    from ppt_gym.ram_learning import tensor_hash

    frozen = {name: value for name, value in policy.state_dict().items() if not name.startswith("actor.")}
    before = tensor_hash(frozen)
    policy.requires_grad_(False)
    with torch.no_grad():
        x = torch.cat([fork_features(policy, record) for record in records]).double()
        y = x.new_tensor([trial["reward"] for record in records for trial in record["trials"]])
        clusters = [record["case"]["cluster"] for record in records for _ in BUNDLES]
        held = [torch.tensor([cluster == value for value in clusters]) for cluster in sorted(set(clusters))]

        def solve(train, penalty):
            a, b = x[train], y[train]
            center = a.mean(0)
            centered = a - center
            gram = centered @ centered.T
            scale = gram.diag().mean().clamp_min(1e-8)
            w = centered.T @ torch.linalg.solve(gram + penalty * scale * torch.eye(len(a), dtype=a.dtype),
                                                b - b.mean())
            return w, b.mean() - center @ w

        errors = {}
        for penalty in (.001, .01, .1, 1.):
            error = 0.
            for mask in held:
                w, bias = solve(~mask, penalty)
                error += float((x[mask] @ w + bias - y[mask]).square().mean())
            errors[penalty] = error
        choice = min(errors, key=errors.get)
        weight, bias = solve(slice(None), choice)
        policy.actor.weight.copy_(weight)
        policy.actor.bias.copy_(bias)
    if tensor_hash(frozen) != before:
        raise ValueError("frozen calibration changed the two-slot dynamics")
    return choice


def fit_returns(policy, records):
    import torch
    from torch.nn import functional as F

    optimizer = torch.optim.AdamW(policy.parameters(), lr=.002)
    targets = torch.tensor([[trial["reward"] for trial in record["trials"]] for record in records])
    for _ in range(12):
        logits = torch.stack([policy.actor(fork_features(policy, row)).squeeze(-1) for row in records])
        loss = F.mse_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def fit(repl, tree, output):
    import torch

    from ppt_gym.two_slot import TwoSlotPolicy

    records = [reference(repl, tree, case, winner) for case, winner in fixtures("train")]
    learned = TwoSlotPolicy()
    calibrated = deepcopy(learned)
    penalty = calibrate(calibrated, records)
    fit_returns(learned, records)
    with output.open("xb") as handle:
        torch.save({"schema": SCHEMA, "training": digest([row["case"] for row in records]),
                    "lean": repl.version, "records": records, "penalty": penalty,
                    "learned": learned.state_dict(), "calibrated": calibrated.state_dict(),
                    "memory": tree.stats()}, handle)


def read_fit(checkpoint):
    import torch

    fit = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (fit["schema"] != SCHEMA or fit["lean"] != LEAN_VERSION
            or fit["training"] != digest([asdict(case) for case, _ in fixtures("train")])):
        raise ValueError("the checkpoint does not match the registered two-slot fit")
    return fit


def fit_trm(tree, source, output):
    import torch

    from ppt_gym.two_slot import TwoSlotTRM

    fit = read_fit(source)
    records = fit["records"]
    if digest([row["case"] for row in records]) != fit["training"]:
        raise ValueError("the archived native returns changed their training cohort")
    policy = TwoSlotTRM()
    fit_returns(policy, records)
    with output.open("xb") as handle:
        torch.save({"schema": TRM_SCHEMA, "parent_sha256": sha256_file(source),
                    "weights": policy.state_dict(), "memory": tree.stats()}, handle)


def score_pairs(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"]["cluster"]].append(row)
    pairs = {}
    for cluster, pair in grouped.items():
        families = {row["case"]["family"] for row in pair}
        if len(pair) != 2 or len(families) != 1:
            raise ValueError("each two-slot evaluation pair requires two worlds from one proof structure")
        family, = families
        proofs = sum(row["success"] for row in pair)
        pairs[cluster] = {"family": family, "proofs": proofs, "complete": proofs == 2}
    return {"complete_pairs": sum(row["complete"] for row in pairs.values()),
            "proofs": sum(row["proofs"] for row in pairs.values()), "pairs": pairs, "rows": rows}


def evaluate(repl, tree, checkpoint, *, trm_checkpoint=None):
    import torch

    from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM

    fit = read_fit(checkpoint)
    if fit["lean"] != repl.version:
        raise ValueError("the fit and live verifier use different Lean versions")
    fit_sha256 = sha256_file(checkpoint)
    results = {}
    cases = list(fixtures("dev"))
    arms = [("persistent", TwoSlotPolicy, fit["learned"], False),
            ("frozen_calibration", TwoSlotPolicy, fit["calibrated"], False),
            ("slot_reset", TwoSlotPolicy, fit["learned"], True)]
    provenance = {"checkpoint_sha256": fit_sha256}
    if trm_checkpoint is not None:
        trm = torch.load(trm_checkpoint, map_location="cpu", weights_only=True)
        if trm["schema"] != TRM_SCHEMA or trm["parent_sha256"] != fit_sha256:
            raise ValueError("the TRM must use the same archived native returns")
        arms.append(("trm", TwoSlotTRM, trm["weights"], True))
        provenance["trm_checkpoint_sha256"] = sha256_file(trm_checkpoint)
    for arm, model, weights, reset in arms:
        policy = model().eval()
        policy.load_state_dict(weights)
        rows = [{"case": asdict(case), **trajectory(repl, tree, case, policy=policy, reset_slots=reset)}
                for case, _ in cases]
        results[arm] = {**score_pairs(rows), "parameters": sum(value.numel() for value in policy.parameters())}
    return {"schema": RESULT_SCHEMA, "lean": repl.version, **provenance,
            "tactic_checks": {family: probe.checks for family, probe in PROBES.items()},
            "mac_limit": MAC_LIMIT, "cpu_limit": CPU_LIMIT,
            "results": results, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    commands.add_parser("verify").add_argument("output", type=Path)
    commands.add_parser("fit").add_argument("output", type=Path)
    control = commands.add_parser("fit-trm")
    control.add_argument("source", type=Path)
    control.add_argument("output", type=Path)
    score = commands.add_parser("evaluate")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("--trm", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("output already exists")
    tree = TreeBudget(2, 1)
    try:
        from ppt_gym.ram_campaign import configure

        configure(tree.threads, SEED)
        if args.stage == "fit-trm":
            fit_trm(tree, args.source, args.output)
            return
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the two-slot probe requires the pinned Lean verifier")
            if args.stage == "fit":
                fit(repl, tree, args.output)
                return
            if args.stage == "verify":
                rows = [reference(repl, tree, case, winner, verify_late=True) for case, winner in fixtures("dev")]
                result = {"schema": RESULT_SCHEMA, "lean": repl.version, "rows": rows}
            else:
                result = evaluate(repl, tree, args.checkpoint, trm_checkpoint=args.trm)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(canonical(result) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
