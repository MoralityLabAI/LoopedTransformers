"""Train the dependent rewrite policy and score complete native proof searches."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_dependent_rewrites import (
    BUDGET, ROUTING_CONFIRMATION_SCHEMA, ROUTING_SCHEMA, SCHEMA, SPUR_SCHEMA, layouts, routing_layouts, spur_layouts,
)
from configs.freeze_ppt_bpd_rewrite import ARMS
from configs.probe_ppt_bpd_conditional_policy import readout_only
from configs.probe_ppt_bpd_mixed_policy import fit
from configs.probe_ppt_bpd_rewrite_revisits import audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def read_witness(path, split):
    item = json.loads(gzip.decompress(path.read_bytes()))
    if item["schema"] in (ROUTING_SCHEMA, ROUTING_CONFIRMATION_SCHEMA) and split == "dev":
        expected = set(routing_layouts(confirmation=item["schema"] == ROUTING_CONFIRMATION_SCHEMA))
    elif item["schema"] in (SCHEMA, SPUR_SCHEMA):
        expected = set((spur_layouts() if item["schema"] == SPUR_SCHEMA else layouts())[split])
    else:
        raise ValueError("unexpected dependent rewrite proof witness schema or split")
    designs = defaultdict(set)
    if item["split"] != split or item["tactic_checks"] != BUDGET:
        raise ValueError("unexpected dependent rewrite proof witness")
    for root, row in item["rows"].items():
        case = SearchCase(**row["case"])
        if case.split != split or case.root != root or row["winner"] not in (0, 1):
            raise ValueError("the dependent rewrite witness changed a registered theorem")
        designs[tuple(row["shape"])].add(row["winner"])
        if len(row["branches"]) != 2:
            raise ValueError("the dependent proof lost one of its opening outcomes")
        for choice, branch in enumerate(row["branches"]):
            cert = branch["certificate"]
            if (not branch["replay_passed"] or branch["choice"] != choice or len(branch["events"]) != BUDGET
                    or branch["closed"] != (choice == row["winner"])
                    or (branch["closed"] and (not cert["passed"] or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]))):
                raise ValueError("a registered dependent branch was not verified")
    if set(designs) != expected or any(values != {0, 1} for values in designs.values()) or len(item["rows"]) != 2 * len(expected):
        raise ValueError("the dependent rewrite cohort lost a paired layout")
    return item


def training_batches(item, *, match_search_menu=False):
    groups = defaultdict(list)
    for root, row in item["rows"].items():
        for branch in row["branches"]:
            events, previous = [], None
            for attempt, step in enumerate(branch["events"], 1):
                packet = json.loads(step["packet"])
                if digest(packet["state"]) != step["before"] or packet["checks_remaining"] != BUDGET - attempt + 1:
                    raise ValueError("a dependent training observation changed during replay")
                menu = step["menu"]
                event = {"input": packet, "candidate_menu": menu, "selected_index": menu.index(step["tactic"]),
                         "kind": "tactic", "attempt": attempt, "status": "accepted_nonterminal"}
                if match_search_menu:
                    eligible = eligible_rewrites(menu, previous)
                    if step["tactic"] not in eligible:
                        raise ValueError("the matched training trace uses a tactic excluded at search time")
                    event["rewrite_choices"] = [i for i, tactic in enumerate(menu) if tactic in eligible]
                events.append(event)
                previous = step["tactic"]
            if events[0]["input"] != json.loads(row["packet"]) or events[0]["selected_index"] != branch["choice"]:
                raise ValueError("dependent training outcomes do not start from the same public decision")
            groups[row["case"]["cluster"]].append({"root": root, "family": row["case"]["family"], "tactic_checks": BUDGET,
                                                    "verified_closed": branch["closed"], "within_cpu_budget": True, "events": events})
    return [groups[key] for key in sorted(groups)]


def inverse(tactic):
    if tactic.startswith("rw [← "):
        return "rw [" + tactic.removeprefix("rw [← ")
    if tactic.startswith("rw ["):
        return "rw [← " + tactic.removeprefix("rw [")
    return None


def eligible_rewrites(menu, previous):
    return tuple(tactic for tactic in menu if tactic != inverse(previous or "")) or tuple(menu)


def trajectory(repl, case, policy, tree, script=None):
    import torch

    from ppt_gym.dependent_rewrite import term_rewrites
    from ppt_gym.live_lean import public_state, state_input
    from ppt_gym.rewrite import rewrite_menu

    repl.reset()
    state = repl.init(case.statement, pp=True)
    depth = 4 if policy.architecture == "hybrid" else 1
    events, macs, previous = [], 0, None
    start = tree.cpu_seconds()
    for attempt in range(BUDGET if script is None else len(script)):
        visible = public_state(repl, state)
        if not visible["goals"] or (script is None and tree.cpu_seconds() - start > 6):
            break
        goal = visible["goals"][0]
        menu = term_rewrites(goal)
        if menu is None:
            menu = rewrite_menu(goal)
        eligible = eligible_rewrites(menu, previous)
        if not eligible:
            break
        packet, before = state_input(repl, state, (), BUDGET - attempt)
        if script is not None:
            tactic = script[attempt]
            if tactic not in eligible:
                raise ValueError("the recorded dependent tactic is no longer public")
        elif len(eligible) == 1:
            tactic, = eligible
        else:
            cost = policy.macs(len(menu), depth)
            if macs + cost > 1_200_000_000:
                break
            macs += cost
            with torch.no_grad():
                logits, _ = policy(packet, menu, depth)
            tactic = max(eligible, key=lambda tactic: float(logits[menu.index(tactic)]))
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError("Lean rejected a tactic predicted by the dependent rewrite grammar")
        state = reply["state"]
        events.append({"tactic": tactic, "before": before, "after": digest(public_state(repl, state))})
        previous = tactic
    closed = not public_state(repl, state)["goals"]
    within = tree.cpu_seconds() - start <= 6
    if closed:
        kernel = repl.kernel(state["sid"])
        if not kernel["ok"] or kernel["axioms"] != []:
            raise ValueError("the selected dependent proof failed kernel checking")
    return events, closed, within, macs


def audit(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.dependent_rewrite import DependentRewritePolicy
    from ppt_gym.gated_dependent_rewrite import GatedDependentRewritePolicy
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    witness, training = read_witness(args.witness, "dev"), read_witness(args.training, "train")
    policy_source = Path(__file__).resolve().parents[1] / "ppt_gym/dependent_rewrite.py"
    if (witness["schema"] not in (SCHEMA, SPUR_SCHEMA) or training["schema"] != witness["schema"]
            or training["policy_sha256"] != witness["policy_sha256"]
            or witness["policy_sha256"] != sha256_file(policy_source)
            or training["source_sha256"] != witness["source_sha256"] or set(training["rows"]) & set(witness["rows"])):
        raise ValueError("the dependent training and held-out witnesses differ in provenance")
    gated_study = witness["schema"] == SPUR_SCHEMA
    batches = training_batches(training)
    source = json.loads(args.audit.read_bytes())
    seed = source["trials"]["trm_rl"]["manifest"]["seed"]
    if source.get("seed", seed) != seed or any(trial["manifest"]["seed"] != seed for trial in source["trials"].values()):
        raise ValueError("the matched checkpoint audit disagrees on its warm seed")
    configure(tree.threads, seed)
    arms = (*ARMS, "hybrid_rl_uniform") if gated_study else ARMS
    if args.arms:
        if set(args.arms) - set(arms):
            raise ValueError("the selected arm does not belong to this dependent study")
        arms = tuple(arm for arm in arms if arm in args.arms)
    selected = tuple(dict.fromkeys(arm.split("_", 1)[0] + "_rl" for arm in arms))
    controls = {arm.split("_", 1)[0]: (model, checkpoint) for arm, model, checkpoint, _ in audited_policies(
        args.sessions, args.audit, source["cohort_sha256"], source["reference_sha256"], selected=selected,
    )}
    args.checkpoints.mkdir(exist_ok=False)
    dev = {root: (row["packet"], row["menu"], row) for root, row in witness["rows"].items()}
    results = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != witness["lean"]:
            raise ValueError("the dependent search verifier changed")
        for arm in arms:
            old, source_checkpoint = controls[arm.split("_", 1)[0]]
            hybrid = old.architecture == "hybrid"
            gated = gated_study and hybrid and arm != "hybrid_rl_uniform"
            policy_type = GatedDependentRewritePolicy if gated else DependentRewritePolicy
            policy = policy_type(old.architecture, old.width, residual_transport=hybrid, premise_interaction=hybrid).eval()
            missing, unused = policy.load_state_dict(old.state_dict(), strict=not gated)
            if gated and (missing != ["graph.transport.evidence_gate.weight"] or unused):
                raise ValueError("the gated policy did not inherit the exact matched initialization")
            result = fit(policy, arm, batches, dev, False, 1, calibrate=readout_only)
            path = args.checkpoints / f"{arm}.pt"
            torch.save(policy.state_dict(), path)
            policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if tensor_hash(policy.state_dict()) != result["weights_after"]:
                raise ValueError("the dependent proof policy changed on CPU reload")
            pairs, paths = defaultdict(list), {}
            for root, (_, _, row) in dev.items():
                case = SearchCase(**row["case"])
                events, closed, within, macs = trajectory(repl, case, policy, tree)
                script = [event["tactic"] for event in events]
                replay, repeated_closure, _, _ = trajectory(repl, case, policy, tree, script)
                if replay != events or repeated_closure != closed:
                    raise ValueError("the chosen dependent search failed native replay")
                cert = certify(case, {"live_kernel": closed, "accepted_script": script})
                if closed and (not cert or not cert["passed"] or cert["axioms"]):
                    raise ValueError("a scored dependent proof lacks an independent certificate")
                opening = row["branches"][result["final"][root]]["closed"]
                pairs[case.cluster].append((opening, closed and within))
                paths[root] = {"events": events, "certificate": cert, "replay_passed": True, "within_cpu_budget": within,
                               "opening_certified": opening, "search_certified": closed and within, "macs": macs}
            counts = {name: {"proofs": sum(sum(values[index] for values in pair) for pair in pairs.values()),
                             "pairs": sum(all(values[index] for values in pair) for pair in pairs.values())}
                      for index, name in enumerate(("opening", "search"))}
            results[arm] = {**result, "source_checkpoint": source_checkpoint, "checkpoint_sha256": sha256_file(path),
                            "native_paths": paths, "counts": counts}
            if gated:
                results[arm]["evidence_gate"] = policy.graph.transport.evidence_gate.weight.detach().tolist()
            print(arm, counts, flush=True)
    return {"scope": "disjoint dependent rewrite layouts with matched trajectory training and live eight-check search",
            "witness_sha256": sha256_file(args.witness), "training_sha256": sha256_file(args.training),
            "source_audit_sha256": sha256_file(args.audit), "training_pairs": len(batches), "epochs": 1,
            "gated_study": gated_study, "seed": seed,
            "results": results, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "training", "audit", "sessions", "output", "checkpoints"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, "hybrid_rl_uniform"))
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
