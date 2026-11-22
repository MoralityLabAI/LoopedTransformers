"""Evaluate sealed MathCan-trained heads on fresh kernel-checked Lean searches."""

import argparse
from collections import defaultdict
from pathlib import Path
import re

import torch

from configs.check_ppt_mathcan_native_costs import invoke_lean
from configs.freeze_ppt_native_progress import (
    AXIOMS, SCHEMA, add_sources, advance, candidate_menu, inventory, native_sources, observe, support_paths,
)
from configs.run_ppt_mathcan_progress import fixed_clock, read_receipt
from configs.run_ppt_mathcan_sparse import Observations, calibration_features, load_warm, summarize
from configs.run_ppt_reuse_plans import apply_ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.mathcan_deadline import deadline_utility
from ppt_gym.mathcan_progress import ProgressRoute, progress_curve, split_completion_curve
from ppt_gym.mathcan_route_policy import RoutePolicy
from ppt_gym.mathcan_sparse import CHECKS, CHECK_COST
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.two_slot import text_chunks


PROGRESS_SHA = "75c658f4689a41d919001763f94b38a12bdaf55be7b31408d67863aa81c0e1b1"
DEADLINE_SHA = "f05bea94b498c39cc928fb5340fbf6377d83c59b244007ec1b7099be40165f6d"
WARM_SHA = "ca60c3a2e29509a5599ef2d27a5781fa9c579c4558d4751d1757998e87567745"
ARMS = ("split", "affine_bellman", "affine_empirical", "trm_bellman", "frozen_progress",
        "prior_trm", "prior_calibration", "shifted_immediate")


class NativeInputs:
    def __init__(self, encoder):
        self.text = Observations(None, encoder)
        self.goal_vectors = {}

    def batch(self, visible, menu, past, remaining):
        active, *pending = [goal["pp"] for goal in visible["goals"]]
        goals = (active, "\n".join(pending))
        for goal in goals:
            if goal not in self.goal_vectors:
                encoder = self.text.encoder
                self.goal_vectors[goal] = encoder(text_chunks(goal, encoder.max_bytes)).mean(0)
        paired = torch.stack([self.goal_vectors[goal] for goal in goals]).unsqueeze(0)
        actions = self.text.action_text(menu).unsqueeze(0)
        history = paired.new_zeros((1, CHECKS, paired.shape[-1]))
        feedback = paired.new_zeros((1, CHECKS))
        if past:
            history[0, :len(past)] = self.text.action_text([name for name, _ in past])
            feedback[0, :len(past)] = paired.new_tensor([reward for _, reward in past])
        return paired, actions, history, feedback, paired.new_tensor([remaining])


def restore_models(args):
    if sha256_file(args.warm) != WARM_SHA:
        raise ValueError("the four-source warm checkpoint changed")
    for path, sha in ((args.progress, PROGRESS_SHA), (args.deadline, DEADLINE_SHA)):
        if sha256_file(path / "summary.json.gz") != sha:
            raise ValueError("the original fit index changed")
    progress = read_receipt(args.progress / "summary.json.gz")
    deadline = read_receipt(args.deadline / "summary.json.gz")
    hybrid, trm, _ = load_warm(args.warm)
    folds = []
    for i, (progress_receipt, deadline_receipt) in enumerate(zip(progress["folds"], deadline["folds"], strict=True)):
        models = {
            "affine_bellman": ProgressRoute(hybrid), "affine_empirical": ProgressRoute(hybrid),
            "trm_bellman": ProgressRoute(trm, trm=True), "frozen_progress": ProgressRoute(hybrid),
            "prior_trm": RoutePolicy(trm, trm=True, deadline=True),
            "prior_calibration": RoutePolicy(hybrid, deadline=True),
        }
        index = (
            (args.progress, progress_receipt, {"affine_bellman": "affine_bellman",
             "affine_empirical": "affine_empirical", "trm_bellman": "trm_bellman", "frozen_progress": "frozen"}),
            (args.deadline, deadline_receipt, {"prior_trm": "trm_rl", "prior_calibration": "frozen"}),
        )
        loaded = []
        for directory, receipt, names in index:
            path = directory / f"fold-{i}-fit.pt"
            if sha256_file(path) != receipt["fit"]:
                raise ValueError("a previously fitted fold changed")
            data = torch.load(path, weights_only=True, map_location="cpu")
            for arm, name in names.items():
                if tensor_hash(data["weights"][name]) != data["weight_hashes"][name]:
                    raise ValueError("an inference model has different weights")
                models[arm].load_state_dict(data["weights"][name])
                models[arm].eval()
            loaded.append(data["calibration"])
        folds.append((models, *loaded))
    return NativeInputs(hybrid.text), folds


def scores(arm, models, factored, incumbent, batch):
    if arm == "prior_trm":
        return deadline_utility(models[arm](*batch), batch[-1])
    if arm == "prior_calibration":
        name, readouts = incumbent
        model = models[arm]
        visible = calibration_features(batch, model.features(model.encode(*batch)))[name]
        curve = torch.stack([apply_ridge(visible, head) for head in readouts[name]], dim=-1)
        return deadline_utility(curve.clamp(0, 1).cummax(-1).values, batch[-1])
    vectors = fixed_clock(batch)
    if arm == "frozen_progress":
        name, readouts = factored
        model = models[arm]
        visible = calibration_features(vectors, model.features(model.encode(*vectors)))[name]
        acceptance = apply_ridge(visible, readouts["accept"]).clamp(0, 1)
        curve = torch.stack([apply_ridge(visible, head) for head in readouts["conditional"]], dim=-1).clamp(0, 1).cummax(-1).values
    else:
        model = models["affine_bellman"] if arm in {"split", "shifted_immediate"} else models[arm]
        logits, curve = model.factors(model.encode(*vectors))
        acceptance = logits.sigmoid()
        if arm in {"split", "shifted_immediate"}:
            empirical = models["affine_empirical"]
            immediate = empirical.factors(empirical.encode(*vectors))[1]
            if arm == "shifted_immediate":
                immediate = immediate.roll(1, dims=-2)
            curve = split_completion_curve(immediate, curve)
    return deadline_utility(progress_curve(acceptance, curve), batch[-1])


def replay(repl, case, budget, selector):
    repl.reset()
    state = repl.init(case["statement"], pp=True)
    visible = observe(repl, state)
    if digest(visible) != case["visible_sha256"]:
        raise ValueError("the sealed native starting observation changed")
    steps, script, history = [], [], []
    failed = defaultdict(set)
    for remaining in reversed(range(1, budget + 1)):
        before = digest(visible)
        menu = tuple(move for move in candidate_menu(visible) if move not in failed[before])
        if not menu:
            break
        tactic = selector(visible, menu, history, remaining)
        status, child, after = advance(repl, state, visible, tactic)
        steps.append({"action": tactic, "status": status, "before": before, "after": digest(after)})
        if status in {"rejected", "no_change", "resource_limit"}:
            failed[before].add(tactic)
            history.append((tactic, -1.))
            continue
        script.append(tactic)
        history.append((tactic, 1.))
        state, visible = child, after
        if status == "closed":
            break
    proof = bool(steps and steps[-1]["status"] == "closed")
    return {"start": [str(case["module"]), case["theorem"], []], "budget": budget, "source": case["source"],
            "steps": steps, "script": script, "proof": proof, "censored": False, "checks": len(steps),
            "observed_return": float(proof) - CHECK_COST * len(steps)}


def verify_independent(directory, env, imports, certificates):
    header = "\n".join(f"import {name}" for name in imports)
    declarations = []
    for number, (case, script) in enumerate(certificates):
        theorem = f"native_progress_{number}"
        intros = "intro " + " ".join(f"«{name}»" for name in case["intro_names"])
        actions = (intros, *script) if case["intro_names"] else script
        tactics = "\n".join(f"  {action}" for action in actions)
        declarations.append(f"theorem {theorem} : {case['statement']} := by\n{tactics}\n#print axioms {theorem}")
    source = directory / "PPTIndependentProgress.lean"
    source.write_text(header + "\n\n" + "\n\n".join(declarations), encoding="utf-8")
    report = invoke_lean(directory, env, source.name)
    groups = re.findall(r"^'native_progress_(\d+)' (does not depend on any axioms|depends on axioms: \[[^\]]*\])$",
                        report, flags=re.MULTILINE)
    if {int(index) for index, _ in groups} != set(range(len(certificates))):
        raise ValueError("did not independently check every native certificate")
    for _, verdict in groups:
        if "[" in verdict and not set(verdict.split("[", 1)[1][:-1].split(", ")) <= AXIOMS:
            raise ValueError("a standalone proof depends on an unexpected axiom")
    return {"source_sha256": sha256_file(source), "certificates": len(certificates), "report": report}


@torch.no_grad()
def evaluate(args, tree):
    registry = read_receipt(args.registry)
    modules, _ = inventory(args)
    if (registry["schema"] != SCHEMA or registry["modules"] != modules
            or registry["lean_sha256"] != sha256_file(LEAN_EXE) or registry["repl_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the native inputs changed after cohort registration")
    observation, folds = restore_models(args)
    cases = registry["cases"]
    results = {}
    proofs = {(case["module"], case["theorem"], tuple(case["reference"])): case for case in cases}
    with native_sources(support_paths(args)) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean version")
            for arm in ARMS:
                rows = []
                for index, (models, factored, incumbent) in enumerate(folds):
                    def choose(visible, menu, history, remaining):
                        batch = observation.batch(visible, menu, history, remaining)
                        return menu[int(scores(arm, models, factored, incumbent, batch)[0].argmax())]

                    for case in cases:
                        for slack in ((0, 1) if case["checks"] > 1 else (0,)):
                            row = replay(repl, case, case["checks"] + slack, choose)
                            row.update(fold=index, slack=slack, minimum=case["checks"])
                            rows.append(row)
                            if row["proof"]:
                                proofs[case["module"], case["theorem"], tuple(row["script"])] = case
                results[arm] = rows
                print(arm, "proved", sum(row["proof"] for row in rows), "rollouts", len(rows), flush=True)
        independent = verify_independent(directory, env, imports, [(case, identity[-1]) for identity, case in proofs.items()])
    subsets = {"multi_exact": lambda row: row["minimum"] > 1 and row["slack"] == 0,
               "multi_extra": lambda row: row["minimum"] > 1 and row["slack"] == 1,
               "single": lambda row: row["minimum"] == 1}
    summary = {arm: {name: summarize(selected) for name, select in subsets.items()
                     if (selected := [row for row in rows if select(row)])} for arm, rows in results.items()}
    write_receipt(args.output, {"schema": SCHEMA, "registry_sha256": sha256_file(args.registry),
                                "progress_sha256": PROGRESS_SHA, "deadline_sha256": DEADLINE_SHA,
                                "results": results, "independent": independent, "summary": summary, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sources(parser)
    for name in ("warm", "progress", "deadline", "registry", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    configure(1, 983)
    tree = TreeBudget(2, 1)
    try:
        evaluate(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
