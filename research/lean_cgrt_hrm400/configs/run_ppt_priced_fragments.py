"""Fit price-aware proof search on masked returns from native fragments."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from configs.freeze_ppt_priced_fragments import SCHEMA as CURRICULUM, checked_root
from configs.freeze_ppt_native_progress import advance
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_premise_budget import Example, MixedSourcePrior, calibrate, calibration_features, train
from configs.run_ppt_reuse_plans import apply_ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import candidate_menu
from ppt_gym.mathcan_deadline import deadline_utility
from ppt_gym.native_feedback import behavior_choice
from ppt_gym.premise_budget import BudgetRecurrence, BudgetTRM
from ppt_gym.priced_fragments import FragmentGraph
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.priced-native-fragment-fit.v1"
SEED = 1039


def prepare(source, prior):
    rows = []
    for case in source:
        for log in case["logs"]:
            graph = FragmentGraph(log)
            for key, observed in graph.edges.items():
                if not observed:
                    continue
                state = graph.states[key]
                moves = tuple(observed)
                bellman, empirical = zip(*(graph.labels(key, move) for move in moves), strict=True)
                history = [(move, 1.) for move in state["history"]]
                inputs = prior.state(state["public"], moves, history, prices=graph.prices)
                rows.append(Example(inputs, case["family"], bellman, empirical))
    return rows


def live(repl, case, schedule, select):
    state, public = checked_root(repl, case)
    if digest(public) != case["visible_sha256"]:
        raise ValueError("the sealed price case's initial observation changed")
    left = schedule["budget"]
    trace, script, feedback = [], [], []
    blocked = {}
    while left:
        options = candidate_menu(repl, state)
        before = digest(public)
        failed = blocked.setdefault(before, set())
        menu = tuple(move for move in options if schedule["prices"][move] <= left and move not in failed)
        if not menu:
            break
        action = select(public, menu, feedback, left)
        charge = schedule["prices"][action]
        effect, child, after = advance(repl, state, public, action)
        trace.append({"before": before, "action": action, "cost": charge, "status": effect, "after": digest(after)})
        left -= charge
        if effect in ("closed", "progress"):
            script.append(action)
            feedback.append((action, 1.))
            state, public = child, after
        else:
            feedback.append((action, -1.))
            failed.add(action)
        if effect == "closed":
            break
    proof = bool(trace and trace[-1]["status"] == "closed")
    return {"module": case["module"], "theorem": case["theorem"], "family": case["family"],
            "budget": schedule["budget"], "spent": schedule["budget"] - left,
            "steps": trace, "script": script, "proof": proof}


@torch.no_grad()
def evaluate(args, models, prior, calibration, held, tree, fit_path):
    arms = (*models, "calibrated", "swapped_prices", "mixed_source_prior", "public_behavior")
    results = {}
    # One Lean proof script can witness both prices. The effort checks stay in the rollouts.
    proofs = {(case["module"], tuple(schedule["reference"])): case for case in held for schedule in case["schedules"]}
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        for arm in arms:
            rows = []
            for case in held:
                for variant, schedule in enumerate(case["schedules"]):
                    visible_prices = (case["schedules"][1 - variant] if arm == "swapped_prices" else schedule)["prices"]

                    def choose(public, menu, history, left):
                        if arm == "public_behavior":
                            return behavior_choice(menu)
                        state = prior.state(public, menu, history, prices=visible_prices)
                        if arm == "calibrated":
                            name, heads = calibration
                            features = calibration_features(models["unfitted_budget"], state)[name]
                            curve = torch.stack([apply_ridge(features, head) for head in heads], -1).clamp(0, 1).cummax(-1).values
                        elif arm == "mixed_source_prior":
                            curve = state.prior
                        else:
                            model = models["budget_bellman"] if arm == "swapped_prices" else models[arm]
                            curve = model(state)
                        return menu[int(deadline_utility(curve.unsqueeze(0), torch.tensor([left]))[0].argmax())]

                    row = live(repl, case, schedule, choose)
                    row["variant"] = variant
                    if row["proof"]:
                        proofs[case["module"], tuple(row["script"])] = case
                    rows.append(row)
            results[arm] = rows
            paired = defaultdict(list)
            for row in rows:
                paired[row["module"]].append(row["proof"])
            print(arm, "proofs", sum(row["proof"] for row in rows), "complete pairs",
                  sum(all(pair) for pair in paired.values()), flush=True)
    with TemporaryDirectory(prefix="ppt-priced-certs-") as scratch:
        independent = verify_independent(Path(scratch), lean_env(), (), [(case, key[-1]) for key, case in proofs.items()])
    write_receipt(args.output / "results.json.gz",
                  {"schema": SCHEMA, "fit_sha256": sha256_file(fit_path), "registry_sha256": sha256_file(args.registry),
                   "results": results, "independent": independent, "memory": tree.stats()})


def run(args, tree):
    registry = read_receipt(args.registry)
    if (registry["schema"] != CURRICULUM or registry["lean_sha256"] != sha256_file(LEAN_EXE)
            or registry["repl_sha256"] != sha256_file(REPL_EXE) or registry["generator_sha256"] != sha256_file(Path(fragment.__file__))):
        raise ValueError("the frozen priced registry, generator or toolchain changed")
    held = {row["visible_sha256"] for row in registry["held"]}
    if held & {key for row in registry["source"] for log in row["logs"] for key in log["public_states"]}:
        raise ValueError("a sealed evaluation state entered source experience")
    args.output.mkdir()
    prior = MixedSourcePrior(args)
    rows = prepare(registry["source"], prior)
    arms = {"positive_bellman": sum(1 in labels for row in rows for labels in row.bellman),
            "positive_empirical": sum(1 in labels for row in rows for labels in row.empirical)}
    print("logged decisions", len(rows), "observed moves", sum(len(row.bellman) for row in rows),
          "verified horizon labels", arms, flush=True)
    base = BudgetRecurrence()
    models = {"budget_bellman": base, "budget_empirical": deepcopy(base), "graph_trm": BudgetTRM(priced=True),
              "unfitted_budget": deepcopy(base).eval()}
    for name, model in models.items():
        if name != "unfitted_budget":
            train(model, rows, empirical=name == "budget_empirical", learning_rate=.0015 if name == "graph_trm" else .025)
            model.eval()
            print("fitted", name, flush=True)
    calibration = calibrate(models["unfitted_budget"], rows)
    weights = {name: model.state_dict() for name, model in models.items()}
    fit_path = args.output / "fit.pt"
    with fit_path.open("xb") as handle:
        torch.save({"schema": SCHEMA, "weights": weights, "calibration": calibration,
                    "registry_sha256": sha256_file(args.registry), "prior_sha256": sha256_file(args.prior),
                    "mixed_sources": prior.counts, "source": arms,
                    "weight_hashes": {name: tensor_hash(state) for name, state in weights.items()}}, handle)
    checkpoint = torch.load(fit_path, weights_only=True, map_location="cpu")
    for name, weights in checkpoint["weights"].items():
        if tensor_hash(weights) != checkpoint["weight_hashes"][name]:
            raise ValueError("a sealed priced checkpoint changed")
        models[name].load_state_dict(weights)
    evaluate(args, models, prior, checkpoint["calibration"], registry["held"], tree, fit_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for part in ("registry", "warm", "prior", "output"):
        parser.add_argument(part, type=Path)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
