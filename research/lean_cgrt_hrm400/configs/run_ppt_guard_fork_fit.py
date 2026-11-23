"""Fit matched fork policies on source guard traces and verify live Lean continuations."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from configs.freeze_ppt_computational_guards import paid_control, witness_snapshot
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_live_quote_fit import train, warm_values
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.computational_guards import GUARD_CHECK, OPEN, SCHEMA as BANK_SCHEMA
from ppt_gym.data import sha256_file
from ppt_gym.guard_fork_policy import features, guard_tokens, numeric_description
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.native-guard-fork-fit.v1"
SEED = 1091
UPDATES = 480
BATCH = 32


def paths(args):
    return args.bpd, args.mathcan


def registered(args):
    bank = read_receipt(args.bank)
    pins = {name: sha256_file(getattr(args, name)) for name in ("bank", "warm", "prior")}
    train_pairs = {case["pair"] for case in bank["source"]}
    dev_pairs = {case["pair"] for case in bank["held"]}
    if (bank["schema"] != BANK_SCHEMA or train_pairs & dev_pairs or len(train_pairs) != 48 or len(dev_pairs) != 40
            or bank["imports"] != {path.name: sha256_file(path) for path in paths(args)}
            or bank["lean_sha256"] != sha256_file(LEAN_EXE) or bank["repl_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the frozen theorem cohorts or native definitions changed")
    return bank, pins


def models_for():
    model = TwoSlotQuoteActor()
    return {"bellman_two_slot": model, "suffix_two_slot": deepcopy(model), "bellman_trm": LiveQuoteTRM()}


def truth_head():
    return nn.Sequential(nn.Linear(20, 24), nn.GELU(), nn.Linear(24, 1))


def prepare(source, prior):
    x, backed, empirical, folds = [], [], [], []
    observed = {}
    groups = {key: rank % 6 for rank, key in enumerate(sorted({case["pair"] for case in source}))}
    for case in source:
        log = case["native"]
        for row in case["targets"]:
            state = log["states"][row["state"]]
            public = state["public"]
            actions = tuple(row["labels"])
            packed, _ = features(public, actions, row["remaining"], warm_values(prior, public, actions))
            x.append(packed)
            for action in actions:
                labels = row["labels"][action]
                backed.append(labels["bellman"])
                value = labels["suffix"]["mean"]
                empirical.append(torch.nan if value is None else value)
                folds.append(groups[case["pair"]])
        for tape in log["fragments"]:
            for step in tape["steps"]:
                if step["action"] != GUARD_CHECK or step["status"] not in ("progress", "rejected"):
                    continue
                raw = log["states"][step["before"]]["public"]["goals"][0]["target"]["raw"]
                token = guard_tokens(raw)
                if token is None:
                    raise ValueError("an actual guard observation was not encoded")
                passed = step["status"] == "progress"
                if observed.setdefault(token, passed) != passed:
                    raise ValueError("a closed guard was assigned conflicting native verdicts")
    return torch.cat(x), torch.tensor(backed), torch.tensor(empirical), torch.tensor(folds), observed


def fit(args, tree):
    bank, pins = registered(args)
    prior = MixedSourcePrior(args)
    x, backed, empirical, folds, observed = prepare(bank["source"], prior)
    models = models_for()
    initial = {name: tensor_hash(model.state_dict()) for name, model in models.items()}
    if initial["bellman_two_slot"] != initial["suffix_two_slot"]:
        raise ValueError("the two training objectives have different initial states")
    sampler = torch.Generator().manual_seed(SEED)
    batches = torch.randint(len(x), (UPDATES, BATCH), generator=sampler)
    for name, model in models.items():
        bounds = torch.stack((empirical, empirical), -1) if name == "suffix_two_slot" else backed
        train(model, x, bounds, batches, rate=.002)
        print("fitted", name, flush=True)
    exact = backed[:, 0] == backed[:, 1]
    calibration = ridge(x[exact].flatten(1), backed[exact, 0], folds[exact])
    reader = truth_head()
    samples = torch.tensor([numeric_description(token) for token in observed])
    results = torch.tensor(list(observed.values()), dtype=torch.float32)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=.002)
    for chosen in torch.randint(len(samples), (UPDATES, BATCH), generator=sampler):
        loss = nn.functional.binary_cross_entropy_with_logits(reader(samples[chosen]).flatten(), results[chosen])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    fitted = {name: model.state_dict() for name, model in models.items()}
    fitted["supervised_guard"] = reader.state_dict()
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "weights": fitted,
                    "weight_hashes": {name: tensor_hash(value) for name, value in fitted.items()},
                    "initial": initial, "sampler": tensor_hash({"batches": batches}), "calibration": calibration,
                    "actions": len(x), "suffix_labels": int(empirical.isfinite().sum()),
                    "verdicts": {str(outcome): list(observed.values()).count(outcome) for outcome in (False, True)},
                    "prior_source_counts": prior.counts, "memory": tree.stats()}, handle)
    print("sealed fork fit", sha256_file(args.fit), flush=True)


@torch.no_grad()
def evaluate(args, tree):
    bank, pins = registered(args)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fitted["schema"] != SCHEMA or fitted["pins"] != pins:
        raise ValueError("the guarded fit does not match its frozen inputs")
    models = models_for()
    models["supervised_guard"] = truth_head()
    for name, model in models.items():
        weights = fitted["weights"][name]
        if tensor_hash(weights) != fitted["weight_hashes"][name]:
            raise ValueError("the frozen guard actor changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    rows, certificates = defaultdict(list), {}
    with native_sources(paths(args)) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected Lean version")
            for case in bank["held"]:
                snapshot = witness_snapshot(repl, case)
                cache = {}
                branch_runs = {}

                def select(public, arm):
                    if not cache:
                        cache["fields"], cache["guards"] = features(public, OPEN, 6, warm_values(prior, public, OPEN))
                        if any(token is None for token in cache["guards"]):
                            raise ValueError("a frozen fork lost its visible terminal guard")
                    fields = cache["fields"]
                    if arm == "guard_swap":
                        changed = fields.clone()
                        changed[:, 0] = fields.flip(0)[:, 0]
                        values = models["bellman_two_slot"](changed)
                    elif arm == "calibrated":
                        values = apply_ridge(fields.flatten(1), fitted["calibration"])
                    elif arm == "supervised_guard":
                        tensor = torch.tensor([numeric_description(token) for token in cache["guards"]])
                        values = models[arm](tensor).flatten()
                    else:
                        values = models[arm](fields)
                    return int(values.argmax())

                for arm in (*models, "calibrated", "guard_swap", "fold_oracle"):
                    branch = case["kernel_winner"] if arm == "fold_oracle" else select(snapshot[1], arm)
                    if branch not in branch_runs:
                        branch_runs[branch] = paid_control(repl, case, False, select=lambda _: branch, snapshot=snapshot)
                    row = branch_runs[branch]
                    rows[arm].append({"module": case["module"], "pair": case["pair"], "family": case["family"],
                                      "guards": case["guards"], **row})
                    if row["proved"]:
                        certificates[case["module"], tuple(row["script"])] = case
        independent = verify_independent(directory, env, imports, [(case, key[1]) for key, case in certificates.items()])
    summary = {}
    for arm, results in rows.items():
        pairs = defaultdict(list)
        for row in results:
            pairs[row["pair"]].append(row["proved"])
        summary[arm] = {"proofs": sum(row["proved"] for row in results),
                        "complete_pairs": sum(all(group) for group in pairs.values()), "count": len(results),
                        "checks": sum(row["checks"] for row in results)}
        print(arm, summary[arm], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "fit_sha256": sha256_file(args.fit), "pins": pins,
                                 "results": rows, "summary": summary, "independent": independent, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "fit", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
