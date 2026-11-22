"""Fit acquisition policies and compare them on newly verified Lean proofs."""

import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.diagnostic_accumulator import DiagnosticAccumulator
from ppt_gym.diagnostic_acquisition import AcquisitionBatch, AcquisitionHead, acquisition_features, action_factors
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget

from configs.freeze_ppt_rollback_length_transfer import SCHEMA as WITNESS_SCHEMA
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import warm, write_receipt
from configs.run_ppt_rollback_length_transfer import (
    SCHEMA as DECODER_SCHEMA, features, groups_for, registry, replay,
)


SCHEMA = "ppt.rollback-acquisition.v1"
SEED = 831
STEPS = 160
WIDTH = 6
TEMPERATURE = .08
TRAIN = ((.02, 0), (.05, 0), (.10, 0), (.05, 1))
TRANSFER = ((.035, 0), (.08, 0), (.14, 0), (.08, 1), (.14, 2))


def witness_rows(args):
    _, original = registry(args.training_witness)
    training = original["train"] + original["dev"]
    witness = json.loads(gzip.decompress(args.witness.read_bytes()))
    if (witness["schema"] != WITNESS_SCHEMA or witness.get("study") != "acquisition-transfer"
            or witness["lean"] != LEAN_VERSION):
        raise ValueError("expected the sealed acquisition transfer witness")
    heldout = witness["rows"]
    roots = {r["payload_root"] for r in heldout}
    layouts = defaultdict(set)
    for row in heldout:
        layouts[row["cluster"]].add(tuple(event["value"] for event in row["observations"]))
    if (len(roots) != 6 or roots & {r["payload_root"] for r in training}
            or {r["arity"] for r in heldout} != {5, 6}
            or len(layouts) != 12
            or any(len(worlds) != 2 ** len(next(iter(worlds))) for worlds in layouts.values())):
        raise ValueError("the fresh acquisition roots or possible worlds are incomplete")
    return training, heldout


def decoders(args, width):
    parent = torch.load(args.decoder, map_location="cpu", weights_only=True)
    if (parent["schema"] != DECODER_SCHEMA or parent["witness_sha256"] != sha256_file(args.training_witness)
            or parent["warm_sha256"] != sha256_file(args.warm)):
        raise ValueError("the frozen decoder does not match its audited training lineage")
    hybrid, trm, _ = warm(args.warm)
    models = {}
    for arm, initial in (("affine", hybrid), ("trm", trm)):
        weights = parent["weights"][arm]
        if tensor_hash(weights) != parent["weight_hashes"][arm]:
            raise ValueError("the frozen proof decoder was modified")
        model = DiagnosticAccumulator(initial, width, kind=arm)
        model.load_state_dict(weights)
        models[arm] = model.eval()
    return models


def setup(args, rows, regimes):
    encoder, _, _ = warm(args.warm)
    inputs = features(rows, encoder, history_width=WIDTH)
    models = decoders(args, inputs.context.shape[-1])
    horizons = (inputs.values >= 0).sum(-1)
    groups = groups_for(rows, history_width=WIDTH)
    batch = AcquisitionBatch.from_regimes(horizons, regimes, WIDTH)
    native = torch.tensor([[float(path["passed"]) for path in row["witness_trials"]] for row in rows])
    forecasts, vectors = {}, {}
    with torch.no_grad():
        for arm, model in models.items():
            witness = torch.empty(len(rows), dtype=torch.long)
            for arity in horizons.unique().tolist():
                index = (horizons == arity).nonzero().flatten()
                padded = max(5, arity)
                scores = model(inputs.context[index], inputs.operators[index], inputs.values[index, :padded],
                               inputs.clock[index, :padded + 1])
                witness[index] = scores[:, arity, :2].argmax(-1)
            prediction = F.one_hot(witness, 2).float()
            forecasts[arm] = action_factors(batch, prediction, groups)
            vectors[arm] = acquisition_features(batch, inputs, forecasts[arm])
    return batch, native, groups, forecasts, vectors


def masked(utilities, batch):
    return utilities.masked_fill(~batch.legal, -torch.inf)


def train(model, batch, native, groups, values, forecast, *, supervised=False):
    active = batch.valid.unsqueeze(-1) & batch.legal
    query = active[..., 2]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    optimum = action_factors(batch, native, groups) if supervised else None
    for _ in range(STEPS):
        utilities, estimate = model(batch, values, forecast)
        with torch.no_grad():
            policy = (masked(utilities, batch) / TEMPERATURE).softmax(-1)
            target = optimum if supervised else action_factors(batch, native, groups, policy)
        if estimate is None:
            loss = F.mse_loss(utilities[active], batch.utility(target)[active])
        else:
            success = F.mse_loss(estimate[..., 0][active], target[..., 0][active])
            work = F.mse_loss(estimate[..., 2, 1][query] / WIDTH, target[..., 2, 1][query] / WIDTH)
            loss = success + work
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def fit(args):
    training, _ = witness_rows(args)
    batch, native, groups, forecasts, vectors = setup(args, training, TRAIN)
    width = vectors["affine"].shape[-1]
    initial = AcquisitionHead(width)
    models = {arm: deepcopy(initial) for arm in ("affine_factor", "affine_supervised", "trm_factor")}
    models["affine_scalar"] = AcquisitionHead(width, scalar=True)
    for arm, model in models.items():
        decoder = "trm" if arm == "trm_factor" else "affine"
        train(model, batch, native, groups, vectors[decoder], forecasts[decoder],
              supervised=arm == "affine_supervised")
        print(arm, STEPS, flush=True)
    # Calibrate the frozen affine forecasts against the same native training data.
    truth = action_factors(batch, native, groups)
    optimal = batch.utility(truth)
    active = batch.legal & batch.valid.unsqueeze(-1)
    source_ids = {name: i for i, name in enumerate(sorted({row["payload_root"] for row in training}))}
    roots = torch.tensor([source_ids[row["payload_root"]] for row in training])[None, :, None]
    folds = roots.expand_as(batch.valid)
    readouts = [ridge(vectors["affine"][active[..., action]], optimal[..., action][active[..., action]],
                      folds[active[..., action]]) for action in range(3)]
    calibrated_success = [ridge(vectors["affine"][active[..., action]], truth[..., action, 0][active[..., action]],
                                folds[active[..., action]]) for action in range(3)]
    query = active[..., 2]
    calibrated_work = ridge(vectors["affine"][query], truth[..., 2, 1][query], folds[query])
    weights = {name: model.state_dict() for name, model in models.items()}
    with args.output.open("xb") as out:
        torch.save({"schema": SCHEMA, "seed": SEED, "steps": STEPS, "train_regimes": TRAIN,
                    "transfer_regimes": TRANSFER, "decoder_sha256": sha256_file(args.decoder),
                    "training_witness_sha256": sha256_file(args.training_witness),
                    "witness_sha256": sha256_file(args.witness), "warm_sha256": sha256_file(args.warm),
                    "weights": weights, "calibrated": readouts,
                    "calibrated_success": calibrated_success, "calibrated_work": calibrated_work,
                    "weight_hashes": {name: tensor_hash(state) for name, state in weights.items()}}, out)


def read_fit(args):
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fit["schema"] != SCHEMA or tuple(map(tuple, fit["transfer_regimes"])) != TRANSFER:
        raise ValueError("the acquisition fit does not match the registered transfer regimes")
    for key in ("decoder", "training_witness", "witness", "warm"):
        if fit[f"{key}_sha256"] != sha256_file(getattr(args, key)):
            raise ValueError(f"the frozen {key} changed after fit")
    return fit


def records_for(q, rows):
    records = []
    for i, (fee, withheld) in enumerate(TRANSFER):
        selected = q[i].argmax(-1)
        for j, row in enumerate(rows):
            steps = next(step for step, action in enumerate(selected[j].tolist()) if action < 2)
            witness = int(selected[j, steps])
            trial = row["witness_trials"][witness]
            cap = row["arity"] - withheld
            if steps > cap or trial["checks"] + steps > row["budget"] - withheld:
                raise ValueError("the policy exceeded the registered proof reserve")
            records.append({"tid": row["tid"], "cluster": row["cluster"], "payload_root": row["payload_root"],
                            "arity": row["arity"], "operator": row["operator"], "regime": i,
                            "fee": fee, "cap": cap, "witness": witness, "diagnostics": steps,
                            "success": trial["passed"], "return": float(trial["passed"]) - fee * steps})
    return records


def evaluate(args, tree):
    _, rows = witness_rows(args)
    fit = read_fit(args)
    batch, native, groups, forecasts, vectors = setup(args, rows, TRANSFER)
    with torch.no_grad():
        scores = {f"{arm}_predictive": batch.utility(value) for arm, value in forecasts.items()}
        scores["oracle"] = batch.utility(action_factors(batch, native, groups))
        scores["calibrated"] = torch.stack([apply_ridge(vectors["affine"], head)
                                            for head in fit["calibrated"]], dim=-1)
        success = torch.stack([apply_ridge(vectors["affine"], head)
                               for head in fit["calibrated_success"]], dim=-1).clamp(0., 1.)
        work = apply_ridge(vectors["affine"], fit["calibrated_work"]).clamp_min(1.)
        work = torch.minimum(work, batch.remaining.clamp_min(1).float())
        work_by_action = torch.cat((torch.zeros_like(success[..., :2]), work.unsqueeze(-1)), dim=-1)
        scores["linear_factor"] = batch.utility(torch.stack((success, work_by_action), dim=-1))
        audit = {}
        for name, state in fit["weights"].items():
            if tensor_hash(state) != fit["weight_hashes"][name]:
                raise ValueError("the stopped-policy weights changed")
            family = "trm" if name == "trm_factor" else "affine"
            model = AcquisitionHead(vectors[family].shape[-1], scalar=name == "affine_scalar")
            model.load_state_dict(state)
            values, estimate = model(batch, vectors[family], forecasts[family])
            scores[name] = values
            if estimate is not None:
                # Isolate the price multiplier. Proof and work estimates stay fixed.
                keep, work = estimate[0, ..., 0], estimate[0, ..., 1]
                active = batch.legal[0, ..., 2]
                low = (keep - .035 * work).argmax(-1)[active] == 2
                high = (keep - .14 * work).argmax(-1)[active] == 2
                if (high & ~low).any():
                    raise ValueError("raising only the price created a new probe")
                audit[name] = {"eligible": int(active.sum()), "buy_at_low_fee": int(low.sum()),
                               "buy_at_high_fee": int(high.sum())}
        policies = {arm: masked(q, batch) for arm, q in scores.items()}

    results = {}
    for arm, q in policies.items():
        records = records_for(q, rows)
        results[arm] = records
        print(arm, sum(row["success"] for row in records),
              round(sum(row["return"] for row in records) / len(records), 6), flush=True)

    # Re-run one preselected valuation per layout in Lean at the middle full-cap fee.
    checked = set()
    native_audit = []
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("the native spot check requires the pinned Lean")
        for j, row in enumerate(rows):
            if row["cluster"] in checked:
                continue
            checked.add(row["cluster"])
            result = replay(repl, tree, row, policies["affine_factor"][1, j])
            expected = results["affine_factor"][len(rows) + j]
            if (result["witness"], result["observations"], result["success"]) != (
                    expected["witness"], expected["diagnostics"], expected["success"]):
                raise ValueError("the native proof changed its archived policy outcome")
            native_audit.append({**result, "fee": expected["fee"], "return": expected["return"]})
    write_receipt(args.output, {"schema": SCHEMA, "fit_sha256": sha256_file(args.fit),
                                "witness_sha256": sha256_file(args.witness), "regimes": TRANSFER,
                                "results": results, "price_intervention": audit, "native_audit": native_audit,
                                "memory": tree.stats()})


def forecast_swap(args):
    """Exchange the frozen forecast while retaining the acquisition head."""
    _, rows = witness_rows(args)
    fit = read_fit(args)
    result = json.loads(gzip.decompress(args.evaluation.read_bytes()))
    if (result["fit_sha256"] != sha256_file(args.fit) or result["witness_sha256"] != sha256_file(args.witness)):
        raise ValueError("the forecast intervention requires the scored acquisition fit")
    batch, native, _, forecasts, vectors = setup(args, rows, TRANSFER)
    results = {}
    horizons = torch.tensor([row["arity"] for row in rows])
    worlds = torch.arange(len(rows))
    decoder_accuracy = {}
    with torch.no_grad():
        for arm, forecast in forecasts.items():
            choice = forecast[0, worlds, horizons, :2, 0].argmax(-1)
            proved = native[worlds, choice]
            decoder_accuracy[arm] = {str(n): int(proved[horizons == n].sum()) for n in (5, 6)}
        for arm, source, donor in (("affine_factor", "affine", "trm"), ("trm_factor", "trm", "affine")):
            change_start = vectors[source].shape[-1] - 6
            torch.testing.assert_close(vectors[source][..., :change_start], vectors[donor][..., :change_start],
                                       atol=0., rtol=0.)
            model = AcquisitionHead(vectors[source].shape[-1]).eval()
            weights = fit["weights"][arm]
            if tensor_hash(weights) != fit["weight_hashes"][arm]:
                raise ValueError("intervention requires the unchanged acquisition head")
            model.load_state_dict(weights)
            scores, _ = model(batch, vectors[source], forecasts[source])
            identity = records_for(masked(scores, batch), rows)
            if identity != result["results"][arm]:
                raise ValueError("the identity forecast failed to replay the natural policy")
            scores, _ = model(batch, vectors[donor], forecasts[donor])
            results[arm] = records_for(masked(scores, batch), rows)
    write_receipt(args.output, {"schema": "ppt.rollback-acquisition-forecast-swap.v1",
                                "fit_sha256": sha256_file(args.fit), "evaluation_sha256": sha256_file(args.evaluation),
                                "witness_sha256": sha256_file(args.witness), "results": results,
                                "decoder_at_full_history": decoder_accuracy})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    trainer, evaluator, probe = (commands.add_parser(name) for name in ("fit", "evaluate", "forecast-swap"))
    for command in (trainer, evaluator, probe):
        for path in ("training_witness", "witness", "decoder", "warm", "output"):
            command.add_argument(path, type=Path)
    evaluator.add_argument("--fit", type=Path, required=True)
    probe.add_argument("--fit", type=Path, required=True)
    probe.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        if args.stage == "fit":
            fit(args)
        elif args.stage == "evaluate":
            evaluate(args, tree)
        else:
            forecast_swap(args)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
