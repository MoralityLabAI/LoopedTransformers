"""Fit and compare Lean witness and cache decisions on native return tables."""

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, sha256_file
from ppt_gym.dependent_witness import PREFIX, proof_view
from ppt_gym.live_lean import public_state
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.reuse_policy import ReusePolicy, world_features

from configs.freeze_ppt_reuse_plans import CACHE_INDEX, SCHEMA as WITNESS_SCHEMA
from configs.run_ppt_slot_continuations import read_continuation_fit, restore_policy
from configs.run_ppt_two_slot import CPU_LIMIT, request_state


SCHEMA = "ppt.reuse-policy.v1"
SEED = 831
EPOCHS = 12
TEMPERATURE = .4


def registry(path):
    witness = json.loads(gzip.decompress(path.read_bytes()))
    if witness["schema"] != WITNESS_SCHEMA or witness["lean"] != LEAN_VERSION:
        raise ValueError("the cache comparison requires the matching verified plans")
    rows = {"train": [], "dev": []}
    for row in witness["rows"]:
        rows[row["split"]].append(row)
    seen = set()
    for split, count in (("train", 16), ("dev", 24)):
        cases = rows[split]
        pairs = {row["cluster"] for row in cases}
        if len(pairs) != count or len(cases) != 2 * count or seen & pairs:
            raise ValueError("the reuse layouts overlap or are incomplete")
        if any(left["cluster"] != right["cluster"] or [left["world"], right["world"]] != [0, 1]
               for left, right in zip(cases[::2], cases[1::2], strict=True)):
            raise ValueError("a layout lost its two-world pairing")
        seen.update(pairs)
    return rows


def certified_returns(rows):
    return torch.tensor([[[float(plan["closed_in_budget"]) for plan in plans] for plans in row["receipts"]]
                         for row in rows])


def loss_for(model, features, returns, objective):
    openings, caches, _ = model(features)
    if objective == "return":
        return F.mse_loss(openings, returns.max(-1).values) + F.mse_loss(caches, returns)
    if objective == "imitation":
        witnesses = returns.max(-1).values.argmax(-1)
        world = torch.arange(len(returns))
        chosen = returns[world, witnesses].argmax(-1)
        return F.cross_entropy(openings, witnesses) + F.cross_entropy(caches[world, witnesses], chosen)
    opening_prob = (openings / TEMPERATURE).softmax(-1)
    cache_prob = (caches / TEMPERATURE).softmax(-1)
    continuation = (cache_prob * returns).sum(-1)
    return -((opening_prob * continuation).sum(-1).mean() + .25 * continuation.mean())


def train_model(model, features, returns, objective):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    rng = random.Random(SEED)
    layouts = list(range(len(features) // 2))
    for _ in range(EPOCHS):
        rng.shuffle(layouts)
        for layout in layouts:
            indices = slice(2 * layout, 2 * layout + 2)
            loss = loss_for(model, features[indices], returns[indices], objective)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    return EPOCHS * len(layouts)


def ridge(features, targets, layout):
    """Choose the ridge penalty by holding out both worlds of a layout."""
    x = features.flatten(0, -2).double()
    y = targets.flatten().double()
    x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    regularizer = x.T @ x
    scale = regularizer.diag()[:-1].mean().clamp_min(1e-8)
    penalty_matrix = torch.eye(x.shape[1], dtype=x.dtype)
    penalty_matrix[-1, -1] = 0.

    def solve(train, penalty):
        train_x, train_y = x[train], y[train]
        return torch.linalg.solve(train_x.T @ train_x + scale * penalty * penalty_matrix, train_x.T @ train_y)

    errors = {}
    for penalty in (.0001, .001, .01, .1, 1.):
        error = 0.
        for group in layout.unique():
            test = layout == group
            error += float((x[test] @ solve(~test, penalty) - y[test]).square().mean())
        errors[penalty] = error
    selected = min(errors, key=errors.get)
    weights = solve(torch.ones(len(x), dtype=torch.bool), selected)
    return {"weights": weights.float(), "penalty": selected, "cv_error": errors}


def apply_ridge(features, readout):
    weights = readout["weights"]
    return features @ weights[:-1] + weights[-1]


def warm_models(checkpoint):
    source = read_continuation_fit(checkpoint, LEAN_VERSION)
    if not source.get("anonymous_witness") or not source.get("mixed_source_counts"):
        raise ValueError("reuse learning requires the audited mixed-source witness checkpoint")
    hybrid = restore_policy(source, "live_rl", source["weights"]["live_rl"])
    trm = restore_policy(source, "trm_workspace", source["weights"]["trm_workspace"])
    return ReusePolicy(hybrid), ReusePolicy(trm, trm=True), source["mixed_source_counts"]


def fit(args):
    rows = registry(args.witness)["train"]
    features = torch.stack([world_features(row) for row in rows])
    returns = certified_returns(rows)
    hybrid, trm, sources = warm_models(args.warm)
    arms = {
        "rl": (deepcopy(hybrid), "rl"),
        "imitation": (deepcopy(hybrid), "imitation"),
        "return_regression": (deepcopy(hybrid), "return"),
        "trm": (trm, "rl"),
    }
    updates = {}
    for name, (model, objective) in arms.items():
        updates[name] = train_model(model, features, returns, objective)
        print(name, updates[name], flush=True)
    layouts = torch.arange(len(rows)) // 2
    with torch.no_grad():
        context, plans, _ = hybrid.encode(features)
        frozen = {
            "opening": ridge(context, returns.max(-1).values, layouts.repeat_interleave(2)),
            "cache": ridge(plans, returns, layouts.repeat_interleave(6)),
            "raw": ridge(features.flatten(-2), returns, layouts.repeat_interleave(6)),
        }
    weights = {name: model.state_dict() for name, (model, _) in arms.items()}
    weights["warm"] = hybrid.state_dict()
    with args.output.open("xb") as out:
        torch.save({"schema": SCHEMA, "lean": LEAN_VERSION, "witness_sha256": sha256_file(args.witness),
                    "warm_sha256": sha256_file(args.warm), "mixed_source_counts": sources, "updates": updates,
                    "frozen": frozen, "weights": weights, "weight_hashes": {
                        name: tensor_hash(state) for name, state in weights.items()}}, out)


def replay(repl, tree, row, witness, cache):
    receipt = row["receipts"][witness][cache]
    start = tree.cpu_seconds()
    repl.reset()
    state = repl.init(row["statement"])
    actions = receipt["actions"][:row["checks"]]
    for index, action in enumerate(actions):
        if index == len(PREFIX) and proof_view(public_state(repl, state)["goals"]) != row["fork"]:
            raise ValueError("the live witness decision changed its public observation")
        if index == CACHE_INDEX and public_state(repl, state)["goals"] != row["cache_states"][witness]["goals"]:
            raise ValueError("the live cache decision changed its public observation")
        state = request_state(repl.run(state["sid"], action))
    goals = public_state(repl, state)["goals"]
    closed = not goals
    if (closed != receipt["closed_in_budget"] or goals != receipt["budget_goals"]
            or tree.cpu_seconds() - start > CPU_LIMIT):
        raise ValueError("the chosen plan disagreed with the native certificate or compute envelope")
    if closed:
        kernel = repl.kernel(state["sid"])
        if not kernel["ok"] or kernel["axioms"] or receipt["axioms"]:
            raise ValueError("the chosen plan could not be certified")
    return {"cluster": row["cluster"], "world": row["world"], "witness": witness, "cache": cache,
            "checks": len(actions), "success": closed, "source_digest": receipt["source_digest"] if closed else None}


def choices_for(name, model, features, frozen):
    if name == "raw_calibration":
        cache = apply_ridge(features.flatten(-2), frozen["raw"])
        return cache.max(-1).values, cache
    if name in {"calibrated", "option_calibration"}:
        context, plans, _ = model.encode(features)
        cache = apply_ridge(plans, frozen["cache"])
        opening = apply_ridge(context, frozen["opening"]) if name == "calibrated" else cache.max(-1).values
        return opening, cache
    openings, caches, _ = model(features, reset=name == "slot_reset")
    return openings, caches


def load_models(args):
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["lean"] != LEAN_VERSION
            or checkpoint["witness_sha256"] != sha256_file(args.witness)
            or checkpoint["warm_sha256"] != sha256_file(args.warm)):
        raise ValueError("the policy checkpoint does not match the frozen plans and mixed warm weights")
    initial, trm, _ = warm_models(args.warm)
    models = {}
    for arm, weights in checkpoint["weights"].items():
        if tensor_hash(weights) != checkpoint["weight_hashes"][arm]:
            raise ValueError("a trained arm's weights changed")
        model = deepcopy(trm if arm == "trm" else initial)
        model.load_state_dict(weights)
        models[arm] = model.eval()
    for name in ("calibrated", "option_calibration", "raw_calibration"):
        models[name] = models["warm"]
    models["slot_reset"] = models["rl"]
    return checkpoint, models


def evaluate(args, tree):
    rows = registry(args.witness)["dev"]
    checkpoint, models = load_models(args)
    features = torch.stack([world_features(row) for row in rows])
    results = {}
    with LeanRepl() as repl, torch.no_grad():
        if repl.version != LEAN_VERSION:
            raise ValueError("the evaluation requires the pinned Lean verifier")
        for name, model in models.items():
            opening, cache = choices_for(name, model, features, checkpoint["frozen"])
            records = []
            for i, row in enumerate(rows):
                branch = int(opening[i].argmax())
                continuation = int(cache[i, branch].argmax())
                records.append({**replay(repl, tree, row, branch, continuation),
                                "opening_scores": opening[i].tolist(), "cache_scores": cache[i, branch].tolist()})
            pairs = Counter(r["cluster"] for r in records if r["success"])
            results[name] = {"proofs": pairs.total(), "pairs": sum(n == 2 for n in pairs.values()), "rows": records}
            print(name, results[name]["proofs"], results[name]["pairs"], flush=True)
    report = {"schema": SCHEMA, "checkpoint_sha256": sha256_file(args.fit),
              "witness_sha256": checkpoint["witness_sha256"], "warm_sha256": checkpoint["warm_sha256"],
              "results": results, "memory": tree.stats()}
    with args.output.open("xb") as out:
        out.write(gzip.compress((canonical(report) + "\n").encode(), mtime=0))


def bpd(args):
    rows = registry(args.witness)["dev"]
    checkpoint, models = load_models(args)
    native = json.loads(gzip.decompress(args.evaluation.read_bytes()))
    if (native["schema"] != SCHEMA or native["checkpoint_sha256"] != sha256_file(args.fit)
            or native["witness_sha256"] != checkpoint["witness_sha256"]):
        raise ValueError("the cache intervention requires the matching native evaluation")
    features = torch.stack([world_features(row) for row in rows])
    returns = certified_returns(rows)
    results = {}
    with torch.no_grad():
        for arm in ("rl", "calibrated", "trm"):
            model, frozen = models[arm], checkpoint["frozen"]
            opening, cache = choices_for(arm, model, features, frozen)
            _, _, memory = model.encode(features)
            donor = memory.unflatten(0, (-1, 2)).flip(1).flatten(0, 1)
            delta = donor - memory
            edits = {"identity": memory, "opposite": donor, "same_structure": donor.flip(1),
                     "norm_sham": memory + delta.roll(3, dims=-1)}
            torch.testing.assert_close((edits["norm_sham"] - memory).norm(dim=-1), delta.norm(dim=-1))
            variants = {}
            for name, carry in edits.items():
                context, plans, _ = model.encode(features, donor=carry)
                if arm == "calibrated":
                    scores = apply_ridge(context, frozen["opening"])
                    variants[name] = apply_ridge(plans, frozen["cache"])
                else:
                    scores = model.actor(context).squeeze(-1)
                    variants[name] = model.cache_actor(plans).squeeze(-1)
                torch.testing.assert_close(scores, opening, atol=0., rtol=0.)
            torch.testing.assert_close(variants["identity"], cache, atol=0., rtol=0.)
            records = []
            for i, receipt in enumerate(native["results"][arm]["rows"]):
                chosen = int(opening[i].argmax())
                original = int(cache[i, chosen].argmax())
                if (receipt["cluster"] != rows[i]["cluster"] or receipt["world"] != rows[i]["world"]
                        or receipt["witness"] != chosen or receipt["cache"] != original
                        or receipt["success"] != bool(returns[i, chosen, original])):
                    raise ValueError("the natural intervention choice did not match Lean")
                edits_at_choice = {}
                for name, scores in variants.items():
                    plan = int(scores[i, chosen].argmax())
                    edits_at_choice[name] = {"cache": plan, "success": bool(returns[i, chosen, plan]),
                                             "scores": scores[i, chosen].tolist()}
                records.append({"cluster": receipt["cluster"], "world": receipt["world"], "witness": chosen,
                                "cache": original, "success": receipt["success"],
                                "donor_dose": float(delta[i, chosen].norm()), "edits": edits_at_choice})
            if len(records) != len(rows):
                raise ValueError("the native intervention comparison is incomplete")
            results[arm] = records
    report = {"schema": "ppt.reuse-policy-bpd.v1", "checkpoint_sha256": native["checkpoint_sha256"],
              "native_sha256": sha256_file(args.evaluation), "witness_sha256": checkpoint["witness_sha256"],
              "results": results}
    with args.output.open("xb") as out:
        out.write(gzip.compress((canonical(report) + "\n").encode(), mtime=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    train = commands.add_parser("fit")
    score = commands.add_parser("evaluate")
    probe = commands.add_parser("bpd")
    for command in (train, score, probe):
        command.add_argument("witness", type=Path)
        command.add_argument("warm", type=Path)
        command.add_argument("output", type=Path)
    for command in (score, probe):
        command.add_argument("--fit", type=Path, required=True)
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
            bpd(args)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
