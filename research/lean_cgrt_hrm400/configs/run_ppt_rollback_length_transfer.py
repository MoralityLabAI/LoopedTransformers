"""Train on short diagnostic histories and replay longer ones in native Lean."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from lcgrt.config import LEAN_VERSION, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.diagnostic_accumulator import DiagnosticAccumulator, MAX_OBSERVATIONS
from ppt_gym.live_lean import public_state
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rollback_diagnostics import DIAGNOSTIC_COST

from configs.check_ppt_mathcan_native_costs import NAMESPACE, PREFIX_MODULE, declaration, invoke_lean
from configs.freeze_ppt_rollback_length_transfer import (
    ALIASES, OPERATORS, SCHEMA as WITNESS_SCHEMA, diagnostic, expression, native_commit,
)
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import open_certificate, warm, write_receipt
from configs.run_ppt_two_slot import CPU_LIMIT


SCHEMA = "ppt.rollback-length-policy.v1"
SEED = 831
STEPS = 180
TEMPERATURE = .55
Q_TEMPERATURE = .10


@dataclass
class Batch:
    context: torch.Tensor
    operators: torch.Tensor
    values: torch.Tensor
    clock: torch.Tensor
    valid: torch.Tensor
    legal: torch.Tensor
    raw: torch.Tensor

    def inputs(self):
        return self.context, self.operators, self.values, self.clock


def diagnostic_operator(raw_goal, arity):
    _, separator, term = raw_goal.rpartition("∧ n =")
    formatting = str.maketrans("", "", "() \t\r\n")
    if not separator:
        raise ValueError(f"expected a public witness formula: {raw_goal}")
    matches = [name for name in OPERATORS if term.translate(formatting) == expression(arity, name).translate(formatting)]
    if len(matches) != 1:
        raise ValueError(f"unregistered public witness formula: {term}")
    return matches[0]


def registry(path):
    report = json.loads(gzip.decompress(path.read_bytes()))
    if report["schema"] != WITNESS_SCHEMA or report["lean"] != LEAN_VERSION:
        raise ValueError("expected the frozen native length-transfer witness")
    sides = {split: [row for row in report["rows"] if row["split"] == split] for split in ("train", "dev")}
    if {row["arity"] for row in sides["train"]} != {2, 3} or {row["arity"] for row in sides["dev"]} != {4, 5}:
        raise ValueError("the training and development horizons changed")
    roots = {split: {row["payload_root"] for row in rows} for split, rows in sides.items()}
    if len(roots["train"]) != 12 or len(roots["dev"]) != 16 or roots["train"] & roots["dev"]:
        raise ValueError("the source proofs are incomplete or overlap")
    return report, sides


def features(rows, encoder):
    values, operators, roots = [], [], []
    embeddings = {}
    with torch.no_grad():
        for row in rows:
            goals = row["public"]["goals"]
            raw_goal = " ".join(goals[0]["target"]["raw"].split())
            parsed = diagnostic_operator(raw_goal, row["arity"])
            if parsed != row["operator"]:
                raise ValueError(f"the registered operator is missing from Lean's visible goal: {raw_goal}")
            if row["cluster"] not in embeddings:
                embeddings[row["cluster"]] = F.normalize(encoder.encode(goals).mean(0), dim=0)
            roots.append(embeddings[row["cluster"]] * .2)
            operators.append(OPERATORS.index(parsed))
            values.append([event["value"] for event in row["observations"]]
                          + [-1] * (MAX_OBSERVATIONS - row["arity"]))
    operators = torch.tensor(operators)
    values = torch.tensor(values)
    horizons = (values >= 0).sum(-1, keepdim=True)
    time = torch.arange(MAX_OBSERVATIONS + 1).expand(len(rows), -1)
    clock = torch.stack((time / MAX_OBSERVATIONS, (horizons - time).clamp_min(0) / MAX_OBSERVATIONS), dim=-1)
    valid = time <= horizons
    legal = torch.stack((torch.ones_like(valid), torch.ones_like(valid), time < horizons), dim=-1)
    root = torch.cat((F.one_hot(operators, len(OPERATORS)).float(), torch.stack(roots)), dim=-1)
    counts = F.one_hot(values.clamp_min(0), 2).float() * (values >= 0).unsqueeze(-1)
    counts = F.pad(counts.cumsum(1), (0, 0, 1, 0))
    raw = torch.cat((root[:, None].expand(-1, MAX_OBSERVATIONS + 1, -1), clock, counts), dim=-1)
    return Batch(root, operators, values, clock, valid, legal, raw)


def groups_for(rows):
    groups = []
    for time in range(MAX_OBSERVATIONS + 1):
        ids, states = {}, []
        for row in rows:
            bits = tuple(observation["value"] for observation in row["observations"][:time])
            states.append(ids.setdefault((row["cluster"], bits), len(ids)))
        index = torch.tensor(states)
        groups.append((index, torch.bincount(index).unsqueeze(-1)))
    return groups


def action_values(batch, rewards, groups, *, policy=None):
    """Average native returns over worlds indistinguishable at each prefix."""
    q = rewards.new_zeros(batch.legal.shape)
    q[..., :2] = rewards[:, None, :]
    continuation = rewards.new_zeros(len(rewards))
    for time in range(MAX_OBSERVATIONS, -1, -1):
        q[:, time, 2] = continuation - DIAGNOSTIC_COST
        index, count = groups[time]
        mean = q.new_zeros((len(count), q.shape[-1])).index_add(0, index, q[:, time]) / count
        if policy is None:
            q[:, time] = mean[index]
            continuation = q[:, time].masked_fill(~batch.legal[:, time], -torch.inf).amax(-1)
        else:
            continuation = (policy[:, time] * q[:, time]).sum(-1)
            q[:, time] = mean[index]
    return q


def model_kind(name):
    if name == "trm":
        return "trm"
    return "recurrent" if name == "recurrent" else "affine"


def train(model, batch, rewards, groups, teacher, *, imitate=False):
    legal = batch.legal
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    for _ in range(STEPS):
        logits = model(*batch.inputs()).masked_fill(~legal, -torch.inf)
        with torch.no_grad():
            if imitate:
                best = teacher.masked_fill(~legal, -torch.inf)
                ties = (best >= best.amax(-1, keepdim=True) - 1e-7).float()
                targets = ties / ties.sum(-1, keepdim=True)
            else:
                q = action_values(batch, rewards, groups, policy=(logits / TEMPERATURE).softmax(-1))
                targets = (q.masked_fill(~legal, -torch.inf) / Q_TEMPERATURE).softmax(-1)
        log_probabilities = (logits / TEMPERATURE).log_softmax(-1).masked_fill(~legal, 0.)
        loss = -(targets * log_probabilities).sum(-1)[batch.valid].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def heads_for(values, targets, batch, ids):
    active = batch.legal & batch.valid.unsqueeze(-1)
    return [ridge(values[active[..., i]], targets[..., i][active[..., i]], ids[active[..., i]]) for i in range(3)]


def fit(args):
    _, rows = registry(args.witness)
    hybrid, trm, mixed = warm(args.warm)
    data = rows["train"]
    batch = features(data, hybrid)
    rewards = torch.tensor([[float(path["passed"]) for path in row["witness_trials"]] for row in data])
    posterior = groups_for(data)
    teacher = action_values(batch, rewards, posterior)
    width = batch.context.shape[-1]
    base = DiagnosticAccumulator(hybrid, width)
    models = {"affine": deepcopy(base), "imitation": deepcopy(base), "warm": base,
              "recurrent": DiagnosticAccumulator(hybrid, width, kind="recurrent"),
              "trm": DiagnosticAccumulator(trm, width, kind="trm")}
    recurrent_weights = models["recurrent"].state_dict()
    recurrent_weights.update({name: value for name, value in base.state_dict().items() if name in recurrent_weights})
    models["recurrent"].load_state_dict(recurrent_weights)
    for name in ("affine", "imitation", "recurrent", "trm"):
        train(models[name], batch, rewards, posterior, teacher, imitate=name == "imitation")
        print(name, STEPS, flush=True)
    unique = {root: i for i, root in enumerate(sorted({row["payload_root"] for row in data}))}
    ids = torch.tensor([unique[row["payload_root"]] for row in data])[:, None].expand_as(batch.valid)
    with torch.no_grad():
        commit, acquire = base.reads(base.encode(*batch.inputs()[:3]), batch.clock)
        frozen = {"calibrated": heads_for(torch.cat((commit, acquire), dim=-1), teacher, batch, ids),
                  "raw_calibrated": heads_for(batch.raw, teacher, batch, ids)}
    weights = {name: model.state_dict() for name, model in models.items()}
    with args.output.open("xb") as out:
        torch.save({"schema": SCHEMA, "lean": LEAN_VERSION, "steps": STEPS, "seed": SEED, "mixed_sources": mixed,
                    "witness_sha256": sha256_file(args.witness), "warm_sha256": sha256_file(args.warm),
                    "weights": weights, "frozen": frozen,
                    "weight_hashes": {k: tensor_hash(v) for k, v in weights.items()}}, out)


def loaded(args, data):
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (fit["schema"] != SCHEMA or fit["lean"] != LEAN_VERSION
            or fit["witness_sha256"] != sha256_file(args.witness) or fit["warm_sha256"] != sha256_file(args.warm)):
        raise ValueError("the fit and source receipts do not match")
    hybrid, trm, _ = warm(args.warm)
    batch = features(data, hybrid)
    models = {}
    for name, weights in fit["weights"].items():
        model = DiagnosticAccumulator(trm if name == "trm" else hybrid, batch.context.shape[-1], kind=model_kind(name))
        if tensor_hash(weights) != fit["weight_hashes"][name]:
            raise ValueError("the learned weights changed")
        model.load_state_dict(weights)
        models[name] = model.eval()
    return models, fit["frozen"], batch


def replay(repl, tree, row, policy):
    start = tree.cpu_seconds()
    repl.reset()
    root = repl.init(open_certificate(row))
    if public_state(repl, root) != row["public"]:
        raise ValueError("the native initial observation changed")
    history = []
    for step, values in enumerate(policy):
        action = int(values.argmax())
        if action < 2:
            break
        if step >= row["arity"]:
            raise ValueError("the policy spent part of the reserved proof allowance")
        root, observed = diagnostic(repl, root, ALIASES[step])
        if observed != row["observations"][step]:
            raise ValueError("the native diagnostic no longer matches its receipt")
        history.append(observed)
    expected = row["witness_trials"][action]
    winning = row["witness_trials"][row["winning_witness"]]
    result = native_commit(repl, root, winning["actions"][2:-1], action, winning["axioms"])
    charge = len(history) + result["checks"]
    if digest(result) != digest(expected) or charge > row["budget"] or tree.cpu_seconds() - start > CPU_LIMIT:
        raise ValueError(f"{row['tid']} changed its Lean outcome or budget: "
                         f"witness {action}, charged {charge}, budget {row['budget']}")
    return {"tid": row["tid"], "cluster": row["cluster"], "payload_root": row["payload_root"],
            "arity": row["arity"], "source": row["source"], "operator": row["operator"], "witness": action,
            "observations": len(history), "success": result["passed"], "checks": charge,
            "return": float(result["passed"]) - DIAGNOSTIC_COST * len(history)}


def scored(models, calibrations, batch):
    with torch.no_grad():
        scores = {name: model(*batch.inputs()) for name, model in models.items()}
        root = models["warm"]
        commit, acquire = root.reads(root.encode(*batch.inputs()[:3]), batch.clock)
        frozen = torch.cat((commit, acquire), dim=-1)
        for arm, vals in (("calibrated", frozen), ("raw_calibrated", batch.raw)):
            scores[arm] = torch.stack([apply_ridge(vals, head) for head in calibrations[arm]], dim=-1)
        return {arm: values.masked_fill(~batch.legal, -torch.inf) for arm, values in scores.items()}


def summarize(records, counts):
    solved = Counter(row["cluster"] for row in records if row["success"])
    return {"proofs": solved.total(), "layouts": sum(solved[key] == n for key, n in counts.items()),
            "diagnostics": sum(row["observations"] for row in records), "rows": records}


def evaluate(args, tree):
    witness, sides = registry(args.witness)
    data = sides["dev"]
    models, calibrations, batch = loaded(args, data)
    scores = scored(models, calibrations, batch)
    before, _, _ = declaration(args.mathcan_source.read_text(encoding="utf-8"), "full_horizon_recovers")
    prelude = before + f"end {NAMESPACE}\n"
    if digest(prelude) != witness["prelude_sha256"] or sha256_file(args.mathcan_source) != witness["source_hashes"]["mathcan_source"]:
        raise ValueError("the audited MathCan source changed")
    results = {}
    counts = Counter(row["cluster"] for row in data)
    with TemporaryDirectory() as scratch:
        directory = Path(scratch)
        source = directory / f"{PREFIX_MODULE}.lean"
        source.write_text(prelude, encoding="utf-8")
        env = lean_env()
        invoke_lean(directory, env, "-o", f"{PREFIX_MODULE}.olean", source.name)
        env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
        with LeanRepl(imports=(PREFIX_MODULE,), cwd=directory, env=env) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the transfer evaluation requires the pinned Lean")
            for arm, policy in scores.items():
                records = [replay(repl, tree, row, actions) for row, actions in zip(data, policy, strict=True)]
                results[arm] = summarize(records, counts)
                print(arm, {key: val for key, val in results[arm].items() if key != "rows"}, flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "fit_sha256": sha256_file(args.fit),
                                "witness_sha256": sha256_file(args.witness), "results": results,
                                "memory": tree.stats()})


def consensus(args):
    """Replay a policy that commits when all of its possible completions agree."""
    _, sides = registry(args.witness)
    data = sides["dev"]
    models, calibrations, batch = loaded(args, data)
    scores = scored(models, calibrations, batch)
    groups = defaultdict(set)
    histories = []
    for row in data:
        bits = tuple(observation["value"] for observation in row["observations"])
        histories.append(bits)
        groups[row["cluster"]].add(bits)
    if any(len(group) != 2 ** len(next(iter(group))) for group in groups.values()):
        raise ValueError("lookahead requires an unweighted exhaustive set of possible bits")
    counts = {cluster: len(paths) for cluster, paths in groups.items()}
    results = {}
    for arm, logits in scores.items():
        # The only information used by the lookahead table is the model's
        # prediction for each hypothetical bit sequence, never its certificate.
        votes = defaultdict(set)
        for row, bits, predicted in zip(data, histories, logits, strict=True):
            decision = int(predicted[len(bits), :2].argmax())
            for length in range(len(bits) + 1):
                votes[row["cluster"], bits[:length]].add(decision)
        records = []
        for row, bits in zip(data, histories, strict=True):
            for length in range(len(bits) + 1):
                ballot = votes[row["cluster"], bits[:length]]
                if len(ballot) == 1:
                    witness, = ballot
                    break
            trial = row["witness_trials"][witness]
            records.append({"tid": row["tid"], "cluster": row["cluster"], "payload_root": row["payload_root"],
                            "operator": row["operator"], "arity": row["arity"], "witness": witness,
                            "observations": length, "success": trial["passed"], "checks": trial["checks"] + length,
                            "return": float(trial["passed"]) - DIAGNOSTIC_COST * length})
        results[arm] = summarize(records, counts)
        print(arm, {key: val for key, val in results[arm].items() if key != "rows"}, flush=True)
    write_receipt(args.output, {"schema": "ppt.rollback-length-consensus.v1", "fit_sha256": sha256_file(args.fit),
                                "witness_sha256": sha256_file(args.witness), "results": results})


def bpd(args):
    _, sides = registry(args.witness)
    data = sides["dev"]
    models, calibrations, batch = loaded(args, data)
    keys = {(row["cluster"], tuple(p["value"] for p in row["observations"])): i for i, row in enumerate(data)}
    donor, stops = [], []
    for row in data:
        first, *rest = [p["value"] for p in row["observations"]]
        donor.append(keys[row["cluster"], (1 - first, *rest)])
        stops.append(row["arity"])
    order = torch.arange(len(data))
    results = {}
    with torch.no_grad():
        for arm in ("affine", "recurrent", "trm", "calibrated"):
            model = models["warm"] if arm == "calibrated" else models[arm]
            natural = model.encode(*batch.inputs()[:3])[order, stops]
            clock = batch.clock[order, stops].unsqueeze(1)
            actual = natural.clone()
            actual[:, 1] = natural[donor, 1]
            delta = actual[:, 1] - natural[:, 1]
            sham = natural.clone()
            sham[:, 1] += delta.roll(3, dims=-1)
            torch.testing.assert_close(delta.norm(dim=-1), (sham[:, 1] - natural[:, 1]).norm(dim=-1))
            scores = {}
            for name, tokens in (("identity", natural), ("opposite", actual), ("norm_sham", sham)):
                commit, acquire = model.reads(tokens.unsqueeze(1), clock)
                if arm == "calibrated":
                    features = torch.cat((commit, acquire), dim=-1)
                    readout = torch.stack([apply_ridge(features, head) for head in calibrations[arm][:2]], dim=-1)
                else:
                    readout = model.commit(commit)
                scores[name] = readout.squeeze(1).tolist()
            results[arm] = [{"tid": row["tid"], "cluster": row["cluster"], "arity": row["arity"],
                             "operator": row["operator"], "winner": row["winning_witness"],
                             "donor_winner": data[donor[i]]["winning_witness"], "norm": float(delta[i].norm()),
                             "scores": {name: values[i] for name, values in scores.items()}}
                            for i, row in enumerate(data)]
    write_receipt(args.output, {"schema": "ppt.rollback-length-bpd.v1", "fit_sha256": sha256_file(args.fit),
                                "witness_sha256": sha256_file(args.witness), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    train_parser, score, probe, replay_parser = (
        commands.add_parser(name) for name in ("fit", "evaluate", "bpd", "consensus")
    )
    for command in (train_parser, score, probe, replay_parser):
        command.add_argument("witness", type=Path)
        command.add_argument("warm", type=Path)
        command.add_argument("output", type=Path)
    for command in (score, probe, replay_parser):
        command.add_argument("--fit", type=Path, required=True)
    score.add_argument("--mathcan-source", type=Path, required=True)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        if args.stage == "fit":
            fit(args)
        elif args.stage == "evaluate":
            evaluate(args, tree)
        elif args.stage == "bpd":
            bpd(args)
        else:
            consensus(args)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
