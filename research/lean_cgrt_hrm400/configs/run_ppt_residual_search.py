"""Learn from native sibling feedback and replay residual proof continuations."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import fsum
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.freeze_ppt_native_progress import advance, observe
from configs.freeze_ppt_residual_search import SCHEMA as BANK_SCHEMA
from configs.freeze_ppt_scope_recovery import PREFIX_MODULE, native_scope, prefix_text
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from configs.run_ppt_scope_recovery import improve
from lcgrt.config import LEAN_EXE, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.residual_search import PROBE, STOP, ResidualSearch, State
from ppt_gym.residual_search_policy import NativeEvidence, disposition


SCHEMA = "ppt.residual-search.v1"
SEED = 1201
STEPS = 120
BATCH = 48
TEMPERATURE = .12
FLOOR = .025
BUDGETS = (1, 2)
STOP_GRID = (0., .025, .05, .10, .15, .25)


def inputs(args, split):
    bank = read_receipt(args.bank)
    prelude = prefix_text(args.mathcan)
    if (bank["schema"] != BANK_SCHEMA or bank["prelude"] != digest(prelude)
            or bank["lean"] != sha256_file(LEAN_EXE) or bank["repl"] != sha256_file(REPL_EXE)):
        raise ValueError("the original native residual bank or runtime changed")
    roots = {key: {row["module"] for row in bank["rows"] if row["split"] == key} for key in ("train", "dev")}
    if roots["train"] & roots["dev"]:
        raise ValueError("a held native proof entered source fitting")
    cases = [(proof, task) for proof in bank["rows"] if proof["split"] == split for task in proof["tasks"]]
    pins = {name: sha256_file(getattr(args, name)) for name in ("bank", "warm", "prior")}
    return cases, prelude, pins


def pack(cases, prior, *, opaque=False):
    entries, features, anchors, numbers = [], [], [], {}
    for _, task in cases:
        search = ResidualSearch(task)
        evidence = NativeEvidence(search, prior, task["case"] if opaque else None)

        def visit(state):
            key = task["case"], state
            if key in numbers:
                return numbers[key]
            actions = search.menu(state)
            rows, base = evidence.describe(state)
            edges, rewards, exact = [], [], []
            for action in actions:
                reward, child, _ = search.transition(state, action)
                index = visit(child) if child is not None else 0
                edges.append(index)
                rewards.append(reward)
                exact.append(reward + (max(entries[index - 1][3]) if index else 0.))
            columns = list(range(len(features), len(features) + len(rows)))
            features.extend(rows)
            anchors.extend(base)
            height = 1 + max((entries[child - 1][-1] for child in edges if child), default=0)
            entries.append((columns, edges, rewards, exact, height))
            numbers[key] = len(entries)
            return len(entries)

        for budget in BUDGETS:
            visit(State(budget))
    width = max(len(columns) for columns, *_ in entries)
    mask = torch.tensor([[index < len(row[0]) for index in range(width)] for row in entries])
    packed = [torch.tensor([row[i] + [0] * (width - len(row[i])) for row in entries]) for i in range(4)]
    indices, edges, immediate, optimum = packed
    if not torch.equal(indices[mask].sort().values, torch.arange(len(features))):
        raise ValueError("the residual action packing lost its original order")
    return (torch.tensor(features), torch.tensor(anchors), indices, mask, edges[..., None],
            immediate, optimum, torch.tensor([row[-1] for row in entries])), numbers


def models():
    actor = TwoSlotQuoteActor()
    trm = LiveQuoteTRM()
    for model in (actor, trm):
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    return {"two_slot_reward": actor, "two_slot_imitation": deepcopy(actor),
            "trm_reward": trm, "frozen_encoder": deepcopy(actor)}


def layout(values, indices, mask):
    return values[indices].masked_fill(~mask, -torch.inf)


def flattened_scores(model, features):
    return torch.cat([model(chunk) for chunk in features.split(1024)])


def train(model, name, packed, batches):
    features, anchors, indices, mask, edges, immediate, optimum, height = packed
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    target = optimum
    probabilities = mask[..., None].float()
    for step, batch in enumerate(batches):
        if name != "two_slot_imitation" and step % 12 == 0:
            with torch.no_grad():
                current = layout(anchors + flattened_scores(model, features), indices, mask)
                target = improve(current, immediate, edges, probabilities, height)
        available = mask[batch]
        ids = indices[batch][available]
        prediction = model(features[ids]) + anchors[ids]
        logits = prediction.new_full(available.shape, -torch.inf).masked_scatter(available, prediction)
        teacher = (target[batch].masked_fill(~available, -torch.inf) / TEMPERATURE).softmax(-1)
        log_policy = (logits / TEMPERATURE).log_softmax(-1).masked_fill(~available, 0.)
        loss = -(teacher * log_policy).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


@torch.no_grad()
def calibration_features(features, anchors, name, frozen):
    memory = features[:, 1, 16:21]
    context = features[:, 2, 3:5]
    interaction = (memory[..., None] * context[:, None]).flatten(1)
    raw = torch.cat((torch.ones_like(anchors[:, None]), anchors[:, None], features.flatten(1), interaction), dim=-1)
    if name == "frozen_slot_calibration":
        hidden = [frozen.representation(frozen.embed(chunk) + frozen.roles) for chunk in features.split(1024)]
        return torch.cat((raw, torch.cat(hidden)), dim=-1)
    return raw


@torch.no_grad()
def calibrate(packed, frozen):
    features, anchors, indices, mask, _, _, optimum, _ = packed
    ordering = indices[mask].argsort()
    target = (optimum[mask][ordering] - anchors).double()
    weights = {}
    for name in ("raw_calibration", "frozen_slot_calibration"):
        design = calibration_features(features, anchors, name, frozen).double()
        gram = design.T @ design + .1 * torch.eye(design.shape[1], dtype=design.dtype)
        weights[name] = torch.linalg.solve(gram, design.T @ target)
    return weights


def fit(args, tree):
    cases, _, pins = inputs(args, "train")
    packed, numbers = pack(cases, MixedSourcePrior(args))
    readers = models()
    initial = {name: tensor_hash(model.state_dict()) for name, model in readers.items()}
    calibration = calibrate(packed, readers["frozen_encoder"])
    batches = torch.randint(len(numbers), (STEPS, BATCH), generator=torch.Generator().manual_seed(SEED))
    for name in ("two_slot_reward", "two_slot_imitation", "trm_reward"):
        train(readers[name], name, packed, batches)
        print("fitted", name, flush=True)
    weights = {name: model.state_dict() for name, model in readers.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
                    "weight_sha256": {name: tensor_hash(state) for name, state in weights.items()},
                    "calibration": calibration, "calibration_sha256": tensor_hash(calibration),
                    "source_cases": len(cases), "source_states": len(numbers),
                    "sampler_sha256": tensor_hash({"batches": batches}), "memory": tree.stats()}, handle)
    print("sealed residual fit", sha256_file(args.fit), flush=True)


def replay(repl, row, case, trial):
    def open_at(prefix, expected):
        repl.reset()
        state = repl.init(row["statement"], pp=True)
        shown = observe(repl, state)
        for action in prefix:
            status, state, shown = advance(repl, state, shown, action)
            if status != "progress":
                raise ValueError("native prefix no longer reaches a residual")
        if digest(shown) != expected:
            raise ValueError("native proof residual observation changed")
        return state, shown

    remaining = trial["trace"]
    if remaining and remaining[0]["role"] == PROBE:
        check, *remaining = remaining
        node = case["sibling"]
        state, shown = open_at(case["sibling_prefix"], node["key"])
        observed, _, after = advance(repl, state, shown, check["action"])
        if observed != check["status"] or digest(after) != node["actions"][check["action"]]["after"]:
            raise ValueError("native sibling evidence changed")
    node = case["target"]
    state, shown = open_at(case["target_prefix"], node["key"])
    for i, event in enumerate(remaining):
        observed, state, shown = advance(repl, state, shown, event["action"])
        registered = node["actions"][event["action"]]
        if observed != event["status"] or digest(shown) != registered["after"]:
            raise ValueError("native continuation outcome changed")
        if observed == "progress" and i + 1 < len(remaining):
            node = registered["child"]


def audit(cases, records, prelude):
    by_case = {task["case"]: (row, task) for row, task in cases}
    wanted = {}
    for trials in records.values():
        for trial in trials:
            if trial["proof"]:
                row, task = by_case[trial["case"]]
                script = tuple([*task["target_prefix"], *trial["script"]])
                wanted[row["module"], script] = row

    def check(directory, env):
        native = []
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            for name, trials in records.items():
                for source in sorted({trial["source"] for trial in trials}):
                    candidates = [trial for trial in trials if trial["source"] == source]
                    row = max(candidates, key=lambda item: (len(item["trace"]), item["proof"]))
                    replay(repl, *by_case[row["case"]], row)
                    native.append({"arm": name, **row})
        return native, verify_independent(directory, env, (PREFIX_MODULE,),
                                          [(row, key[-1]) for key, row in wanted.items()])

    return native_scope(prelude, check)


def pick(scores, search, state, threshold=0.):
    moves = search.menu(state)
    action, value = max(zip(moves, scores, strict=True), key=lambda item: round(float(item[1]), 9))
    return STOP if float(value - scores[0]) < threshold else action


def patches(cases, numbers, packed, model, values):
    features, anchors, indices, mask = packed[:4]
    arms = defaultdict(list)
    witnessed = defaultdict(set)
    for proof, task in cases:
        witnessed[proof["source"]].add(task["sibling"]["actions"]["omega"]["status"])
    for proof, task in cases:
        current = task["sibling"]["actions"]["omega"]["status"]
        donors = sorted(witnessed[proof["source"]] - {current})
        if not donors:
            continue
        search = ResidualSearch(task)
        patched = State(2, probe=("omega", current))
        key = numbers[task["case"], patched] - 1
        ids = indices[key, mask[key]]
        original = features[ids]
        donor = original.clone()
        donor[:, 1, 16:21] = donor.new_tensor(disposition(donors[0]))
        cue = model.embed(original) + model.roles
        delta = (model.embed(donor) + model.roles)[:, 1] - cue[:, 1]
        dose = delta.norm()
        direction = F.normalize(delta.flatten(), dim=0)
        orthogonal = torch.zeros_like(direction)
        orthogonal[int(direction.abs().argmin())] = 1.
        sham = F.normalize(orthogonal - (orthogonal @ direction) * direction, dim=0).reshape_as(delta) * dose
        for name, edit in (("forced_probe", torch.zeros_like(delta)), ("probe_donor", delta), ("probe_sham", sham)):
            edited = cue.clone()
            edited[:, 1] += edit
            if name != "forced_probe" and not torch.isclose((edited - cue).norm(), dose, atol=1e-6):
                raise ValueError("the sibling intervention dose changed")
            changed = model.read(edited) + anchors[ids]

            def choose(state):
                if state == State(2):
                    return PROBE, "omega"
                if state == patched:
                    return pick(changed, search, state)
                index = numbers[task["case"], state] - 1
                return pick(values[index, mask[index]], search, state)

            trial = search.run(2, choose)
            arms[name].append({"case": task["case"], "module": proof["module"], "budget": 2,
                               "source": proof["source"], "dose": float(dose), "observation": current,
                               "donor": donors[0], **trial})
    return arms


def loaded_scores(packed, readers, checkpoint):
    features, anchors, indices, mask, _, _, optimum, _ = packed
    frozen = readers["frozen_encoder"]
    readout = {name: layout(anchors + flattened_scores(model, features), indices, mask)
               for name, model in readers.items() if name != "frozen_encoder"}
    for name, weights in checkpoint["calibration"].items():
        raw = calibration_features(features, anchors, name, frozen).double() @ weights
        readout[name] = layout(anchors + raw, indices, mask)
    readout["native_exact"] = optimum
    readout["mixed_source_prior"] = layout(anchors, indices, mask)
    reset = features.clone()
    reset[:, 1, 16:21] = 0.
    readout["probe_feedback_reset"] = layout(
        anchors + flattened_scores(readers["two_slot_reward"], reset), indices, mask)
    return readout


def trajectories(cases, numbers, mask, values, threshold):
    records = []
    for proof, task in cases:
        search = ResidualSearch(task)
        for budget in BUDGETS:
            def choose(state):
                index = numbers[task["case"], state] - 1
                return pick(values[index, mask[index]], search, state, threshold)

            trial = search.run(budget, choose)
            records.append({"case": task["case"], "module": proof["module"], "budget": budget,
                            "source": proof["source"], **trial})
    return records


def summarize(rows):
    roots = defaultdict(list)
    for row in rows:
        roots[row["module"]].append(row["utility"])
    return {"proofs": sum(row["proof"] for row in rows), "cases": len(rows),
            "calls": sum(len(row["trace"]) for row in rows),
            "probes": sum(any(event["role"] == PROBE for event in row["trace"]) for row in rows),
            "root_utility": fsum(fsum(v) / len(v) for v in roots.values()) / len(roots)}


def source_thresholds(args, prior, readers, checkpoint):
    cases, _, _ = inputs(args, "train")
    packed, numbers = pack(cases, prior, opaque=args.opaque_binders)
    mask = packed[3]
    readout = loaded_scores(packed, readers, checkpoint)
    results = {}
    for name, values in readout.items():
        if name == "native_exact":
            continue
        grid = {threshold: summarize(trajectories(cases, numbers, mask, values, threshold))
                for threshold in STOP_GRID}
        chosen = max(grid, key=lambda threshold: round(grid[threshold]["root_utility"], 12))
        results[name] = {"threshold": chosen, "grid": grid}
    return results


@torch.no_grad()
def evaluate(args, tree):
    cases, prelude, pins = inputs(args, "dev")
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["pins"] != pins
            or checkpoint["calibration_sha256"] != tensor_hash(checkpoint["calibration"])):
        raise ValueError("the residual fit or calibration changed")
    readers = models()
    for name, model in readers.items():
        weights = checkpoint["weights"][name]
        if tensor_hash(weights) != checkpoint["weight_sha256"][name]:
            raise ValueError("frozen residual model weights changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    stopping = source_thresholds(args, prior, readers, checkpoint) if args.stop_match else {}
    packed, numbers = pack(cases, prior, opaque=args.opaque_binders)
    mask = packed[3]
    readout = loaded_scores(packed, readers, checkpoint)
    records = {}
    for name, values in readout.items():
        records[name] = trajectories(cases, numbers, mask, values, 0.)
        if name in stopping:
            records[name + "_source_stop"] = trajectories(cases, numbers, mask, values, stopping[name]["threshold"])
    records.update(patches(cases, numbers, packed, readers["two_slot_reward"], readout["two_slot_reward"]))
    summaries = {name: summarize(rows) for name, rows in records.items()}
    for name, row in summaries.items():
        print(name, row, flush=True)
    replayed, independent = audit(cases, records, prelude)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit), "floor": FLOOR,
                                 "summary": summaries, "results": records, "replays": replayed, "source_stop": stopping,
                                 "independent": independent, "opaque_binders": args.opaque_binders,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "mathcan", "warm", "prior", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--stop-match", action="store_true", help="fit each stopping margin on source rows only")
    parser.add_argument("--opaque-binders", action="store_true", help="hide original binder names from the frozen prior")
    args = parser.parse_args()
    if args.phase == "fit" and (args.stop_match or args.opaque_binders):
        raise ValueError("stopping and binder interventions require a frozen fit")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
