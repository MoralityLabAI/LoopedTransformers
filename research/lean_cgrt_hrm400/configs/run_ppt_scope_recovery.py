"""Fit scope-aware reward policies and replay their native witness choices."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from math import fsum
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.freeze_ppt_rollback_length_transfer import diagnostic, native_commit
from configs.freeze_ppt_scope_recovery import ALIASES, PREFIX_MODULE, ROOTS, SCHEMA as NATIVE_SCHEMA, native_scope, prefix_text
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import public_state
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.scope_policy import ScopeSlots, ScopeTRM, calibrated_features, inputs
from ppt_gym.scope_recovery import BUDGETS, CHECK_INDEX, EARLY, N_ACTIONS, OPEN, RESOLVE, TENTATIVE
from ppt_gym.scope_recovery import Layout, State, legal, rollout


SCHEMA = "ppt.scope-recovery.v1"
SEED = 1171
UPDATES = 180
TEMPERATURE = .10
FLOOR = .025
CALIBRATORS = ("raw_history", "aligned_history", "frozen_slots")


def checked_bank(args, split):
    bank = read_receipt(args.bank)
    prelude = prefix_text(args.mathcan)
    if (bank["schema"] != NATIVE_SCHEMA or bank["prelude"] != digest(prelude)
            or bank["lean"] != sha256_file(LEAN_EXE) or bank["repl"] != sha256_file(REPL_EXE)):
        raise ValueError("the scope bank has different Lean or import pins")
    counts = {name: {row["payload_root"] for row in bank["rows"] if row["split"] == name} for name in ROOTS}
    if any(len(counts[name]) != ROOTS[name] for name in ROOTS) or counts["train"] & counts["dev"]:
        raise ValueError("scope source roots are missing or mixed between partitions")
    return [row for row in bank["rows"] if row["split"] == split], prelude, sha256_file(args.bank)


def signatures(rows):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster"]].append(row)
    representatives = {}
    for worlds in clusters.values():
        layout = Layout(worlds)
        if any(row["public"] != worlds[0]["public"] or "standalone_axioms" not in row for row in worlds):
            raise ValueError("a native scope observation or certificate changed")
        reference = representatives.setdefault(layout.edited, layout)
        if any([path["passed"] for path in row["witness_trials"]]
               != [path["passed"] for path in reference.worlds[bits]["witness_trials"]]
               for bits, row in layout.worlds.items()):
            raise ValueError("same-address layouts have inconsistent native outcomes")
    if set(representatives) != {0, 1}:
        raise ValueError("both edited addresses must appear")
    return representatives


def pack(layouts):
    rows, index = [], {}

    def visit(edited, state):
        if state is None:
            return 0
        key = edited, state
        if key in index:
            return index[key]
        layout = layouts[edited]
        immediate, edges, probabilities = [0.] * N_ACTIONS, [[0, 0] for _ in range(N_ACTIONS)], [
            [0., 0.] for _ in range(N_ACTIONS)]
        for action in legal(state):
            immediate[action], candidates = layout.transitions(state, action)
            for branch, (probability, child) in enumerate(candidates):
                edges[action][branch] = visit(edited, child)
                probabilities[action][branch] = probability
        height = 1 + max((rows[child - 1][-1] for children in edges for child in children if child), default=0)
        values = layout.optimal(state)
        context, history = inputs(state, edited)
        rows.append((context, history, [action in values for action in range(N_ACTIONS)],
                     immediate, edges, probabilities, [values.get(action, 0.) for action in range(N_ACTIONS)], height))
        index[key] = len(rows)
        return len(rows)

    for edited in layouts:
        for budget in BUDGETS:
            visit(edited, State(budget))
    packed = tuple(torch.tensor(column) for column in zip(*rows, strict=True))
    return packed, index


@torch.no_grad()
def improve(logits, immediate, edges, probabilities, height):
    values = logits.new_zeros(len(logits) + 1)
    q = torch.zeros_like(logits)
    policy = (logits / TEMPERATURE).softmax(-1)
    for depth in range(1, int(height.max()) + 1):
        rows = (height == depth).nonzero().flatten()
        q[rows] = immediate[rows] + (probabilities[rows] * values[edges[rows]]).sum(-1)
        values[rows + 1] = (policy[rows] * q[rows]).sum(-1)
    return q


def models():
    scope = ScopeSlots()
    return {"scope_reward": scope, "scope_imitation": deepcopy(scope),
            "trm_reward": ScopeTRM(), "calibration_encoder": deepcopy(scope)}


def train(model, name, packed):
    context, history, mask, immediate, edges, probabilities, optimum, height = packed
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(UPDATES):
        logits = model(context, history).masked_fill(~mask, -torch.inf)
        q = optimum if name == "scope_imitation" else improve(logits, immediate, edges, probabilities, height)
        target = (q.masked_fill(~mask, -torch.inf) / TEMPERATURE).softmax(-1)
        log_prob = logits.log_softmax(-1).masked_fill(~mask, 0.)
        loss = -(target * log_prob).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def features(name, encoder, context, history):
    if name == "frozen_slots":
        activation = encoder.read(context, *encoder.memory(context, history))
        return torch.cat((torch.ones_like(context[:, :1]), context, activation), dim=1)
    return calibrated_features(context, history, aligned=name == "aligned_history")


def calibrate(encoder, context, history, support, optimum):
    fitted = {}
    with torch.no_grad():
        for name in CALIBRATORS:
            x = features(name, encoder, context, history).double()
            columns = []
            for action in range(N_ACTIONS):
                allowed = support[:, action]
                design = x[allowed]
                system = design.T @ design + .01 * torch.eye(x.shape[1], dtype=x.dtype)
                columns.append(torch.linalg.solve(system, design.T @ optimum[allowed, action].double()))
            fitted[name] = torch.stack(columns, dim=1)
    return fitted


def fit(args, tree):
    rows, _, bank = checked_bank(args, "train")
    packed, index = pack(signatures(rows))
    readers = models()
    initial = {name: tensor_hash(model.state_dict()) for name, model in readers.items()}
    if initial["scope_reward"] != initial["scope_imitation"]:
        raise ValueError("the two matched scope objectives changed their initialization")
    context, history, mask, _, _, _, optimum, _ = packed
    calibration = calibrate(readers["calibration_encoder"].eval(), context, history, mask, optimum)
    for name in ("scope_reward", "scope_imitation", "trm_reward"):
        train(readers[name], name, packed)
        print("fitted", name, flush=True)
    weights = {name: model.state_dict() for name, model in readers.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "bank": bank, "weights": weights, "initial": initial,
                    "seal": {name: tensor_hash(values) for name, values in weights.items()},
                    "calibration": calibration, "calibration_sha256": tensor_hash(calibration),
                    "source_states": len(index), "memory": tree.stats()}, handle)
    print("sealed scope fit", sha256_file(args.fit), flush=True)


def selected(logits, state):
    return max(legal(state), key=lambda action: round(float(logits[action]), 9))


def bit_patch(row, model):
    left, right, tentative, keep = (quote["value"] for quote in row["observations"])
    edited = row["edited"]
    if (left, right)[edited] == tentative:
        return None
    state = State(0, 2, (left, right), tentative, keep)
    context, history = inputs(state, edited)
    context, history = torch.tensor([context]).float(), torch.tensor([history]).float()
    _, donor = inputs(replace(state, keep=1 - keep), edited)
    active, pending = model.memory(context, history)
    donor_active, _ = model.memory(context, torch.tensor([donor]).float())
    vector = donor_active[0, edited] - active[0, edited]
    dose = vector.norm()
    unit = F.normalize(vector, dim=0)
    noise = torch.zeros_like(unit)
    index = int(unit.abs().argmin())
    noise[index] = 1. if int(row["cluster"][-1], 16) % 2 else -1.
    sham = F.normalize(noise - (noise @ unit) * unit, dim=0) * dose
    choices = {}
    for name, displacement in (("natural", torch.zeros_like(vector)), ("donor", vector), ("sham", sham)):
        replacement = active.clone()
        replacement[0, edited] += displacement
        if name != "natural" and not torch.isclose((replacement - active).norm(), dose, atol=1e-6):
            raise ValueError("the scope donor and sham used different physical doses")
        action = selected(model.head(model.read(context, replacement, pending))[0], state)
        choices[name] = {"witness": action, "proved": row["witness_trials"][action]["passed"],
                         "donor_aligned": action == 1 - row["winning_witness"]}
    return {"tid": row["tid"], "root": row["payload_root"], "kept": bool(keep), "dose": float(dose), **choices}


def replay(repl, row, result):
    source = row["certificate_source"]
    if digest(source) != row["certificate_source_digest"]:
        raise ValueError("the native scope proof changed")
    statement = source.removeprefix(f"theorem {row['tid']} : ").split(" := by\n", 1)[0]
    repl.reset()
    state = repl.init(statement)
    if public_state(repl, state) != row["public"]:
        raise ValueError("the native scope proof observation changed")
    paid = 0
    for event in result["trace"]:
        action = event["action"]
        if action == OPEN:
            continue
        index = CHECK_INDEX[action]
        state, observed = diagnostic(repl, state, ALIASES[index])
        if observed != row["observations"][index] or observed["value"] != event["bit"]:
            raise ValueError("a paid scoped observation changed on native replay")
        paid += 1
    chosen = row["witness_trials"][result["witness"]]
    winning = row["witness_trials"][row["winning_witness"]]
    path = native_commit(repl, state, winning["actions"][2:-1], result["witness"], winning["axioms"])
    if digest(path) != digest(chosen) or paid != result["diagnostics"] or paid + path["checks"] != result["checks"]:
        raise ValueError("the native scope trajectory changed outcome or cost")


def audit(prelude, rows, results, lesions):
    by_tid = {row["tid"]: row for row in rows}

    def verify(directory, env):
        selected_rows = []
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            for name, records in results.items():
                for kept in (False, True):
                    eligible = [item for item in records if item["kept"] == kept]
                    chosen = max(eligible, key=lambda item: (item["diagnostics"], item["proved"]))
                    replay(repl, by_tid[chosen["tid"]], chosen)
                    selected_rows.append({"arm": name, **chosen})
            for address in (0, 1):
                for kept in (False, True):
                    patch = next(item for item in lesions
                                 if item["kept"] == kept and by_tid[item["tid"]]["edited"] == address)
                    row = by_tid[patch["tid"]]
                    for arm in ("natural", "donor", "sham"):
                        actions = iter((*EARLY, OPEN, TENTATIVE, RESOLVE, patch[arm]["witness"]))
                        trial = rollout(row, 3, lambda _: next(actions))
                        replay(repl, row, trial)
                        selected_rows.append({"arm": "bpd_" + arm, "tid": row["tid"], **trial})
        return selected_rows

    return native_scope(prelude, verify)


@torch.no_grad()
def evaluate(args, tree):
    rows, prelude, bank = checked_bank(args, "dev")
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["bank"] != bank
            or checkpoint["calibration_sha256"] != tensor_hash(checkpoint["calibration"])):
        raise ValueError("the fitted scope state or calibration changed")
    readers = models()
    for name, model in readers.items():
        weights = checkpoint["weights"][name]
        if tensor_hash(weights) != checkpoint["seal"][name]:
            raise ValueError("frozen scope policy weights changed")
        model.load_state_dict(weights)
        model.eval()
    (context, history, _, _, _, _, exact, _), index = pack(signatures(rows))
    encoder = readers.pop("calibration_encoder")
    all_scores = {name: model(context, history) for name, model in readers.items()}
    reset = history.clone()
    reset[context[:, 0] == 0, :2, :2] = 0.
    all_scores["checkpoint_reset"] = readers["scope_reward"](context, reset)
    all_scores.update({name: features(name, encoder, context, history).double() @ weight
                       for name, weight in checkpoint["calibration"].items()})
    all_scores["exact"] = exact
    results = defaultdict(list)
    for row in rows:
        for budget in BUDGETS:
            for name, scores in all_scores.items():
                trial = rollout(row, budget, lambda state: selected(scores[index[row["edited"], state] - 1], state))
                results[name].append({"tid": row["tid"], "root": row["payload_root"], "budget": budget,
                                      "kept": bool(row["observations"][3]["value"]), **trial})
    summaries = {name: {"proofs": sum(row["proved"] for row in outcomes), "rows": len(outcomes),
                        "diagnostics": sum(row["diagnostics"] for row in outcomes),
                        "utility": fsum(row["utility"] for row in outcomes) / len(outcomes)}
                 for name, outcomes in results.items()}
    lesions = [patch for row in rows if (patch := bit_patch(row, readers["scope_reward"])) is not None]
    for name, report in summaries.items():
        print(name, report, flush=True)
    print("scope BPD", len(lesions), Counter((patch["natural"]["proved"], patch["donor"]["proved"],
                                            patch["sham"]["proved"]) for patch in lesions), flush=True)
    verified = audit(prelude, rows, results, lesions)
    write_receipt(args.results, {"schema": SCHEMA, "bank": bank, "fit_sha256": sha256_file(args.fit),
                                 "floor": FLOOR, "summary": summaries, "results": results, "bpd": lesions,
                                 "live_replays": verified, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "mathcan", "fit", "results"):
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
