"""Fit, verify and perturb a proof policy that retains diagnostic observations."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from lcgrt.config import LEAN_VERSION, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.diagnostic_policy import HISTORIES, DiagnosticPolicy, evidence
from ppt_gym.live_lean import public_state
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rollback_diagnostics import DIAGNOSTIC_COST, SCHEMA as WITNESS_SCHEMA

from configs.check_ppt_mathcan_native_costs import NAMESPACE, PREFIX_MODULE, declaration, invoke_lean
from configs.freeze_ppt_rollback_diagnostics import OPERATIONS, policy_bound, probe
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_slot_continuations import read_continuation_fit, restore_policy
from configs.run_ppt_two_slot import CPU_LIMIT


SCHEMA = "ppt.rollback-diagnostic-policy.v1"
SEED = 831
STEPS = 160
TEMPERATURE = .55
LEGAL = ((1, 1, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 0, 0), (1, 1, 0, 0))


def load_witness(path):
    report = json.loads(gzip.decompress(path.read_bytes()))
    if report["schema"] != WITNESS_SCHEMA or report["lean"] != LEAN_VERSION:
        raise ValueError("expected the pinned native rollback witness")
    groups = {split: [row for row in report["rows"] if row["split"] == split] for split in ("train", "dev")}
    train = {row["payload_root"] for row in groups["train"]}
    dev = {row["payload_root"] for row in groups["dev"]}
    if len(train) != 15 or len(dev) != 18 or train & dev:
        raise ValueError("the original payload split is incomplete")
    return report, groups


def warm(checkpoint):
    fit = read_continuation_fit(checkpoint, LEAN_VERSION)
    if not fit.get("mixed_source_counts"):
        raise ValueError("the diagnostic planner requires a mixed-source warm start")
    hybrid = restore_policy(fit, "live_rl", fit["weights"]["live_rl"]).eval()
    trm = restore_policy(fit, "trm_workspace", fit["weights"]["trm_workspace"]).eval()
    return hybrid, trm, fit["mixed_source_counts"]


def features(rows, encoder):
    """Construct features from the structured goal and the probe results only."""
    legal = torch.tensor(LEGAL, dtype=torch.bool)
    contexts, histories = [], []
    embeddings = {}
    with torch.no_grad():
        for row in rows:
            public = row["public"]["goals"]
            raw = public[0]["target"]["raw"]
            known = [name for name, (expression, _) in OPERATIONS.items() if raw.startswith(f"∃ n, n = {expression} ∧ ")]
            if known != [row["operation"]]:
                raise ValueError("the proof observation did not expose the registered operator")
            cluster = row["cluster"]
            if cluster not in embeddings:
                embeddings[cluster] = F.normalize(encoder.encode(public).mean(0), dim=0)
            operation = F.one_hot(torch.tensor(tuple(OPERATIONS).index(known[0])), len(OPERATIONS)).float()
            budgets = torch.tensor([[(row["proof_budget"] - len(history)) / 16 for history in HISTORIES]]).T
            root = torch.cat((operation, embeddings[cluster] * .2))
            contexts.append(torch.cat((root.expand(len(HISTORIES), -1), budgets, legal[:, 2:].float()), dim=-1))
            histories.append([[(i, row["probe_receipts"][i]["observed_value"]) for i in history]
                              + [(-1, 0)] * (2 - len(history)) for history in HISTORIES])
    events = torch.tensor(histories)
    context = torch.stack(contexts)
    raw = torch.cat((context, evidence(events)[0].sum(-3).flatten(-2)), dim=-1)
    return context, events, raw


def teacher(rows):
    """Compute return targets conditional on each training observation."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["cluster"]].append(row)
    targets = torch.zeros(len(rows), len(HISTORIES), 4)
    for i, row in enumerate(rows):
        layout = groups[row["cluster"]]
        for j, history in enumerate(HISTORIES):
            posterior = [other for other in layout if all(other["probe_receipts"][probe_index]["observed_value"]
                         == row["probe_receipts"][probe_index]["observed_value"] for probe_index in history)]
            for value in (0, 1):
                targets[i, j, value] = sum(other["witness_trials"][value]["passed"] for other in posterior) / len(posterior)
            for index in (0, 1):
                if index in history:
                    continue
                branches = defaultdict(list)
                for other in posterior:
                    branches[other["probe_receipts"][index]["observed_value"]].append(other)
                rest = tuple(k for k in (0, 1) if k not in (*history, index))
                optimal = [policy_bound(branch, len(rest), rest) for branch in branches.values()]
                targets[i, j, index + 2] = (sum(p - DIAGNOSTIC_COST * c for p, c, _ in optimal)
                                           / len(posterior) - DIAGNOSTIC_COST)
    return targets


def policy_loss(logits, rewards):
    probability = (logits / TEMPERATURE).softmax(-1)
    stop = (probability[..., :2] * rewards[:, None, :]).sum(-1)
    left = stop[:, 1] + probability[:, 1, 3] * (stop[:, 3] - DIAGNOSTIC_COST)
    right = stop[:, 2] + probability[:, 2, 2] * (stop[:, 4] - DIAGNOSTIC_COST)
    initial = stop[:, 0] + probability[:, 0, 2] * (left - DIAGNOSTIC_COST)
    initial = initial + probability[:, 0, 3] * (right - DIAGNOSTIC_COST)
    # Retain a training signal at diagnostic continuations that the current
    # policy rarely visits.
    return -(initial.mean() + .20 * torch.stack((left, right, stop[:, 3], stop[:, 4])).mean())


def train(model, x, errors, rewards, targets, objective):
    legal = torch.tensor(LEGAL, dtype=torch.bool).expand(len(x), -1, -1)
    masked_targets = targets.masked_fill(~legal, -torch.inf)
    best = (masked_targets >= masked_targets.amax(-1, keepdim=True) - 1e-7).float()
    teacher_probs = best / best.sum(-1, keepdim=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    for _ in range(STEPS):
        full = model(x, errors)
        logits = full.masked_fill(~legal, -torch.inf)
        if objective == "rl":
            loss = policy_loss(logits, rewards)
        elif objective == "imitation":
            log_probs = logits.log_softmax(-1).masked_fill(~legal, 0.)
            loss = -(teacher_probs * log_probs).sum(-1).mean()
        else:
            loss = F.mse_loss(full[legal], targets[legal])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def fit(args):
    _, groups = load_witness(args.witness)
    rows = groups["train"]
    hybrid, trm, counts = warm(args.warm)
    x, events, raw = features(rows, hybrid)
    targets = teacher(rows)
    rewards = torch.tensor([[float(path["passed"]) for path in row["witness_trials"]] for row in rows])
    base = DiagnosticPolicy(hybrid, x.shape[-1])
    models = {"rl": deepcopy(base), "imitation": deepcopy(base), "return_regression": deepcopy(base),
              "trm": DiagnosticPolicy(trm, x.shape[-1], trm=True), "warm": base}
    for name, objective in (("rl", "rl"), ("imitation", "imitation"), ("return_regression", "return"), ("trm", "rl")):
        train(models[name], x, events, rewards, targets, objective)
        print(name, STEPS, flush=True)
    roots = {root: i for i, root in enumerate(sorted({row["payload_root"] for row in rows}))}
    ids = torch.tensor([roots[row["payload_root"]] for row in rows])[:, None].expand(-1, len(HISTORIES))
    legal = torch.tensor(LEGAL, dtype=torch.bool).expand(len(rows), -1, -1)
    with torch.no_grad():
        _, frozen = base.read(base.memory(x, events), x)
        readouts = {name: [ridge(values[legal[..., i]], targets[..., i][legal[..., i]], ids[legal[..., i]])
                           for i in range(4)] for name, values in (("calibrated", frozen), ("raw_calibrated", raw))}
    weights = {name: model.state_dict() for name, model in models.items()}
    with args.output.open("xb") as out:
        torch.save({"schema": SCHEMA, "lean": LEAN_VERSION, "seed": SEED, "steps": STEPS,
                    "witness_sha256": sha256_file(args.witness), "warm_sha256": sha256_file(args.warm),
                    "mixed_sources": counts, "weights": weights, "frozen": readouts,
                    "weight_hashes": {name: tensor_hash(state) for name, state in weights.items()}}, out)


def loaded(args, rows):
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["lean"] != LEAN_VERSION
            or checkpoint["witness_sha256"] != sha256_file(args.witness)
            or checkpoint["warm_sha256"] != sha256_file(args.warm)):
        raise ValueError("the fit requires the matching native witness and warm checkpoint")
    hybrid, trm, _ = warm(args.warm)
    x, events, raw = features(rows, hybrid)
    models = {}
    for arm, weights in checkpoint["weights"].items():
        model = DiagnosticPolicy(trm if arm == "trm" else hybrid, x.shape[-1], trm=arm == "trm")
        if tensor_hash(weights) != checkpoint["weight_hashes"][arm]:
            raise ValueError("a proof policy's registered weight digest changed")
        model.load_state_dict(weights)
        models[arm] = model.eval()
    return models, checkpoint["frozen"], x, events, raw


def open_certificate(row):
    certificate = row["certificate_source"]
    if digest(certificate) != row["certificate_source_digest"]:
        raise ValueError("the native certificate changed")
    return certificate.removeprefix(f"theorem {row['tid']} : ").split(" := by\n", 1)[0]


def native_choice(repl, tree, row, scores):
    start = tree.cpu_seconds()
    repl.reset()
    root = repl.init(open_certificate(row))
    if public_state(repl, root) != row["public"]:
        raise ValueError("Lean changed the visible starting proof state")
    history, diagnostics = (), []
    while True:
        index = HISTORIES.index(history)
        mask = torch.tensor(LEGAL[index], dtype=torch.bool)
        action = int(scores[index].masked_fill(~mask, -torch.inf).argmax())
        if action < 2:
            break
        root, observed = probe(repl, root, action - 2)
        if observed != row["probe_receipts"][action - 2]:
            raise ValueError("the native diagnostic diverged from its verified receipt")
        diagnostics.append(observed)
        history = (*history, action - 2)

    receipt = row["witness_trials"][action]
    accepted = row["witness_trials"][row["winning_witness"]]["actions"]
    script = [f"refine ⟨{action}, ?_⟩", *accepted[1:receipt["checks"]]]
    state, passed = root, False
    for i, tactic in enumerate(script, 1):
        result = repl.run(state["sid"], tactic)
        if not result["ok"]:
            if i != receipt["checks"] or receipt["passed"]:
                raise ValueError("a chosen witness failed at a different native decision")
            break
        state = result["state"]
    else:
        kernel = repl.kernel(state["sid"])
        passed = not state["goals"] and kernel["ok"] and sorted(kernel["axioms"]) == receipt.get("axioms")
    checks = len(history) + receipt["checks"]
    if passed != receipt["passed"] or checks > row["proof_budget"] or tree.cpu_seconds() - start > CPU_LIMIT:
        raise ValueError("the chosen path violated its certificate or check budget")
    return {"tid": row["tid"], "cluster": row["cluster"], "payload_root": row["payload_root"],
            "source": row["source"], "operation": row["operation"], "witness": action,
            "diagnostics": diagnostics, "success": passed, "checks": checks,
            "reward": float(passed) - DIAGNOSTIC_COST * len(history)}


def write_receipt(path, report):
    with path.open("xb") as out:
        out.write(gzip.compress((canonical(report) + "\n").encode(), mtime=0))


def evaluate(args, tree):
    report, groups = load_witness(args.witness)
    rows = groups["dev"]
    models, readouts, x, events, raw = loaded(args, rows)
    with torch.no_grad():
        all_scores = {name: model(x, events) for name, model in models.items()}
        all_scores["slot_reset"] = models["rl"](x, events, reset=True)
        _, frozen = models["warm"].read(models["warm"].memory(x, events), x)
        for name, vals in (("calibrated", frozen), ("raw_calibrated", raw)):
            all_scores[name] = torch.stack([apply_ridge(vals, head) for head in readouts[name]], dim=-1)
    results = {}
    before, _, _ = declaration(args.mathcan_source.read_text(encoding="utf-8"), "full_horizon_recovers")
    prelude = before + f"end {NAMESPACE}\n"
    if digest(prelude) != report["prelude_sha256"] or sha256_file(args.mathcan_source) != report["source_hashes"]["mathcan_source"]:
        raise ValueError("the original MathCan source changed")
    with TemporaryDirectory() as scratch:
        directory = Path(scratch)
        source = directory / f"{PREFIX_MODULE}.lean"
        source.write_text(prelude, encoding="utf-8")
        env = lean_env()
        invoke_lean(directory, env, "-o", f"{PREFIX_MODULE}.olean", source.name)
        env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
        with LeanRepl(imports=(PREFIX_MODULE,), cwd=directory, env=env) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("native evaluation requires the pinned Lean verifier")
            for arm, scores in all_scores.items():
                records = [native_choice(repl, tree, row, values) for row, values in zip(rows, scores, strict=True)]
                complete = Counter(row["cluster"] for row in records if row["success"])
                count = {"proofs": complete.total(), "layouts": sum(n == 4 for n in complete.values()),
                         "checks": sum(len(row["diagnostics"]) for row in records)}
                results[arm] = {**count, "rows": records}
                print(arm, count, flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "fit_sha256": sha256_file(args.fit),
                                "witness_sha256": sha256_file(args.witness), "results": results,
                                "memory": tree.stats()})


def bpd(args):
    _, groups = load_witness(args.witness)
    rows = groups["dev"]
    models, frozen, x, events, _ = loaded(args, rows)
    indices = {(row["cluster"], tuple(row["world"])): i for i, row in enumerate(rows)}
    donors = torch.tensor([indices[row["cluster"], (1 - row["world"][0], row["world"][1])] for row in rows])
    results = {}
    # Intervene after the fixed left-then-right schedule at the commit step.
    with torch.no_grad():
        for arm in ("rl", "calibrated", "trm"):
            model = models["warm"] if arm == "calibrated" else models[arm]
            natural = model.memory(x[:, 3], events[:, 3])
            delta = natural[donors, 0] - natural[:, 0]
            variants = {"identity": natural.clone(), "opposite_left": natural.clone(), "norm_sham": natural.clone()}
            variants["opposite_left"][:, 0] = natural[donors, 0]
            variants["norm_sham"][:, 0] += delta.roll(3, dims=-1)
            torch.testing.assert_close((variants["norm_sham"][:, 0] - natural[:, 0]).norm(dim=-1), delta.norm(dim=-1))
            baseline, features_at_read = model.read(natural, x[:, 3])
            if arm == "calibrated":
                baseline = torch.stack([apply_ridge(features_at_read, head) for head in frozen[arm]], dim=-1)
            patched = {}
            for name, state in variants.items():
                logits, features_at_read = model.read(state, x[:, 3])
                if arm == "calibrated":
                    logits = torch.stack([apply_ridge(features_at_read, head) for head in frozen[arm]], dim=-1)
                patched[name] = logits[:, :2]
            torch.testing.assert_close(patched["identity"], baseline[:, :2], atol=0., rtol=0.)
            results[arm] = [{"tid": row["tid"], "cluster": row["cluster"], "source": row["source"],
                             "operation": row["operation"], "winner": row["winning_witness"],
                             "donor_winner": rows[int(donors[i])]["winning_witness"],
                             "dose": float(delta[i].norm()),
                             "scores": {name: logits[i].tolist() for name, logits in patched.items()}}
                            for i, row in enumerate(rows)]
    write_receipt(args.output, {"schema": "ppt.rollback-diagnostic-bpd.v1", "fit_sha256": sha256_file(args.fit),
                                "witness_sha256": sha256_file(args.witness), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    train_parser = commands.add_parser("fit")
    score = commands.add_parser("evaluate")
    probe_parser = commands.add_parser("bpd")
    for command in (train_parser, score, probe_parser):
        command.add_argument("witness", type=Path)
        command.add_argument("warm", type=Path)
        command.add_argument("output", type=Path)
    for command in (score, probe_parser):
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
        else:
            bpd(args)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
