"""Train COG-03 and preserve every charged evaluation attempt.

Run from the repository root with Python and CPU PyTorch installed:
  python -m research.cog03_ledger_router.run --corpus tasks.jsonl --output results
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import torch
import torch.nn.functional as F

from .environment import (SPLITS, available_actions, canonical, check_step,
                          goal_ancestors, load_tasks, public_observation,
                          sha256, useful_actions)
from .model import FEATURES, LedgerRouter, encode_observation


ARMS = ("looped", "one_pass", "reset_history", "reset_inner", "shuffled_history",
        "random", "ready_first", "goal_ancestry")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def gz_rows(path, rows):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((canonical(row) + "\n").encode())


def history_event(action, receipt):
    return {"action": action, **{k: receipt[k] for k in ("accepted", "code", "new_fact")}}


def train_model(tasks, *, seed, depth, updates, budget):
    """Fit on train roots. Truncate gradients at previous verifier decisions."""
    if any(task["split"] != "train" for task in tasks):
        raise ValueError("model updates require train roots")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = LedgerRouter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    optimizer_rows = []
    labelled_calls = transition_calls = 0
    started = time.monotonic()
    model.train()
    for update in range(updates):
        task = tasks[rng.randrange(len(tasks))]
        world = task["world"]
        known, history, previous = set(world["facts"]), [], None
        losses = []
        for step in range(budget):
            useful = useful_actions(world, known)
            if world["goal"] in known or not useful:
                break
            observation = public_observation(task, known, history)
            features, attention_mask = encode_observation(observation, budget - step, budget)
            out = model(features, attention_mask,
                        previous if depth > 1 else None, depth=depth)
            labels = [check_step(world, known, i)["accepted"] for i in range(len(world["rules"]))]
            labelled_calls += len(labels)
            legal_targets = torch.tensor(labels, dtype=torch.float32)
            # Several useful next actions can be correct in the same state.
            ranking = torch.logsumexp(out["logits"], 0) - torch.logsumexp(out["logits"][useful], 0)
            loss = ranking + 0.25 * F.binary_cross_entropy_with_logits(out["acceptance_logits"], legal_targets)
            losses.append(loss)
            previous = out["state"].detach()
            # Wrong attempts expose real checker feedback during training.
            action = rng.randrange(len(world["rules"])) if rng.random() < 0.20 else rng.choice(useful)
            receipt = check_step(world, known, action)
            transition_calls += 1
            history.append(history_event(action, receipt))
            known = set(receipt["next_known"])
        if not losses:
            raise ValueError("training task supplied no nontrivial action")
        total = torch.stack(losses).mean()
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite train loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_rows.append({"update": update, "root_id": task["root_id"],
                               "loss": float(total.detach()), "teacher_steps": len(losses)})
    model.eval()
    receipt = {"seed": seed, "depth": depth, "optimizer_updates": updates,
               "parameters": model.parameter_count(), "train_roots": len(tasks),
               "train_roots_observed": len({r["root_id"] for r in optimizer_rows}),
               "loss_first20": sum(r["loss"] for r in optimizer_rows[:20]) / min(20, updates),
               "loss_last20": sum(r["loss"] for r in optimizer_rows[-20:]) / min(20, updates),
               "label_collection_verifier_calls": labelled_calls,
               "teacher_transition_verifier_calls": transition_calls,
               "training_seconds": time.monotonic() - started,
               "backpropagation": "full inner recurrence; detach between verifier decisions",
               "checkpoint_selection": "final fixed update; no calibration or test selection"}
    return model, receipt, optimizer_rows


def rollout(task, *, arm, seed, model, budget):
    if arm not in ARMS or budget < 1:
        raise ValueError("unregistered arm or budget")
    world = task["world"]
    known = set(world["facts"])
    needed = goal_ancestors(world)  # audit and named symbolic control only
    history, failed, previous = [], set(), None
    source = int(hashlib.sha256((task["root_id"] + str(seed)).encode()).hexdigest()[:16], 16)
    rng = random.Random(source)
    rows = []
    started = time.monotonic()
    inner_calls = 0
    for step in range(budget):
        if world["goal"] in known:
            break
        candidates = available_actions(world, known, failed)
        if not candidates:
            break
        observation = public_observation(task, known, history)
        passes = 0
        if arm == "random":
            action = rng.choice(candidates)
        elif arm == "ready_first":
            ready = [i for i in candidates if set(world["rules"][i]["premises"]) <= known]
            action = (ready or candidates)[0]
        elif arm == "goal_ancestry":
            choices = useful_actions(world, known)
            action = (choices or candidates)[0]
        else:
            if model is None:
                raise ValueError("neural arm requires a checkpoint")
            passes = 1 if arm == "one_pass" else 4
            carry = None if arm in ("one_pass", "reset_history") else previous
            features, attention_mask = encode_observation(observation, budget - step, budget)
            with torch.no_grad():
                out = model(features, attention_mask, carry, depth=passes,
                            reset_inner=arm == "reset_inner",
                            shuffled_history=arm == "shuffled_history")
            previous = out["state"].detach()
            scores = out["logits"].tolist()
            action = max(candidates, key=lambda i: (scores[i], -i))
            inner_calls += passes
        receipt = check_step(world, known, action)
        new = receipt["new_fact"]
        rows.append({
            "schema_version": "cog03.attempt.v1", "arm": arm, "seed": seed,
            "root_id": task["root_id"], "task_id": task["task_id"],
            "split": task["split"], "family": task["family"], "step": step,
            "budget": budget, "observation": observation, "action": action,
            "receipt": receipt, "verifier_calls": 1,
            "new_unique_fact": new is not None,
            "goal_relevant_fact": new is not None and new in needed,
            "solved": receipt["solved"],
            "policy_compute": {"inner_passes": passes,
                               "carry_history": arm in ("looped", "reset_inner", "shuffled_history")},
        })
        history.append(history_event(action, receipt))
        if new is not None:
            failed.clear()
        elif not receipt["accepted"]:
            failed.add(action)
        known = set(receipt["next_known"])
    counts = Counter(r["receipt"]["code"] for r in rows)
    outcome = {
        "schema_version": "cog03.episode.v1", "root_id": task["root_id"],
        "task_id": task["task_id"], "task_digest": task["task_digest"],
        "family": task["family"], "split": task["split"], "arm": arm, "seed": seed,
        "budget": budget, "verifier_calls": len(rows), "solved": world["goal"] in known,
        "accepted": sum(r["receipt"]["accepted"] for r in rows),
        "rejected": sum(not r["receipt"]["accepted"] for r in rows),
        "duplicates": counts["duplicate_lemma"],
        "unique_verified_facts": len(known - set(world["facts"])),
        "goal_relevant_verified_facts": len((known - set(world["facts"])) & needed),
        "final_known": sorted(known), "inner_cell_calls": inner_calls,
        "elapsed_seconds": time.monotonic() - started,
    }
    return rows, outcome


def summarize(episodes):
    grouped = defaultdict(list)
    for row in episodes:
        grouped[(row["split"], row["arm"], row["seed"])].append(row)
    result = []
    for (split, arm, seed), rows in sorted(grouped.items()):
        calls = sum(r["verifier_calls"] for r in rows)
        unique = sum(r["unique_verified_facts"] for r in rows)
        relevant = sum(r["goal_relevant_verified_facts"] for r in rows)
        result.append({"split": split, "arm": arm, "seed": seed,
                       "roots": len(rows), "solved": sum(r["solved"] for r in rows),
                       "accuracy": sum(r["solved"] for r in rows) / len(rows),
                       "verifier_calls": calls, "mean_calls": calls / len(rows),
                       "unique_facts_per_call": unique / calls if calls else 0.0,
                       "goal_relevant_facts_per_call": relevant / calls if calls else 0.0,
                       "inner_cell_calls": sum(r["inner_cell_calls"] for r in rows),
                       "seconds": sum(r["elapsed_seconds"] for r in rows)})
    return result


def report(output, summary, fits, registration):
    by = defaultdict(list)
    for r in summary:
        by[(r["split"], r["arm"])].append(r)
    lines = ["# COG-03: recurrent routing over checked finite proof ledgers", "",
             "The run trains a 24-wide transformer cell and reuses its weights across four passes.",
             "Each proposed rule application is checked against the current established facts.",
             "The tasks are synthetic Horn proof DAGs exported by DataGoblin COG-02.", "",
             "## Measured run", "",
             "| Split | Arm | Completed roots / trials | Mean verifier calls | Relevant facts per call |",
             "| --- | --- | ---: | ---: | ---: |"]
    for split in ("calibration", "ordinary_test", "ood_test"):
        for arm in ARMS:
            rows = by[(split, arm)]
            if not rows:
                continue
            solved, trials = sum(r["solved"] for r in rows), sum(r["roots"] for r in rows)
            calls = sum(r["verifier_calls"] for r in rows)
            relevant = sum(r["goal_relevant_facts_per_call"] * r["verifier_calls"] for r in rows)
            lines.append(f"| {split} | {arm} | {solved}/{trials} | {calls/trials:.3f} | {relevant/calls:.4f} |")
    lines += ["", "Seeds reuse the same root worlds. The trial denominator includes those repeated measurements.",
              f"Each split contains {registration['roots_per_split']} distinct roots, with one presentation per root.", "",
              "## Training", "", "| Seed | Inner passes | Parameters | Updates | First-20 loss | Last-20 loss | Teacher checker calls |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fit in fits:
        lines.append(f"| {fit['seed']} | {fit['depth']} | {fit['parameters']} | {fit['optimizer_updates']} | "
                     f"{fit['loss_first20']:.4f} | {fit['loss_last20']:.4f} | "
                     f"{fit['label_collection_verifier_calls']+fit['teacher_transition_verifier_calls']} |")
    lines += ["", "The fixed final checkpoint supplies each evaluation. Training receives only train roots.",
              "The recurrent model and the one-pass model share the architecture and number of optimizer updates.",
              "Their inner-cell compute differs by depth. Every policy receives the same maximum verifier calls.",
              "A shared public rule prevents retrying a rejected action until the ledger changes; conclusions already known are skipped.",
              "The policy sees current facts, public rules and previous checker feedback. Source IDs, split names and teacher labels stay outside its input.", "",
              "## Controls and evidence", "",
              "`reset_history` clears carried state at each verifier decision. `reset_inner` resets state before every inner pass while paying for four passes.",
              "`shuffled_history` rolls the same carried vectors across rule positions. `one_pass` is fitted independently at one inner pass.",
              "The symbolic `goal_ancestry` policy supplies a structural ceiling for this finite task family.",
              "The independent Python checker establishes validity in the declared Horn calculus. DataGoblin's separately reported Lean replay covers its exported certificates.",
              "Evidence class: public development pilot. The architecture was frozen before this run; a calibrated causal power gate remains open.",
              "These measurements concern routing in generated propositional tasks. They provide a training and replay interface for later native Lean corpus experiments.", "",
              "## Reproduce", "", "```bash",
              "python -m research.cog03_ledger_router.run --corpus research/cog03_ledger_router/data/tasks.jsonl.gz --output /tmp/cog03-replay",
              "python -m unittest discover -s research/cog03_ledger_router/tests -v", "```", "",
              "`registration.json` freezes roots and arms. `episodes.jsonl` retains completed and unfinished roots; `attempts.jsonl.gz` records every checker call.",
              "`training.json` counts teacher label checks separately from evaluation calls. Checkpoints and optimizer traces support a full replay.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def run(corpus, output, *, seeds=(17, 29), updates=180, budget=24, threads=2):
    output = Path(output)
    if output.exists():
        raise ValueError("use a fresh output directory to preserve prior receipts")
    if updates < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive updates and distinct seeds required")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    tasks = load_tasks(corpus)
    split_tasks = {split: [t for t in tasks if t["split"] == split] for split in SPLITS}
    if len({len(v) for v in split_tasks.values()}) != 1:
        raise ValueError("registered campaign requires equal root counts across split roles")
    output.mkdir(parents=True)
    registration = {
        "schema_version": "cog03.registration.v1", "experiment_id": "COG-03",
        "created_before_training": True, "corpus_sha256": sha256(corpus),
        "roots_per_split": len(split_tasks["train"]),
        "roots": {split: [t["root_id"] for t in values] for split, values in split_tasks.items()},
        "arms": list(ARMS), "seeds": list(seeds), "budget": budget,
        "optimizer_updates_per_model": updates, "width": 24, "heads": 4,
        "inner_depth": 4, "learning_rate": 0.002, "weight_decay": 0.001,
        "features": list(FEATURES), "source_code": {
            name: sha256(Path(__file__).with_name(name)) for name in ("environment.py", "model.py", "run.py")},
        "primary_contrast": "looped minus independently trained one_pass root completion at budget24",
        "physical_dose": "4 tied inner passes and previous-decision carry",
        "minimum_meaningful_effect": 0.10,
        "independent_unit": "source Horn DAG root; seeds are repeated measurements",
        "planned_comparisons": 4, "hypothesis_preregistered": True,
        "sealed_readout": False, "readout_selection": "public development; no checkpoint selection on readouts",
        "matched_sham": "shuffled_history preserves carried-vector norm and four-pass compute",
        "matched_sham_floor": "unestimated in this pilot",
        "estimated_resolution": "power gate not passed; report descriptive paired root outcomes",
        "verdict": "REDESIGN_OR_PILOT_ONLY",
        "exact_evidence_provenance": "finite Horn interpreter; synthetic COG-02 public corpus",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "torch_threads": threads},
    }
    write_json(output / "registration.json", registration)
    models, fits = {}, []
    for seed in seeds:
        for depth in (4, 1):
            model, fit, optimizer_rows = train_model(split_tasks["train"], seed=seed,
                                                    depth=depth, updates=updates, budget=budget)
            name = f"router-s{seed}-d{depth}.pt"
            torch.save({"schema_version": "cog03.checkpoint.v1", "state_dict": model.state_dict(),
                        "architecture": {"width": 24, "heads": 4, "depth": depth},
                        "registration_sha256": sha256(output / "registration.json")}, output / name)
            fit["checkpoint"] = name
            fit["checkpoint_sha256"] = sha256(output / name)
            fits.append(fit)
            models[(seed, depth)] = model
            gz_rows(output / f"optimizer-s{seed}-d{depth}.jsonl.gz", optimizer_rows)
            print(canonical({"stage": "trained", "seed": seed, "depth": depth,
                             "loss_first20": fit["loss_first20"], "loss_last20": fit["loss_last20"]}), flush=True)
    write_json(output / "training.json", fits)
    episodes, attempts = [], []
    for split in ("calibration", "ordinary_test", "ood_test"):
        for seed in seeds:
            for arm in ARMS:
                model = models[(seed, 1 if arm == "one_pass" else 4)]
                for task in split_tasks[split]:
                    rows, outcome = rollout(task, arm=arm, seed=seed, model=model, budget=budget)
                    attempts.extend(rows)
                    episodes.append(outcome)
                print(canonical({"stage": "evaluated", "split": split, "seed": seed, "arm": arm}), flush=True)
    gz_rows(output / "attempts.jsonl.gz", attempts)
    (output / "episodes.jsonl").write_text("".join(canonical(r) + "\n" for r in episodes))
    summary = summarize(episodes)
    write_json(output / "summary.json", summary)
    # One paired row per root and seed preserves the common-root design.
    lookup = {(r["split"], r["seed"], r["root_id"], r["arm"]): r for r in episodes}
    paired = []
    for row in episodes:
        if row["arm"] == "looped":
            control = lookup[(row["split"], row["seed"], row["root_id"], "one_pass")]
            paired.append({"case_id": f"{row['root_id']}:s{row['seed']}", "root_id": row["root_id"],
                           "split": row["split"], "seed": row["seed"],
                           "candidate_correct": int(row["solved"]), "control_correct": int(control["solved"])})
    (output / "paired.jsonl").write_text("".join(canonical(r) + "\n" for r in paired))
    report(output, summary, fits, registration)
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "artifacts.json", {"schema_version": "cog03.artifacts.v1", "files": files,
                                           "attempt_rows": len(attempts), "episode_rows": len(episodes)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--updates", type=int, default=180)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.corpus, args.output, seeds=tuple(args.seeds), updates=args.updates,
        budget=args.budget, threads=args.threads)


if __name__ == "__main__":
    main()
