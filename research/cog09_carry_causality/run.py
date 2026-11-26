"""COG-09: frozen-checkpoint, matched-dose carry perturbation pilot."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from collections import defaultdict

import torch

from research.cog03_ledger_router.environment import (
    available_actions, canonical, check_step, goal_ancestors, load_tasks,
    public_observation, useful_actions,
)
from research.cog03_ledger_router.model import LedgerRouter, encode_observation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO / "research/cog03_ledger_router"
SPEC = json.loads((HERE / "protocol.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("external module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def record(stream, data):
    stream.write(canonical(data) + "\n")


def get_injection(model, previous):
    if previous is None:
        return None
    return torch.sigmoid(model.carry_gate) * model.carry(previous)


def forward_at_site(model, features, mask, injection):
    """Same four tied cells as COG-03, with an explicit post-gate carry site."""
    x = model.input(features).unsqueeze(0)
    if injection is not None:
        if injection.shape != x.shape:
            raise ValueError("carry shape mismatch")
        x = x + injection
    state = x
    for _ in range(SPEC["depth"]):
        state = model.recurrent(state, src_mask=mask)
    state = model.norm(state)
    return {
        "state": state, "logits": model.policy(state).squeeze(0).squeeze(-1),
        "acceptance_logits": model.acceptance(state).squeeze(0).squeeze(-1),
    }


def event(action, receipt):
    return {"action": action, **{k: receipt[k] for k in ("accepted", "code", "new_fact")}}


def advance(world, known, history, failed, action):
    receipt = check_step(world, known, action)
    history.append(event(action, receipt))
    if receipt["new_fact"] is not None:
        failed.clear()
    elif not receipt["accepted"]:
        failed.add(action)
    return set(receipt["next_known"]), receipt


def checkpoint(seed):
    path = OLD / "results" / f"router-s{seed}-d4.pt"
    if sha(path) != SPEC["checkpoints"][str(seed)]:
        raise ValueError("checkpoint digest mismatch")
    item = torch.load(path, map_location="cpu", weights_only=True)
    if item["architecture"] != {"width": 24, "heads": 4, "depth": 4}:
        raise ValueError("unexpected architecture")
    model = LedgerRouter()
    model.load_state_dict(item["state_dict"], strict=True)
    model.eval()
    return model


def sham_directions(direction):
    out = []
    for seed in SPEC["sham_seeds"]:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(direction.shape, generator=g, dtype=direction.dtype)
        v = v - torch.dot(v, direction) * direction
        if v.norm() <= 1e-7:
            raise ValueError("degenerate sham")
        out.append(v / v.norm())
    return out


def fit_probe(model, tasks):
    """Fixed mean-difference direction, using ready, fresh rule tokens only."""
    if not tasks or any(t["split"] != "train" for t in tasks):
        raise ValueError("probe fit requires original training roots")
    groups = [[], []]
    fit_checks = cell_calls = states = 0
    for task in tasks:
        world = task["world"]
        known, history, previous = set(world["facts"]), [], None
        ancestors = goal_ancestors(world)
        for i in range(24):
            useful = useful_actions(world, known)
            if world["goal"] in known or not useful:
                break
            f, mask = encode_observation(public_observation(task, known, history), 24-i, 24)
            injection = get_injection(model, previous)
            if injection is not None:
                states += 1
                for j, rule in enumerate(world["rules"]):
                    if rule["conclusion"] not in known and set(rule["premises"]) <= known:
                        label = int(rule["conclusion"] in ancestors)
                        groups[label].append(injection[0, j].detach().clone())
            result = forward_at_site(model, f, mask, injection)
            cell_calls += SPEC["depth"]
            previous = result["state"]
            action = useful[0]
            receipt = check_step(world, known, action)
            fit_checks += 1
            history.append(event(action, receipt))
            known = set(receipt["next_known"])
    neg, pos = (torch.stack(g) for g in groups)
    direction = pos.mean(0) - neg.mean(0)
    if not torch.isfinite(direction).all() or direction.norm() < 1e-7:
        raise ValueError("degenerate probe direction")
    direction /= direction.norm()
    center = torch.cat([neg, pos]).mean(0)
    return center, direction, sham_directions(direction), {
        "negative_tokens": len(neg), "positive_tokens": len(pos),
        "states": states, "teacher_checks": fit_checks, "cell_calls": cell_calls,
        "training_roots": len(tasks), "center": center.tolist(),
        "direction": direction.tolist(),
    }


def patch(injection, arm, center, direction, shams):
    """Erase centered projection or add exactly the same per-token norm orthogonally."""
    if injection is None or arm == "baseline":
        return injection, 0.0, 0.0
    if arm == "reset_history":
        return None, float(injection.norm()), 0.0
    amplitude = SPEC["dose"] * ((injection - center) @ direction).unsqueeze(-1)
    v = direction if arm == "erase" else shams[int(arm.removeprefix("sham"))]
    delta = -amplitude * v
    actual = delta.norm(dim=-1)
    expected = amplitude.abs().squeeze(-1)
    error = float((actual - expected).abs().max())
    if not torch.allclose(actual, expected, rtol=2e-6, atol=2e-7):
        raise ValueError("matched dose violated")
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite intervention")
    return injection + delta, float(delta.norm()), error


def choose(logits, candidates):
    return max(candidates, key=lambda i: (float(logits[i]), -i))


def same_state(model, task, seed, probe, stream):
    """Every direction sees exactly the same pre-action state and carry."""
    center, direction, shams = probe
    world = task["world"]
    known, history, previous, failed = set(world["facts"]), [], None, set()
    arms = SPEC["arms"][:6]
    checks = passes = matched_states = 0
    acc = defaultdict(lambda: [0, 0.0])
    max_norm_gap = 0.0
    for step in range(SPEC["budget"]):
        useful = useful_actions(world, known)
        if world["goal"] in known or not useful:
            break
        obs = public_observation(task, known, history)
        f, mask = encode_observation(obs, SPEC["budget"]-step, SPEC["budget"])
        injection = get_injection(model, previous)
        base = forward_at_site(model, f, mask, injection)
        passes += SPEC["depth"]
        if previous is not None:
            matched_states += 1
            candidates = available_actions(world, known, failed)
            per_arm = {}
            for arm in arms:
                patched, norm, error = patch(injection, arm, center, direction, shams)
                if arm == "baseline":
                    out = base
                else:
                    out = forward_at_site(model, f, mask, patched)
                    passes += SPEC["depth"]
                logits = out["logits"]
                action = choose(logits, candidates)
                mass = float(torch.softmax(logits[candidates], dim=0)[
                    torch.tensor([i in useful for i in candidates], dtype=torch.bool)].sum())
                hit = int(action in useful)
                acc[arm][0] += hit
                acc[arm][1] += mass
                per_arm[arm] = {"action": action, "useful": hit, "useful_mass": mass,
                                "dose_l2": norm, "dose_error": error}
                if arm.startswith("sham"):
                    max_norm_gap = max(max_norm_gap, abs(norm - per_arm["erase"]["dose_l2"]))
            if max_norm_gap > 2e-5:
                raise ValueError("matched-state sham dose mismatch")
            record(stream, {"root_id": task["root_id"], "seed": seed, "step": step,
                            "observation": obs, "useful": useful, "arms": per_arm})
        previous = base["state"]
        known, _ = advance(world, known, history, failed, useful[0])
        checks += 1
    return {"root_id": task["root_id"], "seed": seed, "states": matched_states,
            "teacher_checks": checks, "cell_calls": passes, "max_norm_gap": max_norm_gap,
            "arms": {arm: {"useful_hits": acc[arm][0], "useful_mass_sum": acc[arm][1]} for arm in arms}}


def rollout(model, task, seed, arm, probe, verifier, stream):
    center, direction, shams = probe
    world = task["world"]
    known, history, previous, failed = set(world["facts"]), [], None, set()
    rows = []
    total_passes = 0
    max_error = 0.0
    for step in range(SPEC["budget"]):
        if world["goal"] in known:
            break
        candidates = available_actions(world, known, failed)
        if not candidates:
            raise ValueError("unsolved rollout exhausted legal menu")
        obs = public_observation(task, known, history)
        norm = error = 0.0
        passes = 0
        if arm == "goal_ancestry":
            action = (useful_actions(world, known) or candidates)[0]
        else:
            f, mask = encode_observation(obs, SPEC["budget"]-step, SPEC["budget"])
            injection = get_injection(model, previous)
            patched, norm, error = patch(injection, arm, center, direction, shams)
            out = forward_at_site(model, f, mask, patched)
            previous = out["state"]
            action = choose(out["logits"], candidates)
            passes = SPEC["depth"]
        known, receipt = advance(world, known, history, failed, action)
        total_passes += passes
        max_error = max(max_error, error)
        rows.append({"root_id": task["root_id"], "task_id": task["task_id"],
                     "split": task["split"], "family": task["family"], "seed": seed,
                     "arm": arm, "budget": SPEC["budget"], "step": step,
                     "observation": obs, "action": action, "receipt": receipt,
                     "verifier_calls": 1, "inner_passes": passes,
                     "dose_l2": norm, "dose_error": error})
    # Independent implementation from EvalMath, not the producer's verdict.
    audit = verifier.replay_episode(task, rows, SPEC["budget"], check_reported=True)
    if audit["stopped_early_unsolved"] or audit["post_solution_calls"]:
        raise ValueError("incomplete episode accounting")
    if arm != "goal_ancestry" and total_passes != SPEC["depth"] * audit["calls"]:
        raise ValueError("unexpected compute count")
    for row in rows:
        record(stream, row)
    first = audit["first_solution_call"]
    auc = Fraction(SPEC["budget"] - first + 1, SPEC["budget"]) if first is not None else Fraction(0)
    if not 0 <= auc <= 1:
        raise ValueError("invalid completion AUC")
    return {"root_id": task["root_id"], "seed": seed, "arm": arm,
            "solved": audit["solved"], "calls": audit["calls"], "first_solution": first,
            "completion_auc": str(auc), "rejected": audit["rejected"],
            "goal_facts": audit["goal_ancestor_verified_facts"], "cell_calls": total_passes,
            "max_dose_error": max_error}


def summarize(episodes):
    result = {}
    for arm in SPEC["arms"]:
        rows = [r for r in episodes if r["arm"] == arm]
        if len(rows) != SPEC["pilot_roots"] * len(SPEC["seeds"]):
            raise ValueError("missing registered episode")
        result[arm] = {"runs": len(rows), "solved": sum(r["solved"] for r in rows),
                       "calls": sum(r["calls"] for r in rows),
                       "cell_calls": sum(r["cell_calls"] for r in rows),
                       "rejected": sum(r["rejected"] for r in rows),
                       "mean_auc": float(sum(Fraction(r["completion_auc"]) for r in rows) / len(rows))}
    return result


def planner_input(tasks, episodes):
    lookup = {(r["root_id"], r["seed"], r["arm"]): Fraction(r["completion_auc"]) for r in episodes}
    expected = {(t["root_id"], s, a) for t in tasks for s in SPEC["seeds"] for a in SPEC["arms"]}
    if set(lookup) != expected or len(lookup) != len(episodes):
        raise ValueError("episode registry mismatch")
    pilot = []
    for task in tasks:
        root = task["root_id"]
        def diff(seed, arm):
            return lookup[(root, seed, "baseline")] - lookup[(root, seed, arm)]
        effect = sum(diff(s, "erase") for s in SPEC["seeds"]) / len(SPEC["seeds"])
        sham_effects = [str(diff(s, f"sham{i}")) for s in SPEC["seeds"]
                        for i in range(len(SPEC["sham_seeds"]))]
        pilot.append({"unit_id": root, "paired_effect": str(effect), "sham_effects": sham_effects})
    return {"schema_version": 1, "experiment_id": "COG-09", "independent_unit": SPEC["independent_unit"],
            **{k: SPEC[k] for k in ("alpha", "scientific_min_effect", "target_effect",
                                  "measurement_resolution", "sham_quantile", "sham_confidence",
                                  "planned_comparisons", "confirm_units_planned", "heldout_sealed",
                                  "pilot_from_development_only")},
            "units_independent": True, "pilot": pilot}


@torch.no_grad()
def run(generator_path, evalmath, output, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    corpus = OLD / "data/tasks.jsonl.gz"
    if sha(corpus) != SPEC["corpus_sha256"] or sha(generator_path) != SPEC["generator_sha256"]:
        raise ValueError("pinned corpus or generator digest mismatch")
    original = load_tasks(corpus)
    generator = external("cog09_generator", generator_path)
    verifier_path = evalmath / "src/spiral_ln/cogentic_ledger.py"
    planner_path = evalmath / "skills/evalmath-mechinterp-power/scripts/preflight.py"
    verifier = external("cog09_independent_verifier", verifier_path)
    planner = external("cog09_preflight", planner_path)
    tasks = [t for t in generator.generate_tasks(SPEC["pilot_roots"], SPEC["pilot_seed"])
             if t["split"] == "ood_test"]
    if len(tasks) != SPEC["pilot_roots"] or len({t["root_id"] for t in tasks}) != len(tasks):
        raise ValueError("pilot root count mismatch")
    if {t["root_id"] for t in original} & {t["root_id"] for t in tasks}:
        raise ValueError("old root overlap")
    if {t["task_id"] for t in original} & {t["task_id"] for t in tasks}:
        raise ValueError("old presentation overlap")
    dump(output / "registration.json", {"protocol": SPEC, "protocol_sha256": sha(HERE/"protocol.json"),
         "runner_sha256": sha(__file__), "generator_sha256": sha(generator_path),
         "verifier_sha256": sha(verifier_path), "planner_sha256": sha(planner_path),
         "root_ids": [t["root_id"] for t in tasks], "torch_version": torch.__version__,
         "threads": threads, "confirmation_generated_or_scored": False})
    with (output / "pilot_tasks.jsonl").open("w") as stream:
        for task in tasks:
            record(stream, task)
    episodes, fixed, fits = [], [], []
    train = [t for t in original if t["split"] == "train"]
    with gzip.open(output / "attempts.jsonl.gz", "wt", encoding="utf-8") as attempts, \
         gzip.open(output / "matched_states.jsonl.gz", "wt", encoding="utf-8") as states:
        for seed in SPEC["seeds"]:
            model = checkpoint(seed)
            # Check the explicit intervention site preserves the unmodified forward.
            obs = public_observation(train[0], set(train[0]["world"]["facts"]), [])
            f, mask = encode_observation(obs, 24, 24)
            previous = model(f, mask, depth=4)["state"]
            expected = model(f, mask, previous, depth=4)
            observed = forward_at_site(model, f, mask, get_injection(model, previous))
            for key in expected:
                torch.testing.assert_close(expected[key], observed[key], rtol=0, atol=0)
            center, direction, shams, fit = fit_probe(model, train)
            fits.append({"seed": seed, **fit, "shams": [v.tolist() for v in shams]})
            for task in tasks:
                fixed.append(same_state(model, task, seed, (center, direction, shams), states))
                for arm in SPEC["arms"]:
                    episodes.append(rollout(model, task, seed, arm, (center, direction, shams), verifier, attempts))
            print(f"COG09 seed {seed} completed", flush=True)
    dump(output / "fits.json", fits)
    dump(output / "episodes.json", episodes)
    dump(output / "fixed_states.json", fixed)
    p = planner_input(tasks, episodes)
    decision = planner.preflight(p)
    dump(output / "pilot.json", p)
    dump(output / "preflight.json", decision)
    same = {}
    for arm in SPEC["arms"][:6]:
        hits = sum(r["arms"][arm]["useful_hits"] for r in fixed)
        n = sum(r["states"] for r in fixed)
        same[arm] = {"states": n, "hits": hits,
                     "mean_useful_mass": sum(r["arms"][arm]["useful_mass_sum"] for r in fixed) / n}
    report = {"evidence_scope": "development only, public synthetic Horn root population",
              "independent_roots": len(tasks), "seeds": SPEC["seeds"],
              "episodes": len(episodes), "arms": summarize(episodes),
              "same_state": same, "fixed_state_teacher_checks": sum(r["teacher_checks"] for r in fixed),
              "fixed_state_cell_calls": sum(r["cell_calls"] for r in fixed),
              "max_same_state_dose_gap": max(r["max_norm_gap"] for r in fixed),
              "preflight": decision, "confirmation_generated_or_scored": False,
              "native_lean_replay": "not run; independent finite verifier replay only"}
    dump(output / "summary.json", report)
    dump(output / "artifacts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print("COG09_SUMMARY " + canonical(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--evalmath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.generator, args.evalmath, args.output, args.threads)


if __name__ == "__main__":
    main()
