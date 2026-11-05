"""Probes of the v1 gym (gym.py at 7b5b6bd/e134b54) for LOCAL_GYM_AUDIT.md.

Each probe measures one claim made by the v1 code or README. The output is a
JSON receipt; nothing here changes gym.py. Run from research/lean_cgrt_hrm400:

    python audit/v1_probes.py --out results/v1_audit/probes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gym  # noqa: E402


def structure_hash(p: gym.Problem) -> str:
    """WL hash of the rule hypergraph, invariant to atom names and rule order."""
    atoms = sorted({a for pre, out in p.rules for a in pre + [out]} | set(p.facts) | {p.goal})
    color = {a: f"{a in p.facts}|{a == p.goal}" for a in atoms}
    for _ in range(len(atoms)):
        new = {}
        for a in atoms:
            prod = sorted("|".join(sorted(color[x] for x in pre)) for pre, out in p.rules if out == a)
            cons = sorted(color[out] + "<" + "|".join(sorted(color[x] for x in pre if x != a))
                          for pre, out in p.rules if a in pre)
            new[a] = hashlib.sha1(json.dumps([color[a], prod, cons]).encode()).hexdigest()[:16]
        color = new
    return hashlib.sha1(json.dumps(sorted(color.values())).encode()).hexdigest()


def probe_parameters():
    out = {}
    for name, width in gym.PRESETS.items():
        m = gym.TwoClockHRM(width)
        total = sum(p.numel() for p in m.parameters())
        out[name] = {"width": width, "parameters": total}
    # Which parameters get no gradient from the training loss?
    m = gym.TwoClockHRM(32)
    p = gym.make_problem(5, 4, "mix")
    known, correct = gym.teacher_states(p)[0]
    x, mask, legal = gym.features(p, known, 48)
    y = torch.zeros(48, dtype=torch.bool); y[correct] = True
    gym.set_loss(m(x[None], mask[None], legal[None]), y[None]).backward()
    dead = [n for n, q in m.named_parameters() if q.grad is None or float(q.grad.abs().sum()) == 0.0]
    out["parameters_without_gradient"] = dead
    out["halt_head_used_in_forward"] = False  # TwoClockHRM.forward never calls self.halt
    return out


def probe_recurrence():
    """Is slow state persistent across proof steps, and is it separated from fast?"""
    m = gym.TwoClockHRM(32).eval()
    p = gym.make_problem(9, 4, "chain")
    states = gym.teacher_states(p)
    with torch.no_grad():
        x, mask, legal = gym.features(p, states[0][0], 48)
        a = m(x[None], mask[None], legal[None])
        b = m(x[None], mask[None], legal[None])
    return {
        "slow_state_reset_each_call": True,  # z = zeros_like(h) inside forward
        "same_input_same_output_after_previous_call": bool(torch.equal(a, b)),
        "fast_and_slow_token_sets": "both over the same rule tokens with the same mask",
        "fast_block_applications_per_decision": m.fast_steps * m.slow_steps,
        "slow_block_applications_per_decision": m.slow_steps,
    }


def probe_cgrt_noop(n=24):
    """Branches permute rule order; the network is permutation-equivariant."""
    torch.manual_seed(0)
    m = gym.TwoClockHRM(32).eval()
    same_action = 0; max_dev = 0.0; total = 0
    for i in range(n):
        p = gym.make_problem(1000 + i, 4, "mix")
        known = set(p.facts)
        with torch.no_grad():
            x, mm, ll = gym.features(p, known, 48)
            base = m(x[None], mm[None], ll[None])[0]
            rnd = random.Random(i)
            perm = list(range(len(p.rules))); rnd.shuffle(perm)
            q = gym.Problem(p.facts, [p.rules[k] for k in perm], p.goal, [], p.depth, p.family)
            xx, m2, l2 = gym.features(q, known, 48)
            v = m(xx[None], m2[None], l2[None])[0]
        mapped = torch.full_like(base, -1e4)
        for new, old in enumerate(perm):
            mapped[old] = v[new]
        real = ll.clone()
        max_dev = max(max_dev, float((base[real] - mapped[real]).abs().max()))
        same_action += int(base.argmax()) == int(mapped.argmax())
        total += 1
    res_g = gym.evaluate(m, seed=5, n=n, max_rules=48, depth_min=2, depth_max=5, device="cpu")
    res_c = gym.evaluate(m, seed=5, n=n, max_rules=48, depth_min=2, depth_max=5, device="cpu",
                         mode="cgrt", branches=3)
    return {"problems": total, "branch_argmax_identical": same_action,
            "max_abs_logit_deviation_across_branches": max_dev,
            "greedy_solved": res_g["solved"], "cgrt_solved": res_c["solved"],
            "greedy_calls": res_g["model_calls"], "cgrt_calls": res_c["model_calls"]}


def probe_split_overlap(train_steps=2000, batch=4, accumulate=1):
    """Structural overlap between the training stream and the fixed eval seeds."""
    args = argparse.Namespace(seed=17, batch=batch, max_train_depth=5, depth_step_interval=500,
                              max_rules=48)
    train = set()
    for step in range(1, train_steps + 1):
        for accum in range(accumulate):
            s = step * accumulate + accum
            rng = random.Random(args.seed + step * 1019 + accum * 117)
            hi = min(args.max_train_depth, 2 + s // max(1, args.depth_step_interval))
            for j in range(batch):
                p = gym.make_problem(args.seed + s * 1_000_003 + j * 101, rng.randint(2, hi),
                                     rng.choice(("chain", "fork", "mix")))
                train.add(structure_hash(p))

    def eval_set(seed, lo, hi, n=24):
        return [gym.make_problem(seed + 104729 * i, lo + i % (hi - lo + 1), ("chain", "fork", "mix")[i % 3])
                for i in range(n)]
    val = eval_set(73_000_000, 2, 5)
    ood = eval_set(83_000_000, 6, 7)
    return {"train_problems_hashed": train_steps * batch * accumulate,
            "train_unique_structures": len(train),
            "validation_in_train": sum(structure_hash(p) in train for p in val), "validation_n": len(val),
            "depth_ood_in_train": sum(structure_hash(p) in train for p in ood), "depth_ood_n": len(ood),
            "families_shared_between_train_and_eval": ["chain", "fork", "mix"]}


def probe_curriculum(accumulate=8, interval=500):
    """sample_batch receives step*accumulate+accum, so depth unlocks `accumulate`x early."""
    def hi(s):
        return min(5, 2 + s // interval)
    first_full = next(step for step in range(1, 10_000) if hi(step * accumulate) == 5)
    return {"accumulate": accumulate, "depth_step_interval": interval,
            "optimizer_step_where_depth5_unlocks": first_full, "intended": 3 * interval}


def probe_resume():
    """Uninterrupted 20 steps vs 10 + resume 10, and silent hyperparameter change."""
    common = ["train", "--preset", "smoke", "--device", "cpu", "--batch", "2", "--threads", "1",
              "--eval-every", "10", "--save-every", "5", "--eval-count", "3"]
    with tempfile.TemporaryDirectory() as d:
        a, b, c = Path(d) / "a", Path(d) / "b", Path(d) / "c"
        gym.main(common + ["--steps", "20", "--out", str(a)])
        gym.main(common + ["--steps", "10", "--out", str(b)])
        gym.main(common + ["--steps", "20", "--out", str(b), "--resume"])
        wa = torch.load(a / "latest.pt", weights_only=False)["weights"]
        wb = torch.load(b / "latest.pt", weights_only=False)["weights"]
        diff = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
        gym.main(common + ["--steps", "10", "--out", str(c)])
        accepted = True
        try:
            gym.main(common + ["--steps", "15", "--out", str(c), "--resume", "--lr", "0.1", "--batch", "7"])
        except ValueError:
            accepted = False
        rows_b = [json.loads(x)["step"] for x in (b / "metrics.jsonl").read_text().splitlines()]
    return {"max_abs_weight_diff_uninterrupted_vs_resumed": diff,
            "bitwise_equal": diff == 0.0,
            "resume_with_changed_lr_and_batch_accepted": accepted,
            "metrics_steps_after_resume": rows_b,
            "last_known_good_copy": False, "checkpoint_hash_verified_on_resume": False}


def probe_oracle(n=60):
    """Backward reachability from the goal (no learning) under the same budget."""
    solved = 0
    for i in range(n):
        p = gym.make_problem(83_000_000 + 104729 * i, 6 + i % 2, ("chain", "fork", "mix")[i % 3])
        need = set(); frontier = [p.goal]
        prod = {out: k for k, (pre, out) in enumerate(p.rules)}
        while frontier:
            a = frontier.pop()
            if a in p.facts or a not in prod:
                continue
            k = prod[a]
            if k not in need:
                need.add(k); frontier += p.rules[k][0]
        known = set(p.facts)
        for _ in range(len(p.teacher) + 1):
            legal = [k for k in gym.applicable(p, known) if k in need]
            if not legal or p.goal in known:
                break
            known.add(p.rules[legal[0]][1])
        solved += p.goal in known
    return {"backward_reachability_oracle_solved": solved, "n": n,
            "note": "every atom has at most one producer, so the proof is unique and search-free"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_audit/probes.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    receipt = {"gym_sha256": hashlib.sha256((ROOT / "gym.py").read_bytes()).hexdigest(),
               "parameters": probe_parameters(), "recurrence": probe_recurrence(),
               "cgrt": probe_cgrt_noop(), "split_overlap": probe_split_overlap(),
               "curriculum": probe_curriculum(), "resume": probe_resume(),
               "oracle": probe_oracle(), "torch": torch.__version__}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
