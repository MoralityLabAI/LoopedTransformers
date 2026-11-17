"""AF-30 deterministic listwise train-only linear ranker for legal Horn actions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

from . import gym, features

UPDATES = 500
STEP = .1
L2 = .01
CLIP = 8.
SEEDS = (17, 29, 43, 61, 73, 89, 103, 131)


def init_weights(seed):
    if seed not in SEEDS:
        raise ValueError("unregistered AF30 source seed")
    rng = random.Random(int(gym.digest(["AF30_INITIAL_LINEAR", seed])[:16], 16))
    return [rng.uniform(-.025, .025) for _ in range(features.DIM)]


def _training_rows():
    """Only TRAIN positive graphs; exact physical BFS is an oracle *labeler*."""
    rows = []
    visited = set()
    for case in gym.cases_for("train"):
        if not case.reachable:
            continue
        public = gym.public_features(case)
        state = frozenset(public["initial_physical"])
        for step in range(case.depth):
            key = gym.digest([public, sorted(state)])
            if key in visited:
                raise AssertionError("TRAIN physical graph/state repeated")
            visited.add(key)
            d = gym.public_bitset_bfs(public, state)
            if not d or d > case.depth:
                raise AssertionError("train-only public physical oracle malformed")
            vectors, work = features.extract(public, state)
            if not vectors:
                raise AssertionError("train case has no legal rule")
            targets = []
            by_slot = {slot: (prem, head) for slot, prem, head
                       in public["rules_physical"]}
            for slot in vectors:
                next_state = state | {by_slot[slot][1]}
                if gym.public_bitset_bfs(public, next_state) == d-1:
                    targets.append(slot)
            if not targets:
                raise AssertionError("TRAIN exact shortest action labels empty")
            rows.append({
                "source_graph_sha256": gym.digest(public),
                "state_sha256": gym.digest(sorted(state)),
                "feature_vectors": vectors,
                "target_slots": tuple(sorted(targets)),
                "steps_remaining_training_oracle": d,
            })
            chosen = min(targets)
            state = state | {by_slot[chosen][1]}
        if public["goal_slot"] not in state:
            raise AssertionError("TRAIN canonical optimal proof not completed")
    if len(rows) != 96*5 + 48:  # 48 D5 + 48 D6
        raise AssertionError(f"training example count changed: {len(rows)}")
    return rows


def training_rows(split="train"):
    if split != "train":
        raise ValueError("source rule ranker fitting is TRAIN-only")
    return _training_rows()


def _stats(rows):
    matrix = [vector for row in rows for vector in row["feature_vectors"].values()]
    n = len(matrix)
    if not n:
        raise ValueError("missing legal training actions")
    mean = [sum(row[j] for row in matrix)/n for j in range(features.DIM)]
    std = [
        max((sum((row[j] - mean[j])**2 for row in matrix)/n)**.5, 1e-6)
        for j in range(features.DIM)
    ]
    return mean, std, n


def _normal(x, mean, std):
    return tuple(max(-CLIP, min(CLIP, (a-b)/c))
                 for a,b,c in zip(x,mean,std))


def dot(weights, values):
    return sum(w*x for w,x in zip(weights, values))


def _softmax(z):
    top=max(z)
    p=[math.exp(v-top) for v in z]
    s=sum(p)
    return [v/s for v in p]


def fit(seed, rows=None):
    rows = training_rows() if rows is None else rows
    if (len(rows) != 528 or
        any(len(r["feature_vectors"]) != len(set(r["feature_vectors"])) or
            not r["target_slots"] or
            not set(r["target_slots"]).issubset(r["feature_vectors"])
            for r in rows)):
        raise ValueError("AF30 frozen TRAIN row roster invalid")
    mean, std, candidates = _stats(rows)
    batches = []
    for row in rows:
        order = sorted(row["feature_vectors"])
        xs = [_normal(row["feature_vectors"][slot], mean, std) for slot in order]
        y = [float(slot in row["target_slots"])/len(row["target_slots"])
             for slot in order]
        batches.append((xs, y))
    weights = init_weights(seed)
    initial = weights[:]
    receipt = []
    for epoch in range(UPDATES):
        grad = [0.]*features.DIM
        loss = 0.
        for xs,y in batches:
            pred = _softmax([dot(weights, x) for x in xs])
            for x,p,t in zip(xs,pred,y):
                error = p-t
                for j in range(features.DIM):
                    grad[j] += error*x[j]
                if t:
                    loss -= t*math.log(max(p,1e-300))
        for j in range(features.DIM):
            grad[j] = grad[j]/len(batches) + 2*L2*weights[j]
            weights[j] -= STEP*grad[j]
        if epoch in (0, 99, 199, 299, 399, 499):
            receipt.append({"update":epoch+1,
                            "pre_update_mean_softmax_loss":loss/len(batches)})
    if not all(math.isfinite(w) for w in weights):
        raise ArithmeticError("AF30 trained scorer nonfinite")
    return {
        "experiment":"AF-30", "seed":seed,
        "feature_names":list(features.FEATURE_NAMES),
        "feature_dimension":features.DIM,
        "training_data":"train_only",
        "train_rows":len(rows), "train_action_candidates":candidates,
        "train_pair_graphs":96,
        "training_row_sha256":gym.digest(rows),
        "initial_weights":initial,
        "weights":weights,
        "mean":mean, "std":std,
        "updates":UPDATES, "step":STEP,"l2":L2,
        "loss_receipt":receipt,
        "input_type":"public_physical_incidence_and_current_known_bits",
    }


def score(model, vector, *, untrained=False):
    weights = model["initial_weights"] if untrained else model["weights"]
    return dot(weights, _normal(vector, model["mean"],model["std"]))


def choose(model, vectors, *, untrained=False):
    if not vectors:
        return gym.STOP
    return min(vectors, key=lambda a:(-score(model,vectors[a],untrained=untrained), a))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--out", type=Path, required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("AF30 ranker output must be immutable")
    model=fit(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"seed":args.seed,"train_rows":model["train_rows"],
                      "model_sha256":gym.digest(model)}))


if __name__=="__main__":
    main()
