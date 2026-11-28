"""Offline routing mass oracle and causal last-window simulator.

Each JSONL row: root (str), turn (str), token (nonnegative int),
layer (0..23), pass (0..2; 0 only outside loop), experts (8 distinct
integers in [0,127]), weights (8 finite nonnegative native selected weights).
One row per visit. Tokens must use unique positions across the turn. Include
only one phase per input file or use disjoint token positions; never combine
teacher-forced and generated samples in one turn. Missing visits are allowed
for pilot subsets; use --strict-visits for complete teacher traces.
"""
import argparse
import json
import math
from collections import defaultdict


def parse_visit(r):
    root, turn, t, layer, p = (r[k] for k in ("root", "turn", "token", "layer", "pass"))
    if not all(isinstance(x, str) and x for x in (root, turn)):
        raise ValueError("root/turn must be nonempty strings")
    if not all(type(x) is int for x in (t, layer, p)):
        raise ValueError("indices must be integers")
    if t < 0 or not 0 <= layer < 24 or not 0 <= p < (3 if 6 <= layer < 18 else 1):
        raise ValueError("invalid visit index")
    ids, weights = r["experts"], r["weights"]
    if len(ids) != 8 or len(set(ids)) != 8 or any(type(e) is not int or not 0 <= e < 128 for e in ids):
        raise ValueError("invalid top-8 expert IDs")
    if len(weights) != 8 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("invalid weights")
    z = sum(weights)
    if not math.isfinite(z) or z <= 0:
        raise ValueError("invalid routing mass")
    return (root, turn, t, layer, p), ids, [w / z for w in weights]


def parse_rows(lines, strict=False):
    seen, rows = set(), []
    for line in lines:
        if not line.strip():
            continue
        key, ids, weights = parse_visit(json.loads(line))
        if key in seen:
            raise ValueError("duplicate visit")
        seen.add(key)
        rows.append((key, dict(zip(ids, weights))))
    if not rows:
        raise ValueError("empty trace")
    if strict:
        counts = defaultdict(int)
        for key, _ in rows:
            counts[key[:3]] += 1
        if any(n != 48 for n in counts.values()):
            raise ValueError("expected all 48 physical visits for each token")
    return sorted(rows, key=lambda r: r[0])


def select(scores, k):
    return frozenset(sorted(range(128), key=lambda e: (-scores.get(e, 0.0), e))[:k])


def simulate(rows, resident, horizon, expert_bytes, policy="oracle", initial=None):
    """Cold-start per turn; one shared physical cache across loop passes.
    oracle uses entire current segment (noncausal). previous uses only the
    prior segment, with a frozen initial prior or deterministic ID tie-break.
    Counts actual admissions including initial fill; no simulated bandwidth overlap.
    """
    if not 8 <= resident <= 128 or horizon < 0 or expert_bytes <= 0:
        raise ValueError("invalid capacity, horizon or byte cost")
    if policy not in ("oracle", "previous", "global"):
        raise ValueError("unknown policy")
    groups = defaultdict(list)
    for (root, turn, t, layer, p), weights in rows:
        groups[root, turn, layer, t // horizon if horizon else 0].append(weights)
    last, prev, stats = {}, {}, defaultdict(lambda: [0.0, 0, 0])
    by_root = defaultdict(lambda: [0.0, 0, 0])
    for (root, turn, layer, seg), visits in sorted(groups.items()):
        key = root, turn, layer
        scores = defaultdict(float)
        for weights in visits:
            for e, w in weights.items():
                scores[e] += w
        prior = (initial or {}).get(layer, {})
        demand = scores if policy == "oracle" else prev.get(key, prior) if policy == "previous" else prior
        chosen = select(demand, resident)
        mass = sum(sum(w for e, w in weights.items() if e in chosen) for weights in visits)
        added = len(chosen - last.get(key, frozenset()))
        for s in (stats[root, turn], by_root[root]):
            s[0] += mass
            s[1] += len(visits)
            s[2] += added
        last[key], prev[key] = chosen, scores
    def summarize(s):
        return {"retained_mass": s[0] / s[1], "visits": s[1],
                "admissions": s[2], "transfer_bytes": s[2] * expert_bytes}
    return {"policy": policy, "resident": resident, "horizon": horizon,
            "root_mean_retained_mass": sum(s[0] / s[1] for s in by_root.values()) / len(by_root),
            "n_roots": len(by_root),
            "n_turns": len(stats),
            "roots": {root: summarize(s) for root, s in sorted(by_root.items())},
            "turns": [{"root": root, "turn": turn, **summarize(s)} for (root, turn), s in sorted(stats.items())],
            "total_transfer_bytes": sum(s[2] * expert_bytes for s in by_root.values())}


def train_prior(rows):
    counts = defaultdict(lambda: defaultdict(float))
    for (_, _, _, layer, _), weights in rows:
        for e, w in weights.items():
            counts[layer][e] += w
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--resident", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--expert-bytes", type=int, required=True)
    ap.add_argument("--prior-trace", help="disjoint training roots, for causal controls")
    ap.add_argument("--strict-visits", action="store_true")
    args = ap.parse_args()
    with open(args.trace, encoding="utf8") as f:
        rows = parse_rows(f, args.strict_visits)
    initial = None
    if args.prior_trace:
        with open(args.prior_trace, encoding="utf8") as f:
            training = parse_rows(f, args.strict_visits)
        if {k[0] for k, _ in rows} & {k[0] for k, _ in training}:
            raise ValueError("training and evaluation roots overlap")
        initial = train_prior(training)
    print(json.dumps([simulate(rows, args.resident, args.horizon, args.expert_bytes, p, initial)
                      for p in ("oracle", "previous", "global")], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
