"""Audit 10 - Horizon semantics: Pass k of the fibered domain is not "<= k concrete steps".

1. Word-level witness with a single rule.  R = {ab <-> cd}, w0 = "abeeab".  The two Pass-1
   redexes sit at p = 0 and p = 4, i.e. in the same congruence fiber (0 = 4 mod 4).  Their merged
   2-gram table concretises to four words, including "cdeecd", which is at concrete distance 2.
   So the Pass-1 fiber already contains a 2-step word; "Pass-1 exactness" holds for symbol
   marginals only.
2. For the K = 3 symbol-level false positives of each registered symbolic domain on the open
   calibration endpoints, the exact concrete distance at which the cell first becomes reachable
   (searched to depth 8), with concrete examples.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DEPTH = 8


def gamma2_words(table: torch.Tensor) -> list[str]:
    Lm1 = table.shape[0]
    words = [a + b for a in "abcdef" for b in "abcdef" if table[0, "abcdef".index(a), "abcdef".index(b)]]
    for p in range(1, Lm1):
        words = [w + b for w in words for b in "abcdef" if table[p, "abcdef".index(w[-1]), "abcdef".index(b)]]
    return sorted(words)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(args.repo)
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    out: dict = {}

    rules = (("ab", "cd"),)
    w0 = "abeeab"
    base_a2, fibers = m17.symbolic_prov4_adj2(rules, w0, 1)[1]
    ball = common.bfs_ball(rules, w0, 4)
    f0 = gamma2_words(fibers[0])
    out["word_level_witness"] = {
        "rules": [list(r) for r in rules], "w0": w0,
        "pass1_redex_positions": [0, 4], "fiber_of_both": 0,
        "pass1_fiber0_concretisation": f0,
        "concrete_distance_of_each": {w: ball.get(w) for w in f0},
        "contains_word_at_distance_2": any(ball.get(w) == 2 for w in f0),
    }

    cases, _ = common.load_open_split(repo, rg, "calibration")
    hist = {d: Counter() for d in ("sym_pos1", "sym_adj2", "sym_sub3", "sym_prov4_adj2")}
    examples = []
    for c in cases:
        for w in (c.source, c.target):
            L = len(w)
            ball = common.bfs_ball(c.rules, w, DEPTH)
            first = {}
            for z, d in ball.items():
                for p, ch in enumerate(z):
                    key = (p, ch)
                    if key not in first or d < first[key]:
                        first[key] = d
            p42 = m17.symbolic_prov4_adj2(c.rules, w, 3)[3]
            p42_pos = m17.adj2_to_pos1(p42[0])
            for fb in p42[1]:
                if fb.any():
                    p42_pos |= m17.adj2_to_pos1(fb)
            preds = {
                "sym_pos1": m17.symbolic_pos1(c.rules, w, 3)[3],
                "sym_adj2": m17.adj2_to_pos1(m17.symbolic_adj2(c.rules, w, 3)[3]),
                "sym_sub3": m17.sub3_to_pos1(m17.symbolic_sub3(c.rules, w, 3)[3]),
                "sym_prov4_adj2": p42_pos,
            }
            for name, pr in preds.items():
                for p, s in pr.nonzero().tolist():
                    d = first.get((p, "abcdef"[s]))
                    if d is not None and d <= 3:
                        continue
                    label = f"first_reachable_at_{d}" if d is not None else f"not_reachable_within_{DEPTH}"
                    hist[name][label] += 1
                    if name == "sym_prov4_adj2" and len(examples) < 6:
                        examples.append({"root": c.root, "word": w, "position": p, "symbol": "abcdef"[s],
                                         "first_concrete_distance": d})
    out["k3_false_positive_first_concrete_distance"] = {k: dict(sorted(v.items())) for k, v in hist.items()}
    out["k3_false_positive_share_that_is_horizon_overshoot"] = {
        k: (sum(n for lab, n in v.items() if lab.startswith("first_reachable")) / max(1, sum(v.values())))
        for k, v in hist.items()}
    out["prov4_adj2_examples"] = examples
    out["search_depth"] = DEPTH
    common.write_json(Path(args.out), out)
    import json
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
