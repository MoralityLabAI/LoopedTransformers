"""Audit 11 - Two accounting facts taken directly from the archive (standard library only).

1. RULE-WINDOW MATCHING.  The results table reports "rule-window checks" for the learned arms and
   0 for bidir / astar / geom_mm / exact_lookahead.  Those controls call the simulator's
   `neighbors`, which performs 16(L-1) + 8(L-2) rule-window comparisons per expanded word.  We put
   both sides in the same unit: model-side checks (as archived) + simulator-side checks.
2. STATE SIZE.  Size of the exact concrete k-balls around the calibration endpoints, and the bits
   needed to list them, against the size of the abstract fiber bank kept per word per pass.
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = common.lbf_dir(Path(args.repo)) / "results" / "lbf17"
    cal = common.read_jsonl_gz(res / "suite" / "calibration.jsonl.gz")   # open split
    L = {r["root"]: len(r["source"]) for r in cal}
    eps = common.read_jsonl_gz(res / "episodes.jsonl.gz")
    acc = defaultdict(lambda: [0, 0, 0, 0, 0.0])
    for e in eps:
        a = acc[e["condition"]]
        n = L[e["root"]]
        a[0] += 1
        a[1] += e["substring_checks"]
        a[2] += e["expanded"] * (16 * (n - 1) + 8 * (n - 2))
        a[3] += e["expanded"]
        a[4] += e["penalized_work"]
    windows = {}
    for c, a in sorted(acc.items()):
        windows[c] = {"episodes": a[0], "model_side_window_checks_per_episode": a[1] / a[0],
                      "simulator_side_window_checks_per_episode": a[2] / a[0],
                      "total_window_checks_per_episode": (a[1] + a[2]) / a[0],
                      "expanded_words_per_episode": a[3] / a[0], "mean_penalized_work": a[4] / a[0]}
    ref = windows["bidir"]["total_window_checks_per_episode"]
    for c in windows:
        windows[c]["ratio_to_bidir"] = windows[c]["total_window_checks_per_episode"] / ref

    sizes = {k: [] for k in (1, 2, 3, 4, 6)}
    bank_bits, lens = [], []
    for r in cal:
        rules = [tuple(x) for x in r["rules"]]
        for w in (r["source"], r["target"]):
            ball = common.bfs_ball(rules, w, 6)
            for k in sizes:
                sizes[k].append(sum(1 for d in ball.values() if d <= k))
            bank_bits.append(4 * (len(w) - 1) * 36)
            lens.append(len(w))
    bits_per_word = st.mean(lens) * math.log2(6)
    state = {f"ball_le_{k}": {"mean_words": st.mean(v), "median_words": st.median(v), "max_words": max(v),
                              "bits_to_enumerate_mean": st.mean(v) * bits_per_word} for k, v in sizes.items()}
    state["abstract_fiber_bank_boolean_cells_per_word_per_pass"] = st.mean(bank_bits)
    state["positionwise_glyph_cells_per_word"] = st.mean(lens) * 6
    common.write_json(Path(args.out), {"rule_window_matching": windows, "state_size": state})
    for c in ("bidir", "geom_mm", "graph_bridge", "exact_lookahead", "sym_prov_readout", "param_nonrecurrent",
              "compute_nonrecurrent", "lbf16_pos1_opened", "rel_prov_opened"):
        w = windows[c]
        print(f"{c:22s} total window checks/ep {w['total_window_checks_per_episode']:10.0f}  x{w['ratio_to_bidir']:.1f} bidir")
    print({k: (round(v["mean_words"], 1), round(v["bits_to_enumerate_mean"])) for k, v in state.items() if k.startswith("ball")},
          state["abstract_fiber_bank_boolean_cells_per_word_per_pass"])


if __name__ == "__main__":
    main()
