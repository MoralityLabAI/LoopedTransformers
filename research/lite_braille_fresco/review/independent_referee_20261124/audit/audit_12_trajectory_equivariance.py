"""Audit 12 - Does alphabet-permutation equivariance hold at the level of whole search episodes?

The unit tests verify S6 equivariance of single forward passes to atol = 1e-5.  A controller
selects by argmax, so tolerance-level differences could in principle change a decision and hence
a trajectory.  Here every open calibration root is re-run under random alphabet permutations
(source, target and all rule sides relabelled together).  The simulator's move order depends only
on (rule index, direction, position), which a relabelling preserves, so an exactly equivariant
controller must reproduce the same (closed, work, path length) on the relabelled problem.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from audit_03_replication_factorial import clean_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=3)
    args = ap.parse_args()
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    common.set_single_thread()
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    cases, _ = common.load_open_split(repo, rg, "calibration")
    rng = random.Random(20261124)
    out = {}
    for arch in ("rel_prov_opened", "sym_prov_readout", "lbf16_pos1_opened"):
        model, _ = common.load_checkpoint(repo, m17, 107, arch, scratch)
        same = diff = 0
        max_work_gap = 0
        examples = []
        for c in cases:
            ref = clean_episode(rg, m15, c, model, arch, "normal", 128)
            for _ in range(args.perms):
                perm = list("abcdef")
                rng.shuffle(perm)
                tr = str.maketrans("abcdef", "".join(perm))
                c2 = rg.Case(c.root, c.lineage, c.split, c.family, c.source.translate(tr), c.target.translate(tr),
                             tuple((a.translate(tr), b.translate(tr)) for a, b in c.rules)).validate()
                r2 = clean_episode(rg, m15, c2, model, arch, "normal", 128)
                ok = (r2["closed"] == ref["closed"] and r2["work"] == ref["work"] and len(r2["path"]) == len(ref["path"]))
                same += int(ok)
                diff += int(not ok)
                if not ok:
                    max_work_gap = max(max_work_gap, abs(r2["work"] - ref["work"]))
                    if len(examples) < 5:
                        examples.append({"root": c.root, "perm": "".join(perm), "work": [ref["work"], r2["work"]],
                                         "closed": [ref["closed"], r2["closed"]]})
        out[arch] = {"episodes_compared": same + diff, "identical_trajectory_outcome": same,
                     "different": diff, "max_abs_work_gap": max_work_gap, "examples": examples}
        print(arch, {k: v for k, v in out[arch].items() if k != "examples"}, flush=True)
    common.write_json(Path(args.out), out)


if __name__ == "__main__":
    main()
