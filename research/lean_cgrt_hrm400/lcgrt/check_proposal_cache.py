"""Check a proposal cache against the proposals an earlier evaluation stored.

    python -m lcgrt.check_proposal_cache --cache D:/lean_cgrt/runs/c1/propcache_c4 \
        --eval D:/lean_cgrt/runs/c1/evalM_tlm3_s17_29_43 --out results/c4/proposal_cache_check.json

For every run present in both, each cached state's beam (first 8, as the evaluation stores it)
must equal the stored policy for the same state, in the same order of states. Also checks that
each cache entry's key names the run's current best.pt hash. Exit 0 only when all runs match.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from .mathlib_eval import KS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--eval", required=True, help="an evaluation dir with states.jsonl.gz")
    ap.add_argument("--runs-root", default="D:/lean_cgrt/runs/c1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in gzip.open(Path(a.eval) / "states.jsonl.gz", "rt", encoding="utf-8")]
    report, ok = {"eval": a.eval, "cache": a.cache, "states": len(rows), "runs": {}}, True
    for path in sorted(Path(a.cache).glob("*.json")):
        run = path.stem
        if run not in rows[0]["policies"]:
            report["runs"][run] = {"compared": False, "reason": "not in that evaluation"}
            continue
        entry = json.loads(path.read_text(encoding="utf-8"))
        best = json.loads((Path(a.runs_root) / run / "best.json").read_text())["sha256"]
        props = entry["proposals"]
        same = sum(p[:max(KS)] == r["policies"][run] for p, r in zip(props, rows))
        diff = [i for i, (p, r) in enumerate(zip(props, rows)) if p[:max(KS)] != r["policies"][run]]
        good = len(props) == len(rows) and same == len(rows) and entry["key"]["ckpt_sha256"] == best
        ok &= good
        report["runs"][run] = {"compared": True, "identical_states": same, "of": len(rows), "first_differences": diff[:10],
                               "key_matches_best_pt": entry["key"]["ckpt_sha256"] == best, "key": entry["key"], "ok": good}
    report["all_ok"] = ok and any(v.get("compared") for v in report["runs"].values())
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
