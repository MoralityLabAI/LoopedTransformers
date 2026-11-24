"""Audit 13 - Record-level recomputation of the LBF-14, LBF-15 and LBF-16 archives.

For each archive: closures and mean penalised work per condition recomputed from
episodes.jsonl.gz and compared with manifest.json; record counts compared with the counts
stated in the results documents; registered screen outcomes echoed.  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

STATED_RECORDS = {"lbf14": 672, "lbf15": 1560, "lbf16": 1560}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for d in ("lbf14", "lbf15", "lbf16"):
        res = common.lbf_dir(Path(args.repo)) / "results" / d
        m = json.loads((res / "manifest.json").read_text())
        rows = common.read_jsonl_gz(res / "episodes.jsonl.gz")
        n, clos, pen = Counter(), Counter(), defaultdict(float)
        for r in rows:
            k = r["condition"]
            n[k] += 1
            clos[k] += int(r["native_certified"])
            pen[k] += r["penalized_work"]
        per, mism = {}, []
        for k in sorted(n):
            mc = m.get("costs", {}).get(k, {})
            mean_pen = pen[k] / n[k]
            per[k] = {"episodes": n[k], "certified_closures": clos[k], "mean_penalized_work": mean_pen}
            if "mean_penalized_work" in mc and abs(mc["mean_penalized_work"] - mean_pen) > 1e-9:
                mism.append(k + ":mean_penalized_work")
            if "certified_closures" in mc and mc["certified_closures"] != clos[k]:
                mism.append(k + ":certified_closures")
        out[d] = {
            "records_archived": len(rows), "records_stated_in_results_document": STATED_RECORDS[d],
            "roots": len({r["root"] for r in rows}), "mismatches_against_manifest": mism,
            "conditions": per, "screens": m.get("screens", {k: m[k] for k in ("primary_screen", "memory_screen") if k in m}),
            "sealed_outcomes_opened_flag": m.get("sealed_outcomes_opened"),
            "suite_directory_archived": (res / "suite").exists(),
        }
        print(d, "records", len(rows), "stated", STATED_RECORDS[d], "mismatches", mism, "suite archived", (res / "suite").exists())
    common.write_json(Path(args.out), out)


if __name__ == "__main__":
    main()
