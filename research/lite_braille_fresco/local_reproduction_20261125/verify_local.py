"""Local receipt verifier for LBF-20M runs (stand-in for the missing bundled audit_20m_receipts.py).

Checks only what can be established from the run's own files. NOT checked (needs the bundle):
retrospective cases vs inherited 20L receipts, fresh answers vs the separately generated structural
reference, and the original CI archive comparison.
"""
import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

TASK_SHA = "dbc74a0dae5bd1f5e1d2ca4079b07e256579caa07bc53fdad8c130601f088107"
ARMS = ("dfs_cover", "direct_tree", "flat_forest", "mdd_forest", "mdd_components", "mdd_query_cache")
DIAGRAM = ("mdd_forest", "mdd_components", "mdd_query_cache")
AUDIT = {"systems": 56, "worlds": 32768, "queries": 1944, "support_checks": 5832, "label_map_checks": 168,
         "cross_terminal_checks": 6, "boundary_checks": 12, "cap_checks": 2, "mismatches": 0}
STAT_KEYS = ("diagram_states", "diagram_edges", "compatible_labels", "terminals", "compiled_units",
             "base_calibrations", "base_hits", "query_hits", "query_misses", "conditioned_calls", "messages",
             "forward_edge_scans", "backward_edge_scans", "support_probes", "dfs_nodes", "branch_calibrations",
             "residual_groups")


def agg(rows):
    a = [r for r in rows if r["status"] == "exact"]
    out = {"roots": len(rows), "exact": len(a), "caps": len(rows) - len(a),
           "queries": sum(r["queries"] for r in a),
           "base_bits": sum(r["base_bits"] for r in a), "query_bits": sum(r["query_bits"] for r in a),
           "raw_product": sum(r["raw_product"] for r in a)}
    for k in STAT_KEYS:
        out[k] = sum(r["stats"].get(k, 0) for r in a)
    out["mean_ms"] = 1000 * statistics.mean(r["total_seconds"] for r in a) if a else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", type=Path, required=True)
    ap.add_argument("--theory-log", type=Path, required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--recorded-head", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    d = a.receipts
    summary = json.loads((d / "summary.json").read_text())
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines()]
    oracle = [json.loads(l) for l in (d / "oracle.jsonl").read_text().splitlines()]
    tasks_bytes = (d / "tasks.json").read_bytes()

    check("source sha equals recorded checkout", a.source_sha == a.recorded_head, f"{a.source_sha} vs {a.recorded_head}")
    check("task digest (summary)", summary["task_sha256"] == TASK_SHA, summary["task_sha256"])
    check("task digest (recomputed from tasks.json)", hashlib.sha256(tasks_bytes).hexdigest() == TASK_SHA)
    for k, v in AUDIT.items():
        check(f"audit.{k}=={v}", summary["audit"].get(k) == v, str(summary["audit"].get(k)))
    check("new_training_fits==0", summary["new_training_fits"] == 0)
    check("caps recorded", (summary["state_cap"], summary["edge_cap"], summary["raw_cap"]) == (4096, 16384, 256))
    theory = a.theory_log.read_text(encoding="utf-8", errors="replace")
    check("six axiom-free declarations in theory.log", theory.count("does not depend on any axioms") == 6)

    roots = {r["root"] for r in rows}
    check("62 distinct roots", len(roots) == 62, str(len(roots)))
    keys = {(r["root"], r["arm"]) for r in rows}
    check("372 unique root-by-arm rows", len(rows) == 372 and len(keys) == 372, f"{len(rows)} rows {len(keys)} unique")
    check("every root has all six arms", all(collections.Counter(r["arm"] for r in rows if r["root"] == x) == {k: 1 for k in ARMS} for x in roots))
    ex = [r for r in rows if r["status"] == "exact"]
    cp = [r for r in rows if r["status"] == "cap"]
    check("298 exact rows", len(ex) == 298, str(len(ex)))
    check("74 cap rows", len(cp) == 74, str(len(cp)))
    check("cap rows keep reason, 0 queries, empty answers",
          all(r["reason"] and r["queries"] == 0 and r["answers"] == [] for r in cp))
    check("cap reasons known", set(r["reason"] for r in cp) <= {"width_cap", "raw_label_cap", "diagram_state_cap", "diagram_edge_cap", "residual_width_cap"},
          json.dumps(dict(collections.Counter(f'{r["arm"]}:{r["reason"]}' for r in cp)), sort_keys=True))

    orc = {o["root"]: o for o in oracle}
    check("oracle covers every root once", len(orc) == len(oracle) == 62)
    nq = sum(len(o["queries"]) for o in oracle)
    nfresh = sum(len(o["queries"]) for o in oracle if o["cohort"] == "fresh_explosion")
    check("342 oracle queries", nq == 342, str(nq))
    check("96 fresh oracle queries", nfresh == 96, str(nfresh))
    mism = []
    cmp_n = 0
    for r in ex:
        o = orc[r["root"]]
        if r["queries"] != len(o["queries"]):
            mism.append({"root": r["root"], "arm": r["arm"], "problem": "query count", "got": r["queries"], "want": len(o["queries"])})
            continue
        for qi, (got, want, q) in enumerate(zip(r["answers"], o["answers"], o["queries"])):
            cmp_n += 1
            if got != want:
                mism.append({"root": r["root"], "arm": r["arm"], "query_index": qi, "query": q, "expected": want, "observed": got})
    check("1614 exact answer-set comparisons", cmp_n == 1614, str(cmp_n))
    check("every exact row equals oracle answers", not mism, json.dumps(mism[:10]))
    for arm in DIAGRAM:
        n = sum(1 for r in ex if r["arm"] == arm)
        check(f"{arm} answers all 62 roots", n == 62, str(n))

    # cohort/family aggregates recomputed from rows
    ok = True; bad = []
    for co, block in summary["cohorts"].items():
        for arm in ARMS:
            want = agg([r for r in rows if r["cohort"] == co and r["arm"] == arm])
            got = block[arm]
            for k, v in want.items():
                if k == "mean_ms":
                    same = (v is None and got[k] is None) or (v is not None and got[k] is not None and abs(v - got[k]) < 1e-6 * max(1, abs(v)))
                else:
                    same = got[k] == v
                if not same: ok = False; bad.append((co, arm, k, got[k], v))
        for fam, fb in block["families"].items():
            for arm in ARMS:
                want = agg([r for r in rows if r["cohort"] == co and r["family"] == fam and r["arm"] == arm])
                for k, v in want.items():
                    got = fb[arm][k]
                    same = (abs(v - got) < 1e-6 * max(1, abs(v))) if k == "mean_ms" and v is not None and got is not None else got == v
                    if not same: ok = False; bad.append((co, fam, arm, k, got, v))
    check("cohort and family aggregates recompute from rows", ok, json.dumps(bad[:10]))
    check("cohort sizes 8/18/24/12", [len({r['root'] for r in rows if r['cohort'] == c}) for c in ("retro_capped", "retro_bipartite", "retro_mixed", "fresh_explosion")] == [8, 18, 24, 12])
    skipped = ["50 retrospective cases identical to inherited 20L cases", "fresh answers vs separately generated structural reference",
               "largest coherent fixture 262,144/1,024/19/36/2 (audit_20m_diagrams.py)", "comparison with original CI archive hash"]
    report = {"receipts": str(d), "passed": all(c["pass"] for c in checks), "n_checks": len(checks),
              "failed": [c for c in checks if not c["pass"]], "checks": checks,
              "not_checked_bundle_missing": skipped, "verifier": "verify_local.py (not the bundled audit_20m_receipts.py)"}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "n_checks": len(checks), "failed": [c["check"] for c in report["failed"]]}))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
