"""Aggregate completed LBF-20M repetitions into row-level and paired summaries + a Markdown report."""
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\patri\lbf20m-run-20261125")
ARMS = ("dfs_cover", "direct_tree", "flat_forest", "mdd_forest", "mdd_components", "mdd_query_cache")
COHORTS = ("retro_capped", "retro_bipartite", "retro_mixed", "fresh_explosion")
STAT = ("messages", "base_calibrations", "base_hits", "query_hits", "query_misses", "conditioned_calls",
        "diagram_states", "diagram_edges", "compatible_labels", "terminals", "compiled_units", "dfs_nodes",
        "support_probes", "branch_calibrations", "residual_groups", "component_checks", "component_entries",
        "feasible_component_entries", "whole_forest_entries", "forward_edge_scans", "backward_edge_scans")
DET = ("status", "reason", "base_bits", "query_bits", "queries", "raw_product", "cutset_size", "answers", "shared_plan_seconds_NOT")


def med(xs): return statistics.median(xs)
def ms(x): return 1000 * x


def main():
    reps = sorted(int(p.name.split("-")[1]) for p in ROOT.glob("repeat-??") if (p / "run.json").exists())
    good, bad = [], []
    for n in reps:
        run = json.loads((ROOT / f"repeat-{n:02d}" / "run.json").read_text())
        audit = json.loads((ROOT / f"repeat-{n:02d}" / "receipt-audit.json").read_text()) if (ROOT / f"repeat-{n:02d}" / "receipt-audit.json").exists() else None
        ok = run["exit_code"] == 0 and audit and audit["passed"]
        (good if ok else bad).append(n)
    data = {}
    for n in good:
        d = ROOT / f"repeat-{n:02d}" / "lbf20m"
        data[n] = {(r["root"], r["arm"]): r for r in (json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines())}
    keys = sorted(data[good[0]], key=lambda k: (k[0], ARMS.index(k[1])))
    # determinism of non-timing fields across repetitions
    nondet = []
    for k in keys:
        base = data[good[0]][k]
        for n in good[1:]:
            r = data[n][k]
            for f in DET:
                if f in base and base[f] != r.get(f):
                    nondet.append({"key": list(k), "field": f, "rep": n})
            for s in STAT:
                if base["stats"].get(s, 0) != r["stats"].get(s, 0):
                    nondet.append({"key": list(k), "field": "stats." + s, "rep": n})
    row_table = []
    for k in keys:
        rs = [data[n][k] for n in good]
        b = rs[0]
        ent = {"root": k[0], "arm": k[1], "cohort": b["cohort"], "family": b["family"], "status": b["status"],
               "reason": b["reason"], "reps": len(rs), "n": b["n"], "cutset_size": b["cutset_size"],
               "raw_product": b["raw_product"], "queries": b["queries"], "base_bits": b["base_bits"],
               "query_bits": b["query_bits"], "total_bits": b["base_bits"] + b["query_bits"],
               "shared_plan_seconds_median": med([r["shared_plan_seconds"] for r in rs]),
               "stats": {s: b["stats"].get(s, 0) for s in STAT if s in b["stats"]}}
        for f in ("setup_seconds", "query_seconds", "total_seconds"):
            v = [r[f] for r in rs]
            ent[f] = {"median": med(v), "min": min(v), "max": max(v), "per_rep": v}
        row_table.append(ent)
    idx = {(e["root"], e["arm"]): e for e in row_table}
    roots_by = collections.defaultdict(list)
    for e in row_table:
        if e["arm"] == ARMS[0]: roots_by[e["cohort"]].append(e["root"])
    fam_of = {e["root"]: e["family"] for e in row_table}

    def arm_summary(roots):
        out = {}
        for arm in ARMS:
            es = [idx[(r, arm)] for r in roots]
            ex = [e for e in es if e["status"] == "exact"]
            o = {"roots": len(es), "exact": len(ex), "caps": len(es) - len(ex),
                 "cap_reasons": dict(collections.Counter(e["reason"] for e in es if e["status"] == "cap")),
                 "base_bits": sum(e["base_bits"] for e in ex), "query_bits": sum(e["query_bits"] for e in ex)}
            o["total_bits"] = o["base_bits"] + o["query_bits"]
            for s in ("messages", "query_hits", "query_misses", "base_calibrations", "base_hits", "diagram_states", "diagram_edges", "compatible_labels", "terminals", "compiled_units"):
                o[s] = sum(e["stats"].get(s, 0) for e in ex)
            for f in ("setup_seconds", "query_seconds", "total_seconds"):
                o[f + "_sum_of_medians_ms"] = ms(sum(e[f]["median"] for e in ex))
            o["shared_plan_seconds_sum"] = sum(idx[(r, arm)]["shared_plan_seconds_median"] for r in roots) if arm == ARMS[0] else None
            out[arm] = o
        return out

    def paired(roots, a, b):
        both = [r for r in roots if idx[(r, a)]["status"] == "exact" and idx[(r, b)]["status"] == "exact"]
        if not both: return {"a": a, "b": b, "roots": 0}
        ta = sum(idx[(r, a)]["total_seconds"]["median"] for r in both)
        tb = sum(idx[(r, b)]["total_seconds"]["median"] for r in both)
        sa = sum(idx[(r, a)]["setup_seconds"]["median"] for r in both); sb = sum(idx[(r, b)]["setup_seconds"]["median"] for r in both)
        qa = sum(idx[(r, a)]["query_seconds"]["median"] for r in both); qb = sum(idx[(r, b)]["query_seconds"]["median"] for r in both)
        wins = sum(idx[(r, a)]["total_seconds"]["median"] < idx[(r, b)]["total_seconds"]["median"] for r in both)
        return {"a": a, "b": b, "roots": len(both), "total_ms_a": ms(ta), "total_ms_b": ms(tb), "ratio_a_over_b": ta / tb,
                "setup_ms_a": ms(sa), "setup_ms_b": ms(sb), "query_ms_a": ms(qa), "query_ms_b": ms(qb),
                "roots_where_a_faster": wins,
                "base_bits_a": sum(idx[(r, a)]["base_bits"] for r in both), "base_bits_b": sum(idx[(r, b)]["base_bits"] for r in both),
                "query_bits_a": sum(idx[(r, a)]["query_bits"] for r in both), "query_bits_b": sum(idx[(r, b)]["query_bits"] for r in both),
                "messages_a": sum(idx[(r, a)]["stats"].get("messages", 0) for r in both), "messages_b": sum(idx[(r, b)]["stats"].get("messages", 0) for r in both)}

    pairs = [("mdd_components", "mdd_forest"), ("mdd_query_cache", "mdd_forest"), ("mdd_query_cache", "mdd_components"),
             ("mdd_forest", "dfs_cover"), ("mdd_components", "dfs_cover"), ("mdd_query_cache", "dfs_cover"), ("flat_forest", "mdd_forest")]
    result = {"schema": "lbf20m.local-report.v1", "completed_repetitions": good, "excluded_repetitions": bad,
              "nondeterministic_nontiming_fields": nondet, "cohorts": {}, "families": {}, "all_roots": {}}
    for co in COHORTS:
        rs = roots_by[co]
        result["cohorts"][co] = {"arms": arm_summary(rs), "paired": [paired(rs, a, b) for a, b in pairs]}
        for fam in sorted({fam_of[r] for r in rs}):
            fr = [r for r in rs if fam_of[r] == fam]
            result["families"][f"{co}/{fam}"] = {"arms": arm_summary(fr), "paired": [paired(fr, a, b) for a, b in pairs]}
    retro = [r for co in COHORTS[:3] for r in roots_by[co]]
    result["all_roots"]["retro_all_50"] = {"arms": arm_summary(retro), "paired": [paired(retro, a, b) for a, b in pairs]}
    result["row_level"] = row_table
    runs = {n: json.loads((ROOT / f"repeat-{n:02d}" / "run.json").read_text()) for n in reps}
    result["process"] = {str(n): {k: runs[n][k] for k in ("exit_code", "wall_seconds", "peak_working_set_bytes", "peak_pagefile_bytes", "start_utc")} for n in reps}
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"completed": good, "excluded": bad, "nondeterministic": len(nondet), "rows": len(row_table)}))


if __name__ == "__main__":
    main()
