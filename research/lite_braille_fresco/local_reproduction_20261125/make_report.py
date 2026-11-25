import csv
import json
from pathlib import Path

R = Path(r"C:\Users\patri\lbf20m-run-20261125")
s = json.loads((R / "report" / "summary.json").read_text())
env = json.loads((R / "environment.json").read_text(encoding="utf-8-sig"))
ARMS = ("dfs_cover", "direct_tree", "flat_forest", "mdd_forest", "mdd_components", "mdd_query_cache")

with open(R / "report" / "row_timings.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cohort", "family", "root", "arm", "status", "reason", "reps", "n", "cutset_size", "raw_product", "queries",
                "setup_ms_median", "setup_ms_min", "setup_ms_max", "query_ms_median", "query_ms_min", "query_ms_max",
                "total_ms_median", "total_ms_min", "total_ms_max", "base_bits", "query_bits", "total_bits", "messages",
                "query_hits", "query_misses", "base_calibrations", "diagram_states", "diagram_edges", "compatible_labels",
                "terminals", "shared_plan_ms_median"])
    for e in s["row_level"]:
        st = e["stats"]
        g = lambda k: round(1000 * e[k]["median"], 4)
        lo = lambda k: round(1000 * e[k]["min"], 4)
        hi = lambda k: round(1000 * e[k]["max"], 4)
        w.writerow([e["cohort"], e["family"], e["root"], e["arm"], e["status"], e["reason"] or "", e["reps"], e["n"], e["cutset_size"],
                    e["raw_product"], e["queries"], g("setup_seconds"), lo("setup_seconds"), hi("setup_seconds"),
                    g("query_seconds"), lo("query_seconds"), hi("query_seconds"), g("total_seconds"), lo("total_seconds"), hi("total_seconds"),
                    e["base_bits"], e["query_bits"], e["total_bits"], st.get("messages", 0), st.get("query_hits", 0), st.get("query_misses", 0),
                    st.get("base_calibrations", 0), st.get("diagram_states", 0), st.get("diagram_edges", 0), st.get("compatible_labels", 0),
                    st.get("terminals", 0), round(1000 * e["shared_plan_seconds_median"], 4)])


def table(blk):
    rows = ["| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        o = blk["arms"][a]
        rows.append(f"| `{a}` | {o['exact']}/{o['roots']} | {', '.join(f'{k} ×{v}' for k, v in o['cap_reasons'].items()) or '-'} | {o['base_bits']:,} | {o['query_bits']:,} | {o['total_bits']:,} | "
                    f"{o['messages']:,} | {o['query_hits']} / {o['query_misses']} | {o['diagram_states']} / {o['diagram_edges']} | {o['compatible_labels']} / {o['terminals']} | "
                    f"{o['total_seconds_sum_of_medians_ms']:.1f} ({o['setup_seconds_sum_of_medians_ms']:.1f} + {o['query_seconds_sum_of_medians_ms']:.1f}) |")
    return "\n".join(rows)


def ptable(blk):
    rows = ["| A vs B (roots both exact) | roots | Σ median total ms A | B | A/B | roots A faster | base+query bits A | B | messages A | B |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in blk["paired"]:
        if not p["roots"]:
            continue
        rows.append(f"| `{p['a']}` vs `{p['b']}` | {p['roots']} | {p['total_ms_a']:.1f} | {p['total_ms_b']:.1f} | {p['ratio_a_over_b']:.2f} | {p['roots_where_a_faster']} | "
                    f"{p['base_bits_a']:,}+{p['query_bits_a']:,} | {p['base_bits_b']:,}+{p['query_bits_b']:,} | {p['messages_a']:,} | {p['messages_b']:,} |")
    return "\n".join(rows)


def plan_ms(co):
    return sum(e["shared_plan_seconds_median"] for e in s["row_level"] if e["cohort"] == co and e["arm"] == "dfs_cover") * 1000


proc = s["process"]
walls = [v["wall_seconds"] for v in proc.values()]
peaks = [v["peak_working_set_bytes"] / 2**20 for v in proc.values()]
fresh = s["cohorts"]["fresh_explosion"]["arms"]
retro = s["all_roots"]["retro_all_50"]["arms"]
parts = []
parts.append(f"""# LBF-20M local reproduction report

Source `40824b86e72da8fe54874ec9d9619d41b2706488` (tested commit of PR #383; merge commit `061aa65d…` not used). Development-only, standard-library CPU experiment; no training, no sealed or held-out data touched.

## Status

- **Completed repetitions: 5 of 5** (exit 0, fresh process each, run sequentially). Every repetition passes all 32 local receipt checks; no non-timing field differs between repetitions.
- **Not done (bundle missing):** `lbf20m-run-and-audit.zip` was not on this machine (searched `C:\\Users\\patri` and `D:\\`). So `audit_20m_receipts.py`, `audit_20m_diagrams.py`, the inherited 20L receipts, the independent fresh reference and the original CI archive were unavailable. The following requirements are therefore **unverified**: 50 retrospective cases identical to inherited 20L cases; fresh answers equal to the separately generated structural reference; the largest-coherent-fixture diagram check (262,144 / 1,024 / 19 / 36 / 2); comparison with the original CI archive hash. `verify_local.py` stands in for the receipt checks that need only the run's own files and is less independent than the bundled verifier.
- **Deviation:** Python **3.12.x** was used instead of the registered 3.11 (3.11 not installed; 3.12 chosen by the user). `sys.flags.optimize == 0`, `PYTHONOPTIMIZE` cleared, `PYTHONHASHSEED=0` (see `preflight.txt`).
- Lean 4.19.0 (`--threads=1`, exit 0): all six declarations report no axioms (`theory.log`). These are abstract statements; the Python implementation is checked separately by the finite audit below.

## Environment

OS {env['os']['Caption']} {env['os']['Version']}; CPU {env['cpu']['Name']} ({env['cpu']['NumberOfCores']} cores / {env['cpu']['NumberOfLogicalProcessors']} threads); RAM {int(env['os']['TotalVisibleMemorySize'])/1048576:.1f} GiB; Lean {env['lean']}; source worktree status clean (0 entries) at `git rev-parse HEAD` = {env['source_head']}.
Invocation: `python -u research/lite_braille_fresco/lbf20m_factor.py --out <run-root>/repeat-NN/lbf20m`, driven by `launch_repeats.py`.

## Verification (every repetition)

Finite audit: 56 systems, 32,768 worlds, 1,944 queries, 5,832 support checks, 168 label-map comparisons, 6 cross-terminal, 12 boundary, 2 cap checks, 0 mismatches, 0 new training fits. Benchmark: 62 distinct roots; 372 unique root-by-arm rows; 298 exact, 74 cap rows (all `width_cap` for `direct_tree` ×62 and `raw_label_cap` for `flat_forest` ×12 fresh); 1,614 exact answer-set comparisons, every exact row equal to its oracle answers; 342 oracle queries including 96 fresh; cap rows keep reasons, zero queries and empty answers; the three diagram arms each answer all 62 roots; cohort and family aggregates recompute from rows; task digest `dbc74a0dae5bd1f5e1d2ca4079b07e256579caa07bc53fdad8c130601f088107` matches. No semantic mismatch occurred, so there is no root/arm/query to report.

## Recorded CI result as context vs this machine (fresh cohort, 12 roots)

| Arm | Fresh base bits (CI) | here | Fresh query-cache bits (CI) | here | Fresh messages (CI) | here |
|---|---:|---:|---:|---:|---:|---:|
| `mdd_forest` | 11,209,296 | {fresh['mdd_forest']['base_bits']:,} | 0 | {fresh['mdd_forest']['query_bits']:,} | 21,779 | {fresh['mdd_forest']['messages']:,} |
| `mdd_components` | 965,184 | {fresh['mdd_components']['base_bits']:,} | 0 | {fresh['mdd_components']['query_bits']:,} | 3,099 | {fresh['mdd_components']['messages']:,} |
| `mdd_query_cache` | 965,184 | {fresh['mdd_query_cache']['base_bits']:,} | 225,396 | {fresh['mdd_query_cache']['query_bits']:,} | 1,849 | {fresh['mdd_query_cache']['messages']:,} |

The retrospective numbers from the original run also reproduce: component caching saves {retro['mdd_forest']['base_bits'] - retro['mdd_components']['base_bits']} nominal bits across the 50 retrospective roots ({retro['mdd_forest']['base_bits']:,} vs {retro['mdd_components']['base_bits']:,}) with identical message counts ({retro['mdd_forest']['messages']:,} vs {retro['mdd_components']['messages']:,}); query memoization stores {retro['mdd_query_cache']['query_bits']:,} extra bits for {retro['mdd_query_cache']['query_hits']} hits. Timings below are measured here and are independent of CI timings.

## Measurement conditions

- `base_bits` and `query_bits` are nominal capacity of calibrated bag tables and unary masks; they exclude Python objects, static tables, diagram storage and index structures. Reported separately and summed.
- Fresh cutsets were supplied (structural selector cutsets); fresh planning time is not measured (0). The common retrospective profile-greedy planner is timed separately: {plan_ms('retro_capped'):.1f} / {plan_ms('retro_bipartite'):.1f} / {plan_ms('retro_mixed'):.1f} ms (median across repetitions, summed over roots) for capped / bipartite / mixed, excluded from arm timings.
- Timings: per root and arm, median across the 5 repetitions of setup, query and total seconds; spread (min/max) is in `row_timings.csv` and `summary.json`. Tables show the sum of per-root medians, and timing comparisons use only roots completed by both arms. Arm order rotates per root inside the experiment. The oracle and witness audit are excluded from arm times.
- Process wall time per repetition (includes audit, planner, DFS oracle, all arms): {', '.join(f'{w:.0f}' for w in walls)} s. The preliminary attempt-0 repeat-01 took 374 s with the same code. The spread (374 to 657 s) was not isolated to a cause; the laptop is shared with other jobs (the GPU was near full) and repetition 1 overlapped a filesystem search, so absolute wall times and small per-root timings should be read with that noise in mind. Whole-process peak working set (not per arm): {min(peaks):.1f}–{max(peaks):.1f} MiB.
- Fresh fixtures are deliberate structural stress cases, not representative or lineage-disjoint generalization. No claim of automatic solver speedup from a supplied cutset.

## Fresh explosion cohort (12 roots)

{table(s['cohorts']['fresh_explosion'])}

{ptable(s['cohorts']['fresh_explosion'])}

Where each method helps on fresh: component caching cuts nominal base capacity {fresh['mdd_forest']['base_bits']/fresh['mdd_components']['base_bits']:.1f}× and messages {fresh['mdd_forest']['messages']/fresh['mdd_components']['messages']:.1f}× against the full forest, and was faster on all 12 roots (about {s['cohorts']['fresh_explosion']['paired'][0]['ratio_a_over_b']:.2f}× the time) because the six-vertex residual blocks are independent and each distinct block signature is calibrated once instead of once per product of signatures. Query memoization removes a further {fresh['mdd_components']['messages']-fresh['mdd_query_cache']['messages']:,} messages ({fresh['mdd_query_cache']['query_hits']} hits, {fresh['mdd_query_cache']['query_misses']} misses) but stores {fresh['mdd_query_cache']['query_bits']:,} extra bits; its total time is within noise of components ({s['cohorts']['fresh_explosion']['paired'][2]['ratio_a_over_b']:.2f}×, faster on {s['cohorts']['fresh_explosion']['paired'][2]['roots_where_a_faster']} of 12). `direct_tree` (width cap) and `flat_forest` (raw label cap) cap on every fresh root; DFS answers all but costs ~{fresh['dfs_cover']['total_seconds_sum_of_medians_ms']/1000:.0f} s in total versus about {fresh['mdd_components']['total_seconds_sum_of_medians_ms']/1000:.2f} s.

### Fresh families
""")
for k, blk in s["families"].items():
    if k.startswith("fresh_explosion/"):
        parts.append(f"\n**{k.split('/')[1]}**\n\n{table(blk)}\n")
parts.append("\n## Retrospective cohorts (50 roots, equal weight)\n")
for co, title in (("retro_capped", "20J capped (8 roots)"), ("retro_bipartite", "20K bipartite (18 roots)"), ("retro_mixed", "20L mixed (24 roots)")):
    parts.append(f"\n### {title}\n\n{table(s['cohorts'][co])}\n\n{ptable(s['cohorts'][co])}\n")
parts.append(f"""
### All 50 retrospective roots

{table(s['all_roots']['retro_all_50'])}

{ptable(s['all_roots']['retro_all_50'])}

Reading the retrospective result: the retrospective roots have almost one residual component each (components arm compiles 8 units for 8 capped roots, 18 for 18 bipartite, 37 for 24 mixed, versus 70 for 12 fresh roots), so splitting into components buys almost nothing — {retro['mdd_forest']['base_bits'] - retro['mdd_components']['base_bits']} bits out of {retro['mdd_forest']['base_bits']:,} and identical message counts. The component path still pays its own overhead: extra unit compilation and signature bookkeeping makes it slower than the full forest on the 20J capped cohort ({s['cohorts']['retro_capped']['paired'][0]['ratio_a_over_b']:.2f}× the time, a few milliseconds) and about equal elsewhere. Query memoization saves {retro['mdd_forest']['messages']-retro['mdd_query_cache']['messages']} messages through {retro['mdd_query_cache']['query_hits']} hits but stores {retro['mdd_query_cache']['query_bits']:,} extra nominal bits (about {retro['mdd_query_cache']['query_bits']/retro['mdd_forest']['base_bits']:.0%} of base capacity), a poor trade here; on the capped cohort it stores {s['cohorts']['retro_capped']['arms']['mdd_query_cache']['query_bits']:,} bits with 0 hits. Its per-root timing advantage on some cohorts is within run-to-run noise. DFS-cover is slower than every diagram arm in total on the bipartite and mixed cohorts (about 5× and 70×), because it searches again for every query; on the 20J capped roots it is comparable (the components arm is slower than DFS there). Setup dominates the diagram arms; query time is a few percent.

## Files

`summary.json` (machine-readable: cohort, family and row-level medians/spread/stats), `row_timings.csv` (row-level timing table), per-repetition `repeat-NN/` (`lbf20m/` raw receipts, `console.log`, `console.err`, `run.json` with exit status and peak memory, `receipt-audit.json`), `theory.log`, `environment.json`, `preflight.txt`, `attempt0-powershell-launcher/` (preliminary attempt kept: repeat-01 complete but exit status not captured; repeats 02/03 killed mid-run with `tasks.json` only, incomplete), scripts `launch_repeats.py`, `verify_local.py`, `aggregate.py`, `make_report.py`, and `SHA256SUMS.txt`.
""")
(R / "report" / "REPORT.md").write_text("".join(parts), encoding="utf-8")
print("report written", len("".join(parts)))
