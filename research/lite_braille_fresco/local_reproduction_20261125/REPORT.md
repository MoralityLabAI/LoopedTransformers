# LBF-20M local reproduction report

Source `40824b86e72da8fe54874ec9d9619d41b2706488` (tested commit of PR #383; merge commit `061aa65d…` not used). Development-only, standard-library CPU experiment; no training, no sealed or held-out data touched.

## Status

- **Completed repetitions: 5 of 5** (exit 0, fresh process each, run sequentially). Every repetition passes all 32 local receipt checks; no non-timing field differs between repetitions.
- **Not done (bundle missing):** `lbf20m-run-and-audit.zip` was not on this machine (searched `C:\Users\patri` and `D:\`). So `audit_20m_receipts.py`, `audit_20m_diagrams.py`, the inherited 20L receipts, the independent fresh reference and the original CI archive were unavailable. The following requirements are therefore **unverified**: 50 retrospective cases identical to inherited 20L cases; fresh answers equal to the separately generated structural reference; the largest-coherent-fixture diagram check (262,144 / 1,024 / 19 / 36 / 2); comparison with the original CI archive hash. `verify_local.py` stands in for the receipt checks that need only the run's own files and is less independent than the bundled verifier.
- **Deviation:** Python **3.12.x** was used instead of the registered 3.11 (3.11 not installed; 3.12 chosen by the user). `sys.flags.optimize == 0`, `PYTHONOPTIMIZE` cleared, `PYTHONHASHSEED=0` (see `preflight.txt`).
- Lean 4.19.0 (`--threads=1`, exit 0): all six declarations report no axioms (`theory.log`). These are abstract statements; the Python implementation is checked separately by the finite audit below.

## Environment

OS Microsoft Windows 11 Home 10.0.26200; CPU AMD Ryzen AI 9 365 w/ Radeon 880M               (10 cores / 20 threads); RAM 47.1 GiB; Lean Lean (version 4.19.0, x86_64-w64-windows-gnu, commit 6caaee842e94, Release); source worktree status clean (0 entries) at `git rev-parse HEAD` = 40824b86e72da8fe54874ec9d9619d41b2706488.
Invocation: `python -u research/lite_braille_fresco/lbf20m_factor.py --out <run-root>/repeat-NN/lbf20m`, driven by `launch_repeats.py`.

## Verification (every repetition)

Finite audit: 56 systems, 32,768 worlds, 1,944 queries, 5,832 support checks, 168 label-map comparisons, 6 cross-terminal, 12 boundary, 2 cap checks, 0 mismatches, 0 new training fits. Benchmark: 62 distinct roots; 372 unique root-by-arm rows; 298 exact, 74 cap rows (all `width_cap` for `direct_tree` ×62 and `raw_label_cap` for `flat_forest` ×12 fresh); 1,614 exact answer-set comparisons, every exact row equal to its oracle answers; 342 oracle queries including 96 fresh; cap rows keep reasons, zero queries and empty answers; the three diagram arms each answer all 62 roots; cohort and family aggregates recompute from rows; task digest `dbc74a0dae5bd1f5e1d2ca4079b07e256579caa07bc53fdad8c130601f088107` matches. No semantic mismatch occurred, so there is no root/arm/query to report.

## Recorded CI result as context vs this machine (fresh cohort, 12 roots)

| Arm | Fresh base bits (CI) | here | Fresh query-cache bits (CI) | here | Fresh messages (CI) | here |
|---|---:|---:|---:|---:|---:|---:|
| `mdd_forest` | 11,209,296 | 11,209,296 | 0 | 0 | 21,779 | 21,779 |
| `mdd_components` | 965,184 | 965,184 | 0 | 0 | 3,099 | 3,099 |
| `mdd_query_cache` | 965,184 | 965,184 | 225,396 | 225,396 | 1,849 | 1,849 |

The retrospective numbers from the original run also reproduce: component caching saves 16 nominal bits across the 50 retrospective roots (2,381,556 vs 2,381,540) with identical message counts (4,782 vs 4,782); query memoization stores 1,661,860 extra bits for 15 hits. Timings below are measured here and are independent of CI timings.

## Measurement conditions

- `base_bits` and `query_bits` are nominal capacity of calibrated bag tables and unary masks; they exclude Python objects, static tables, diagram storage and index structures. Reported separately and summed.
- Fresh cutsets were supplied (structural selector cutsets); fresh planning time is not measured (0). The common retrospective profile-greedy planner is timed separately: 16.4 / 160.9 / 177.0 ms (median across repetitions, summed over roots) for capped / bipartite / mixed, excluded from arm timings.
- Timings: per root and arm, median across the 5 repetitions of setup, query and total seconds; spread (min/max) is in `row_timings.csv` and `summary.json`. Tables show the sum of per-root medians, and timing comparisons use only roots completed by both arms. Arm order rotates per root inside the experiment. The oracle and witness audit are excluded from arm times.
- Process wall time per repetition (includes audit, planner, DFS oracle, all arms): 599, 585, 539, 567, 657 s. The preliminary attempt-0 repeat-01 took 374 s with the same code. The spread (374 to 657 s) was not isolated to a cause; the laptop is shared with other jobs (the GPU was near full) and repetition 1 overlapped a filesystem search, so absolute wall times and small per-root timings should be read with that noise in mind. Whole-process peak working set (not per arm): 31.1–31.4 MiB.
- Fresh fixtures are deliberate structural stress cases, not representative or lineage-disjoint generalization. No claim of automatic solver speedup from a supplied cutset.

## Fresh explosion cohort (12 roots)

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 12/12 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 206972.8 (18.4 + 206936.3) |
| `direct_tree` | 0/12 | width_cap ×12 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 0/12 | raw_label_cap ×12 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `mdd_forest` | 12/12 | - | 11,209,296 | 0 | 11,209,296 | 21,779 | 0 / 980 | 1008 / 2176 | 17616 / 392 | 588.5 (510.9 + 76.0) |
| `mdd_components` | 12/12 | - | 965,184 | 0 | 965,184 | 3,099 | 0 / 980 | 1008 / 2176 | 17616 / 392 | 285.0 (216.7 + 65.9) |
| `mdd_query_cache` | 12/12 | - | 965,184 | 225,396 | 1,190,580 | 1,849 | 916 / 64 | 1008 / 2176 | 17616 / 392 | 276.0 (227.2 + 50.0) |

| A vs B (roots both exact) | roots | Σ median total ms A | B | A/B | roots A faster | base+query bits A | B | messages A | B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mdd_components` vs `mdd_forest` | 12 | 285.0 | 588.5 | 0.48 | 12 | 965,184+0 | 11,209,296+0 | 3,099 | 21,779 |
| `mdd_query_cache` vs `mdd_forest` | 12 | 276.0 | 588.5 | 0.47 | 12 | 965,184+225,396 | 11,209,296+0 | 1,849 | 21,779 |
| `mdd_query_cache` vs `mdd_components` | 12 | 276.0 | 285.0 | 0.97 | 6 | 965,184+225,396 | 965,184+0 | 1,849 | 3,099 |
| `mdd_forest` vs `dfs_cover` | 12 | 588.5 | 206972.8 | 0.00 | 12 | 11,209,296+0 | 0+0 | 21,779 | 0 |
| `mdd_components` vs `dfs_cover` | 12 | 285.0 | 206972.8 | 0.00 | 12 | 965,184+0 | 0+0 | 3,099 | 0 |
| `mdd_query_cache` vs `dfs_cover` | 12 | 276.0 | 206972.8 | 0.00 | 12 | 965,184+225,396 | 0+0 | 1,849 | 0 |

Where each method helps on fresh: component caching cuts nominal base capacity 11.6× and messages 7.0× against the full forest, and was faster on all 12 roots (about 0.48× the time) because the six-vertex residual blocks are independent and each distinct block signature is calibrated once instead of once per product of signatures. Query memoization removes a further 1,250 messages (916 hits, 64 misses) but stores 225,396 extra bits; its total time is within noise of components (0.97×, faster on 6 of 12). `direct_tree` (width cap) and `flat_forest` (raw label cap) cap on every fresh root; DFS answers all but costs ~207 s in total versus about 0.29 s.

### Fresh families

**coherent_alias**

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 4/4 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 124154.1 (7.6 + 124129.1) |
| `direct_tree` | 0/4 | width_cap ×4 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 0/4 | raw_label_cap ×4 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `mdd_forest` | 4/4 | - | 329,040 | 0 | 329,040 | 635 | 0 / 20 | 64 / 120 | 1920 / 8 | 167.2 (159.7 + 6.8) |
| `mdd_components` | 4/4 | - | 329,040 | 0 | 329,040 | 635 | 0 / 20 | 64 / 120 | 1920 / 8 | 81.5 (73.9 + 6.6) |
| `mdd_query_cache` | 4/4 | - | 329,040 | 65,904 | 394,944 | 624 | 4 / 16 | 64 / 120 | 1920 / 8 | 90.9 (83.5 + 7.5) |

**coupled_modes**

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 4/4 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 20624.8 (5.5 + 20619.0) |
| `direct_tree` | 0/4 | width_cap ×4 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 0/4 | raw_label_cap ×4 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `mdd_forest` | 4/4 | - | 4,913,664 | 0 | 4,913,664 | 9,592 | 0 / 480 | 472 / 1028 | 7848 / 192 | 174.6 (148.4 + 27.8) |
| `mdd_components` | 4/4 | - | 394,848 | 0 | 394,848 | 1,352 | 0 / 480 | 472 / 1028 | 7848 / 192 | 103.7 (76.0 + 26.6) |
| `mdd_query_cache` | 4/4 | - | 394,848 | 88,128 | 482,976 | 760 | 448 / 32 | 472 / 1028 | 7848 / 192 | 94.8 (74.1 + 21.1) |

**independent_modes**

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 4/4 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 62193.9 (5.3 + 62188.2) |
| `direct_tree` | 0/4 | width_cap ×4 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 0/4 | raw_label_cap ×4 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `mdd_forest` | 4/4 | - | 5,966,592 | 0 | 5,966,592 | 11,552 | 0 / 480 | 472 / 1028 | 7848 / 192 | 246.6 (202.7 + 41.3) |
| `mdd_components` | 4/4 | - | 241,296 | 0 | 241,296 | 1,112 | 0 / 480 | 472 / 1028 | 7848 / 192 | 99.9 (66.9 + 32.7) |
| `mdd_query_cache` | 4/4 | - | 241,296 | 71,364 | 312,660 | 465 | 464 / 16 | 472 / 1028 | 7848 / 192 | 90.4 (69.7 + 21.5) |

## Retrospective cohorts (50 roots, equal weight)

### 20J capped (8 roots)

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 8/8 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 46.5 (1.6 + 44.9) |
| `direct_tree` | 0/8 | width_cap ×8 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 8/8 | - | 71,420 | 0 | 71,420 | 333 | 0 / 0 | 0 / 0 | 16 / 0 | 121.0 (118.6 + 2.5) |
| `mdd_forest` | 8/8 | - | 71,420 | 0 | 71,420 | 333 | 0 / 24 | 28 / 20 | 16 / 16 | 25.2 (22.8 + 2.4) |
| `mdd_components` | 8/8 | - | 71,420 | 0 | 71,420 | 333 | 0 / 24 | 28 / 20 | 16 / 16 | 52.3 (49.9 + 2.4) |
| `mdd_query_cache` | 8/8 | - | 71,420 | 192,180 | 263,600 | 333 | 0 / 24 | 28 / 20 | 16 / 16 | 18.6 (16.3 + 2.3) |

| A vs B (roots both exact) | roots | Σ median total ms A | B | A/B | roots A faster | base+query bits A | B | messages A | B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mdd_components` vs `mdd_forest` | 8 | 52.3 | 25.2 | 2.08 | 6 | 71,420+0 | 71,420+0 | 333 | 333 |
| `mdd_query_cache` vs `mdd_forest` | 8 | 18.6 | 25.2 | 0.74 | 5 | 71,420+192,180 | 71,420+0 | 333 | 333 |
| `mdd_query_cache` vs `mdd_components` | 8 | 18.6 | 52.3 | 0.36 | 5 | 71,420+192,180 | 71,420+0 | 333 | 333 |
| `mdd_forest` vs `dfs_cover` | 8 | 25.2 | 46.5 | 0.54 | 6 | 71,420+0 | 0+0 | 333 | 0 |
| `mdd_components` vs `dfs_cover` | 8 | 52.3 | 46.5 | 1.12 | 5 | 71,420+0 | 0+0 | 333 | 0 |
| `mdd_query_cache` vs `dfs_cover` | 8 | 18.6 | 46.5 | 0.40 | 6 | 71,420+192,180 | 0+0 | 333 | 0 |
| `flat_forest` vs `mdd_forest` | 8 | 121.0 | 25.2 | 4.81 | 1 | 71,420+0 | 71,420+0 | 333 | 333 |

### 20K bipartite (18 roots)

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 18/18 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1601.2 (10.6 + 1589.2) |
| `direct_tree` | 0/18 | width_cap ×18 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 18/18 | - | 2,011,820 | 0 | 2,011,820 | 2,119 | 0 / 0 | 0 / 0 | 144 / 0 | 323.0 (306.0 + 17.0) |
| `mdd_forest` | 18/18 | - | 2,011,820 | 0 | 2,011,820 | 2,119 | 0 / 65 | 189 / 191 | 144 / 109 | 302.8 (293.3 + 9.2) |
| `mdd_components` | 18/18 | - | 2,011,820 | 0 | 2,011,820 | 2,119 | 0 / 65 | 189 / 191 | 144 / 109 | 301.3 (292.4 + 8.6) |
| `mdd_query_cache` | 18/18 | - | 2,011,820 | 913,740 | 2,925,560 | 2,084 | 9 / 56 | 189 / 191 | 144 / 109 | 304.7 (295.8 + 8.5) |

| A vs B (roots both exact) | roots | Σ median total ms A | B | A/B | roots A faster | base+query bits A | B | messages A | B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mdd_components` vs `mdd_forest` | 18 | 301.3 | 302.8 | 1.00 | 11 | 2,011,820+0 | 2,011,820+0 | 2,119 | 2,119 |
| `mdd_query_cache` vs `mdd_forest` | 18 | 304.7 | 302.8 | 1.01 | 11 | 2,011,820+913,740 | 2,011,820+0 | 2,084 | 2,119 |
| `mdd_query_cache` vs `mdd_components` | 18 | 304.7 | 301.3 | 1.01 | 11 | 2,011,820+913,740 | 2,011,820+0 | 2,084 | 2,119 |
| `mdd_forest` vs `dfs_cover` | 18 | 302.8 | 1601.2 | 0.19 | 18 | 2,011,820+0 | 0+0 | 2,119 | 0 |
| `mdd_components` vs `dfs_cover` | 18 | 301.3 | 1601.2 | 0.19 | 18 | 2,011,820+0 | 0+0 | 2,119 | 0 |
| `mdd_query_cache` vs `dfs_cover` | 18 | 304.7 | 1601.2 | 0.19 | 18 | 2,011,820+913,740 | 0+0 | 2,084 | 0 |
| `flat_forest` vs `mdd_forest` | 18 | 323.0 | 302.8 | 1.07 | 6 | 2,011,820+0 | 2,011,820+0 | 2,119 | 2,119 |

### 20L mixed (24 roots)

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 24/24 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 12828.5 (12.7 + 12814.6) |
| `direct_tree` | 0/24 | width_cap ×24 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 24/24 | - | 298,316 | 0 | 298,316 | 2,322 | 0 / 0 | 0 / 0 | 336 / 0 | 181.1 (158.1 + 22.6) |
| `mdd_forest` | 24/24 | - | 298,316 | 0 | 298,316 | 2,330 | 0 / 128 | 148 / 244 | 336 / 61 | 186.4 (164.8 + 21.1) |
| `mdd_components` | 24/24 | - | 298,300 | 0 | 298,300 | 2,330 | 0 / 128 | 148 / 244 | 336 / 61 | 180.4 (161.5 + 19.0) |
| `mdd_query_cache` | 24/24 | - | 298,300 | 555,940 | 854,240 | 2,317 | 6 / 122 | 148 / 244 | 336 / 61 | 187.6 (167.8 + 19.7) |

| A vs B (roots both exact) | roots | Σ median total ms A | B | A/B | roots A faster | base+query bits A | B | messages A | B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mdd_components` vs `mdd_forest` | 24 | 180.4 | 186.4 | 0.97 | 13 | 298,300+0 | 298,316+0 | 2,330 | 2,330 |
| `mdd_query_cache` vs `mdd_forest` | 24 | 187.6 | 186.4 | 1.01 | 12 | 298,300+555,940 | 298,316+0 | 2,317 | 2,330 |
| `mdd_query_cache` vs `mdd_components` | 24 | 187.6 | 180.4 | 1.04 | 10 | 298,300+555,940 | 298,300+0 | 2,317 | 2,330 |
| `mdd_forest` vs `dfs_cover` | 24 | 186.4 | 12828.5 | 0.01 | 24 | 298,316+0 | 0+0 | 2,330 | 0 |
| `mdd_components` vs `dfs_cover` | 24 | 180.4 | 12828.5 | 0.01 | 24 | 298,300+0 | 0+0 | 2,330 | 0 |
| `mdd_query_cache` vs `dfs_cover` | 24 | 187.6 | 12828.5 | 0.01 | 24 | 298,300+555,940 | 0+0 | 2,317 | 0 |
| `flat_forest` vs `mdd_forest` | 24 | 181.1 | 186.4 | 0.97 | 15 | 298,316+0 | 298,316+0 | 2,322 | 2,330 |

### All 50 retrospective roots

| Arm | Exact / roots | Cap reasons | base_bits | query_bits | sum | messages | cache hits / misses | states / edges | labels / terminals | Σ median total ms (setup + query) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dfs_cover` | 50/50 | - | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 14476.3 (24.9 + 14448.8) |
| `direct_tree` | 0/50 | width_cap ×50 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.0 (0.0 + 0.0) |
| `flat_forest` | 50/50 | - | 2,381,556 | 0 | 2,381,556 | 4,774 | 0 / 0 | 0 / 0 | 496 / 0 | 625.1 (582.7 + 42.1) |
| `mdd_forest` | 50/50 | - | 2,381,556 | 0 | 2,381,556 | 4,782 | 0 / 217 | 365 / 455 | 496 / 186 | 514.4 (481.0 + 32.6) |
| `mdd_components` | 50/50 | - | 2,381,540 | 0 | 2,381,540 | 4,782 | 0 / 217 | 365 / 455 | 496 / 186 | 534.1 (503.9 + 30.0) |
| `mdd_query_cache` | 50/50 | - | 2,381,540 | 1,661,860 | 4,043,400 | 4,734 | 15 / 202 | 365 / 455 | 496 / 186 | 510.9 (479.9 + 30.6) |

| A vs B (roots both exact) | roots | Σ median total ms A | B | A/B | roots A faster | base+query bits A | B | messages A | B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mdd_components` vs `mdd_forest` | 50 | 534.1 | 514.4 | 1.04 | 30 | 2,381,540+0 | 2,381,556+0 | 4,782 | 4,782 |
| `mdd_query_cache` vs `mdd_forest` | 50 | 510.9 | 514.4 | 0.99 | 28 | 2,381,540+1,661,860 | 2,381,556+0 | 4,734 | 4,782 |
| `mdd_query_cache` vs `mdd_components` | 50 | 510.9 | 534.1 | 0.96 | 26 | 2,381,540+1,661,860 | 2,381,540+0 | 4,734 | 4,782 |
| `mdd_forest` vs `dfs_cover` | 50 | 514.4 | 14476.3 | 0.04 | 48 | 2,381,556+0 | 0+0 | 4,782 | 0 |
| `mdd_components` vs `dfs_cover` | 50 | 534.1 | 14476.3 | 0.04 | 47 | 2,381,540+0 | 0+0 | 4,782 | 0 |
| `mdd_query_cache` vs `dfs_cover` | 50 | 510.9 | 14476.3 | 0.04 | 48 | 2,381,540+1,661,860 | 0+0 | 4,734 | 0 |
| `flat_forest` vs `mdd_forest` | 50 | 625.1 | 514.4 | 1.22 | 22 | 2,381,556+0 | 2,381,556+0 | 4,774 | 4,782 |

Reading the retrospective result: the retrospective roots have almost one residual component each (components arm compiles 8 units for 8 capped roots, 18 for 18 bipartite, 37 for 24 mixed, versus 70 for 12 fresh roots), so splitting into components buys almost nothing — 16 bits out of 2,381,556 and identical message counts. The component path still pays its own overhead: extra unit compilation and signature bookkeeping makes it slower than the full forest on the 20J capped cohort (2.08× the time, a few milliseconds) and about equal elsewhere. Query memoization saves 48 messages through 15 hits but stores 1,661,860 extra nominal bits (about 70% of base capacity), a poor trade here; on the capped cohort it stores 192,180 bits with 0 hits. Its per-root timing advantage on some cohorts is within run-to-run noise. DFS-cover is slower than every diagram arm in total on the bipartite and mixed cohorts (about 5× and 70×), because it searches again for every query; on the 20J capped roots it is comparable (the components arm is slower than DFS there). Setup dominates the diagram arms; query time is a few percent.

## Files

`summary.json` (machine-readable: cohort, family and row-level medians/spread/stats), `row_timings.csv` (row-level timing table), per-repetition `repeat-NN/` (`lbf20m/` raw receipts, `console.log`, `console.err`, `run.json` with exit status and peak memory, `receipt-audit.json`), `theory.log`, `environment.json`, `preflight.txt`, `attempt0-powershell-launcher/` (preliminary attempt kept: repeat-01 complete but exit status not captured; repeats 02/03 killed mid-run with `tasks.json` only, incomplete), scripts `launch_repeats.py`, `verify_local.py`, `aggregate.py`, `make_report.py`, and `SHA256SUMS.txt`.
