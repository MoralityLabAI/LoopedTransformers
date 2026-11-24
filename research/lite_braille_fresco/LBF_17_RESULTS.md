# LBF-17: Relational Abstract Interpretation and Provenance-Preserving Equivariant Fresco Results

**Date:** 2026-11-24 · **Disposition:** `ENGINEERING_PASS; ALL_FIVE_REGISTERED_SCREENS_PASSED (1A, 1B, 2A, 2B, 2C)` · Prospective calibration milestone (`SEED = 20261227`; sealed `ordinary` and `ood` cohorts of both LBF-16 and LBF-17 preserved strictly unopened).

[Mathematical theory & Galois hierarchy](LBF_17_RELATIONAL_ABSTRACT_INTERPRETATION.md) · [Standalone zero-axiom Lean 4.19.0 formalization](lean/LBF17AbstractInterp.lean) · [Pre-fit registration](LBF_17_REGISTRATION.md) · [Frozen LBF-16 baseline (`53,047`p)](lbf16_equivariant_fresco.py) · [LBF-17 implementation](lbf17_relational_fresco.py) · [Unit & regression tests (`30/30`)](tests/test_lbf17_relational_fresco.py) · Preserved in-repo archive [`results/lbf17/`](results/lbf17/manifest.json) (`18.81 MB`: 24 checkpoints across 8 architectures $\times$ 3 seeds, compressed continuous fresco tensor archive, abstraction diagnostics, 2,400+ calibration episode records, Braille glyph traces, illuminated manuscript SVG).

Suite manifest SHA-256 `a85f402f674423efd4967f6d3b2128450964a200a40f2b0d47f40ab2b7cfc8e2` · Teacher summary SHA-256 `fbec5397d67f53afaac3987e451b9eaf478c8e13c19a36924f4fb7aa16497901` · Lean formalization SHA-256 `1d34145d5747f25f50c991971406207bae3e3740d03acc9615e5767dafb96025` · Abstraction diagnostics SHA-256 `fbadd2a2314568fdf630151c4b9c8f564a7a7f30b3bb617e543154f930fcd4b0` · Raw fresco tensor archive SHA-256 `5822544eee025b5efba9731fc96ac7230738a7d7e983f10de0fd5cd3528942f5`.

---

## 1. Custody, Formal Lean 4.19.0 Verification, and Symmetry Invariants

- **Fresh disjoint lineage suite (`SEED = 20261227`):** Generated `144` unique canonical rewrite-system lineages (`64` train, `48` calibration, `16` sealed ordinary, `16` sealed OOD), cryptographically verified to have `0` lineage overlap against all `720` historical lineages from seeds `20261204`, `20261205`, `20261207`, `20261209`, `20261215`, and `20261221`. Both `sealed_outcomes_opened = false` and `lbf16_sealed_cohorts_preserved_unopened = true` were enforced throughout fitting and scoring.
- **Standalone zero-axiom Lean 4.19.0 formalization (`2,584` total compiled declarations, `0` axioms):**
  - **Relational abstract interpretation theory (`15/15` constructive theorems in [`lean/LBF17AbstractInterp.lean`](lean/LBF17AbstractInterp.lean)):** Compiled standalone in Lean 4.19.0 (`6caaee842e94`) with `#print axioms` reporting `'...' does not depend on any axioms` (zero `propext`, zero `Classical.choice`, zero `Quot.sound`) for `abstract_iter_soundness`, `best_transformer_soundness`, `galois_pos1`, `galois_adj2`, `galois_sub3`, `adj2_refines_pos1`, `sub3_refines_adj2`, `provenance_fibers_refine_merged`, `repaired_lbf16_horizon_support_soundness`, `nonbacktracking_provenance_soundness`, `cex1_concrete_component_exact` (`w_0 = "abc"`, rules `ab ↔ de`, `bc ↔ fa`, `df ↔ cc`, concrete component `{abc, dec, afa}`), `cex1_pos1_pass1_exact_pass2_spurious`, `cex1_relational_and_provenance_exact`, `cex2_adj2_spurious_sub3_and_prov_exact` (`w_0 = "abbc"`), and `cex3_sub3_loses_global_consistency_prov_exact` (`w_0 = "aebcd"`).
  - **Proof-search certificates (`2,569` witness theorems):** `64` training reference proofs, `24` counterfactual continuation proofs, **`2,433` closed calibration policy proofs at `B = 128`**, and all `48` calibration reference proofs compiled standalone and axiom-free in Lean 4.19.0 with `0` online tactic calls during search.
- **Verified algebraic equivariances & tight parameter matching (`30/30` unit tests):** Exact $\mathfrak{S}_6$ alphabet-permutation invariance (`atol = 1e-5`), exact $\mathfrak{S}_{m_0} \times \mathfrak{S}_{m_1}$ frontier-set equivariance (`atol = 1e-5`), exact zero-preserving support (`env_zero` freezes multi-hop support at $k=1$ without affecting $B_m$; `mat_zero` zeroes $B_m$ while preserving 3-hop relational envelope propagation), rational-saturation trellis exactness on Counterexample 1 (`abc` $\to$ `{abc, dec, afa}` with `cc` support strictly `0.0`), and `compute_nonrecurrent` single-pass FLOPs $\ge 1.0\times$ 3-pass `rel_prov_opened`. All 8 learned architectures lie within **`53,047..53,093` parameters** (`46`-parameter / `0.087%` spread around frozen LBF-16's `53,047` budget) and share identical per-seed training frame draw hashes (`107`, `109`, `113`).

---

## 2. Primary Calibration Results (`48` Roots, `B = 128`, `DEPTH = 16`)

| Condition | Params | Passes $K$ | Certified closures | Closure rate | Seed `107 / 109 / 113` | Mean pen. work | Seed `107 / 109 / 113` work | Model calls | Pair evals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`rel_prov_opened` (primary LBF-17)** | **53,078** | **3** | **144/144** | **100.00%** | **100.0 / 100.0 / 100.0%** | **29.15** | **29.83 / 29.15 / 28.48** | **910** | **35,533** |
| `sym_prov_readout` (exact `prov4_adj2` + $K=1$ readout) | 53,078 | 1 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 30.15 | 30.98 / 30.44 / 29.02 | 909 | 45,014 |
| `adj2_noprov_opened` (merged `adj2`, no provenance) | 53,078 | 3 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 30.22 | 30.92 / 30.73 / 29.00 | 935 | 42,878 |
| `rel_prov_mat_zero` ($B_m = 0$, intact env bank) | 53,078 | 3 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 30.38 | 29.94 / 30.98 / 30.21 | 939 | 39,495 |
| `rel_prov_mat_roll` ($B_m$ rolled, intact env bank) | 53,078 | 3 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 30.38 | 30.63 / 30.25 / 30.25 | 936 | 40,080 |
| `rel_prov_scramble_ribbon` (`ribbon_conv` scrambled) | 53,078 | 3 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 31.23 | 30.98 / 31.52 / 31.19 | 972 | 45,068 |
| `compute_nonrecurrent` (compute- & param-matched $K=1$) | 53,070 | 1 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 31.69 | 33.52 / 30.38 / 31.19 | 985 | 56,339 |
| `param_nonrecurrent` (param-matched $K=1$) | 53,093 | 1 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 31.83 | 31.38 / 32.44 / 31.69 | 991 | 55,491 |
| `rel_prov_standard` (`gate_bias = -6.0`) | 53,078 | 3 | 144/144 | 100.00% | 100.0 / 100.0 / 100.0% | 32.30 | 31.67 / 33.02 / 32.21 | 998 | 63,720 |
| `rel_prov_ablate_axial` (diagonal axial only) | 53,078 | 3 | 143/144 | 99.31% | 100.0 / 97.9 / 100.0% | 31.53 | 31.67 / 33.10 / 29.83 | 997 | 60,147 |
| `rel_prov_env_roll` (env banks rolled, $B_m$ intact) | 53,078 | 3 | 143/144 | 99.31% | 97.9 / 100.0 / 100.0% | 32.23 | 34.92 / 31.06 / 30.71 | 1,008 | 52,473 |
| `rel_prov_env_zero` (env banks zeroed, $B_m$ intact) | 53,078 | 3 | 143/144 | 99.31% | 97.9 / 100.0 / 100.0% | 34.07 | 34.88 / 34.46 / 32.88 | 1,051 | 33,008 |
| `lowrank_opened` (learned bilinear low-rank, no trellis) | 53,061 | 3 | 142/144 | 98.61% | 95.8 / 100.0 / 100.0% | 34.36 | 34.17 / 34.75 / 34.17 | 1,049 | 85,216 |
| **`lbf16_pos1_opened` (frozen LBF-16 baseline arch)** | **53,047** | **3** | **141/144** | **97.92%** | **100.0 / 97.9 / 95.8%** | **35.66** | **35.00 / 35.42 / 36.56** | **1,090** | **90,354** |
| `rel_prov_scramble_window` (rewrite-window scrambled) | 53,078 | 3 | 133/144 | 92.36% | 91.7 / 91.7 / 93.8% | 49.12 | 53.19 / 48.96 / 45.21 | 1,459 | 182,771 |
| **`graph_bridge` (strongest explicit heuristic)** | **0** | — | **39/48** | **81.25%** | — | **58.92** | — | **0** | **98,169** |
| `geom_mm` (explicit front-to-front geometric MM) | 0 | — | 39/48 | 81.25% | — | 59.10 | — | 0 | 93,753 |
| **`exact_lookahead` (charged 1-step exact set probe)** | **0** | — | **37/48** | **77.08%** | — | **71.67** | — | **0** | **231,814** |
| **`bidir` (canonical smaller-frontier BFS)** | **0** | — | **30/48** | **62.50%** | — | **84.52** | — | **0** | **0** |
| `astar` (symbolic unilateral A*) | 0 | — | 25/48 | 52.08% | — | 90.10 | — | 0 | 0 |
| `uniform` (5 pseudorandom fringe streams) | 0 | — | 122/240 | 50.83% | — | 95.04 | — | 0 | 0 |

### Root-paired bootstrap contrasts (`rel_prov_opened` minus comparator at `B = 128`, `48` calibration roots)

| Comparator | Closure $\Delta$ (pp) | Closure 95% CI | Penalized work $\Delta$ | Work 95% CI | Root wins / losses / ties |
|---|---:|---:|---:|---:|---:|
| `uniform` | **+49.17** | `[+38.33, +60.00]` | **-65.88** (`-69.3%`) | `[-72.28, -58.94]` | `35 / 0 / 13` |
| `astar` | **+47.92** | `[+33.33, +62.50]` | **-60.95** (`-67.6%`) | `[-69.58, -51.76]` | `23 / 0 / 25` |
| **`bidir`** | **+37.50** | **`[+25.00, +52.08]`** | **-55.37** (**`-65.5%`**) | **`[-64.31, -46.78]`** | **`18 / 0 / 30`** |
| **`exact_lookahead` (charged)** | **+22.92** | **`[+12.50, +35.42]`** | **-42.51** (**`-59.3%`**) | **`[-51.57, -34.15]`** | **`11 / 0 / 37`** |
| **`geom_mm`** | **+18.75** | **`[+8.33, +31.25]`** | **-29.95** (**`-50.7%`**) | **`[-38.58, -21.84]`** | **`9 / 0 / 39`** |
| **`graph_bridge`** | **+18.75** | **`[+8.33, +31.25]`** | **-29.76** (**`-50.5%`**) | **`[-38.67, -21.42]`** | **`9 / 0 / 39`** |
| `rel_prov_scramble_window` | **+7.64** | `[+1.39, +14.58]` | **-19.97** (`-40.6%`) | `[-26.31, -14.25]` | `5 / 0 / 43` |
| **`lbf16_pos1_opened` (LBF-16)** | **+2.08** | **`[0.00, +5.56]`** | **-6.51** (**`-18.3%`**) | **`[-10.99, -3.01]`** | **`2 / 0 / 46`** |
| `lowrank_opened` | **+1.39** | `[0.00, +3.47]` | **-5.21** (`-15.2%`) | `[-9.19, -1.94]` | `2 / 0 / 46` |
| `rel_prov_env_zero` | **+0.69** | `[0.00, +2.08]` | **-4.92** (`-14.4%`) | `[-8.20, -2.41]` | `1 / 0 / 47` |
| `rel_prov_standard` | 0.00 | `[0.00, 0.00]` | **-3.15** (`-9.7%`) | **`[-5.74, -1.08]`** | `0 / 0 / 48` |
| `rel_prov_env_roll` | **+0.69** | `[0.00, +2.08]` | **-3.08** (`-9.5%`) | **`[-5.40, -1.31]`** | `1 / 0 / 47` |
| **`param_nonrecurrent`** | 0.00 | `[0.00, 0.00]` | **-2.68** (**`-8.4%`**) | **`[-5.30, -0.72]`** | `0 / 0 / 48` |
| **`compute_nonrecurrent`** | 0.00 | `[0.00, 0.00]` | **-2.54** (**`-8.0%`**) | **`[-4.86, -0.67]`** | `0 / 0 / 48` |
| `rel_prov_ablate_axial` | **+0.69** | `[0.00, +2.08]` | **-2.38** (`-7.6%`) | **`[-5.16, -0.39]`** | `1 / 0 / 47` |
| `rel_prov_scramble_ribbon` | 0.00 | `[0.00, 0.00]` | **-2.08** (`-6.6%`) | **`[-4.21, -0.25]`** | `0 / 0 / 48` |
| `rel_prov_mat_zero` | 0.00 | `[0.00, 0.00]` | **-1.22** (`-4.0%`) | **`[-2.42, -0.23]`** | `0 / 0 / 48` |
| `rel_prov_mat_roll` | 0.00 | `[0.00, 0.00]` | **-1.22** (`-4.0%`) | `[-2.66, +0.01]` | `0 / 0 / 48` |
| `adj2_noprov_opened` | 0.00 | `[0.00, 0.00]` | **-1.06** (`-3.5%`) | **`[-2.40, -0.04]`** | `0 / 0 / 48` |
| `sym_prov_readout` | 0.00 | `[0.00, 0.00]` | -0.99 (`-3.3%`) | `[-2.43, +0.22]` | `0 / 0 / 48` |

---

## 3. Matched Compute-Budget Frontier (`B ∈ {16, 32, 64, 96, 128}`) and Actual Resource Accounting

All neural FLOPs are measured via PyTorch `FlopCounterMode` during evaluation; `exact_lookahead` charges every concrete successor generated during 1-step intersection probing (`work += 1`, including duplicates).

| Condition | $B=16$ | $B=32$ | $B=64$ | $B=96$ | $B=128$ | **5-Budget AUC** | MFLOPs / dec | Total GFLOPs ($B=128$) | Rule-window checks | Wall ms / ep ($B=128$) | Peak tensor KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`rel_prov_opened` ($K=3$)** | 11.81% | 65.97% | **97.92%** | **100.00%** | **100.00%** | **75.14%** | **150.3** | **136.80** | 27,332,700 | **1,713.2** | **4,096.9** |
| `adj2_noprov_opened` ($K=3$) | 11.81% | **68.06%** | 95.14% | 99.31% | **100.00%** | 74.86% | 126.4 | 118.15 | 15,524,964 | 1,607.7 | 2,832.4 |
| `rel_prov_mat_zero` ($K=3$) | 11.11% | 64.58% | 96.53% | **100.00%** | **100.00%** | 74.44% | 158.8 | 149.07 | 29,558,100 | 992.9 | 6,545.1 |
| **`compute_nonrecurrent` ($K=1$)** | 13.89% | 64.58% | 93.06% | 98.61% | **100.00%** | **74.03%** | **262.4** | **258.46** | 6,828,060 | 1,514.8 | **25,536.0** |
| `rel_prov_scramble_ribbon` ($K=3$) | 13.19% | 63.89% | 94.44% | 97.92% | **100.00%** | 73.89% | 166.7 | 162.08 | 31,771,100 | 1,207.9 | 5,182.8 |
| `rel_prov_ablate_axial` ($K=3$) | 11.11% | 66.67% | 92.36% | 99.31% | 99.31% | 73.75% | 186.4 | 185.83 | 34,700,000 | 1,385.2 | 5,781.9 |
| `rel_prov_mat_roll` ($K=3$) | 11.11% | 63.19% | 94.44% | **100.00%** | **100.00%** | 73.75% | 159.6 | 149.35 | 29,527,400 | 1,050.5 | 5,014.9 |
| `sym_prov_readout` (sym $K=3$ + $K=1$) | 10.42% | 63.89% | 93.75% | **100.00%** | **100.00%** | 73.61% | 26.1 | 23.73 | 8,369,452 | 696.7 | 5,016.0 |
| `rel_prov_standard` ($K=3$, `bias=-6`) | 12.50% | 63.19% | 93.75% | 97.92% | **100.00%** | 73.47% | 193.1 | 192.69 | 35,549,700 | 2,520.9 | 6,190.1 |
| **`param_nonrecurrent` ($K=1$)** | 13.89% | 62.50% | 90.97% | 97.22% | **100.00%** | **72.92%** | **54.7** | **54.25** | 6,826,060 | 665.3 | 5,640.1 |
| `rel_prov_env_roll` ($K=3$) | 10.42% | 64.58% | 90.97% | 98.61% | 99.31% | 72.78% | 179.8 | 181.25 | 34,996,400 | 1,345.6 | 5,640.1 |
| `lowrank_opened` ($K=3$) | **14.58%** | 62.50% | 91.67% | 95.14% | 98.61% | 72.50% | 156.1 | 163.76 | 11,601,828 | 1,056.5 | 3,687.2 |
| `rel_prov_env_zero` ($K=3$) | 11.81% | 59.03% | 91.67% | 97.22% | 99.31% | 71.81% | 159.6 | 167.76 | 35,778,000 | 1,378.7 | 5,124.9 |
| **`lbf16_pos1_opened` ($K=3$ LBF-16)** | 12.50% | 60.42% | 90.28% | 95.14% | 97.92% | **71.25%** | **167.3** | **182.39** | 1,779,108 | 1,390.1 | 3,445.3 |
| `rel_prov_scramble_window` ($K=3$) | 9.72% | 43.06% | 75.69% | 86.11% | 92.36% | 61.39% | 311.0 | 453.68 | 75,973,300 | 3,894.5 | 9,235.0 |
| **`graph_bridge`** | 10.42% | 37.50% | 62.50% | 75.00% | 81.25% | **53.33%** | 0.0 | 0.00 | 1,367,636 | 57.0 | 0.0 |
| `geom_mm` | 10.42% | 35.42% | 60.42% | 77.08% | 81.25% | 52.92% | 0.0 | 0.00 | 0 | 21.3 | 0.0 |
| **`exact_lookahead` (charged)** | 2.08% | 20.83% | 54.17% | 66.67% | 77.08% | **44.17%** | 0.0 | 0.00 | 0 | 19.5 | 3.1 |
| **`bidir`** | 2.08% | 14.58% | 39.58% | 52.08% | 62.50% | **34.17%** | 0.0 | 0.00 | 0 | 1.3 | 0.0 |
| `astar` | 2.08% | 10.42% | 33.33% | 47.92% | 52.08% | 29.17% | 0.0 | 0.00 | 0 | 1.8 | 0.0 |

**Key compute-matched findings:**
1. **`rel_prov_opened` outperforms `compute_nonrecurrent` despite using `42.7%` fewer FLOPs/decision (`150.3` vs `262.4` MFLOPs), `47.1%` fewer total episode GFLOPs (`136.8` vs `258.5`), and `84.0%` less peak tensor memory (`4.10 MB` vs `25.54 MB`):** Widening single-pass 1-hop convolutions to exceed 3-pass FLOPs cannot substitute for multi-hop relational abstract interpretation — at `B = 64`, `rel_prov_opened` leads `compute_nonrecurrent` by **`+4.86 pp`** (`97.92%` vs `93.06%`) and `param_nonrecurrent` by **`+6.94 pp`** (`97.92%` vs `90.97%`), while saving `-2.54` (`95% CI [-4.86, -0.67]`) and `-2.68` (`[-5.30, -0.72]`) penalized work at `B = 128`.
2. **Why relational abstract interpretation beats charged exact set lookahead (`+22.92 pp` at `B = 128`, `+43.75 pp` at `B = 64`):** `exact_lookahead` achieves high local precision per expansion (`340` expanded states vs `813` for `bidir`), but probing exact concrete 1-step sets consumes successor budget (`3,429` work across `48` episodes), starving multi-hop search at `B ∈ {16, 32, 64}`. The $4 \times 6 \times 6$ relational provenance fresco propagates conservative $3$-hop subterm reachability across both frontiers in **zero** concrete successor expansions.
3. **Efficiency gain over frozen LBF-16 (`lbf16_pos1_opened`):** Eliminating positionwise chimera bridges raises `B = 64` closure by **`+7.64 pp`** (`97.92%` vs `90.28%`), raises `B = 128` closure from `97.92%` to `100.00%`, cuts penalized work by **`-18.3%`** (`29.15` vs `35.66`, `[-10.99, -3.01]`), cuts pairwise frontier evaluations by **`-60.7%`** (`35,533` vs `90,354`), and reduces total episode neural GFLOPs by **`-25.0%`** (`136.80` vs `182.39`).

---

## 4. Relational Abstraction Error, Glyph Brightness Calibration, and Conditional Closure

### 4.1 Concrete $k = 3$ reachability vs relational hierarchy (`train`: `64` roots; `calibration`: `48` unseen roots)

| Abstract domain ($K = 3$ horizon) | Train recall (`fn`) | Train precision (`fp`) | Train false bridge rate | Calib recall (`fn`) | Calib precision (`fp`) | Calib FP reduction vs $\mathcal{A}_1$ | Calib false bridge rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sym_pos1` ($\mathcal{A}_1$ Cartesian $1$-gram) | 100.00% (`0`) | 68.99% (`1,724`) | 99.38% (`161/162`) | 100.00% (`0`) | 68.12% (`2,724`) | `0.00%` | 100.00% (`209/209`) |
| `neural_lbf16_pos1` (LBF-16 envelope) | 100.00% (`0`) | 68.99% (`1,724`) | 99.38% (`161/162`) | 100.00% (`0`) | 68.12% (`2,724`) | `0.00%` | 100.00% (`209/209`) |
| `sym_prov_pos1` ($1$-step branch prov $\otimes \mathcal{A}_1$) | 100.00% (`0`) | 82.53% (`812`) | 98.15% (`159/162`) | 100.00% (`0`) | 81.89% (`1,287`) | `-52.75%` | 98.56% (`206/209`) |
| `sym_adj2` ($\mathcal{A}_2$ adjacent $2$-gram) | 100.00% (`0`) | 96.00% (`160`) | 92.59% (`150/162`) | 100.00% (`0`) | 95.28% (`288`) | `-89.43%` | 90.43% (`189/209`) |
| `neural_adj2_noprov` (merged trellis) | 100.00% (`0`) | 96.00% (`160`) | 92.59% (`150/162`) | 100.00% (`0`) | 95.28% (`288`) | `-89.43%` | 90.43% (`189/209`) |
| **`sym_prov4_adj2` ($F_p=4$ prov $\otimes \mathcal{A}_2$)** | **100.00% (`0`)** | **98.87% (`44`)** | **90.74% (`147/162`)** | **100.00% (`0`)** | **99.05% (`56`)** | **`-97.94%`** | **86.12% (`180/209`)** |
| **`neural_rel_prov` (`rel_prov_opened` trellis)** | **100.00% (`0`)** | **98.87% (`44`)** | **90.74% (`147/162`)** | **100.00% (`0`)** | **99.05% (`56`)** | **`-97.94%`** | **86.12% (`180/209`)** |
| `sym_sub3` ($\mathcal{A}_3$ length-$3$ subterm) | 100.00% (`0`) | 99.61% (`15`) | 80.25% (`130/162`) | 100.00% (`0`) | 99.64% (`21`) | `-99.23%` | 74.16% (`155/209`) |

Rational saturation $\rho_\varepsilon(x) = x/(x + 10^{-4})$ with exact Viterbi max-product trellis boundary coupling makes the continuous neural support of `rel_prov_opened` **cell-for-cell identical in Boolean support** to exact symbolic `sym_prov4_adj2` (`100.00%` recall, `0` symbol or bridge false negatives, **`99.05%` precision**, cutting calibration false positives from `2,724` to `56` — a **`97.94%` reduction**, capturing `98.7%` of $\mathcal{A}_3$'s false-positive reduction at $6\times$ smaller per-position state than $6^3 = 216$).

### 4.2 Can fresco glyph brightness distinguish certified reachable subterms from $\mathcal{A}_1$-plausible chimeras?

| Split (`evaluated_plausible_subterms`) | Certified / Spurious count | `lbf16_pos1` mean brightness (cert vs spur) | `lbf16_pos1` AUROC | `adj2_noprov` AUROC | **`rel_prov` mean brightness (cert vs spur)** | **`rel_prov` continuous AUROC** | **`rel_prov` 4-LED quantized AUROC** | Within-`rel_prov`-supported AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train (`4,217` non-root $\mathcal{A}_1$ cells)** | `2,493` / `1,724` | `0.8657` vs `0.5776` ($1.50\times$) | `0.8597` | `0.9941` | **`0.9511` vs `0.0225` ($42.2\times$)** | **`0.9986`** | **`0.9872`** | **`0.9451`** |
| **Calibration (`6,580` non-root $\mathcal{A}_1$ cells)** | `3,856` / `2,724` | `0.8615` vs `0.6008` ($1.43\times$) | `0.8405` | `0.9921` | **`0.9498` vs `0.0181` ($52.6\times$)** | **`0.9985`** | **`0.9899`** | **`0.9287`** |

Even after 4-level Lite-Brite LED quantization (`0..3`), `rel_prov_opened` separates certified concrete-reachable subterms from $\mathcal{A}_1$-plausible chimeras at **`0.9899` AUROC**; even *inside* the `56` residual cells where `prov4_adj2` Boolean support remains nonzero, continuous gate-modulated brightness separates certified from spurious subterms at **`0.9287` AUROC** (`0.9451` on train).

### 4.3 Oracle action-ranking regret and proof closure stratified by root $\mathcal{A}_1$ abstraction error

- **Calibration single-frame oracle action-ranking regret** (mean utility gap vs exact bounded continuation teacher on non-tied calibration frames): `sym_prov_readout` `0.0519`, `compute_nonrecurrent` `0.0519`, `adj2_noprov_opened` `0.0553`, `lbf16_pos1_opened` `0.0578`, `rel_prov_standard` `0.0604`, `rel_prov_opened` `0.0621`, `lowrank_opened` `0.0638`, `param_nonrecurrent` `0.0697`. Notably, `sym_prov_readout` and `adj2_noprov_opened` achieve lower single-frame teacher imitation regret than `lbf16_pos1_opened` and `param_nonrecurrent`, while `rel_prov_opened` achieves the **lowest closed-loop trajectory work (`29.15`) and highest 5-budget AUC (`75.14%`)** because eliminating false multi-step bridges prevents off-distribution search detours over 10–30 sequential expansions.
- **Closure stratified by root $\mathcal{A}_1$ false-positive rate (median `pos1` FPR = `0.4579`; `25` low-error vs `23` high-error roots at `B = 128`):**
  - Classical & lookahead baselines degrade sharply on high-chimera roots: `bidir` drops from `68.00%` (`83.08` work) $\to$ **`56.52%`** (`86.09` work); `exact_lookahead` drops from `84.00%` (`68.40` work) $\to$ **`69.57%`** (`75.22` work); `geom_mm` drops from `84.00%` $\to$ **`78.26%`**.
  - **`rel_prov_opened` achieves `100.00%` closure on both strata**, cutting penalized work relative to `lbf16_pos1_opened` on high-chimera roots (`28.35` vs `32.03`, `-11.5%`) and eliminating `lbf16_pos1_opened`'s remaining failures on low-FPR/longer-diameter roots (`100.00%` & `29.89` work vs `96.00%` & `39.00` work, `-23.4%`).

---

## 5. Factored Causal Mechanism Separation and Continuous Tensor Probes

### 5.1 Five orthogonal causal interventions on `rel_prov_opened` (`95` multi-action probed calibration decisions)

| Factored causal intervention | Isolated mechanism | Top-action changes (`/95`) | Max logit $\Delta$ | 5-Budget AUC ($\Delta$ vs intact) | $B=64$ closure ($\Delta$) | $B=128$ pen. work ($\Delta$ vs `29.15`) |
|---|---|---:|---:|---:|---:|---:|
| **Intact `rel_prov_opened`** | Full relational provenance fresco | `0 / 95` | `0.000` | **`75.14%`** (`0.00 pp`) | **`97.92%`** (`0.00 pp`) | **`29.15`** (`0.00`) |
| 1. `scramble_window` | Local rewrite-window subword coordinates | **`63 / 95` (`66.3%`)** | **`7.359`** | `61.39%` (**`-13.75 pp`**) | `75.69%` (**`-22.22 pp`**) | `49.12` (**`+19.97 [+14.25, +26.31]`**) |
| 4a. `env_zero` | Multi-pass envelope banks $(B_w, B_{w,\text{bi}})=0$ ($B_m$ intact) | **`36 / 95` (`37.9%`)** | `1.766` | `71.81%` (**`-3.33 pp`**) | `91.67%` (**`-6.25 pp`**) | `34.07` (**`+4.92 [+2.41, +8.20]`**) |
| 4b. `env_roll` | Envelope banks rolled across positions ($B_m$ intact) | **`28 / 95` (`29.5%`)** | `2.889` | `72.78%` (**`-2.36 pp`**) | `90.97%` (**`-6.94 pp`**) | `32.23` (**`+3.08 [+1.31, +5.40]`**) |
| 3. `ablate_axial` | Cross-frontier axial partner competition & centering | **`29 / 95` (`30.5%`)** | `2.973` | `73.75%` (**`-1.39 pp`**) | `92.36%` (**`-5.56 pp`**) | `31.53` (**`+2.38 [+0.39, +5.16]`**) |
| 2. `scramble_ribbon` | 1D spatial `ribbon_conv` subterm continuity | **`41 / 95` (`43.2%`)** | `2.796` | `73.89%` (**`-1.25 pp`**) | `94.44%` (**`-3.47 pp`**) | `31.23` (**`+2.08 [+0.25, +4.21]`**) |
| 5b. `mat_roll` | Pairwise matrix bank $B_m$ rolled (env banks intact) | `16 / 95` (`16.8%`) | `1.321` | `73.75%` (**`-1.39 pp`**) | `94.44%` (**`-3.47 pp`**) | `30.38` (`+1.22 [-0.01, +2.66]`) |
| 5a. `mat_zero` | Pairwise matrix bank $B_m = 0$ (env banks intact) | `9 / 95` (`9.5%`) | `1.463` | `74.44%` (**`-0.69 pp`**) | `96.53%` (**`-1.39 pp`**) | `30.38` (**`+1.22 [+0.23, +2.42]`**) |

**Mechanistic decomposition:** Separating LBF-16's coarse spatial/memory shams reveals a clean quantitative hierarchy:
1. **Rewrite-window coordinate alignment (`scramble_window`: `-13.75 pp` AUC, `+19.97` work)** is the foundational grounding invariant.
2. **Within-decision envelope recurrence (`env_zero`: `-3.33 pp` AUC, `+4.92` work) vs matrix-bank recurrence (`mat_zero`: `-0.69 pp` AUC, `+1.22` work):** **`~80.1%`** (`4.92 / (4.92 + 1.22)`) of recurrent work savings come from multi-pass propagation of the relational provenance envelope $(B_w, B_{w,\text{bi}})$, while **`~19.9%`** come from iterative cross-frontier matrix refinement $B_m$.
3. **Cross-frontier axial set competition (`ablate_axial`: `-5.56 pp` at `B = 64`, `+2.38` work)** and **subterm ribbon continuity (`scramble_ribbon`: `-3.47 pp` at `B = 64`, `+2.08` work)** each contribute statistically significant (`95% CI` strictly excluding `0`) independent efficiency gains.
4. **Provenance fibers (`rel_prov_opened` vs `adj2_noprov_opened`) & trellis structure (`vs lowrank_opened`):** Adding $F_p = 4$ spatial-congruence provenance fibers cuts residual adjacent-bigram false positives by another **`80.6%`** (`288 -> 56`) and reduces `B = 128` penalized work by **`-1.06` (`95% CI [-2.40, -0.04]`)** while lifting `B = 64` closure from `95.14%` to `97.92%`. Explicit Viterbi bigram trellises outperform unstructured equivariant bilinear low-rank coupling (`lowrank_opened`) by **`+2.64 pp` AUC** and **`-5.21` work (`[-9.19, -1.94]`)**.

### 5.2 Continuous fresco tensor mechanics (`95` probed decisions per arm)

| Continuous fresco probe metric | `rel_prov_standard` (`bias = -6.0`) | `adj2_noprov_opened` (`bias = -1.5`) | **`rel_prov_opened` (`bias = -1.5`)** | Mechanistic meaning |
|---|---:|---:|---:|---|
| Pass-3 bigram envelope gate $\sigma(g_{\text{bi}})$ | `0.00585` | `0.23890` | **`0.23375`** ($39.9\times$) | Opens multi-hop $6 \times 6$ relational envelope |
| Pass-3 positional gate $\sigma(g_{\text{pos}})$ | `0.00518` | `0.31234` | **`0.27099`** ($52.3\times$) | Transmits calibrated marginal glyph brightness |
| Pass-3 pairwise fresco gate $\sigma(g_{\text{mat}})$ | `0.01615` | `0.19770` | **`0.19300`** ($11.9\times$) | Propagates axial partner competition across passes |
| Active $6 \times 6$ bigram support (Pass 1 $\to$ Pass 3) | `6.83% -> 16.72%` | `6.83% -> 19.08%` | **`6.79% -> 17.21%`** | Provenance fibers prune `~10%` chimera bigrams by Pass 3 |
| Effective rank: Fresco / Pos-env / Prov-bigram | `2.418 / 1.230 / 1.010` | `1.822 / 1.285 / 1.018` | **`1.586 / 1.324 / 1.026`** | Non-degenerate multi-mode fresco bank (`> 1.58`) |
| Subterm spatial Moran's $I$ (Pass 1 $\to$ Pass 3) | `+0.5398 -> +0.5202` | `+0.5222 -> +0.6272` | **`+0.5262 -> +0.6158`** | Opened gates *increase* spatial bridge coherence (`+0.090`); closed gates degrade (`-0.020`) |
| Inter-pass brightening $\mathbb{E}|\sigma(M^{(3)})-\sigma(M^{(1)})|$ | `0.06667` | `0.11403` | **`0.10552`** ($1.58\times$) | Multi-hop meeting pairs brighten selectively over passes |

---

## 6. Pre-Declared Screen Verdict (`5 / 5 PASS`)

1. **Primary Screen 1A (`rel_prov_opened` vs `bidir` & charged `exact_lookahead`): `PASS`** — `100.00%` (`100 / 100 / 100%`) vs `62.50%` `bidir` (`+37.50 pp >= +15.0 pp`, `-65.5%` work `>= 30%`, `18` wins / `0` losses) and vs `77.08%` charged `exact_lookahead` (`+22.92 pp` at `B = 128`, `75.14%` vs `44.17%` 5-budget AUC, `-59.3%` work, `11` wins / `0` losses).
2. **Primary Screen 1B (`rel_prov_opened` vs strongest explicit heuristic `graph_bridge` / `geom_mm`): `PASS`** — `100.00%` vs `81.25%` (`+18.75 pp >= +5.0 pp`, `-50.5%` work `>= 20%`, `9` wins / `0` losses, all 3 seeds `100% > 81.25%`, and `rel_prov_opened` strictly leads at every budget `B ∈ {16, 32, 64, 96, 128}`).
3. **Secondary Screen 2A (Multi-pass recurrence vs `param_nonrecurrent`, `compute_nonrecurrent`, & `rel_prov_standard`): `PASS`** — Higher 5-budget closure AUC (`75.14%` vs `72.92%`, `74.03%`, and `73.47%`) and statistically lower `B = 128` penalized work (`29.15` vs `31.83`, `31.69`, and `32.30`; all three `95%` bootstrap CIs strictly negative), while using `42.7%` fewer FLOPs/decision and `84.0%` less memory than `compute_nonrecurrent`.
4. **Secondary Screen 2B (Relational abstraction precision & glyph brightness AUROC): `PASS`** — `100.00%` concrete $3$-step recall (`fn = 0`), **`99.05%` precision (`>= 95%`)**, **`97.94%` false-positive reduction (`>= 90%`)** vs `lbf16_pos1_opened` (`56` vs `2,724` FPs), and **`0.9985` continuous / `0.9899` 4-LED quantized glyph AUROC (`>= 0.90`)**.
5. **Secondary Screen 2C (Five factored causal interventions): `PASS`** — Every single corruption (`scramble_window`, `scramble_ribbon`, `ablate_axial`, `env_zero`/`env_roll`, `mat_zero`/`mat_roll`) reduces 5-budget AUC (`61.39%..74.44% < 75.14%`) and increases `B = 128` penalized work (`30.38..49.12 > 29.15`) with `0` root losses for intact `rel_prov_opened`.
