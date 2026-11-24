# LBF-16: $\mathfrak{S}_6$-Equivariant Horizon-Separated Meeting Fresco Results

**Date:** 2026-11-24 · **Disposition:** `ENGINEERING_PASS; ALL_FOUR_REGISTERED_SCREENS_PASSED (1A, 1B, 2A, 2B)` · Development calibration milestone (sealed `ordinary`/`ood` cohorts preserved unopened for separate confirmation).

[Mathematical specification & registration](LBF_16_EQUIVARIANT_FRESCO_REGISTRATION.md) · [Prior LBF-15 falsifying diagnosis](LBF_15_MEETING_POTENTIAL_RESULTS.md) · Tested implementation `c8bc1c18` · Preserved in-repo archive [`results/lbf16/`](results/lbf16/manifest.json) (12.09 MB: 12 checkpoints, continuous fresco tensor archive, 1,560 episode records, Braille glyph traces, illuminated manuscript SVG).
Suite manifest SHA-256 `ed10fcf5bc72c9cc0f0ba7847fa5ebc11d9ec04c6a3a98a3b17c0ef316957375` · Teacher summary SHA-256 `23b67d208397c02a4c1b31ae8855f9f1e67286687149d7c0f80ff2bd7384826d` · Raw fresco tensor archive SHA-256 `e1953c1c67a5936f6be0ac86a25322fade6b18c8424d696e6765d70a3b2bcb97`.

---

## 1. Custody, Symmetry Invariants, and Standalone Lean 4.19.0 Certification

- **Untouched lineage-disjoint suite:** Fresh generator seed `20261221` produced `144` unique canonical rule-system lineages (`64` train, `48` calibration, `16` sealed ordinary, `16` sealed OOD), verified disjoint against all `576` historical lineages from `20261204`, `20261205`, `20261207`, `20261209`, and `20261215`.
- **Verified algebraic symmetries & budget matching:** All 25 unit tests passed, verifying exact $\mathfrak{S}_6$ alphabet-permutation invariance (`atol = 1e-5`), exact $\mathfrak{S}_{m_0}\times\mathfrak{S}_{m_1}$ frontier-set equivariance (`atol = 1e-5`), exact zero-mode support freezing ($\operatorname{supp}(B_w^{(3)}) = \operatorname{supp}(B_w^{(1)})$), Pass-1 sham identity, and multi-pass `roll` Frobenius-norm isometry. Parameter counts were matched within `60` parameters (`0.11%`) inside the `57,475` cap: `53,047` (`recurrent_opened`, `recurrent_standard`), `53,007` (`unstructured_recurrent`), and `53,067` (`nonrecurrent`).
- **Train-only teacher & Lean 4.19.0 certificates:** `64` training reference certificates and `24` counterfactual continuation certificates compiled standalone and axiom-free in Lean 4.19.0 (`6caaee842e94`). Across `605` training frontier frames (`489,903` oracle successor checks), `343` (`56.69%`) required multi-hop bridges ($d_R^* \ge 3$), `414` (`68.43%`) were non-tied, and `109` (`18.02%`) had mixed closure. On calibration (`48` roots), **`1,333` policy proof witnesses and all `48` calibration reference witnesses** compiled standalone and axiom-free in Lean 4.19.0 (`1,469` total Lean 4.19.0 certificates).

---

## 2. Primary Calibration Results (`48` roots, `WORK = 128`, `DEPTH = 16`)

| Condition | Params | Certified closures | Closure rate | Seed 107 / 109 / 113 | Mean pen. work | Total work | Model calls | Pair evals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`opened_normal` (primary)** | **53,047** | **138/144** | **95.83%** | **93.8 / 97.9 / 95.8%** | **33.94** | **4,881** | **1,044** | **106,765** |
| `opened_no_persist` | 53,047 | 138/144 | 95.83% | 95.8 / 95.8 / 95.8% | 33.34 | 4,795 | 1,024 | 97,106 |
| `opened_zero` (zeroed bank) | 53,047 | 137/144 | 95.14% | 95.8 / 91.7 / 97.9% | 33.88 | 4,872 | 1,047 | 90,310 |
| `nonrecurrent` ($K=1$) | 53,067 | 136/144 | 94.44% | 93.8 / 93.8 / 95.8% | 34.81 | 5,005 | 1,068 | 119,809 |
| `recurrent_standard` ($\text{bias}=-6$) | 53,047 | 136/144 | 94.44% | 93.8 / 95.8 / 93.8% | 35.22 | 5,064 | 1,065 | 129,021 |
| `unstructured_recurrent` | 53,007 | 136/144 | 94.44% | 95.8 / 93.8 / 93.8% | 35.47 | 5,099 | 1,071 | 130,088 |
| `opened_roll` (rolled bank) | 53,047 | 135/144 | 93.75% | 93.8 / 91.7 / 95.8% | 35.78 | 5,143 | 1,075 | 115,251 |
| `opened_scramble` (subterm scramble) | 53,047 | 132/144 | 91.67% | 93.8 / 89.6 / 91.7% | 46.81 | 6,729 | 1,404 | 191,606 |
| **`geom_mm` (strongest heuristic)** | **0** | **40/48** | **83.33%** | — | **66.94** | **3,205** | **0** | **110,849** |
| `graph_bridge` (explicit graph) | 0 | 39/48 | 81.25% | — | 63.25 | 3,027 | 0 | 95,726 |
| **`bidir` (symbolic smaller-layer)** | **0** | **28/48** | **58.33%** | — | **85.56** | **4,087** | **0** | **0** |
| `uniform` (5 fringe streams) | 0 | 116/240 | 48.33% | — | 94.23 | 22,490 | 0 | 0 |
| `astar` (symbolic A*) | 0 | 22/48 | 45.83% | — | 90.77 | 4,331 | 0 | 0 |

### Root-paired contrasts (`opened_normal` minus comparator at `B = 128`)
| Comparator | Closure $\Delta$ (pp) | Closure 95% bootstrap | Penalized work $\Delta$ | Work 95% bootstrap | Root wins / losses / ties |
|---|---:|---:|---:|---:|---:|
| `astar` | **+50.00** | [+36.11, +63.89] | **-56.83** (-62.6%) | [-67.15, -46.27] | 25 / 0 / 23 |
| `uniform` | **+47.50** | [+36.94, +57.64] | **-60.29** (-64.0%) | [-68.18, -52.11] | 36 / 0 / 12 |
| `bidir` | **+37.50** | [+24.31, +50.69] | **-51.63** (-60.3%) | [-61.44, -41.60] | 19 / 0 / 29 |
| `graph_bridge` | **+14.58** | [+5.56, +25.00] | **-29.31** (-46.3%) | [-37.08, -21.38] | 8 / 0 / 40 |
| `geom_mm` | **+12.50** | [+4.17, +22.22] | **-33.00** (-49.3%) | [-41.55, -24.43] | 7 / 0 / 41 |
| `opened_scramble` | **+4.17** | [+0.69, +8.33] | **-12.88** (-27.5%) | [-18.40, -8.12] | 5 / 0 / 43 |
| `opened_roll` | **+2.08** | [0.00, +5.56] | **-1.84** (-5.1%) | [-3.39, -0.60] | 2 / 0 / 46 |
| `unstructured_recurrent` | **+1.39** | [-1.39, +4.86] | **-1.53** (-4.3%) | [-4.23, +0.49] | 2 / 1 / 45 |
| `recurrent_standard` | **+1.39** | [-1.39, +4.86] | **-1.28** (-3.6%) | [-3.44, +0.25] | 2 / 1 / 45 |
| `nonrecurrent` | **+1.39** | [-1.39, +4.86] | **-0.88** (-2.5%) | [-2.45, +0.44] | 2 / 1 / 45 |
| `opened_zero` | **+0.69** | [0.00, +2.08] | +0.06 (+0.2%) | [-1.10, +1.31] | 1 / 0 / 47 |
| `opened_no_persist` | 0.00 | [-2.08, +2.08] | +0.60 (+1.8%) | [+0.04, +1.27] | 1 / 1 / 46 |

---

## 3. Matched Compute-Budget Frontier (`B ∈ {32, 64, 96, 128}`, Zero Refitting)

| Condition | $B = 32$ | $B = 64$ | $B = 96$ | $B = 128$ | Mean Closure AUC |
|---|---:|---:|---:|---:|---:|
| **`opened_normal`** | **65.28%** | 90.28% | **93.75%** | **95.83%** | **86.28%** |
| `opened_no_persist` | **65.97%** | 91.67% | 93.06% | **95.83%** | **86.63%** |
| `nonrecurrent` ($K=1$) | 63.89% | **92.36%** | 93.06% | 94.44% | 85.94% |
| `opened_zero` | 60.42% | 91.67% | 93.06% | 95.14% | 85.07% |
| `recurrent_standard` | 63.19% | 88.89% | 92.36% | 94.44% | 84.72% |
| `unstructured_recurrent` | 59.72% | 89.58% | **93.75%** | 94.44% | 84.38% |
| `opened_roll` | 57.64% | 90.97% | 93.06% | 93.75% | 83.85% |
| `opened_scramble` | 48.61% | 76.39% | 88.19% | 91.67% | 76.22% |
| `graph_bridge` | 33.33% | 56.25% | 75.00% | 81.25% | 61.46% |
| `geom_mm` | 33.33% | 56.25% | 68.75% | 83.33% | 60.42% |
| `bidir` | 18.75% | 37.50% | 47.92% | 58.33% | 40.63% |
| `astar` | 16.67% | 35.42% | 41.67% | 45.83% | 34.90% |

---

## 4. Continuous Fresco Tensor Mechanics and Causal Attribution

| Continuous boundary / fresco metric (`90` probed decisions/arm) | `recurrent_standard` ($\text{bias}=-6$) | `recurrent_opened` ($\text{bias}=-1.5$) | Causal interpretation |
|---|---:|---:|---|
| Pass-3 envelope gate mean $\sigma(g_{\text{env}})$ | `0.00273` | **`0.25248`** ($92.3\times$) | Opens multi-hop $\mathfrak{S}_6$-equivariant rewrite envelope |
| Pass-3 fresco gate mean $\sigma(g_{\text{mat}})$ | `0.03310` | **`0.21428`** ($6.5\times$) | Transmits cross-frontier axial competition across passes |
| Active envelope support fraction (Pass 1 $\to$ Pass 3) | `15.25% -> 53.62%` | **`15.31% -> 67.07%`** | Multi-step reachable symbol support expands $4.38\times$ |
| Effective rank: Fresco bank / Envelope bank | `2.468` / `1.132` | **`2.660` / `1.458`** | Eliminates LBF-14's rank-1 slot collapse ($\operatorname{erank}=1.07$) |
| Subterm spatial Moran's $I$ along $p=0..L-1$ (Pass 1 $\to$ Pass 3) | `+0.3534 -> +0.3898` | **`+0.3697 -> +0.5795`** | Recurrent passes form spatially coherent subword bridge islands |
| Mean inter-pass potential brightening $\mathbb{E}|\sigma(M^{(3)})-\sigma(M^{(1)})|$ | `0.06069` | **`0.09307`** ($1.53\times$) | Stronger recurrent refinement on multi-hop pairs |
| Top-action changes under `zero` / `roll` / `scramble` | `17` / `15` / `42` | **`32` / `24` / `40`** (of `90`) | High causal reliance on intact envelope & subterm geometry |

### Summary of pre-declared screen outcomes and where improvement comes from
1. **Primary Screen 1A (`opened_normal` vs symbolic `bidir`): PASS** (`+37.50 pp` closure, `-60.34%` penalized work, `19` wins / `0` losses, all 3 seeds `93.8%–97.9%` > `58.3%`).
2. **Primary Screen 1B (`opened_normal` vs strongest explicit heuristic `geom_mm`): PASS** (`+12.50 pp` closure, `-49.30%` penalized work, `7` wins / `0` losses, all 3 seeds `93.8%–97.9%` > `83.3%`; also `+14.58 pp` and `-46.34%` work vs `graph_bridge`).
3. **Secondary Screen 2A (Recurrence & Structure): PASS** (`opened_normal` beats `nonrecurrent`, `recurrent_standard`, and `unstructured_recurrent` on both $B=128$ closure `95.83%` vs `94.44%`, lower penalized work `33.94` vs `34.81–35.47`, and higher budget-curve AUC `86.28%` vs `84.38%–85.94%`, with largest recurrent advantage at tight budget $B=32$: `+5.56 pp` over `unstructured_recurrent` and `+2.08 pp` over `recurrent_standard`).
4. **Secondary Screen 2B (Causal Memory & Spatial Coherence): PASS** (`opened_normal` beats `opened_zero`, `opened_roll`, and `opened_scramble` on both $B=128$ closure and budget-curve AUC: at $B=32$, zeroing the message bank costs `-4.86 pp`, rolling costs `-7.64 pp`, and scrambling subterm positions costs `-16.67 pp`; at $B=128$, `opened_scramble` loses `-4.17 pp` and `+12.88` penalized work with `0` wins / `5` losses, completely eliminating LBF-15's sibling-blurring inversion).
5. **Honest negative finding on cross-step scalar persistence:** `opened_no_persist` tied `opened_normal` (`95.83%`, `33.34` vs `33.94` work; `86.63%` vs `86.28%` AUC), demonstrating that **within-decision multi-pass $\mathfrak{S}_6$-equivariant rewrite propagation, subterm spatial continuity ($I: +0.37 \to +0.58$), and feasibility-masked axial set competition** drive the algorithmic advantage without needing scalar carry-over from $t-1$.
