# LBF-15: Geometric Meeting-Potential Fresco Results

**Date:** 2026-11-24 · **Disposition:** `ENGINEERING_PASS; PRIMARY_1A_VS_BIDIR_PASSED (+11.11 pp, -31.6% WORK); HEURISTIC_1B_AND_MECHANISM_SCREENS_FALSIFIED` · Development only.

[Mathematical specification](LBF_15_MATHEMATICAL_SPECIFICATION.md) · [Pre-scoring registration](LBF_15_MEETING_POTENTIAL_REGISTRATION.md) · Tested implementation `c98cc997` · Preserved in-repo archive [`results/lbf15/`](results/lbf15/manifest.json) (8.97 MB, 12 checkpoints + raw fresco tensors + 1,560 episode records + manuscript SVG).
Suite manifest SHA-256 `510307a206245c6ce3ab3f856e46fb4ba0d985ff621215f512ce932dbf26796f` · Teacher summary SHA-256 `3b79b88009839c40886514efa42e0f071c46d5b91bf2507313e07a89014ea778` · Raw fresco tensor archive SHA-256 `1f3bf37412b498a6f2b2a848c5103783c1e23581e6c35861f81a3a00d462469d`.

---

## 1. Custody and Counterfactual Marginal-Utility Census

- **Suite & lineage custody:** Fresh generator seed `20261215` produced `144` unique canonical rule-system lineages (`64` train, `48` calibration, `16` sealed ordinary, `16` sealed OOD), verified disjoint against all `432` historical lineages from `20261204`, `20261205`, `20261207`, and `20261209`. Sealed `ordinary` and `ood` files were never opened.
- **Train-only teacher:** `64` axiom-free training reference certificates and `24` axiom-free counterfactual continuation certificates (`12` equality, `12` dependent transport) compiled standalone in Lean 4.19.0 (`6caaee842e94`). Across `658` open-frontier training states (`740,461` oracle successor checks; `42,371` matrix pairs, `26,705` positive $\le 6$-hop bridges), **`478/658` frames (`72.64%`) were non-tied** and **`126` (`19.15%`) exhibited mixed closure** (versus `24.68%` non-tied and `2.13%` mixed in LBF-14's whole-layer formulation), yielding `1,813` positive-marginal-utility and `3,953` negative-marginal-utility node actions.
- **Calibration certificates:** Across `1,560` calibration episodes (`48` roots × `32.5` effective tracks), **`1,341` policy proof witnesses and all `48` calibration reference witnesses** compiled standalone and axiom-free in Lean 4.19.0 (`1,477` total Lean 4.19.0 certificates across the campaign). All 12 checkpoints matched exact draw schedules per seed (`31f3b370...`, `db0cf687...`, `4e52bf6c...`) and exact pre-bias initializations between `recurrent_standard` and `recurrent_opened`.

---

## 2. Frozen Calibration Search (`48` roots, `WORK = 128`, `DEPTH = 16`)

| Condition | Params | Certified closures | Closure rate | Seed 107 / 109 / 113 | Mean pen. work | Total work | Model calls | Pair evals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `opened_scramble` (spatial sham) | 55,859 | 134/144 | **93.06%** | 95.8 / 93.8 / 89.6% | 41.68 | 5,992 | 1,182 | 53,516 |
| `unstructured_recurrent` | 55,865 | 133/144 | **92.36%** | 91.7 / 95.8 / 89.6% | 43.18 | 6,207 | 1,197 | 34,139 |
| `recurrent_standard` ($\text{bias}=-6$) | 55,859 | 132/144 | **91.67%** | 91.7 / 93.8 / 89.6% | **39.53** | 5,681 | 1,096 | 37,766 |
| `opened_zero` (zeroed bank) | 55,859 | 131/144 | **90.97%** | 91.7 / 93.8 / 87.5% | 41.69 | 5,990 | 1,163 | 42,456 |
| `opened_no_persist` | 55,859 | 128/144 | 88.89% | 93.8 / 87.5 / 85.4% | 43.47 | 6,244 | 1,215 | 54,575 |
| **`opened_normal` (primary)** | **55,859** | **127/144** | **88.19%** | **89.6 / 89.6 / 85.4%** | **44.15** | **6,340** | **1,232** | **61,835** |
| `opened_roll` (rolled bank) | 55,859 | 127/144 | 88.19% | 91.7 / 89.6 / 83.3% | 43.65 | 6,268 | 1,227 | 54,122 |
| `nonrecurrent` ($K=1$) | 55,941 | 127/144 | 88.19% | 87.5 / 91.7 / 85.4% | 44.84 | 6,440 | 1,259 | 30,293 |
| `graph_bridge` (explicit graph) | 0 | 42/48 | 87.50% | — | 44.75 | 2,142 | 0 | 72,485 |
| `geom_mm` (explicit geometric) | 0 | 42/48 | 87.50% | — | 46.71 | 2,236 | 0 | 74,276 |
| **`bidir` (symbolic smaller-layer)** | **0** | **37/48** | **77.08%** | — | **64.52** | **3,086** | **0** | **0** |
| `astar` (symbolic A*) | 0 | 34/48 | 70.83% | — | 67.81 | 3,241 | 0 | 0 |
| `uniform` (5 fringe streams) | 0 | 147/240 | 61.25% | — | 79.30 | 18,940 | 0 | 0 |

### Secondary budget-scaling curve (closure rate, zero refitting)
| Budget $B$ | `astar` | `bidir` | `geom_mm` | `graph_bridge` | `nonrecurrent` | `unstructured_rec` | `recurrent_standard` | `opened_normal` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 25.00% | 27.08% | 62.50% | 62.50% | 55.56% | 59.03% | 61.11% | 58.33% |
| 64 | 60.42% | 64.58% | 72.92% | 75.00% | 79.86% | 79.17% | **81.25%** | 77.08% |
| 96 | 70.83% | 70.83% | 83.33% | 85.42% | 86.11% | 85.42% | **89.58%** | 86.11% |
| 128 | 70.83% | 77.08% | 87.50% | 87.50% | 88.19% | **92.36%** | 91.67% | 88.19% |

---

## 3. Pre-Declared Screens and Falsifying Causal Diagnostics

1. **Primary Screen 1A (`opened_normal` vs symbolic `bidir`): PASS.**
   `opened_normal` beat `bidir` by **+11.11 pp** (`95% CI [+2.78, +21.53]`, `8` wins / `2` losses / `38` ties) and cut mean penalized work by **31.58%** (`44.15` vs `64.52`, $\Delta = -20.38$, `95% CI [-27.68, -13.53]`), with all three seeds (`89.58%`, `89.58%`, `85.42%`) above `77.08%`.
2. **Primary Screen 1B (`opened_normal` vs strongest explicit heuristic `graph_bridge`): MISSED.**
   `opened_normal` (`88.19%`, `44.15` pen. work) edged `graph_bridge` (`87.50%`, `44.75` pen. work) by only `+0.69 pp` (`-1.35%` work), short of the `+4.0 pp` or `-10%` work gate — even though `recurrent_standard` (`91.67%`, `39.53` pen. work: `+4.17 pp` and `-11.65%` work!) and `unstructured_recurrent` (`92.36%`: `+4.86 pp`) exceeded that bar.
3. **Secondary Screens 2A & 2B (Recurrence, Structure, and Causal Shams): FALSIFIED — REVEALING TWO EXACT ARCHITECTURAL DEFECTS.**
   Although the 2D fresco resolved LBF-14's rank-1 slot collapse (`word_bank_erank = 2.484`, `fresco_bank_erank = 1.528`, `zero` max score delta `0.734` flipping `7/91` top choices, `scramble` flipping `21/91`), **ablating the spatial convolution (`opened_scramble`: `93.06%`, $+4.86\text{ pp}$, `95% CI [+1.39, +9.72]`) and zeroing the opened message bank (`opened_zero`: `90.97%`, $+2.78\text{ pp}$, `95% CI [+0.69, +5.56]`) IMPROVED closure over `opened_normal` (`88.19%`)!**

### Mechanistic diagnosis motivating LBF-16
1. **Sibling blurring vs. sibling competition ($I = -0.025$):** Planar fringe siblings $u_{i-1}, u_i, u_{i+1}$ are **competing alternative rewrites of the same parent word**, so true potential on the planar grid is *anti-correlated* across adjacent siblings (`morans_i_passK_mean = -0.0252` in `opened_normal`). Applying an un-signed $3 \times 3$ `Conv2d` smoothed a true bridge child's high activation onto its adjacent dead-end siblings; `opened_scramble` and `unstructured_recurrent` broke that sibling-blurring leak and jumped to `93.06%` and `92.36%`. **Fix:** Replace smooth fringe `Conv2d` with **parent-sibling contrastive lateral inhibition** ($H_{ij} - \bar{H}_{\text{sib}(i), j}$) while keeping spatial continuity strictly along the **1D subterm position lattice $p \in \{0,\dots,L-1\}$** where overlapping rules genuinely compose.
2. **$\mathfrak{S}_6$ alphabet-symmetry violation in `sym_emb`:** Every root in `rewrite_gym` samples an independent alphabet-permuted rule system over `a..f`. Embedding raw characters via `nn.Embedding(7, 40)` broke $\mathfrak{S}_6$ permutation equivariance, allowing opened recurrent gates (`word_gate = 0.157`, `fresco_gate = 0.105`) under batch-1 SGD to memorize training-set letter correlations that behaved as OOD noise on fresh calibration lineages (explaining why `opened_zero` and `recurrent_standard` beat `opened_normal`). **Fix:** Formulate LBF-16 around an **exact $\mathfrak{S}_6$-equivariant soft-rewrite envelope tensor** $P_w^{(k)} \in \mathbb{R}^{L \times |\Sigma| \times C}$ whose gated message bank transmits strictly alphabet-equivariant multi-hop rewrite reachability across passes $k=1,\dots,K$.
