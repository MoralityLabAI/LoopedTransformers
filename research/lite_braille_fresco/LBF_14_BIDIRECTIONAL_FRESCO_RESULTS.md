# LBF-14: bidirectional frontier utility and recurrent gate fresco results
Date 2026-11-24. Disposition: ENGINEERING_PASS; PRIMARY_AND_MEMORY_SCREENS_MISSED. Development only.

[Registration](LBF_14_BIDIRECTIONAL_FRESCO_REGISTRATION.md) · [Passing CPU/Lean CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45285311131) · Tested head 812ad4a04e0584d3a7f5cb662cd0351f908e1637.
[CI workflow artifact 14871586749](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45285311131/artifacts/14871586749) (13,058,685 bytes, expires 2026-12-08) · In-repo preserved archive [`results/lbf14/`](results/lbf14/manifest.json).
Suite manifest SHA-256 `c23915dbf6d79d52718cdd16605610ee60ab343aefb12eb219d72d63f25c9a47` · Teacher frames SHA-256 `d0c563dcd84c6f58446692c525c65bc1d396f20d1dc8f5ecf8f0c1ecd0493dcb` · Raw boundary tensor ledger SHA-256 `505815e2a01b751dd239d3931b4764536043bcedcd04ebd2aa13c74db256aa0e`.

## Custody and counterfactual teacher census
All 19 unit tests passed (`test_lbf_gym`, `test_lbf12_rewrite_fresco`, `test_lbf13_budget_fresco`, `test_lbf14_bidirectional_fresco`). Source pinned to commit `cd9ac784c5804ec2fe595bb6a01d2697fc016610` (blob `204a1d012a4ebe8e80d8bd722590742b7e7f02ba`), fresh generator seed `20261209` with 112 distinct canonical rule-system lineages (48 train, 32 calibration, 16 sealed ordinary, 16 sealed OOD).

- **Training counterfactual teacher:** 48 axiom-free training reference certificates, 235 layer-decision frames along smaller-layer trajectories, 29,597 forced-continuation oracle successor checks, 14,388 pairwise Hamming evaluations, and 24 axiom-free counterfactual proof certificates (12 equality, 12 dependent transport).
- **Counterfactual tie structure:** Out of 235 training layer decisions, **177 (75.32%) were exact continuation-cost ties** ($C_0 = C_1$) and only **5 (2.13%) produced mixed closure** ($C_0 \le r$ vs $C_1 = r+1$), with mean absolute utility gap $\mathbb{E}|u_0 - u_1| = 0.02756$. Because complete-layer expansion must traverse the same forward/backward layer pairs $(d_0, d_1)$ to close a shortest witness of length $d^* = d_0 + d_1$, forced single-layer order permutations rarely alter total layer work except near budget truncation.
- **Matched initialization and draw custody:** Both seeds (`107`, `109`) matched pre-gate initial fingerprints (`1100b0c2...`, `af8096d2...`) and exact 360-step draw schedules (`dcf26248...`, `bda35e49...`) across `standard` (gate bias $-6$) and `opened` (gate bias $-2$), differing at initialization solely at `gate_a.gate.bias` and `gate_b.gate.bias`.
- **Calibration verification:** 672 frozen search episodes across 32 lineage-disjoint calibration roots; 432 policy proof certificates and all 32 calibration reference certificates compiled standalone and axiom-free in Lean 4.19.0 (`6caaee842e94`). Zero online Lean tactic calls; sealed `ordinary` and `ood` splits untouched.

| Fit arm | Seed | Shared initial SHA-256 | Checkpoint SHA-256 | Final fingerprint |
|---|---:|---|---|---|
| `standard` | 107 | `1100b0c28a0aefa2...` | `44fc4a790e9a160685034ce4c774582c8aa8d655f3c5cb11df25e925521dc40d` | `7b0013b2e2364119...` |
| `opened` | 107 | `1100b0c28a0aefa2...` | `12e5caa75215bb8018d5878d98e261d0fe42b140106e319653d82ea7241265e4` | `edd5fd867e38dab1...` |
| `standard` | 109 | `af8096d25012313b...` | `5299d34b9ba9479c08e210dbb294ea09db02f4b4f29057de03f1e2af0dfae7a1` | `5b27d90cda848787...` |
| `opened` | 109 | `af8096d25012313b...` | `915c978355b06866436e64e06cc398ef6b0d1c2fa435d172a5ef812f82eb51fb` | `45120b40e4c17959...` |

## Frozen calibration search (32 roots, work cap 128, path cap 16)
| Condition | Certified closures | Closure rate | Seed 107 | Seed 109 | Total work | Mean penalized work | Expanded | Model calls | Hamming evals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bidir` (smaller layer) | 23/32 | 71.88% | — | — | 2,204 | 69.16 | 453 | 0 | 0 |
| `meeting` (nearest cross-Hamming) | 23/32 | 71.88% | — | — | 2,218 | 69.59 | 452 | 0 | 7,610 |
| `depth` (smaller depth) | 22/32 | 68.75% | — | — | 2,323 | 72.91 | 477 | 0 | 0 |
| `ridge` ($\lambda=0.01$) | 21/32 | 65.62% | — | — | 2,325 | 73.00 | 483 | 142 | 6,330 |
| `standard_normal` | 42/64 | 65.62% | 21/32 | 21/32 | 4,722 | 74.12 | 959 | 281 | 12,201 |
| `standard_zero` | 42/64 | 65.62% | 21/32 | 21/32 | 4,722 | 74.12 | 959 | 281 | 12,201 |
| `standard_roll` | 42/64 | 65.62% | 21/32 | 21/32 | 4,722 | 74.12 | 959 | 281 | 12,201 |
| `opened_normal` | 41/64 | 64.06% | 21/32 | 20/32 | 4,771 | 74.91 | 968 | 283 | 12,600 |
| `opened_zero` | 41/64 | 64.06% | 21/32 | 20/32 | 4,755 | 74.66 | 964 | 283 | 12,815 |
| `opened_roll` | 41/64 | 64.06% | 21/32 | 20/32 | 4,755 | 74.66 | 964 | 283 | 12,681 |
| `uniform` (5 streams) | 94/160 | 58.75% | — | — | 13,213 | 82.99 | 2,642 | 0 | 0 |

## Root-paired contrasts (`opened_normal` minus comparator)
| Comparator | Closure $\Delta$ (pp) | Closure 95% bootstrap | Penalized work $\Delta$ | Work 95% bootstrap | Root wins / losses |
|---|---:|---:|---:|---:|---:|
| `bidir` (strongest control) | -7.81 | [-17.19, 0.00] | +5.75 | [+2.44, +9.28] | 0 / 3 |
| `meeting` | -7.81 | [-17.19, 0.00] | +5.31 | [+2.33, +8.83] | 0 / 3 |
| `depth` | -4.69 | [-12.50, 0.00] | +2.00 | [-1.05, +5.16] | 0 / 2 |
| `ridge` | -1.56 | [-12.50, +9.38] | +1.91 | [-2.22, +5.80] | 2 / 2 |
| `standard_normal` | -1.56 | [-4.69, 0.00] | +0.78 | [-0.23, +1.95] | 0 / 1 |
| `opened_zero` | 0.00 | [0.00, 0.00] | +0.25 | [0.00, +0.75] | 0 / 0 |
| `opened_roll` | 0.00 | [0.00, 0.00] | +0.25 | [0.00, +0.75] | 0 / 0 |
| `uniform` | +5.31 | [-2.50, +13.44] | -8.09 | [-12.52, -4.08] | 8 / 3 |

Both registered screens failed: `opened_normal` trailed `bidir` by 7.81 pp (+5.75 mean penalized work), and `opened_normal` tied `opened_zero` and `opened_roll` on closure (0.00 pp; +0.25 penalized work).

## Gate opening and boundary probe audit
Across 128 probed calibration layer decisions per arm (first 2 decisions × 32 roots × 2 seeds):
- **`standard` ($\text{bias}=-6$):** Pass-2 mean gate strengths were `0.003172` (`gate_a`) and `0.002908` (`gate_b`); max outcome-logit displacement was `0.000971` under `zero` (0 top-choice changes) and `0.000177` under `roll` (0 top-choice changes).
- **`opened` ($\text{bias}=-2$):** Pass-2 mean gate strengths rose $47\times$ and $39\times$ to `0.148926` (`gate_a`) and `0.114023` (`gate_b`); max outcome-logit displacement rose $28\times$ to `0.026878` under `zero` (flipping 4/128 top side choices) and `0.006327` under `roll` (0 top-choice changes). First-pass equivalence and rotated-bank norm invariance held across all 256 probed records.

## Research decision
Opening the message gates mechanically restores pass-2 bank sensitivity ($47\times$ gate strength, $28\times$ `zero` logit shift), but neither `opened` nor `ridge` improves proof search because (1) **whole-layer side scheduling is coarse**: 75.3% of training frames tie on exact continuation work, while unsolved roots exhaust the 128-successor budget expanding unpromising cousins inside a wide frontier layer; and (2) **PPT-B collapses frontiers to two byte-encoded side summaries** whose candidate vectors never enter the recurrent workspace (`z` only pools `cand_emb.mean(0)`). Next step (LBF-15): move from coarse two-side layer scheduling to node-level cross-frontier meeting-potential propagation $M_{ij}^{(t)}=\phi_\theta(u_i^{(t)},v_j^{(t)},R,b)$ on a fresh lineage-disjoint suite.
