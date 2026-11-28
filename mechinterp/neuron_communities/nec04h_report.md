# NEC-04H frozen Looper XOR phase-routing results

**FSI-01 REDESIGN_OR_PILOT_ONLY.** The learned phase head generalized to the owned contexts. Its assembled parity accuracy improved by 6.09 percentage points over the frozen native output and finished 0.24 points below the matched shuffled-phase arm. The primary developmental gate failed.

## Zero-forward archive result

Each root used the original 900-step auxiliary-prefix checkpoint and archived logits. The scalar oracle verified the original MUX and phase identities on all 19,968 source and owned rows. The table uses only the parity branch. The diagnostic adds the true phase to the archived stage-5 XOR prediction and uses the true selector. Alternate-branch predictions are the frozen native predictions in this diagnostic.

| Root | Parity rows | Native parity | True phase and selector |
| --- | ---: | ---: | ---: |
| 12 | 480 | 42.92% | 100.00% |
| 13 | 384 | 50.78% | 50.52% |
| 14 | 448 | 50.22% | 50.00% |
| 15 | 192 | 50.00% | 50.00% |
| 16 | 384 | 39.32% | 96.88% |
| 17 | 320 | 44.38% | 96.25% |
| **Root macro** | **2,208 total** | **46.27%** | **73.94%** |

The diagnostic locates the available XOR information in roots 12, 16 and 17 and retains all three XOR failures in the macro. NEC-04G's 61.88% oracle intervened on the fifth Fourier coefficient of the archived **final** logit. Its result and NEC-04F's negative native training result retain their original interpretation. [NEC-04F](https://github.com/MoralityLabAI/LoopedTransformers/pull/312) · [NEC-04G](https://github.com/MoralityLabAI/LoopedTransformers/pull/315)

## Frozen-model repair

The two phase arms shared initial weights, source batches, and a single learned selector per checkpoint. Both used cached final memory from the original 12 visits. The frozen model, native head and auxiliary head had zero updates and zero gradients. PyTorch 2.10.0 CPU replay on Windows changed zero archived native or auxiliary decisions. The maximum absolute error across probabilities was 0.0000512, and the maximum raw logit difference was 0.000320. The independent scalar reconstruction also reproduced all 46,080 routed predictions from the sealed caches and heads.

| Root | Phase head | Shuffled phase head | Selector | Native parity | Learned parity | Shuffled parity | Learned alternate | Learned full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 55.34% | 54.82% | 100.00% | 42.92% | 60.62% | 57.50% | 98.96% | 75.00% |
| 13 | 85.94% | 43.23% | 99.09% | 50.78% | 49.22% | 51.56% | 73.18% | 61.20% |
| 14 | 78.26% | 53.39% | 100.00% | 50.22% | 51.12% | 50.45% | 75.00% | 61.07% |
| 15 | 66.54% | 45.70% | 100.00% | 50.00% | 51.04% | 49.48% | 64.06% | 60.81% |
| 16 | 57.94% | 47.92% | 100.00% | 39.32% | 65.89% | 51.30% | 76.56% | 71.22% |
| 17 | 41.41% | 51.95% | 99.22% | 44.38% | 36.25% | 55.31% | 69.42% | 55.60% |
| **Root macro** | **64.24%** | **49.50%** | **99.72%** | **46.27%** | **52.36%** | **52.60%** | **76.20%** | **64.15%** |

| Stage-900 arm | Parity macro | Alternate macro | Full macro |
| --- | ---: | ---: | ---: |
| Frozen native | 46.27% | 76.28% | 60.70% |
| Learned phase and selector | 52.36% | 76.20% | 64.15% |
| Shuffled phase and learned selector | 52.60% | 76.54% | 64.17% |
| True phase and learned selector, diagnostic | 74.03% | 75.60% | 74.63% |
| Learned phase and true selector, diagnostic | 52.65% | 76.28% | 64.34% |
| True phase and selector, diagnostic | 73.94% | 76.28% | 75.00% |

The 240-step learned and shuffled parity macros were 50.60% and 51.65%. At 900 steps the learned arm retained the alternate branch within 0.09 points of native. Pooled parity counts were 1,015 native, 1,176 learned and 1,169 shuffled out of 2,208. The registered root macro weights all six root pairs equally.

Teacher-aligned normalized fifth-order Walsh coefficients at 900 steps averaged −0.08642 native, +0.04750 learned, and +0.05413 shuffled. Root 16's learned coefficient reached +0.3390 and root 17's fell to −0.2353. These diagnostics describe the same six developmental roots. Both phase bits arrived before the fifth focal bit in every root, so the only observed chronological stratum is before/before. Its six-root accuracy is the macro above.

The assembled task result missed the +5-point floor over the matched shuffled-phase arm. The two heads' source-to-owned transfer on phase labels therefore did not meet the gate for a prospective routing study. The original 18–23 candidate roots and reserved contexts remain sealed. The root count, prior exposure, and lack of an empirical sham distribution keep the evidence at `REDESIGN_OR_PILOT_ONLY`.

The compressed numerical receipt records all six roots, per-case predictions, original model and archive SHA values, 12 feature cache SHA values, and 36 head checkpoint SHA values with matching source sampler receipts. Execution used logical CPU 2. The original artifact lock and frozen protocol are linked below. [Numerical receipt](nec04h_results.json.gz) · [Artifact lock](nec04h_artifacts.json) · [Protocol](nec04h_protocol.md)
