# LBF-12: native equality rewrite fresco pilot results
Date 2026-11-24. Disposition: ENGINEERING_PASS; BOTH_REGISTERED_SCIENTIFIC_SCREENS_MISSED. Development only.

[Registration](LBF_12_REWRITE_FRESCO_REGISTRATION.md) · [Passing CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45282614711) · Tested commit 3350f56f6ff5d2995e1640d686d99b25988442e8.
[Artifact 14870421943](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45282614711/artifacts/14870421943), archive digest SHA-256 ac1a476d982cdfe55a052f6475988b2e769b6e4577ab003cd2cf7a78d7b4200a, expires 2026-12-07.

## Certificate repair and custody
Initial PR workflow 45280869555 passed ten engineering tests but failed at the dependent Type certificate: Lean theorem declarations require a proposition. Repair 3350f56 uses def for the Type-valued transport term, retaining theorem for equality. Eq.mp, congrArg and explicit Eq.trans composition are unchanged. The corrected run passes all ten engineering tests and native axiom audits, two fixtures, 24 training reference certificates (240 frames), 63 successful policy certificates and 16 calibration reference certificates. Same-worker duplicate fit matches. Source blob 204a1d012a4ebe8e80d8bd722590742b7e7f02ba; suite manifest c5cdfec74f4eacce984adda164c95a1cd2c1eff03e2463e87b371af33da9008b; checkpoint e408d7ea2814180b5e93d6ec84c4a97881e8101a6ac700b26d7d44e3be87a9e7. No online Lean tactic calls or sealed scoring.

## Frozen calibration
| Policy | Certified closures | Considered successors | Expanded | Model calls |
|---|---:|---:|---:|---:|
| A* | 7/16 | 1476 | 269 | 0 |
| Bidirectional BFS | 7/16 | 1481 | 265 | 0 |
| Normal | 6/16 | 1676 | 301 | 301 |
| Zero | 6/16 | 1676 | 301 | 301 |
| Roll | 6/16 | 1676 | 301 | 301 |
| Uniform (five streams) | 31/80 | 8152 | 1476 | 0 |

Root-paired normal minus A* and bidir: -6.25 pp, descriptive 95% interval [-18.75,0], 0 wins / 1 loss / 15 ties each. Normal minus uniform: -1.25 pp, interval [-13.75,+8.75], 2 wins / 1 loss / 13 ties (against per-root five-stream mean). Normal minus zero and roll: zero, 16 ties each. Memory screen and competence screen both missed.

## Direct boundary audit
60 decision-local observations, 120 pass records; hook-free outputs agree, actual gate contributions and bank flow checked. Max zero logit delta .0018072128295898438; max roll delta .0005161762237548828. One zero top-rank change, zero roll changes. Both shams leave the residual-state recurrence intact. The result is weak influence of the explicit message bank at this checkpoint under this evaluation, not a claim that all recurrence is irrelevant. The stored boundary measurements summarize tensors; LBF-13 will preserve complete continuous tensors. Current learned search also reorders examined siblings as well as frontier ties, so LBF-13 isolates these effects.

## Next
[Stacked LBF-13 PR #335](https://github.com/MoralityLabAI/LoopedTransformers/pull/335) uses fresh rule systems and fixed-policy budget-conditioned all-action targets, matched witness rank training and two seeds, separate frontier-only and sibling-only search, actual boundary tensor archives. LBF-12 misses both registered screens; no scientific promotion.
