# LBF-06: byte-matched readable-context controls

**Date:** 2026-11-23. **Disposition:** local readable-context advantage missed the +0.05 development screen. The PPT-B checkpoint remains below competence. All locked ordinary/OOD tests remain unopened.

**CI:** https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45184599452

**Artifact:** https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45184599452/artifacts/14824005524

Artifact SHA-256: 9e720f45976e3e470f6f4c0734005536b8837ee323d945b2b24dbf22fb3c6b49. Expiration 2026-12-07.

## Question and preregistered comparison

LBF-05 recorded a +7.8125-point root-paired development gain for hybrid typed-plus-readable Lean state over historical raw JSON. The follow-up asks whether the **readable suffix's actual text ordering** causes that difference. Input variants were frozen before neural scoring:

- **hybrid:** the original LBF-05 public goal/budget, candidate-named typed rule clauses, and up to 112 bytes of Lean's public readable context.
- **hybrid_nopp:** identical header, typed rules and total byte length, replacing the readable suffix with a repeated inert ASCII character.
- **hybrid_shuffle:** identical header, typed rules, final byte count and suffix Unicode character histogram, with a deterministic public-input-hashed shuffle of the readable suffix's codepoints. The permutation never sees source labels.

All variants enforce a 384-byte window. They share the original PPT-B architecture, 57,475 parameters, seed-matched initialization, two training recurrent passes, 720 optimizer steps, public candidate menus, native Lean proof-verifier budget and source-root sampling. The code contains no inference-time optimizer updates or hidden-verdict dependence.

## Source and computation controls

The new frozen Lean G03 suite used seed 20261127. Real Lean 4.19 and the standalone compiler independently certified 48 train-only roots, yielding **282 source frames** with shortest-continuation targets. **40/48** source reference scripts fit the eight-tactic allowance. Source ancestry remains disjoint within this suite; broader generator-family ancestry awaits external audit.

Models were trained on a single GitHub-hosted CPU worker using PyTorch 2.5.1, one thread and deterministic CPU algorithms. A separate duplicate source-only hybrid fit reproduced the original weights and optimizer trace **exactly** on that worker. The new hybrid controls passed tests for identical byte lengths, preserved typed clauses, reproducible public-only shuffle, valid Unicode bytes, and explicit rejection of hidden source labels.

Each trained checkpoint received a real Lean eight-tactic native search over the same 32 calibration roots. Kernel closures underwent independent standalone compilation. All model episodes were independently replayed without inference-model calls. Locked ordinary test/OOD answers remained closed.

## Results

| Seed | Hybrid | No pretty text | Shuffled pretty text | Public auto8 | Native ceiling within budget |
|---|---:|---:|---:|---:|---:|
| 107 | 6/32 | 6/32 | 5/32 | 3/32 | 23/32 |
| 109 | 6/32 | 7/32 | 6/32 | 3/32 | 23/32 |

With independent theorem root as the unit (32 unique roots, 2 matched training seeds):

| Root-averaged effect | Mean | Descriptive 95% bootstrap | Wins / losses / ties |
|---|---:|---:|---:|
| Hybrid minus matched no-pretty | **-0.015625** | [-0.078125, +0.046875] | 2 / 3 / 27 |
| Hybrid minus matched shuffled-pretty | **+0.015625** | [-0.062500, +0.093750] | 4 / 3 / 25 |

The prespecified development requirement was a **+0.05 gain over both shams**. The result **missed**. The scores are development-only; the confidence intervals include zero and seed repetitions use the same theorem roots. The strongest checkpoint closed **7/32 = 21.875%**, below the 40% competence gate. The exact native candidate-uniform baseline remains unmeasured.

## Recurrent fresco probes

The source-trained seed-109 hybrid checkpoint was frozen and re-forwarded at three recurrent passes over native-replayed public packets. Four glyph fresco episodes were rendered, with 114 paired cross-pass zero/roll message probes. Peak absolute candidate-logit movement was **0.0014986991882324219**. All first-pass zero-memory invariance checks passed; no intervention changed the top-ranked tactic. The renderer made zero additional Lean calls.

## Interpretation

**LBF-05 remains a valid scoped positive development comparison of hybrid versus historical raw input.** Its initial association with readable pretty-printed context does not survive this sharper LBF-06 control. The model may benefit from candidate-named typed rules, additional byte structure, or other training-input effects. Readable-text order is currently a poor target for further optimization in this small specialist.

The native proof ceiling at 23/32 and the trained models' 5–7/32 scores indicate substantial policy-search headroom. Further work should prioritize a *typed rule/goal information necessity test* and source-verified model-visited repair training before spending a larger GPU budget on ornate sensory encodings.

### Proposed LBF-07: causal typed-structure ablation

Pre-register new development roots and three matched input conditions: complete candidate dependency graph, candidate names with premise/head associations deterministically shuffled under a public-only transform, and a byte/character-count-preserving placebo. Hold all other training and verification dimensions fixed, with a one-worker deterministic duplicate. The question is whether usable graph incidence, rather than simply exposed public vocabulary, changes native Lean closure. No sealed promotion until a competent checkpoint and a prospective matched-sham power gate.

Related: [LBF-05](LBF_05_NATIVE_HYBRID_RESULTS.md), [hybrid serializer](../lean_cgrt_hrm400/ppt_gym/lbf_hybrid.py), [LBF-06 comparator](lbf06_compare.py), [native workflow](../../.github/workflows/lbf06-pretty-causal.yml).
