# LBF-05: Native Lean hybrid-input pilot

**Date:** 2026-11-23
**Status:** +0.05 development screen passed; checkpoint-competence gate blocked.

**Workflow:** https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182848388

**Machine artifact:** https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182848388/artifacts/14820759625

Artifact ZIP SHA-256: 661d5d58a31d6e82a26459deac9c5da5deba66e7a101f3685bf82a064a9f3ab9. GitHub retains this run artifact for 14 days.

## Source and design

LBF-05 uses a new G03 Lean 4.19.0 source suite, seed 20261126. The native kernel and independent standalone compiler certified **48 train-only roots**, producing **251 shortest-continuation training frames**. Of those reference scripts, 43 fit the eight-tactic budget. Training and calibration root lineages are disjoint within the frozen suite. External ancestry overlap remains unchecked. Ordinary-test and OOD answer sets remain sealed.

Six models shared the same 57,475-parameter PPT-B architecture, AB middle-block order, 384-byte ByteEncoder, two recurrent training passes, 720 FP32 optimizer steps and source-root sampling schedules. Two seeds (107, 109) were each trained on raw, typed, or hybrid public-input strings, with all six jobs executing on one physical GitHub CPU worker. A seed-107 raw duplicate showed **exact named tensor equality and identical optimizer-sampling traces**, using one thread and deterministic PyTorch 2.5.1.

The new lbf05.hybrid_public_state.v1 representation preserves a mandatory public goal/budget header, candidate-named premise/head rules, and a bounded suffix of the Lean server's readable context. Its source validators reject unknown packet fields, invalid tactic candidates and byte-limit violations. No proof outcomes enter the encoder.

## Label-blind source-window audit

Exactly 251 independently sourced training packets were measured through the actual 384-byte model window. The following token coverage is lexical and should not be interpreted as semantic correctness or evidence of neural reliance.

| Root-weighted metric | Raw | Typed | Hybrid |
|---|---:|---:|---:|
| Mean full encoded input bytes | 2,493.17 | 179.93 | 365.19 |
| Fraction exceeding 384-byte window | 100% | 0% | 0% |
| Public candidate-name lexical recall | 7.83% | 100% | 100% |
| Candidate-rule atom lexical recall | 99.33% | 100% | 100% |
| Goal atom lexical recall | 100% | 100% | 100% |

The raw model receives candidate text separately from the state string, an important caveat for the low candidate-name recall in its state-byte tail.

## Independently compiled native proof completion

| Seed | Raw | Typed | Hybrid | Public-order auto8 | Native attainable |
|---|---:|---:|---:|---:|---:|
| 107 | 2/32 | 3/32 | 3/32 | 3/32 | 23/32 |
| 109 | 1/32 | 3/32 | 5/32 | 3/32 | 23/32 |

All six calibration runs scored the same 32 independent roots with eight allowed tactic checks, and all generated proof-episode ledgers passed model-free native replay. Hybrid-minus-raw, averaged within root across the two seeds, was **+0.078125 (7.8125 percentage points)**. The descriptive root-cluster bootstrap interval was **[0, +0.15625]**. There were six root-averaged wins, one loss and 25 ties.

The registered hybrid development threshold was +0.05; the result **passed**. Typed-minus-raw was +0.046875 with descriptive 95% interval [-0.015625, +0.109375], four wins, one loss and 27 ties. Both seeds reuse the same 32 theorem roots and cannot be counted as 64 independent examples. Earlier LSA-02C typed negatives used a different source cohort; they remain valid development evidence.

## Fresco and model-message probes

A frozen, native-source-trained seed-109 hybrid checkpoint was re-forwarded over replay-attested public native episode packets. The LED/Braille fresco adapter rendered four neural-arm episodes and recorded 72 frozen-model cross-pass message interventions. First-pass zero-memory invariance passed. Peak absolute candidate-logit displacement was **0.0006660819053649902**. The top-ranked tactic changed in **zero** probes. The renderer made zero Lean calls and exported no private proof labels.

## Scientific boundary and next gate

**Development result: positive. Competence gate: BLOCKED_INCOMPETENT_CHECKPOINT.** The strongest hybrid checkpoint closed 5/32 (15.625%) against the required 40% and a native reference ceiling of 23/32. The bootstrap interval reaches zero, and a prospective native candidate-uniform baseline and matched shams remain outstanding. This pilot establishes a reproducible, source-verified input-comparison workflow and a useful scoped lead, not a general architecture or model-interpretability claim.

**Next registration: LBF-06.** On newly frozen development roots, train matched hybrid, hybrid-without-readable-context, and norm/length-matched shuffled-readable-context variants. Preserve the same byte allowance, source families, initialization, optimizer exposure, tactic budget and independent kernel receipts. A deterministic public-only shuffle can isolate whether the local order of visible contextual facts contributes to decisions. Keep the locked ordinary/OOD partitions closed unless a separate prospective competence and power gate passes.

Related: [LBF-04](LBF_04_NATIVE_TRAINED_RESULTS.md), [LSA-02C](../../reports/lsa02c_typed_public_state_cpu_20261123.md), [hybrid encoder](../lean_cgrt_hrm400/ppt_gym/lbf_hybrid.py), [paired comparator](lbf05_compare.py), [LBF-05 CI](../../.github/workflows/lbf05-hybrid-native.yml).
