# LBF-08: Native model-visited Lean repairs and action-set supervision

**Date:** 2026-11-23. **Status:** completed development pilot; registered primary effect threshold **MISSED**. Multi-optimal action-set hypothesis **UNINFORMATIVE_ZERO_MULTI_OPTIMAL_VISITS**. Proof competence **BLOCKED_INCOMPETENT_CHECKPOINT**.

**Registration:** [LBF_08_REPAIR_REGISTRATION.md](LBF_08_REPAIR_REGISTRATION.md), committed before scoring.

**Successful verified workflow:** [GitHub Actions #45203538921](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45203538921), commit `9084568be75d0868df7dee4ba03b5dc8c7ffb5c9`.

**Machine artifacts:** [lbf08-model-visited-native-repair](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45203538921/artifacts/14833565116), ZIP SHA-256 `59f146b1fc02a6961cbc3d1fc2474579664721707282023dbb7f36a52115c997`, expires 2026-12-07.

## Frozen native source and model visitation

New G03 suite generator seed **20261129**, with disjoint train, calibration, ordinary-test and OOD partitions. Real Lean 4.19.0 and an independent standalone compiler certified **48/48 train-only reference proofs**, producing **256 native shortest-continuation training frames**. The source reference was within eight tactics for **43/48** roots. Wider generated-family ancestry remains unchecked.

A frozen 360-step, seed-107, native-source-trained PPT-B behavior checkpoint (width 32, AB recurrence, 384-byte `hybrid_rule_intact` public input) acted on all 48 train roots through the real Lean tactic stepper. After those model decisions, native Lean exploration supplied the shortest-action training labels. Every behavior episode was independently replayed without the model. Unreachable visited states were kept censored, rather than turned into false negatives.

| Model-visited source census | Observed |
|---|---:|
| Native train roots visited | 48 |
| Witnessed model-visited frames | **138** |
| Independent roots with witnessed visits | **38** |
| Censored entries and initial-root exclusions retained | **71** |
| Original model-autonomous terminal kernel closures on train | **18** |
| Source-verified full scripts from accepted model prefixes | **23** |
| Of the 23, already model-autonomous scripts | **18** |
| Additional *oracle-assisted* full recoveries | **5** |

The 23 independently certified full scripts **include the 18 autonomous model completions**. There were five additional teacher-forced completions. Teacher-forced proofs are counted only as train-source recovery evidence. They are never credited as autonomous model proof success.

The first engineering collection attempt failed because the generic shortest-script utility tried to sum an unreachable alternative branch's undefined proof cost. The corrected finite-shortest recovery helper skips such branches and has its own regression test. The subsequent experiment passed all native compilation and model-free replay checks without changing the preregistered source seed or training design.

## Matched CPU training

The experiment trained three arms on each of seeds **107** and **109**, with identical initial weights per seed, 57,475 trainable parameters, two recurrent passes, 720 FP32 AdamW updates per arm, one deterministic CPU thread and the same source-root/frame draw roster.

| Arm | Source packets | Visited-state labels |
|---|---|---|
| `source_only` | Compiled native shortest-reference path | Native full shortest-action set |
| `visited_one` | Reference/visited mixed at registered schedule | First eligible shortest public action |
| `visited_set` | Exactly the same packet roster as visited_one | All equally shortest public actions |

For seed 107, each visited arm consumed **296** model-visited updates. For seed 109 it consumed **289**. The remaining steps used reference frames; source-only consumed no visited frames. A repeated source-only seed-107 fit produced **bitwise-identical named parameter tensors and optimizer draw traces** on one physical CPU worker.

**Action-set exposure gate:** All four visited-model fits recorded `multi_positive_visited_updates=0`. Every admitted visited training state had exactly one shortest public action. Consequently, visited_one and visited_set had the same labels and same training history within each seed. The comparison is **uninformative about multi-optimal action-set learning** and should not be presented as a failed action-set mechanism.

## Real Lean calibration

Each trained checkpoint was evaluated on the same **32 independent calibration roots**, with at most eight Lean tactic checks and two recurrent passes per attempt. Whole-proof credit required the actual Lean terminal kernel, independent standalone compilation, and model-free replay. The native reference ceiling was computed separately.

| Seed | Source only | Visited one | Visited set | Public auto8 | Native attainable |
|---|---:|---:|---:|---:|---:|
| 107 | **4/32** | **6/32** | **6/32** | 7/32 | 26/32 |
| 109 | **5/32** | **5/32** | **5/32** | 7/32 | 26/32 |

| Seed | Source tactic checks / model forwards | Visited one | Visited set |
|---|---:|---:|---:|
| 107 | 132 / 132 | 129 / 129 | 129 / 129 |
| 109 | 135 / 135 | 138 / 138 | 138 / 138 |

Effects were paired within 32 unique theorem roots, with repeated seeds averaged inside each root before the 2,000-draw descriptive root bootstrap (seed 20261123).

| Native root-paired contrast | Mean | Descriptive 95% interval | Wins/losses/ties |
|---|---:|---:|---:|
| Visited set minus source only | **+3.125 pp** | **[−4.6875, +12.50] pp** | 3 / 2 / 27 |
| Visited one minus source only | **+3.125 pp** | **[−4.6875, +12.50] pp** | 3 / 2 / 27 |
| Visited set minus visited one | **0 pp** | **[0, 0] pp** | 0 / 0 / 32 |

The registered **+5-point** primary development threshold was missed. The observed +3.125-point gain has uncertainty spanning zero. Every trained arm also trailed public auto8 (7/32) in its native whole-proof count. The best model solved 6/32, against a 26/32 native reference ceiling; the registered >=40% model-competence threshold remains unmet.

## Fresco instrumentation

The replay-attested, frozen seed-109 visited_set checkpoint generated four LBF recurrent fresco episodes and **84** matched zero/roll cross-pass message probes. First-pass zero-bank invariance passed. Peak absolute candidate-logit displacement was **0.1518944501876831**, with **zero changes to the highest-scored tactic**. Renderer Lean calls: zero. Verifier outcome labels passed into the visualization: zero.

These represent instrumented coda-channel measurements and actual model-side communication perturbations. All formal proof authority remains with native Lean and the independent compiler.

## Scientific interpretation and LBF-09

**The source-to-repair pipeline works.** It supplies model-visited native state transitions, censored rejections, source-authoritative shortest actions and separately certified teacher-forced complete scripts. Its current supervised model improves marginally against reference-only, below the registered benefit floor. It remains far from the native proof ceiling.

The multi-action-set mechanism remains empirically untested because the admitted visited states contained **zero multi-positive examples**. LBF-09 should preregister *native Lean sources with multiple equal-length proof paths* and require at least **25% multi-positive witnessed model-visited training frames** before a full-set versus one-action comparison. Keep root lineages, public candidate menus, learner weights, optimizer steps and native proof-check budget matched. Report all generation misses and avoid cherry-picking after native outcomes. Native verifier feedback and model-visited branch repairs are a more promising focus than additional ornamental glyph detail.

**Evidence class:** development only. Ordinary-test and OOD cohorts remained closed; exact native candidate-uniform baseline and FSI-01 confirmatory comparison remain outstanding.

Related: [registered protocol](LBF_08_REPAIR_REGISTRATION.md), [native collector/fit](../lean_cgrt_hrm400/ppt_gym/lbf08_repair.py), [root-paired audit](lbf08_compare.py), [CPU/native workflow](../../.github/workflows/lbf08-native-repair.yml), [earlier LBF-07](LBF_07_NATIVE_RULE_RESULTS.md).
