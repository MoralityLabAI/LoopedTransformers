# LBF-09: Native Lean multi-shortest action-set supervision

**Date:** 2026-11-23  
**Experiment:** independently certified native Lean 4.19 proof branches, frozen model-visited states, two-seed matched CPU proof-completion comparison.  
**Disposition:** **POSITIVE DEVELOPMENT ACTION-SET SCREEN**; source-only superiority over the visited-set learner remains unresolved mechanistically; full scientific competence/power gates remain blocked.

**Preregistration:** [LBF_09_MULTI_SHORTEST_REGISTRATION.md](LBF_09_MULTI_SHORTEST_REGISTRATION.md), committed before source scoring.  
**Verified CI:** [run 45210953625](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45210953625), completed successfully, commit **8e4b7bcd082b36091265b42f64833816da6ab55e**.  
**Reproduction artifacts:** [lbf09-native-multishortest-proofs](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45210953625/artifacts/14836527224), ZIP SHA-256 **d960a78057fda9a86cd79d5549d188e4df11cea1f91daffbe3f677e1b5ddecbb**; GitHub retention until 2026-12-07.

## The source intervention

The G03 Horn fragment generator now accepts opt-in \`equal_alt_p\` while keeping the existing default **0.0**. In the LBF-09 suite, \`equal_alt_p=1.0\` generates alternative *provable* routes by constructing a fresh premise proposition with equal minimum proof cost, then adding a different rule that reaches the original head through that new premise. Each original fact also receives a second separately named hypothesis witness. No arbitrary correctness labels are introduced. Native Lean independently explores the resulting hypotheses and supplies the actual shortest transition costs.

The original goal and its minimum proof size are invariant under the augmentation. Source-law tests check this, the existence of distinct-premise equal-cost rules, duplicate named leaf witnesses, and unchanged legacy generation when \`equal_alt_p=0\`.

The frozen suite uses seed **20261130** with source-disjoint train, calibration, ordinary-test and OOD shards. Its generator and source manifestations are hash-bound; external overlap against older generated families remains unchecked. The new family deliberately makes alternate proof paths abundant, and its source distribution is easier in some respects than previous LBF cohorts. Results cannot be compared across families as if theorem difficulty were matched.

## Native train-only collection

Real Lean **4.19.0** and a separate standalone compiler verified all **48 train-root reference proofs**. The native shortest-continuation collector exported **271 reference-path training frames**; 40/48 roots had a reference proof within the common eight-tactic budget.

A frozen **360-step source-only PPT-B** behavior model (seed 107, width 32, eight typed slots, AB tied middle blocks, byte budget 384) visited each of the 48 train roots through the real Lean stepper. Only **after the model's decisions**, the native AND/OR search computed optimal action sets for the recorded public packets.

| Native train-only source item | Verified |
|---|---:|
| Independently compiled train theorem roots | 48 |
| Shortest-source training frames | 271 |
| Source frames with >=2 shortest native actions | **191 / 271** |
| Witnessed model-visited frames | **172** |
| Model-visited multi-optimal frames | **108 / 172** |
| Witnessed visited root lineages | **45** |
| Multi-optimal visited root lineages | **45** |
| Root-weighted multi-optimal visit fraction | **66.24%** |
| Censored/initial-only ledger entries | 60 |
| Autonomous behavior-model proof closures on train roots | 25 |
| Native/standalone-certified complete scripts from model prefixes | 30 |
| Of those 30, already autonomous model closures | 25 |
| Additional **teacher-assisted** complete recoveries | 5 |

The 30 certified scripts **include the 25 model-autonomous train proofs**; only five required teacher-supplied finishing steps. Those five are counted as **source assistance** and carry no autonomous benchmark credit. All collection episodes and teacher-assisted proofs passed independent model-free native replay.

The registered **25% witnessed multi-optimal-frame**, **12 multi-optimal-root** and **50 multi-positive optimizer updates per seed** gates passed. The source exposure was strong enough to evaluate full-action-set supervision as an actual treatment.

## CPU training: same packets, real multi-positive target difference

Two fixed seeds (**107, 109**) each trained the same 57,475-parameter PPT-B on three arms, with 720 FP32 AdamW updates, two recurrent passes, identical seed-specific initial weights, root/frame draw schedule, and one deterministic CPU thread on the same GitHub worker.

| Arm | Train inputs | Positive target |
|---|---|---|
| \`source_only\` | Native shortest-reference states | All native shortest continuations |
| \`visited_one\` | Native reference and model-visited states | Menu-first shortest continuation only on visited states |
| \`visited_set\` | **Exactly the same** source/visited packets as visited_one | **Every** equally shortest continuation on visited states |

| Seed | Model-visited optimizer updates per visited arm | Multi-positive updates in visited_set | Multi-positive updates in visited_one |
|---|---:|---:|---:|
| 107 | **337** | **236** | 0, by arm design |
| 109 | **332** | **214** | 0, by arm design |

All six learners completed **720 updates** with equal architecture, optimization envelope and predecision source. The exact source-only seed-107 duplicate produced **bitwise-identical named tensors and optimizer draw logs** on the same physical CPU worker. The single-action/full-set comparison had substantial genuine label differences.

## Native proof-completion outcomes

All arms were scored on **the same 32 independent calibration theorem roots** at eight Lean tactic checks and two inference recurrent passes. Native kernel closure, independent standalone compilation and complete model-free replay were mandatory for every proof credit. No ordinary-test or OOD answers were opened.

| Seed | Reference-only | Visited one | Visited full set | Public auto8 | Native attainable |
|---|---:|---:|---:|---:|---:|
| 107 | **15/32** | **12/32** | **13/32** | 8/32 | 24/32 |
| 109 | **13/32** | **9/32** | **12/32** | 8/32 | 24/32 |

Observed native tactic checks / model forwards:

| Seed | Reference-only | Visited one | Visited set |
|---|---:|---:|---:|
| 107 | 179 / 179 | 155 / 155 | 175 / 175 |
| 109 | 168 / 168 | 141 / 141 | 154 / 154 |

The eight-call *per theorem* maximum was identical. Aggregate attempted tactics differed because models reached terminal states, exhausted menus or completed proofs at different points.

## Preregistered root-paired effects

The independent sampling unit was **32 theorem lineages**, with the two matched-seed effects averaged *within each root*. Descriptive 95% intervals came from 2,000 root-bootstrap draws seeded 20261123.

| Matched native completion difference | Effect | Descriptive 95% interval | Wins/losses/ties |
|---|---:|---:|---:|
| **Visited full set minus visited one** (primary) | **+6.2500 pp** | **[−3.125, +15.625] pp** | 7 / 3 / 22 |
| Visited full set minus reference-only (secondary) | **−4.6875 pp** | **[−18.750, +9.375] pp** | 6 / 9 / 17 |
| Visited one minus reference-only | **−10.9375 pp** | **[−25.000, +1.5625] pp** | 5 / 10 / 17 |

The preregistered **+5 percentage-point action-set-over-single-action development screen passed**. The secondary +5-point improvement over reference-only **missed**. Confidence intervals include zero, and the same 32 roots are reused across two seeds. Treat this as a useful pilot result requiring an independently preregistered follow-up.

The source-only checkpoints closed 15/32 and 13/32 proofs, showing that the artificially equal-cost theorem family is materially more accessible to this model than prior native suites. The two full-set checkpoints closed 13/32 and 12/32, and the exact candidate-uniform baseline was not measured. The **full registered competence/power gate remains blocked** regardless of the development gain, pending a new FSI-01-confirmed cohort and native uniform-menu baseline.

## LBF/Fresco instrumentation

The frozen seed-109 visited-set checkpoint produced **four replay-attested recurrent fresco episodes** and **72** zero/roll cross-pass message probes. The first-pass empty-memory equivalence passed. Peak absolute candidate-logit movement was **0.13165509700775146**, with **zero changes in selected top tactic** among these frozen observations. Rendering made zero new Lean calls and consumed no future proof outcome labels.

The glyph board remains an observational compression of coda slots; the native proof certificates provide truth and independent acceptance.

## Run quality and recorded engineering corrections

- First source-law CI required a correction to the unit assertion separating duplicated **original** facts from the new *single-occurrence twin* fact propositions. The source generator and registration were unchanged.
- The initial native campaigns completed all source, training and proof-evaluation stages but stopped on comparator CLI/path handoffs (\`--collection\` and the replay receipt file). Those interfaces were corrected, and the final green run recomputed all source and model artifacts from the same preregistered seed.
- Final campaign independently validated the source hashes, 48 native reference proofs, 45 eligible model-visited root lineages, the multi-optimal source gate, identical same-seed initialization/draw schedules, exact same-worker repeat, six kernel/standalone proof evaluations, and model-free replays. Source and locked-test custody stayed intact.

## Proposed LBF-10: hypothesis discrimination and transport

The present positive effect establishes that **a training label containing several valid actions** can help this small recurrent policy compared with an arbitrary one-action target on the same visited states. Distinguishing the computational mechanism requires a more diagnostic, new-source study.

Preregister independent **structurally nontrivial equal-length branches** where alternatives rely on different *premise graphs* and cannot be recognized by duplicate fact names. Maintain separate indistinguishable-looking decoy branches, exercise the same multiple-optimal exposure gate, then test transfer to the legacy G03 family at identical model size and tactic budgets. Add a native candidate-uniform baseline, source/visited predecision-candidate margin analysis, and matched shams for source multiplicity. Require an FSI-01 power gate before sealed validation.

The result remains development-only. Model competence is conditional on the generated theorem family and proof checks. No sealed ordinary-test or OOD evaluations were performed.

## Implementation / evidence index

- [Preregistration](LBF_09_MULTI_SHORTEST_REGISTRATION.md)
- [Optional native Horn source generator](../lean_cgrt_hrm400/lcgrt/fragment.py)
- [Frozen theorem suite provenance](../lean_cgrt_hrm400/ppt_gym/live_suite.py)
- [Native multi-optimal exposure gate](lbf09_exposure.py)
- [Root-paired and provenance-gated proof comparator](lbf09_compare.py)
- [CPU/native Lean run workflow](../../.github/workflows/lbf09-multishortest.yml)
- [Machine artifact bundle](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45210953625/artifacts/14836527224)
