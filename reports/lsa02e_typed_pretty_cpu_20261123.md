# LSA-02E: typed public premises with bounded Lean pretty-print context

**Date:** November 23, 2026. **Evidence:** exploratory CPU-only native-Lean development comparison. **Registered development screen:** PASS. **Formal native competence:** BLOCKED. **Code:** [PR #326](https://github.com/MoralityLabAI/LoopedTransformers/pull/326), merged as `3a0acd304e93f88d83a2d24f40c56c0b28b2046f`. **Native CI and source artifacts:** [GitHub Actions run 45182664891](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182664891). No locked ordinary/OOD test outcomes were opened.

## Source, computation and controls

A new Horn theorem suite was frozen at generator seed `20261126`, with 64 roots per split. Actual Lean 4.19 `lgym` verified **48 independent training roots**, producing **251 shortest-continuation training frames**. All 48 shortest scripts independently compiled without additional axioms, and **43/48** were within the eight-tactic check budget. The native training teacher manifest SHA-256 is `e62e7358c2b54a94787a8d4395488726233e4ce202ed12ca4dfa7059617620ad`, bound to the suite manifest `6c1931bfe9453db8e59d99877354efd2c5a301f1c1cbc25ed58acd54a5edbf34`.

Three public representations entered the **same 384-byte Plan B byte encoder**:

- `raw`: unchanged canonical JSON of the public Lean goal, using the historical final-384-byte tail crop;
- `typed`: unchanged compact goal, budget and public hypothesis premise/head serialization;
- `typed_pp`: the identical typed *hypothesis grammar*, with every current-goal and candidate type mandatory, plus up to 96 valid UTF-8 bytes from the tail of Lean's **public** pretty-printed goal text, before spending residual bytes on public background hypothesis rules.

Each model used 57,475 unique parameters, two tied AB middle-block passes, width 32, four heads, 720 FP32 AdamW steps, identical source-root/frame sampling and training seed. Seeds `113` and `127` were fitted in one physical GitHub CPU job. Every seed/view model was trained **twice** on that worker, producing 12 fits; all six duplicate pairs passed exact tensor, checkpoint-byte and optimizer-trace equality. The job used CPU PyTorch 2.5.1, one thread, deterministic operators and disabled MKLDNN. The two view controls shared bitwise-identical initial weights with the hybrid within each seed.

Average model input sizes on the 251 training frames:

| Public representation | Mean UTF-8 source length | Input budget | Actual model behavior |
| --- | ---: | ---: | --- |
| Raw canonical JSON | 2,901.45 bytes | 384 bytes | Historical tail truncation |
| Compact typed | 194.40 bytes | 384 bytes | Whole serialized view fits |
| Typed plus pretty | 284.14 bytes | 384 bytes | Whole serialized view fits |

The hybrid adds an average of **89.74 bytes of public text** to the typed state while preserving mandatory candidate premise/head relations. None of the views receives reference proofs, shortest-solution lengths or private generator labels at inference. Verified shortest-continuation labels were used for train-only supervision.

## Kernel-verified native Lean results

Each of the six trained checkpoints was evaluated on the *same 32 fresh calibration theorem roots*, with eight tactic checks and independently replayed native Lean action transcripts. Every claimed closure passed the Lean kernel and standalone axiom-free compilation. A separately compiled, model-free shortest-proof oracle found **23 of 32 roots** attainable within the common action menu and budget; the public-order `auto8` control closed **3 of 32**.

The primary endpoint is **full-proof completion by the trained AB dispatch**.

| Training seed | Raw JSON | Typed | Typed plus pretty | Independent attainable oracle |
| --- | ---: | ---: | ---: | ---: |
| 113 | 2/32 | 4/32 | **6/32** | 23/32 |
| 127 | 5/32 | 4/32 | **6/32** | 23/32 |
| Mean matched-seed completion | 10.9375% | 12.5000% | **18.7500%** | 71.8750% |

Native source and checkpoint hashes were verified before each scoring run. All three `AB`, `BA`, and `auto8` policies were evaluated and replayed for each cell; the comparisons above use trained AB only.

## Predeclared screen and uncertainty

For each theorem lineage, the difference in certified completion was averaged across seeds, and a **descriptive** 2,000-draw root bootstrap was computed over the **32 independent theorem roots**.

| Paired contrast | Effect on native proof closure | Descriptive 95% root-bootstrap interval |
| --- | ---: | ---: |
| Hybrid minus raw | **+7.8125 percentage points** | **0 to +15.625 points** |
| Hybrid minus typed | **+6.25 percentage points** | **−1.5625 to +14.0625 points** |

The prospectively registered development screen required at least **+5 absolute points over both controls**, with the hybrid tying or beating raw in each seed. Its manifest field `registered_screen_passed` is `true`.

The second interval spans zero, and the first touches zero. All 64 seed-root measurements reuse the same 32 theorem lineages. The design is an exploratory *development* study, with no FSI-01 matched-sham calibration or external Mathlib/real-world proof transfer claim.

Training-source discrimination was also limited and seed-dependent. Root-weighted optimal-action accuracy on strictly discriminating **training** forks was:

| Seed | Raw | Typed | Hybrid |
| --- | ---: | ---: | ---: |
| 113 | 36.56% | 41.96% | 54.67% |
| 127 | 44.59% | 46.26% | 42.00% |

These source-only rankings do not substitute for certified proof completion. The hybrid's positive calibration effect at both seeds provides a reason to test public pretty-state retention further.

## Scientific gate and next development step

The strongest native model closed **6/32 = 18.75%**, below the previously registered **40%** trained-order competence requirement. The missing predeclared, independent **uniform-menu native proof baseline** is another gate. The existing `auto8` public-order policy is a different control. Consequently the formal disposition remains `BLOCKED_INCOMPETENT_CHECKPOINT`, even though the **representation development screen passed**.

A useful next **CPU-only** pre-gym test would prospectively register a budget-controlled public pretty-text dosage study with separate, fresh train/calibration root lineages. Candidate-premise coverage must remain mandatory, the same-worker reproducibility contract must hold, and the exact public token/byte allocation should be audited. It should add an independent root-seeded random-menu proof policy to calibrate the native competence baseline, then ask whether extra public pretty context improves proof closure beyond this one hybrid dosage.

Cross-runner bitwise numerical reproducibility remains unresolved despite successful one-worker training. The raw, typed and hybrid learned weights in this study were created on the **same** CPU worker, which protects this matched training comparison from one known source of variability. No larger gym or GPU-scale architecture is promoted by the present data.

## Evidence links

- [Preregistered LSA-02E protocol](../docs/lsa02e_typed_pretty_cpu_protocol.md)
- [Hybrid typed and pretty encoder](../research/lean_cgrt_hrm400/ppt_gym/typed_state.py)
- [One-worker matched CPU fit and duplicate checks](../research/lean_cgrt_hrm400/ppt_gym/pretty_train.py)
- [Native root-level source/seed proof audit](../research/lean_cgrt_hrm400/ppt_gym/pretty_compare.py)
- [Full six-cell GitHub Actions run 45182664891](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182664891)
- [Native source artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182664891/artifacts/14822067529)
- [Same-worker training artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182664891/artifacts/14822706498)
- [Root-paired result artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45182664891/artifacts/14823320416)

Actions artifacts expire according to the workflow's 14-day retention. This committed report retains the exact source hashes, denominators, result and decision boundary.
