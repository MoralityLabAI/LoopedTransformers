# NW-01: original three-seed recurrent witness retention and matched causal-state results

**Original registered retrospective neural pilot: PASS, 5/5 gates.** First prospective-to-model registration `42983fba14f3a616e9697523bce8a533fa8c6a84` was frozen before training, using the already published [NOR42 64-pair source roster](https://github.com/MoralityLabAI/Compactification-Theory/blob/0a330602c72ec34017178857956e306710f724a1/research/nor42_paired_program_ood/prospective/RESULTS.md) as retrospective model evaluation. The [first hosted NW01 model run #44145535924](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44145535924) passed both pre-data CI and its original three-seed model job. The first raw complete `nw01_original_pilot.json` is **162,716 bytes, SHA256 `bf1fc415802df28def64e8f5a2a16a3933d78d2f11ada0df5f0976569baf227d`**, with all three original model checkpoints and full recurrent state trace arrays in [artifact 14357125920](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44145535924/artifacts/14357125920).

## Source and model provenance

| Frozen item | Exact source/evidence |
| --- | --- |
| Original 64 NOR42 public-source pairs | SHA256 `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033`, 9,059,929 bytes |
| NOR42 original first registered grade | SHA256 `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`, 201,956 bytes |
| Immutable first NOR42 source/grade ZIP | SHA256 `e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d`, 2,799,867 bytes |
| Deterministic public-only feature-roster digest | SHA256 `edf2dfe01ac1f04e53ffb3ffab669d5b40891b194266b201e2264e2d84ce4356` |
| Model architecture | Tied 48D GRUCell, 8 passes, fixed first-pass-only witness input |
| Fit | 512 independent synthetic source roots, 200 AdamW updates per model, batch 32 source roots (128 paired-view examples) |
| Model training seeds | 11, 23, 47 |
| Heldout evaluation | 64 previously published NOR42 source roots; 128 equal-prior clean/hidden world rows; 16 original 4-case source replicate blocks |

The source feature extractor only inspects the public query, header, original native mask and input sparsity. The hidden-world first true coefficient residual β, revealed separately in the **authenticated arm**, comes from the original completed NOR42 grade. In the public-only arm the clean and hidden worlds have **exactly identical 50D public features and witness zeros**; the sham arm carries a source-matched, label-independent residual identical in both worlds. Real-world authentic coefficients are an external premise; the synthetic β is an easy positive control.

## All five originally registered model-pilot gates

| Original gate | Empirical result | Frozen outcome |
| --- | --- | --- |
| G1 Public/sham paired information firewall | All seeds public 50.0%, sham 50.0% accuracy; public/sham paired state and logit discrepancies exactly zero in original 8-pass receipt | PASS |
| G2 Witness readout reachable at final loop | Authenticated **128/128 world labels correct for each of 3 seeds**; Brier 0.0000056–0.0000087 | PASS |
| G3 True witness retained across 8 loops | Final public-matched clean/hidden state distances 10.51–10.68 L2; paired hidden-minus-clean probability +0.9946 to +0.9956 | PASS |
| G4 Matched donor state controls versus L2-matched random | Pass-4 donor-minus-random absolute final-logit effect **+10.83**, **+11.34**, **+11.21** across seeds, all well above predeclared +0.10 | PASS |
| G5 Original-source, raw model/trace and block-unit integrity | 64 unique completed source roots, 16 replicate blocks, all three model/NPZ state/checkpoint SHA files retained | PASS |

The original scientific endpoint was a model **pilot**, supported by the frozen experimental source geometry and a reproducible synthetic witness. These gates do not certify a named BPD parameter component, a true deployed target-output oracle, or successful neural induction of the entire RRC25 continuation.

## Depth and Brier outcomes

| Seed | Authenticated accuracy pass 1 | Pass 2 | Pass 4 | Pass 8 | Final authenticated Brier | Public-only Brier | Sham Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000083 | 0.25203 | 0.49758 |
| 23 | 75.78% | 100.00% | 100.00% | 100.00% | 0.0000056 | 0.25103 | 0.49798 |
| 47 | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000087 | 0.25117 | 0.49750 |

The public-only condition achieved exactly the mathematical paired 50% ceiling and its Brier was close to the **exact 0.25 floor**. The sham condition also scored 50%, with Brier near **0.50**, because the witness-trained model strongly trusted a purported coefficient discrepancy that was independent of the true world. This is a substantive negative-control result: an unauthenticated signal can generate highly confident errors. The sham root-paired Brier penalty is a direct overconfidence measure.

Final learned witnessed separation is already high at pass 4 for all three models. The first injected coefficient residual remains in a recurrent state whose clean/hidden L2 separation grows from roughly 2.75–2.88 at pass 1 to 10.51–10.68 by pass 8. Public-only and sham paired state distances are zero for all recorded passes.

## Causal state transport and sham specificity

The donor patch is the **entire 48D hidden state** from the *opposite* clean/hidden world of the SAME public source at the same recurrent pass. All source features are identical within the matched pair, and the witness is only injected at the first pass. Swapping a state at passes 1, 2, 4 or 6, then running the rest of the shared GRU, recovers the donor's own first-run final logit **exactly** in all original runs. The same-world identity patch also has zero final-logit change.

A source/model/pass-seeded random-direction control receives **the same state L2 displacement** as the donor-target difference. At pass 4:

| Seed | Donor mean absolute logit shift | Norm-matched random mean absolute shift | Donor minus random | 95% descriptive interval across 16 source blocks |
| --- | ---: | ---: | ---: | --- |
| 11 | 11.993 | 1.166 | **+10.827** | [10.597, 11.057] |
| 23 | 12.369 | 1.031 | **+11.338** | [11.020, 11.656] |
| 47 | 11.985 | 0.779 | **+11.206** | [11.006, 11.406] |

This is a measured **causal effect of swapping the full informative recurrent state** under a synthetic witness. A matched random vector of equal norm fails to reproduce that magnitude, showing direction-specific signal transport. The identical-public-input structure makes exact donor restoration a particularly strong engineering control while also preventing claims that the model separately inferred an unobserved hidden world.

## Limits and next mechanistic tests

1. The model is a small tied GRU verifier trained to classify candidate correctness after an explicit coefficient observation. The full future program is supplied by the NOR42 benchmark; the model **was not trained to induce the RRC25 clean program from demonstrations**. That is a separate harder model-learning experiment.
2. The NOR42 source roster was already public when NW01 training was registered. The 64 observed pairs are 64 source roots inside 16 generator blocks, so the result is **retrospective**, with uncertainty and hyperparameter reuse limitations. The extremely strong positive witness signal makes heldout witness classification easy.
3. Full-state donor patching can transport a synthetic truth indicator as expected. To test a BPD dark fiber or circuit basis, decompose the shared recurrent matrix and register **named component or subspace interventions**, matched replacement trajectories and independent negative controls. The BPD-LG and EvalMath information-floor companion prevents treating decoder scores as component identifiability.
4. The sham Brier near 0.5 is the most consequential operational finding in this pilot: a model that can use a well-formatted genuine witness can also become confident when a witness-looking residual is false. The next proposed run should train with provenance-aware uncertainty and independently measure its resistance to sham or revoked evidence.

**Source/statistical interpretation:** the [EvalMath exact paired-world result](https://github.com/MoralityLabAI/EvalMath/pull/52) proves an equal-prior 0.5 deterministic public-only accuracy ceiling, Brier floor 0.25, and exactly 0.5 accuracy for a label-independent root-paired sham. The measured recurrent donor effects are a separate empirical evidence class. No numerical claim here establishes a general theorem of neural uncertainty or a real-world trusted-oracle implementation.
