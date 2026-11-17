# ROC-04B: independent reserved-rule test confirms the permutation-diversity gain

**17 November 2026 · frozen twelve-model confirmatory test completed.** First frozen protocol: `configs/roc04b_reserved_permutation_confirmation_v1.json` at commit `f032c52db2ebb544b745b3972f5f86f4c47c8b38`, **before construction of the 384 reserved examples and labels**. Hosted run: [GitHub Actions #44087829448](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44087829448), with full 4,608-row independent audit green. ROC-04A's original twelve optimizer-step-1920 checkpoints are unchanged and all twelve model SHA-256 digests are verified.

## Pre-result provenance repair

The original frozen config inadvertently truncated the stored SHA-256 for seed `2026111719`, `low_affine4` to a 40-character prefix, and the first hosted run [#44087398369](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44087398369) stopped on that checkpoint identity. That first attempt generated the reserved examples and evaluated earlier models in memory, but stopped before emitting a full result or score archive. The amendment at commit `2624b30a72ba5b500684067ff0f186906062e2e3` replaced only the mistyped hash with the **original checkpoint SHA-256 found identically in ROC-04A's saved model receipt, reproducibility manifest and binary file**. Commit `1fff17e286308550abc50f5235735bf249673527` added a preflight check of *all twelve* model hashes **before** test generation. The experiment's source seed, reserved identities, models, primary analysis and gates remained frozen. The successful second run is reported with this complete correction history.

## Registered setup

The independent test uses one record for each of the original ROC-04A source-lock's **384 reserved permutation identities**, from table-pool indices 768–1151. Source event schedules are fresh, with fixed seed `20261117101`, variable delayed A rule queries, and source-B/C distractors. The exact symbolic interpreter and a direct table lookup agree on every answer. These rule tables are disjoint from the 512 high-diversity train rules, the 256 exposed development rules, and all eight original ROC-02 affine rules.

The test cases and labels were generated after the preregistration. Their source JSONL SHA-256 is `95614846bfac6fb66cdcf8c927cdf4a975ba92960dd6a2924ea344228dfef143`. The original ROC-02 separately sealed families 6–7 were never loaded or evaluated.

Both arms reuse the same 51,520-parameter, depth-8 query-conditioned tied recurrent models that were frozen in ROC-04A [PR #155](https://github.com/MoralityLabAI/LoopedTransformers/pull/155). All twelve checkpoint SHA values were pinned in the registration. The new test performs **zero training updates**.

## Primary confirmatory result

| Model seed | Four-rule accuracy | 512-permutation accuracy | Paired improvement | High model exact 8-row table |
|---|---:|---:|---:|---:|
| 2026111705 | 76/384 = 19.79% | 296/384 = 77.08% | +57.29 pp | 79/384 = 20.57% |
| 2026111707 | 75/384 = 19.53% | 265/384 = 69.01% | +49.48 pp | 38/384 = 9.90% |
| 2026111711 | 83/384 = 21.61% | 272/384 = 70.83% | +49.22 pp | 29/384 = 7.55% |
| 2026111713 | 85/384 = 22.14% | 295/384 = 76.82% | +54.69 pp | 55/384 = 14.32% |
| 2026111717 | 82/384 = 21.35% | 291/384 = 75.78% | +54.43 pp | 51/384 = 13.28% |
| 2026111719 | 85/384 = 22.14% | 264/384 = 68.75% | +46.61 pp | 22/384 = 5.73% |
| **Mean** | **21.09%** | **73.05%** | **+51.95 pp** | **11.89%** |

**Primary preregistered gate: PASS.** The high-diversity model outperforms the four-rule comparator in all **6/6 independent model seeds**, with mean improvement `+0.51953125`. The registered test enumerates all 64 sign assignments of the six paired seed differences, with one as or more extreme in the predicted direction: **one-sided exact p=1/64=0.015625**. This meets the frozen conditions requiring 5/6 positive seed directions, ≥20 percentage-point mean gain and p≤0.05. The same 384 test cases recur across models, so the independent model-initialization unit is six, not 2,304.

**Competence gate: FAIL.** Four of six high-diversity models meet ≥70% task accuracy; the criterion required five. Zero of six meet ≥50% full eight-row table correctness. The mean exact table accuracy is 11.89%. The architecture can predict the queried output on many fresh tables without recovering all eight rows of the carried rule.

## Legal edits and control behavior

Each test binding has a legal counterfactual that swaps its A-table output values at rows `q` and `(q+4) mod 8`. The query and all other events stay unchanged; the independent oracle verifies a changed correct label. The high-diversity models achieve **73.05%** mean task accuracy on those modified-rule inputs, compared with **22.22%** for low-four models. Both edits have a natural finite input cost and are described as semantic-task controls rather than physically norm-matched hidden-state interventions.

Changing an intervening B/C distractor's rule preserves the A-rule answer. For eligible cases, the high-diversity model's original prediction is retained under this label-preserving edit **93.55%** of the time on average; the four-rule model retains 51.12%. This suggests the high-diversity trained network is substantially more invariant to nuisance-rule changes on this generator, with the supplied query-to-row readout still an explicit inductive bias.

## Scientific interpretation and limits

ROC-04A exposed-development accuracy improved from 22.92% to 70.70% by changing source-rule diversity at equal model/optimizer/sample budgets. The present test repeats the effect on fresh, previously reserved permutation identities, improving accuracy from 21.09% to 73.05%. This supports a causal **training distribution intervention** claim: broader A-table exposure caused better transfer in this model/task design, under six paired model starts. It does **not** establish closed low-rank recurrent quotients, exact rule representations or an identifiable local Jacobian mechanism.

The underlying task is a literal lookup from a visible 8×8 binding table. A deterministic direct-row interpreter already solves it at 100%. The network is more than 26 percentage points below that exact oracle even in the high-diversity arm, and complete-table recovery remains weak. ROC-04A and ROC-04B share the same model seeds and architecture, so they confirm across unseen *source tables*, not new independently trained model cohorts. The exact six-seed p describes the registered matched model-initialization comparison under this fixed experiment.

## Receipts and reproduction

- Original source registration: [ROC-04A #155](https://github.com/MoralityLabAI/LoopedTransformers/pull/155), source manifest and 44-file SHA-indexed archive, 12 complete original model checkpoints.
- Frozen confirmation protocol: [ROC-04B registration](ROC04B_PROSPECTIVE_PROTOCOL.md) and source-lock commit `f032c52`.
- Independent test source, scoring and receipt scripts: `research_gym/analysis/roc04b_reserved_test.py` and `roc04b_independent_audit.py`.
- Hosted [successful test run #44087829448](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44087829448) and artifact `ROC04B-independent-reserved-permutation-replay`. The standalone archive includes the exact original ROC-04A training ZIP, all twelve checkpoint hashes, all 384 newly generated test inputs/labels, all 4,608 per-case predictions, six-seed CSV, independent audit JSON and a 16-member SHA-256 manifest.
- Full confirmatory standalone ZIP SHA-256: `4d6539560fba91bffe083e77bb219d01823382cef9b1cd72e44c20595920520a`.

A further model-seed replication, richer training-family diversity curve and measured readout-closure interventions could test the degree to which the remaining errors come from incomplete table transfer, query-selective decoding and recurrent-state drift. The original ROC-02 families 6–7 remain sealed.
