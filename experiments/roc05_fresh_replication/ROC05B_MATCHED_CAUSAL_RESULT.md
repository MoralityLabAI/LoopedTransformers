# ROC-05B: learned rule transfer does not confer a same-dose rank-seven steering advantage

**17 November 2026 · preregistered causal primary FAIL.** The ROC-05B protocol was frozen at commit `ef18aef009e72c0b0e75de9331d6135db7977d98` before any ROC-05B intervention measurement. The ROC-05A answer accuracies were already exposed at that time, so this remains a development-family causal diagnostic, not an independent causal rule-identity holdout.

**Original-source and model pin:** twelve ROC-05A first-run model checkpoints from [run #44101721734](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44101721734) and their exact SHA-256 hashes in `configs/roc05b_matched_rule_causal_v1.json`. The second ROC-05A CPU replay changed 10 of twelve checkpoint bytes, so **none** of the repeated-run models were substituted for the original first cohort. Both train corpora and the 256 exposed development records were checked against their source SHA; all six source rule/schedule pairings remain frozen. No model was refit.

## Native recurrent observer and matched intervention

The original `QueryConditionedTableMemory` model receives the explicit early source-A table, B/C distractors, carry events and a final query selecting one row of a learned 8×8 table head. The query is excluded from the recurrent memory update by design. The intervention point is after all pre-query source events plus the first **four of eight** tied residual reasoning steps. The original **four-step suffix** and original table head/query row score all interventions, with staged/unperturbed logits checked against native model.forward within 1e-5.

The primary observer is a **rank-7 train-only supervised covariance subspace** of the 1,024 training midpoints versus their true queried-answer onehot labels, with a separate rank-7 PCA observer and seeded random controls. ROC-05B excludes the arbitrary eighth direction of the centered eight-class output covariance. On 256 exposed development records per model, the legal initial A-table counterfactual swaps outputs at queried row q and (q+4) mod 8. Its new exact answer differs; the query and all intervening events remain fixed. The original full legal edit is scored at its natural physical dose.

For each original midpoint, the altered source input also defines an input-softened interpolation. Its midpoint L2 is matched to **0.05, 0.10 and 0.20** by a 64-cell first-crossing search plus 24 bisections. The soft input is *off the discrete source manifold* and has no independently valid symbolic correct label. It is a physical input-level sensitivity control. Native hidden injections at the same midpoint and L2 norm include (i) the source-change projection onto rank-7 supervised U, (ii) its dark complement, (iii) a PCA rank-7 projection, (iv) the full source-induced hidden direction, (v) SHA-keyed rank-7 random observed perturbation, and (vi) seeded isotropic control. Every arm has an eligibility/norm audit.

The registered directional output measure is **signed target-margin gain** for the legally changed correct label y′ versus the old label y, measured on the original model logits:

\[
g(z';z,y,y')=(z'_{y'}-z'_y)-(z_{y'}-z_y).
\]

The single primary test uses the six **high-512** original first-run models and the mean per seed of rank-7 observed-source steering minus equal-L2 soft source-input steering, at r=0.20. It requires ≥80% physically eligible cases per model, at least five positive model-seed differences, positive six-seed mean and exact one-sided sign-flip p≤0.05 over 64 sign assignments.

## Measured primary result

All **256/256** source-edit cases were physically paired and eligible at radius 0.20 for every high-512 seed.

| Frozen seed | Rank-7 observed-rule gain | Matched soft-input gain | Observed minus input |
|---|---:|---:|---:|
| 20261117231 | +0.117114 | +0.197126 | **−0.080012** |
| 20261117233 | +0.107219 | +0.180734 | **−0.073514** |
| 20261117239 | +0.130885 | +0.186401 | **−0.055517** |
| 20261117241 | +0.120991 | +0.186984 | **−0.065994** |
| 20261117251 | +0.113885 | +0.185136 | **−0.071252** |
| 20261117257 | +0.117713 | +0.179745 | **−0.062032** |
| **Six-seed mean** | **+0.117968** | **+0.186022** | **−0.06805338** |

**Primary registered gate: FAIL.** Zero of six paired model differences favor rank-7 observed steering; the mean observed-minus-soft effect is **−0.06805337873205038** in target-margin units. The one-sided exact 64-sign test for the preregistered *positive* direction gives **p=1.0**. This fails both the directional-count and the exact test conditions while passing the physical eligibility requirement. The reverse direction is a descriptive pattern observed in all six seeds and receives no post-hoc confirmatory promotion.

The comparator is a softened off-manifold input feature edit whose midpoint effect is dose-matched. Its larger new-target directional margin gain is **not** itself semantic correctness under a valid discrete alternative rule. The full legal *discrete* source edit supplies that separate task check: the high-diversity models predict the changed correct answer in **1,157/1,536 = 75.33%** of cases, while their original unedited answer accuracy is **1,147/1,536 = 74.67%**. The four-rule models score **326/1,536 = 21.22%** on full legal edits and **315/1,536 = 20.51%** unedited. This gives useful grounding of the intended rule edit while preserving the distinction between physical interpolation and legitimate symbolic inputs.

## Computed-change and null-calibrated mechanism cards

Every one of twelve frozen models contributes two explicitly indexed exposed shadow states and two train-only rank-7 observer types. For each, ROC-05B computes the dense four-step suffix tangent `T`, computed-change `D=T−I`, the within-observed block `A`, dark-to-observed block `B`, observed-to-dark block `C`, and fixed-query future-logit Gram `H=(R_q T)^T(R_q T)`. Each model/state receives **128** SHA-keyed rank-conditional Gaussian QR null projectors, with six predeclared one-sided add-one p-values and Holm adjustment under ROC-01 Amendment C. All undefineds and abstentions are retained.

Across the 12 **supervised** high-512 shadow cards, the mean rank-matched ratios are **A/null = 4.969**, **B/null = 1.749**, **C/null = 1.550**. Average capture of original fixed-query future-logit Jacobian energy is **26.31%**, with only 7 observed dimensions of 128. However, the full joint ROC-01 diagnostic predicates classify **0 of 48** total supervised/PCA shadow cards: **all 48 remain unclassified**. The large incoming-coupling B relative to random indicates that the task-supervised observer does not form a nearly closed computed-change quotient under this test. Capturing a substantial portion of native output sensitivity alone cannot certify a rule-specific low-rank mechanism.

The lower-competence four-rule models are retained as controls; their rank-7 observed-minus-soft margin differences are small and positive (~+0.0023 averaged over six seeds), under an original task accuracy of only 20.51%. They do not satisfy or reverse the high-512 primary causal claim.

## Verification and limitations

[Successful GitHub Actions replay #44107080894](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44107080894) independently executed all six original first-run paired seed jobs, numerical midpoint/tangent tests and a standard-library receipt auditor. The independent census covers **73,728 per-case model/arm/radius/route rows**, exactly **48** complete rank-null observer cards, original and changed labels, physical L2 errors, and all 64 six-seed sign assignments. The source-only geometric CI also passed. A separate earlier full replay [#44106910429](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44106910429) independently generated the same primary effect, before a packaging improvement added frozen model binaries.

The final **self-contained ROC-05B ZIP** preserves all twelve *first-run* original `.pt` model bytes and receipts, three exact ROC-05 train/dev sources, all per-case causal CSV.gz files, observer/null cards, source and test scripts, the independent summary and a **69-file SHA-256 manifest**. Verified standalone archive SHA-256: `752729e84c30b92ef530591e2b3404df3869c037480eff244042adf07342987a`. The public original model/source hashes in the preregistration agree with the archive bytes.

The conclusion is scoped: **training diversity successfully improved transferable queried-answer competence, while the particular task-supervised rank-seven recurrent observer failed to outperform a same-physical-dose input interpolation in directing changed-rule output margins**. This is a direct negative causal-intervention result for that selected observer, on exposed development cases. It neither excludes other observer ranks/architectures nor proves anything about general natural-language reasoning. The final query-to-row route is an explicit architectural bias; complete eight-row table recovery in ROC-05A remains low.

No ROC-05 reserved test examples were generated and the original ROC-02 sealed source families 6/7 remain unopened. Independent future rule-set evaluation and a broader ROC-01 intervention cover would require new prospective protocols.
