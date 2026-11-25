# LS-NEURON-14 evidence plan

## Question and decision

This is a prospective development pilot over 24 new perturbation seeds, 445 through 468. It asks whether an interval-aware ranking and acceptance rule increases the chance that a fixed-budget search produces a newly certified eight-GELU mod4 cell. The initial perturbation dose is 15 percent, and all four arms share each starting cell.

The primary comparison is `bound_single` versus `point_single`. The secondary comparison is `bound_paired` versus `point_paired`; the test family contains both comparisons. A useful operational gate for each comparison is at least 12 newly certified candidate endpoints out of 24 and an advantage of at least six new certificates over its matched control. The primary operational gate is evaluated on the single-move comparison. A secondary success cannot replace it.

Each arm uses 16 search rounds, two nonlinear scale anchors, exactly five fitted slots per anchor, and at most 40 function evaluations per fit. Candidate ranking and post-fit acceptance use the assigned score on 17 points per state region. The continuous point-loss fitter and cheap profiler stay fixed at nine points per region. Each arm receives 161 continuous fitting calls with a cap of 6,440 function evaluations. Profile slots total 3,360 for single moves and 5,664 for paired moves. Log both the number of fitting calls and the realized function evaluations because equal caps can yield different elapsed work. Preserve no-op candidates and every unsuccessful seed.

## Unit, outcome, and target population

The independent unit is one pseudorandom initialization draw from the declared perturbation generator. There are 24 units, with 24 binary paired cases in each comparison. The same units appear in both comparisons. The 96 final model files, the 32 transition cases per model, and every trajectory prefix remain observations attached to those units.

The primary binary endpoint is `initial POINT_VIOLATION and final CERTIFIED`. Use all 24 assigned seeds in the denominator. An initially unresolved check contributes zero to this repair endpoint even when the final model is certified. Report the broader `initial not CERTIFIED and final CERTIFIED` acquisition count separately, together with final certificates and certificate losses. Final point violations and unresolved checks also count as zero, with their separate verdicts retained. The initial witness establishes a violation that the final exact certificate repairs. Verification incompleteness can affect measured repair rates.

Population language applies only to fresh draws from this perturbation generator around the hand-built mod4 template, with this codebook, scale convention, search algorithm, and budget. The model architecture and its invariant regions were developed on earlier experiments. Native recurrent transformers and sealed ARG data remain outside this study.

## Statistical report

Use the unmodified LoopedTransformers FSI bridge and EvalMath FSI-01 module. Freeze two registration JSON files before generating fresh seeds; retain the local hash and the first remote commit when available. Store outcomes in separate JSONL files and hash both inputs in each report. Local prospective registration does not imply registration in an external service.

For each comparison, show the candidate-only wins, control-only wins, ties, and their seed identifiers. The conditional exact McNemar test targets equal marginal certified-repair probabilities, assuming the independently generated seed pairs are exchangeable under that null. Report its two-sided exact fraction and Bonferroni adjustment for the two planned comparisons. Include the seed-level paired difference and FSI confidence output. Marginal Clopper-Pearson intervals describe the declared seed-generator population. The paired percentile bootstrap interval is descriptive and can have weak coverage with a small cohort.

The scientific effect floor is an advantage of 1/4, or 25 percentage points. The finite count increment at 24 seeds is 1/24. At familywise alpha 0.05 with two comparisons, the FSI one-sided Hoeffding radius is `sqrt(2 log(40)/24)`, approximately 0.55444. Its positive-floor policy therefore requires an observed advantage above approximately 0.80444, which means at least 20 net wins out of 24. The operational gate at six net wins does not satisfy that statistical effect-floor gate. Six wins and zero losses produce two-sided exact p=1/32 and Bonferroni p=1/16; seven wins and zero losses produce p=1/64 and adjusted p=1/32.

Every report retains `PILOT_ONLY` as the study's evidence class, even if the operational count gate passes. The FSI output remains visible with its own exact verdict and reasons.

## Required preflight and its scope

The official EvalMath preflight was fetched at commit `43f08e55975e68a5ac0d5caf20df12370f3d2299`. It was run with `--strict`, using only LS13 development receipts. The input compares the historical `paired_restart` and `single_restart` certificate outcomes on seeds 433 through 444. Their mean paired advantage is 1/12. This is historical design context; LS13 supplies no observed effect of the new interval-aware ranking.

The historical `continuous_restart` versus `continuous` endpoints supply zero observed differences, but their compute differs and they cannot calibrate a matched sham for LS14. To prevent those zeros from implying a calibrated mechanistic null, the manifest supplies the universal support bound of one on the absolute difference of two binary endpoints. This external bound is mathematically valid and deliberately uninformative. It has no empirical quantile-confidence claim. The preflight target effect equals the planned scientific floor of 1/4. The returned decision is `REDESIGN_OR_PILOT_ONLY`, with strict exit status 2. Its field `upper_sham_quantile: 1.0` contains the supplied universal support bound. Its reported approximate resolvable effect belongs to this conservative historical input and must never be presented as an LS14 power estimate.

A direct LS14 power estimate remains unavailable because no development-only bound-ranking effects or physically matched ranking shams exist. The historical preflight is retained as an auditable reason to keep the current experiment at pilot scope. Twelve development units also fail to identify a population 95th-percentile sham envelope with 95 percent confidence. Under an iid continuous sham model, even the sample-maximum tolerance bound needs at least 59 independent units because `1-0.95^59 >= 0.95`.

## Exposure and controls

The final exact certificate is scored after search. The mathematical target regions and transition law are known to the algorithm, so this experiment does not estimate transfer to unseen transition types. Prospective sealing refers to the fresh seed cohort and the absence of algorithm selection using its final outcomes. A protocol change after opening these outcomes must preserve the original run and be labeled as adaptive follow-up.

The point-ranking arms control the candidate generator, continuous fitter, anchor schedule, and fitting caps within each move class. Comparing average point loss to a worst-interval score changes both the aggregation of error and the curvature allowance. Any observed gain belongs to that combined change. The study lacks a point-maximum-only arm that would isolate the curvature term.

## Exact evidence

An exact rational verifier can prove the recurrence maps every point of each region of radius 1/8 into the target region of radius 3/32 for all 32 state/input cases. Adding any real disturbance of magnitude at most 1/32 then returns the state to the radius-1/8 region. Induction extends a successful certificate to every finite sequence of inputs for the specified exact-real recurrence. This property is model-specific and independent of seed-population confidence intervals.

The floating interval-aware ranking score remains a numerical proxy unless every operation is outward-rounded with a justified enclosure. Python rational arithmetic and the analytic bounds are separate from a Lean or proof-kernel certificate, and the exact-real recurrence theorem does not establish floating-point hardware rounding robustness.

## What would change the conclusion

A successful pilot would justify freezing the resulting algorithm and planning a separate cohort against the observed paired discordance distribution. Confirming a population advantage above 25 percentage points would require enough fresh independent initialization draws to clear an effect-floor interval, together with honest null calibration for any mechanistic claim. Broader architectural claims require native model tests and a different target population.

## Provenance

- Parent receipts: LS13 `certificates.json`, SHA-256 `fb1369fa9ca0e68db2afb5c345ce87c07c56e0e6d4bc080c736565c3b9ccefd7`.
- EvalMath source: commit `43f08e55975e68a5ac0d5caf20df12370f3d2299`.
- LoopedTransformers FSI bridge: commit `88aef2c14c6b9d75775c2379c2ee5cc6ce5484a9`, path `research_gym/scripts/evalmath_fsi_gate.py`.
- Exact source byte hashes and Git blob identities appear in `source_provenance.json`.
