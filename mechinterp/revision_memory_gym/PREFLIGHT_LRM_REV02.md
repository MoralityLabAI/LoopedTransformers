# LRM-REV-02: development preflight scope

**Prospective engineering and calibration stage, 19 November 2026 (Singapore).**
The scientific readiness status is
`PENDING_MATCHED_SHAM_AND_CONFIRMATION_DESIGN`.
Source and protocol are frozen before generating the first training or
development episode. Results from this stage remain development evidence.

## Registered scope

| Item | Contract |
|---|---|
| Independent unit | One independently initialized and trained GRU with its own training and development source namespaces |
| Planned units | Four model roots, numbered 0 through 3 |
| Per-root budget | GRU hidden size 32; 800 training steps; batch size 64; 2,048 training episodes and 512 development episodes |
| Evidence law | Twelve named old bits with six corrected positions, using the existing revision gym law |
| Primary contrast | Development mean Brier advantage of six truthful old-value reminders at the corrected set `S`, relative to the mean of four uniformly sampled six-position reminder sets `T` |
| Matched input dose | Six reminder tokens in each arm, using the same 17-dimensional token grammar and cloned post-correction recurrent state |
| Bayes comparator | Full-history posterior `p_H` after the original raw stream, schedule and accepted replacements |
| Scientific effect floor | 0.01 in unhalved multiclass Brier units, retained as a future design requirement |
| Current evidence class | Development-only competence and reminder-policy comparison |
| Scientific confirmation population | Undesigned and unopened |

The 2,048 distinct development episodes improve measurement within four
model roots. Reminder views and disclosure stages are repeated observations
within those roots. Training draws reuse the declared training episodes.
The report preserves each root's paired outcomes and every active control.

## Why strict readiness remains pending

The four random-`T` reminder policies are **active controls**. Each supplies
truthful observations selected by a different policy. These messages can
change a finite model's forecast, even though the full-input Bayes target
stays fixed. Differences between `T` policies therefore measure reminder-policy
variation. They do not constitute a calibrated null for the primary mean
effect.

EvalMath's current `preflight.py` requires nonempty observed pilot rows and
matched `sham_effects` for every independent root. It also requires a declared
confirmation root count. This stage has neither a matched-null assay nor a
confirmation design, so those inputs remain unavailable. The report records
the status above with `matched_sham_floor`, `planned_confirmation_units` and
`estimated_confirmation_power` set to `null`. It records
`strict_preflight_executed=false`. A readiness receipt will require actual
calibration observations and a registered future design.

The skill's prescribed order places independent development calibration
before its planner. Its resolution notes explicitly allow exploratory
pilots that retain all arms and source hashes. This stage supplies competence
measurements for that process. Any architecture or reminder-policy decisions
informed by these results expose this development readout to selection.

## Required inputs for a future mean-effect gate

A future protocol must define a matched-null assay for the same paired
root-mean Brier contrast, including its intervention information and physical
dose. Its uncertainty procedure must sample or resample independent model
roots. Record the complete comparison family, empirical measurement
resolution and a separately reserved readout before scientific scoring.
The root-mean contrast must retain its sign and the equal weighting of roots.

The existing planner first takes the largest absolute sham effect within
each root, then calibrates a quantile across those maxima. Its 59-root rule
supports a 95th-percentile/95%-confidence tolerance statement for individual
roots under the stated assumptions. That quantity is distinct from the
sampling distribution of the primary mean. A future mean-effect readiness
gate needs its matching estimator and uncertainty contract. With four roots,
the stated 95/95 rootwise tolerance bound is also unavailable.

## Units and inference interface

Unhalved multiclass Brier lies in `[0,2]`; a paired difference can lie in
`[-2,2]`. EvalMath's preflight accepts signed effects only in `[-1,1]`.
If that planner is used after an appropriate assay exists, divide every
effect, sham value, target, scientific floor and measurement resolution by
two. The 0.01 Brier floor becomes 0.005 half-Brier. Preserve the original
unhalved values in the receipt and declare the conversion.

FSI-01's manifest CLI accepts binary candidate/baseline outcomes. Brier stays
a continuous root-level endpoint. Its rational-effect sign-flip helper
requires the stated independent symmetric-sign null; its small-root bootstrap
interval is descriptive. This engineering stage makes no confirmation or
causal-mediation decision from four model roots.

## Pinned sources

Reviewed EvalMath commit: `bed334f032d31d777bad0e5152e732f57252092b`.

- [Mechanistic-resolution skill](https://github.com/MoralityLabAI/EvalMath/blob/bed334f032d31d777bad0e5152e732f57252092b/skills/evalmath-mechinterp-power/SKILL.md)
- [Resolution and evidence notes](https://github.com/MoralityLabAI/EvalMath/blob/bed334f032d31d777bad0e5152e732f57252092b/skills/evalmath-mechinterp-power/references/resolution-and-evidence.md)
- [Strict planner](https://github.com/MoralityLabAI/EvalMath/blob/bed334f032d31d777bad0e5152e732f57252092b/skills/evalmath-mechinterp-power/scripts/preflight.py), Git blob `0ff062f76f5534d9f18425f986ae3e750397f50c`
- [FSI-01 implementation](https://github.com/MoralityLabAI/EvalMath/blob/bed334f032d31d777bad0e5152e732f57252092b/src/spiral_ln/evaluation_finite_sample.py), Git blob `ef9ecee84aeefbeeb2a087639a77ad1420a7bc50`
