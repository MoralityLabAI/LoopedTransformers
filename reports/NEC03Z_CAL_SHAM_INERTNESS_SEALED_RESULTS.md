# NEC-03Z-CAL: eight-root sham inertness dose calibration and sealed evaluation stop

**Completed 22 November 2026. Scientific evidence: original source-training in-sample calibration only.** The historical continuous physically matched disjoint SAE quartet controls used in NEC03X can damage functionally correct native model predictions. NEC03Y independently verified source-training additive sham damage in the earlier groups40–47. This study registers an explicit **functional-inertness and physical-variation calibration** on a separate original training-source panel, with a hard stop before any analysis-evaluation data are read.

The original [NEC03Z source-calibration registration commit 5c24641](https://github.com/MoralityLabAI/LoopedTransformers/commit/5c24641c05d995c4fe2c01684c86fec81da527a7) locked historical native/SAE checkpoint identities, the old unchanged 495-support/32-step sham solver, five common dose fractions, source training groups48–55, the sealed group56–63 panel, all four calibration gates and an independent neural audit **before scoring**. Two later audit-only corrections were recorded at [46c8697](https://github.com/MoralityLabAI/LoopedTransformers/commit/46c8697e19bf66ea546ddf8c60d04f9c9f5b2ba6) and [047d27b](https://github.com/MoralityLabAI/LoopedTransformers/commit/047d27b9817779c521d3f91bc3952d6e62bfdb65). The original model scorer, sham selection and all scientific thresholds stayed byte-for-byte unchanged.

[**Final fully successful hosted experiment 44980365568**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44980365568) passed original NEC03O A/B native+SAE source ancestry, **8/8** preflight unit tests (including independent auditor/aggregate imports), **8/8** original-root source scorers with independent checkpoint/SAE/physical dose/sham replay audits, and the final pure-stdlib eight-root calibration decision. Its audited aggregate artifact **14737831835**, named **nec03z-eight-root-calibration-and-sealed-eval-decision**, contains the full immutable JSON gate result and 8-root ×2-arm ×5-dose calibration CSV. Individual root source and independent-audit raw artifacts are also retained in that same Actions run.

## Native/SAE source population and exact cohort isolation

The model population is precisely the old eight independently seeded five-input Boolean development worlds **dev5-000..007**, their paired composed and additive **frozen reset1800 native** teacher models, and model-specific 128-feature TopK16 SAE dictionaries, first40 source-fit/last40 source-rank and historical graph-selected recurrent sites/hubs. Every old world and both native arms remain represented. Original previously source-ineligible roots **3 and 5** retain their historical ITT-zero status if any later source intervention is evaluated; their original native models are still present descriptively.

The **only newly scored input histories** are original native/SAE already-trained source groups **48–55**, eight groups with complete 32-focal-mask contexts each. This yields **256 source inputs per model arm, sixteen arms and 4096 original source-case-arm rows**. These models had originally trained on all 80 source groups, so this is a *method-calibration split within source-trained data*, **not** a fresh population, world or model holdout.

The separate, predeclared candidate analysis panel **source groups 56–63** has **zero model forwards, zero sham interventions, zero task labels read and zero outcome measurements in NEC03Z**. No original eight-world development-evaluation/reserved, 88 recruited-world data, 32 five-input embargoed or old 64 four-input confirmation data were opened.

## Fixed physical sham and five-dose calibration

The old NEC03X control uses the original high-source-ranked active SAE feature and its three most physically cosine-aligned active decoder columns as one coherent source quartet. The continuous disjoint sham searches **495** remaining active four-code supports. Within each support it fits four **nonnegative coefficients bounded by three times the original source code activations**, using **32** deterministic projected gradient steps and the original fixed physical-gradient/angular, group-cohesion, cross-Gram and output-vector cosine gates, with no true-label routing.

Source and sham are removed from the same original native full-affine SAE nonhub donor state with original graph hubs live. Both original output-32D L2 removal vectors are first attenuated to the **same smaller physically attainable dose**, with no amplification and 1% matching tolerance. Both then receive one of the predeclared common multipliers

\[
t\in\{1.0,\ 0.5,\ 0.25,\ 0.125,\ 0.0625\}.
\]

Every original source input must have a physically valid sham on all its historical recurrent source sites, and the scaled shared output dose must remain positive. Invalid source/sham histories receive the identical unmodified full-affine donor on every source site. Each native teacher recorded **14 complete nonlinear 12-visit recurrent modes**: clean, hubs lesion, full native nonhub, full-affine SAE, and five original source/sham dose pairs. Teacher labels were consulted only after all source/sham routing and vector dose choices were fixed.

## Independent source replay and first two failed hosted runs

The initial [Actions run 44980017449](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44980017449) passed **6/6** source protocol tests and executed the frozen original source scorer, then every independent root audit stopped before any independent model replay because the auditor imported a nonexistent JSON helper. That original run remains recorded as failed. Commit **46c8697** defined the auditor-local JSON reader, added an audit import preflight test and revised only the pinned auditor and test identities.

The subsequent [Actions run 44980253153](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44980253153) passed **8/8** original independent root checkpoint/SAE/solver/14-mode native replay audits, then its final root aggregate stopped because the scorer and independent-auditor evidence use different field names for the protected cohort. Commit **047d27b** aligned those *aggregate-only* provenance checks, compared independently generated floating-point physical dose medians within a tight tolerance while preserving exact correctness and case integer counts, and added the aggregate import preflight. These **audit-side repairs** did not alter any of the 495-support solver, original source groups, model/SAE parameters, task truths, five dose multipliers or gate thresholds. The third full run **44980365568** passed all stages without any additional intervention change.

## Calibration: physically matched coverage and baseline headroom

All six historically paired-source-eligible world roots are used for the calibration stop gate. At every one of the five doses, the original continuous sham's **all-site physical admission** remains unchanged:

| Eligible world | Composed matched controls /256 | Additive matched controls /256 | Composed full-affine correct /256 | Additive full-affine correct /256 |
|---|---:|---:|---:|---:|
| dev5-000 | **183** | 220 | **234** | 255 |
| dev5-001 | **131** | 256 | 256 | 256 |
| dev5-002 | **122** | **101** | 256 | 256 |
| dev5-004 | 215 | **171** | 256 | 256 |
| dev5-006 | **103** | **174** | 256 | 256 |
| dev5-007 | 238 | 253 | 256 | 256 |
| **Qualified six-world totals** | **992/1536 (64.58%)** | **1175/1536 (76.50%)** | **1514/1536** | **1535/1536** |

The preregistered physical-coverage gate requires **at least 205/256 matched cases in every original eligible model arm**. Four composed arms and three additive arms fail it, independently of the common physical dose multiplier.

A separate preregistered task-headroom gate requires **at least two of six independent qualified roots per model family** to score in the **60–95% full-affine source accuracy band**. The observed counts are **1/6 composed (root0)** and **0/6 additive**. The original additive teachers are effectively saturated on this source panel. Across all eight historical roots including the original source-ineligible diagnostic roots, full-affine correctness was **1977/2048 composed** and **2047/2048 additive**.

## Sham functional damage across the predeclared dose ladder

The damage denominator is **original full-affine correct AND physically matched source/sham valid** input cases. Incorrect baseline cases and sham wrong-to-correct repairs are recorded separately, and repairs never cancel the sham correct-to-wrong count. These are source-training diagnostics; repeated masks within a world are correlated and do not support treating case arms as independent world roots.

| Original shared-removal fraction | Composed sham correct→wrong | Additive sham correct→wrong | Per-arm functional quality |
|---|---:|---:|---|
| **1.0000** | **33/975 (3.38%)** | **58/1174 (4.94%)** | Some qualified arms fail ≤2% |
| **0.5000** | **5/975 (0.51%)** | **0/1174 (0%)** | Root0 composed still **4/166 (2.41%)**, fails |
| **0.2500** | **0/975 (0%)** | **0/1174 (0%)** | Passes ≤2% in all qualified arms |
| **0.1250** | **0/975** | **0/1174** | Passes damage; physical variation <0.05 in composed roots4/7 |
| **0.0625** | **0/975** | **0/1174** | Passes damage; variation floor fails many model arms |

At **quarter dose**, source/sham output similarity and control selection remain exactly the frozen NEC03X policy. No correct-to-wrong sham flips occurred among the 975 composed or 1174 additive qualified previously correct matched source-training cases. The **median minimum-visit dose / natural source-physical-contribution** exceeds the predeclared **0.05** variation floor in **every eligible arm**, ranging from roughly 0.093 to 0.233. This is a **useful calibration result**: moderate common-dose attenuation reduces observed generic damage while preserving a measurable physical perturbation on the subset of cases for which the old physically strong null is available.

At half dose, pooled damage is small but root0 composed violates the predeclared **per-arm** 2% cap. At eighth dose, some composed arms lose their registered nontrivial variation floor even though sham damage disappears. The analysis did not pick a dose by maximizing the source-minus-sham necessity effect, and no new causal efficacy significance test was performed.

## Frozen scientific decision: STOP and seal

All five original global dose candidates failed the complete **four-gate policy**, even though **t=0.25** passed the sham-inertness and minimum-relative-physical-variation requirements. The original matched-control availability and teacher baseline headroom gates failed independently and could not be repaired by uniform attenuation.

The exact machine-audited aggregate decision is

**STOP_SEAL_ANALYSIS_SOURCE_GROUPS_56_TO_63**

The final output records **selected dose fraction = null**, **0 scored analysis-group source inputs**, **0 newly recruited-world outcomes**, **8/8 independent source-world audits passing**, and scientific gate **FSI-01 REDESIGN_OR_PILOT_ONLY**.

This calibration shows a dose/semantic-damage tradeoff within an already source-trained, near-saturated model population. It establishes neither functional inertness on independent new tasks nor a source-specific causal SAE mechanism. Any next experiment should first create a pre-registered **difficulty-controlled non-saturated Boolean program/world benchmark**, with calibration-world performance and negative-control damage gates established before evaluating a separate root cohort.

[NEC-04A issue #292](https://github.com/MoralityLabAI/LoopedTransformers/issues/292) scopes that generator, world-root split and jointly physically matched/semantically calibrated negative control. The original NEC03Q/R/S/T/U/V/W/X/Y results are unchanged. The **69/88 previously source-eligible NEC03P recruited independent roots retain zero measured causal-sham outcomes**, so the **≥59 independent eligible causal-sham-root outcomes** needed for a future 95/95 tail tolerance estimate remain unmeasured.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
