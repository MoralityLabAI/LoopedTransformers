# RAM-10 source-header-only dependency audit, before all new Lean collection

**November 15, 2026, methodologically post-registration.** The original
unmodified [preregistration](PREREGISTRATION.md) preceded examination of
TRAIN/DEV statements and all TEST module contents. It freezes MathCan commit
`7515eecf51a60e7f8cfcd0801fddf8a3cf039e89` and the exact split.

This audit inspected the public TRAIN/DEV source **imports**, top-level
declaration names and module Git blobs. Four TEST modules remain reserved
without opening their source text or proof scripts. No new Lean compiler
states, original reference proof bodies or model labels have been read for
this campaign. Every selected TRAIN/DEV name below follows the registered
first-six-original-declarations rule.

| Partition | Module | Original top-level declarations selected | Original imports |
|---|---|---:|---|
| TRAIN | BasinCapture | 6 | Init |
| TRAIN | BlockRemainder | 3 | Init |
| TRAIN | CoarseStateDependence | 6 | Init |
| TRAIN | FilteredOrbit | 3 | Init |
| TRAIN | InvariantBall | 4 | Init |
| TRAIN | MemoryProjection | 6 | MathCan.PairedHistory |
| TRAIN | MultiBasin | 4 | Init |
| TRAIN | OrbitStitching | 2 | Init |
| DEV | GradientStability | 6 | MathCan.MemoryProjection |
| DEV | OrderedJet | 6 | Init |

The expected original code-root inventory is therefore **34 TRAIN roots and
12 DEV roots**. The TEST selector remains the first two original theorem/lemma
declarations per each of `NOR07Volterra`, `PeriodOne`, `TacticFibers`, and
`TreeReconstruction`, giving at most eight original TEST theorem cases.

**Dependency limitation:** `MemoryProjection` imports `MathCan.PairedHistory`,
whose original module was reserved as RAM-08 PEF ancestry. The Lean module may
use its imported declarations and constants in proof contexts; its **tactic
observations, theorem roots, proof body scripts and positive labels** are
excluded from RAM-10. `GradientStability` imports `MemoryProjection` and thus
transitively shares that dependency. Original RAM-08 PEF theorem evaluation
is maintained separately, and the new result cannot retroactively modify it.

**Operational guard:** Before source-only data collection the CI TRAIN job
removes all four original TEST module files from its checkout. It builds only
the ten registered TRAIN/DEV modules and their imported dependencies. TEST
checkout occurs in a dependent job only after nine trained models and the
source data hashes have been frozen. This prevents prospective TEST theorem
proof bodies or goal states from entering the new training collector.

The original module selector, family-fair 80-action limit, hard tactic-diversity
gates, nine-model training rule and independent proof certification remain
exactly as registered. No extra roots or tuning were selected after reading
this dependency audit.
