# LOOPSCOPE-13 / LGA-10: curvature-matched shams fail source-only feasibility

**18 November 2026. Source-only engineering verdict `FAIL_SOURCE_FEASIBILITY`**, before first trained-model inference or opening reserved prospective seed `31064`.

The original [prospective protocol](PROTOCOL.md) was committed first at `95d9c6ab1c84d6709861ff0b6478e49cc96c5ea9`, then its exact source-only geometry and test implementations. The experimental target was to find eight distinct matched shams on the 65D first-order future-dark sphere, fixing current-head and learned-energy tangency, all ten first-order future logits, the natural pair dose, and the signed legal donor-vs-original margin **quadratic Hessian response**. A nontrivial sham must have `|cos θ|≤0.85` relative to the original source-donor future-dark component, with 32 independently seeded random-plane attempts per sham and a 64-angle fixed bracket grid.

## Original pilot receipt

The [source-only CI run #44319953707](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44319953707) first passed the seven unit tests: analytic symmetric-quadratic exact curvature matching, matching all first-order outputs and physical L2 dose, zero-null-radius handling, deterministic admission failure, invalid inputs, actual two-native-block FP64 HVP symmetry and central-gradient finite differences, and a native excluded-model smoke that explicitly recorded a matching failure.

The same CI then ran the first registered **8 bases × 8 legal donors = 64 excluded-only source-goal cases**. Its only model was a newly initialized, **untrained** TinyRecursiveModel with artificial quadratic energy; only source generator seed `61064` was used. The prospective `31064`, original trained teacher bundles, LGA-05 potentials and all original LGA-08/LGA-09 source cohorts remain unopened in this feasibility run.

| Registered source-only check | Original result |
|---|---:|
| Independent excluded base strings | 8 |
| Legal donor-goal instances | 64 |
| Cases with all eight nontrivially separated curvature-matched shams | **28/64** |
| First registered native admission requirement | **≥95%** |
| Observed admission | **43.75%** |
| Rejected `INELIGIBLE_HESSIAN_BRACKET` | **36/64** |
| Maximum admitted analytic curvature discrepancy | **1.39×10⁻¹⁶** |
| Mean random-plane attempts per accepted sham, among admitted cases | **4.65** |
| Source-only pilot CPU runtime | **28.26 seconds** |

**Registered source-only engineering FAIL.** The successful shams match the intended quadratic Hessian constraint to numerical precision. Across 36 cases, at least one of the eight shams could not find a sufficiently separated valid quadratic-level-set crossing within 32 registered attempts. The original result carries all failures; no cases were replaced, radii/angle thresholds adjusted, or outcome-dependent directions selected.

The failure appears across all eight excluded bases, with four to five rejected goals per base. Because this is untrained architecture/synthetic-energy feasibility, it offers **no evidence of a trained source-donor curvature circuit**, nor does it refute the earlier source-locked first-order full-J positive result.

## Original immutable artifact

[Actions artifact #14434876017](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44319953707/artifacts/14434876017): first `feasibility.json`, SHA-256 `ee6dfc30920b429f8edd8ff0b42d773763dc186b19abf460ab93d010bb3563f5`; original ZIP SHA-256 `f9a5098aae55c74326c8e05b70c9d20f40bca016ab17a60d2b68b39d42a0c8db`. The JSON preserves all 64 source IDs, goal IDs, admission decisions and failure reasons. The seven unit tests and the feasibility-run capture both passed CI, since a negative feasibility verdict is an allowed engineering result.

## Implication for next study

Strict nontrivial second-order scalar-matching at an unchanged natural pair radius is a **constraint-satisfaction problem**, not simply drawing uniformly random future-dark directions. High-dimensional available nullity (65) alone does not guarantee that an arbitrary fixed quadratic level set contains sufficiently separated admissible directions. A future study should estimate the local restricted Hessian's spectral concentration and feasible angular separation on excluded-only sources, and compare a global constrained-level-set solver against the fixed random two-plane method. That solver must be registered as a separate `LGA-10B` variant and pass source-only admission before accessing a new trained-model test seed. Any nearer angular match would require a frozen angle and physical-dose tolerance and a no-trivial-copy control.

The registered `31064` first-science run is **held**. No source lock for trained artifacts or first model-native test has been generated. Original LGA-09 full-J 3/3 DEVELOPMENT_STRONG_FULL_J_PASS and independent audit remain unchanged.
