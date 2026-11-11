# GG-03 closeout: low-rank endpoint transfer fails the legal-state specificity gate

**November 11, 2026.** The protocol in [GG-03 registration](../docs/rmp_goal_guarding_causal_v0_3.md) was frozen before the new reporting and Bayes-invariant order-swap panels were examined. These are additional tests of GG-02's explicitly **supervised** Bayesian-decision model.

**Registered verdict: MECHANISTIC_SPECIFICITY_NEGATIVE (two of two model seeds).**

## Verification and data provenance

- Source base: LoopedTransformers main at 6dde1f988aa27c07be0e648dea2947adc7597fff, after GG-02's merge.
- [GitHub Actions run 43039492546](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43039492546) completed successfully. **7 focused tests passed in 3.46 seconds.** The full frozen neural replay and the output receipt schema verification both passed.
- [Raw source-bound artifact, both final models, selected fitted bases, and full JSON](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43039492546/artifacts/13863793416). This artifact contains the full source/donor panel identity hashes and all fixed comparator metrics, as well as the trained weights.
- Original GG-02 calibration 256, validation 128 and locked 256 pair IDs were included in generation solely to exclude their identifiers from GG-03's new 256-pair reporting set. New reporting seed **34001**, Bayes-invariant order-swap seed **34002**. Every oracle label is calculated independently with exact rational arithmetic. The new panel is held out at the **pair identifier** level within the same finite task grammar.
- GG-01 base models were retrained from the exact registered seeds **11 and 17**, using 400 supervised AdamW steps. The GG-02 rank-2/4/8 subspace training repeated its original calibration/validation selection rules with **120** subspace-fit steps. Rank selection occurred before encoding the new reporting set.

## Frozen heldout counterfactual results

| Endpoint | Seed 11 | Seed 17 |
|---|---:|---:|
| GG-02 validation-selected rank | 4 | 2 |
| New oracle-discriminating eligible / attempted | 256 / 256 | 251 / 256 |
| Learned rank-k hybrid action | **254 / 256 (99.22%)** | **251 / 251 (100%)** |
| Same-rank calibration PCA hybrid action | 253 / 256 (98.83%) | 247 / 251 (98.41%) |
| Target-blind receiver-suffix output-gradient patch | 106 / 256 (41.41%) | 97 / 251 (38.65%) |
| GG-01 linear posterior-decoding rank-one patch | 0 / 256 | 0 / 251 |
| Learned patches within 0.1 × nearest-distinct-legal-state separation | **0 / 256** | **0 / 251** |
| Learned patch scaled distance to nearest legal prefix state, median | **4.406** | **3.671** |
| Learned patch scaled distance, p95 | **7.820** | **8.944** |

The native full-state replacement equals an explicitly executed donor-prefix/receiver-suffix hybrid to the declared numeric tolerance. The receiver sham is exactly the unmodified receiver continuation. Both full-state and sham states satisfy the exact finite legal-prefix-state proximity check.

**Incremental learned direction:** learned minus matched-rank PCA action accuracy is only **+0.39 percentage points (1/256)** for seed 11 and **+1.59 points (4/251)** for seed 17. The preregistered learned-subspace-specificity threshold was **+5 points on both seeds**, so this part of the gate fails. The target-blind local output-gradient direction is substantially weaker; the results therefore rule out that particular *first-order rank-one* baseline as a complete explanation of the action transfer. They do not establish a uniquely trained causal belief representation.

**On-support condition:** all learned patches fail the threshold in both seeds. The finite legal support is exactly four two-cue prefixes for each of nine prior/reward-ratio contexts, giving **36 legal visit-two states**. Each patched state is compared to its own context's four states. Its distance is normalized by the smallest positive separation between distinct legal states in that context. The support gate demanded at least **95%** with normalized nearest-state distance at most 0.1; the observed proportion is **0%**. The scale of the departures, including medians exceeding 3.6 separations, is material. A strong output action obtained through these off-support points does not certify naturally realized Bayesian state transport.

## Bayes-invariant order-swap negative control

The independent panel swaps 01 with 10 before an unchanged four-cue suffix and an unchanged prior and reward ratio. Both prefixes have **exactly the same rational posterior odds and correct final action**. The model is eligible only if both unpatched full-sequence orderings are independently classified correctly.

| Endpoint | Seed 11 | Seed 17 |
|---|---:|---:|
| Correctly classified natural order pairs | 256 / 256 | 247 / 256 |
| Learned projected intervention keeps the oracle action | 254 / 256 (99.22%) | 242 / 247 (97.98%) |
| Newly introduced action violations | **2 / 256 (0.78%)** | **5 / 247 (2.02%)** |
| Registered maximum violation rate | 1% | 1% |

The seed-17 intervention fails the negative-control requirement; seed 11 meets it. Same-sufficient-statistic order changes should leave the Bayesian decision unchanged. The violations constitute a second indication that rank-k patching distorts the model's decision geometry even though the donor and receiver share the task's exact sufficient statistic.

## Interpretation and next experiment

GG-01 showed that a rational Bayesian belief is almost perfectly *linearly decodable*, even before training. GG-02 showed near-perfect *counterfactual action transfer* through a small hidden-state subspace also found by unsupervised PCA. GG-03 now shows that those low-rank patches can lie **several legal-state separations outside the finite native state support** while continuing to produce correct hybrid actions.

The outcome makes a narrow point: for this particular finite supervised recurrent classifier, **endpoint counterfactual hit rates and probe R² are inadequate mechanistic-fidelity certificates**. A patch can change the output in the expected direction while leaving the set of states the network naturally realizes under its registered input grammar.

The next experimental task should change the model's learning process, not add another fitted patch family to this nearly saturated toy. A candidate is a separately registered two-stage optimization problem with actual, randomized post-action weight updates and independently measured gold reward, crossed with visible training-disclosure cues. That would measure an actual causal optimizer effect and help distinguish self-training expectations from cue-conditioned classification. No trained model in GG-01–03 has demonstrated spontaneously emerging goal guarding, manipulation of its real optimizer, or simulator bargaining.

## Prior art and limitations

The GG-03 test uses ordinary finite-state enumeration, local backpropagated output sensitivities, PCA, and causal interchange on a supervised recurrent network. The 36-point state support is exact **for the discrete GG-01 input grammar at visit two**. Additional continuous-input tasks can have richer state manifolds, so the observed failure is a task-local negative rather than a claim about general latent-state topology. The reporter's eligibility condition selects examples where both receiver and direct hybrid behavior are already correct; unconditional and eligible denominators must remain distinguished.
