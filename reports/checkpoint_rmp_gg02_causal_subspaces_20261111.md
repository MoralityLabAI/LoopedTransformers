# GG-02 closeout: low-rank transfer succeeds, incremental learned-subspace gate fails

**November 11, 2026 | registered finite supervised neural positive control.**

The [GG-02 protocol](../docs/rmp_goal_guarding_causal_v0_2.md) was frozen at
\`ae06734e\` before fitting any causal subspace or inspecting the locked outcome.
It follows [merged GG-01](checkpoint_rmp_gg01_causal_belief_20261111.md),
which learned the exact Bayesian task but failed 256/256 rank-one belief-coordinate
interventions on both seeds. GG-02 trains only small orthogonal projectors on
frozen native \`ExplicitDecoderBlock\` recurrent models.

**Research verdict: \`LOW_RANK_TRANSFER_POSITIVE__INCREMENTAL_CAUSAL_ADVANTAGE_NEGATIVE\`.**
Rank-2/4/8 causal and *untrained PCA* state-difference subspaces carry most of the
counterfactual action effect. The preregistered learned-subspace advantage over
the same-rank PCA baseline fails on both seeds. GG-02 does **not** establish a
uniquely learned belief mediator or spontaneous awareness of training.

## Fixed protocol and provenance

- Train GG-01 seed 11 and 17, 400 optimizer updates per seed, batch 128, using
  unchanged four-cue exact-rational supervised policy task and terminal/prefix CE.
  No GG-02 target labels touch native model weights.
- Freeze full model weights. Intervene on recurrent hidden state at visit 2,
  using donor's first two evidence bits and receiver's remaining four.
  Four- and six-cue generalization tasks remain the original finite cue family.
- Distinct, hash-unique and oracle-discriminating source/donor pair manifests:
  calibration 256, validation 128, locked reporting 256, generated with
  seeds 31001/31002/31003. Exact hybrid target differs from both entire receiver
  and entire donor targets. Data overlap at the level of *individual cue patterns*
  is possible in this finite grammar; complete source/donor/context pair IDs
  remain disjoint between splits.
- Fit rank-2/4/8 orthonormal projector bases with a direct frozen-model
  counterfactual CE objective for 120 steps per rank. Select rank from validation
  **only**; report locked results after selection. GG-01 probe rank-one,
  state-sham, full-state continuation identity, unsupervised same-rank PCA, and
  eight independently seeded per-example norm-matched Gaussian patches serve as
  controls.
- CPU GitHub Actions [run 43037247644](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43037247644)
  passed the six exact/native unit tests and completed the full neural
  experiment plus JSON receipt validation. A second PR-triggered run
  [43037315816](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43037315816)
  also completed successfully.
- [Full JSON, initial/final GG-01 weights, fitted rank-2/4/8 bases](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43037247644/artifacts/13863791132).
  Model weights were produced during GG-02 from the unchanged GG-01 training
  implementation, and the source-bound final tensors are identified below.

## Locked outcomes

| Endpoint | Seed 11 | Seed 17 |
|---|---:|---:|
| Final GG-01 six-cue accuracy | 4092 / 4096 | 4081 / 4096 |
| Calibration eligible | 256 / 256 | 250 / 256 |
| Validation eligible | 128 / 128 | 125 / 128 |
| Locked eligible | 256 / 256 | 245 / 256 |
| Validation-selected rank | **4** | **2** |
| Selected learned subspace, locked | **255 / 256 (99.61%)** | **245 / 245 (100%)** |
| Same-rank PCA subspace, locked | **256 / 256 (100%)** | **239 / 245 (97.55%)** |
| GG-01 rank-one probe, locked | 0 / 256 | 0 / 245 |
| Receiver-state sham, locked | 0 / 256 | 0 / 245 |
| Maximum of eight norm-matched random controls | 20 / 256 (7.81%) | 24 / 245 (9.80%) |
| Inside calibration 99th-percentile clean-state **norm** | 100% | 100% |
| Nearest-clean-state cosine, selected patch | 0.7351 | 0.6649 |
| Selected-rank hybrid-target margin lift (logit units) | +18.8008 | +18.4020 |
| Full counterfactual margin lift | +18.5746 | +18.4791 |
| **Preregistered learned-subspace support** | **FAIL** | **FAIL** |

The validation sets selected rank 4 for seed 11 because rank 4 and rank 8
each hit 128/128 and smaller rank won the tie. Seed 17 selected rank 2:
ranks 2 and 4 each reached 125/125, while rank 8 was 124/125.

All three ranks on the **locked** panel:

| Seed | Rank | Causally fitted patch | PCA patch | Maximum random |
|---|---:|---:|---:|---:|
| 11 | 2 | 253 / 256 (98.83%) | 255 / 256 (99.61%) | 10.16% |
| 11 | 4 | 255 / 256 (99.61%) | 256 / 256 (100%) | 7.81% |
| 11 | 8 | 256 / 256 (100%) | 256 / 256 (100%) | 8.98% |
| 17 | 2 | 245 / 245 (100%) | 239 / 245 (97.55%) | 9.80% |
| 17 | 4 | 245 / 245 (100%) | 241 / 245 (98.37%) | 8.16% |
| 17 | 8 | 245 / 245 (100%) | 245 / 245 (100%) | 6.94% |

The selected learned basis exceeds PCA by **+2.45 percentage points** on
seed 17 and trails PCA by **0.39 points** on seed 11. The protocol required an
improvement of **at least five points above PCA on each seed**. It fails that
test, despite excellent counterfactual reconstruction and random-null
separation.

The state norm ceiling is a limited plausibility check. Low nearest-clean
cosine values, especially **0.665** on seed 17, leave open the possibility
that a patched state is geometrically distant from the clean-state manifold.
The large patch-to-receiver norm ratios (99th percentile approximately
2.06 and 2.01) require stronger multivariate on-manifold tests before claiming
a semantically faithful latent-state explanation.

## Observational versus causal interpretation

GG-01 had a nearly perfectly decodable posterior *at random initialization*.
GG-02 finds that the supervised model's action is causally transferable by
a rank-2 or rank-4 **displacement subspace** between receiver and donor
hidden states. This supports a low-dimensional *empirical state-difference
geometry* for the frozen finite task, beyond a linear probe's decoding score.
The PCA comparison shows that a supervised counterfactual subspace fit was
not necessary to achieve near-perfect transfer in these synthetic panels.

At least two mechanisms remain compatible with the observations: a genuine
low-rank implementation of Bayesian sufficient statistics; or state changes
along directions with strong action-logit effects, including potentially
off-manifold shortcut behavior. A more discriminating experiment should
identify a *minimal causal state* that predicts and transfers exact posterior
updates while preserving control-task variables and clean-state statistics.
Matched-norm random directions are insufficiently strong controls when the
donor/receiver displacement itself has a concentrated covariance spectrum.

A future successor should preregister spectral-rank measurement and PCA
controls as co-primary, require stronger manifold/neighborhood controls and
interventions over a task grammar with genuinely heldout latent dynamics.
After identifying that mechanism, run a separately authorized randomized
optimizer-update experiment with sealed gold labels and a model trained
on actual incentive conflicts. This GG-02 study remains a **supervised
Bayesian positive control** and provides no evidence for autonomous
goal guarding, self-referential RL reasoning, or deceptive intent.

## Auditable hashes

- GG-01 final model seed 11: \`454cc4d2ae01f3072dfd475b9d7309f255eb8265d2c8e0d1bd59886a34d584d1\`.
- GG-01 final model seed 17: \`a03c4b40d66ad2ea8ba796cd4737ea36e5706a8cfedc379fb9fde0ac47da7b08\`.
- Calibration panel ID digest: \`6daa73a718119987887079d2952975f70d7c10daceee9f7d5f2308743ea9b75f\`.
- Validation panel ID digest: \`cc78f8def57c0a432d7a9abdd2facd0c9cfbc436781434ad673bcedbf51588b8\`.
- Locked panel ID digest: \`b3f42b867cd7906d9e297dbb1c37c1f600a07bd654df7758bf7a15bdc42d3429\`.

The full machine-readable result and basis tensors remain in the GitHub
Actions artifact; this report includes the values relevant to the registered
decision gates and all ranks.
