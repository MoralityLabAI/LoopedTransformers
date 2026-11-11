# RMP-GG-03: Causal specificity, finite-state support, and output-gradient controls

**Frozen November 11, 2026 before any GG-03 measurement.** This is a diagnostic successor to the completed, merged [GG-02](../reports/checkpoint_rmp_gg02_causal_subspaces_20261111.md). Earlier GG-02 *locked outcomes have already been seen*; no claim of fresh registration or validation is made for the old panels. This protocol freezes a **new, source/donor-pair-disjoint reporting panel** and diagnostic thresholds before examining its results.

## Motivation and precise question

GG-01 trained a recurrent action classifier with >99.6% six-cue accuracy and near-perfect linear posterior-odds decoding, including near-perfect readout from **untrained** network states. Its trained rank-one belief-probe intervention failed. GG-02's rank-2/4 counterfactual patches succeeded at 99.6–100%, but ordinary covariance PCA matched them closely, so the registered special-learned-subspace gate failed. GG-02 also found that many patches have low cosine similarity to any calibrated clean state, despite satisfying a weaker norm ceiling.

**Question:** does the causal effect of the GG-02 subspace survive an exact finite-support check and target-blind downstream-output-gradient control? Can it preserve a counterfactual intervention that is *Bayes-invariant* even though the sequence order changes?

This is a supervised Bayesian control-task mechanism experiment. It does not supply evidence of independently arising training awareness, simulator bargaining, reward hacking, or gradient manipulation by a language model.

## Frozen source and replay

- Base code: the LoopedTransformers main commit resulting from GG-02's merge \`6dde1f988aa27c07be0e648dea2947adc7597fff\`.
- Source model: unchanged GG-01 \`EvaluationBeliefLooper\`, with one native tied \`ExplicitDecoderBlock\` width 32, four heads, two positions, one new binary cue per visit.
- Seeds **11 and 17**, each trained with the unmodified GG-01 \`train_cell\` protocol for 400 updates. Retain the last checkpoint, with no test-based selection.
- Refit the *GG-02* rank-\(2,4,8\) orthogonal bases using the same original GG-02 **calibration** panel, 120 steps and identical seed schedule; repeat its **validation-only** rank choice. The selected basis is frozen before opening the new GG-03 reporting panel.
- New manifest: call the unchanged \`generate_disjoint_panels\` with the **three already published GG-02 panels** (calibration 256 seed 31001, validation 128 seed 31002, locked 256 seed 31003) followed by a fourth panel \`gg03_reporting\`, **256** pairs seed **34001**. This preserves the prior global uniqueness guard and excludes every earlier GG-02 source/donor/context pair identifier.
- Each reporting pair has a common prior and \(R/G\) ratio, a receiver six-cue sequence, a donor six-cue sequence, and a hybrid obtained by replacing receiver cues 1–2 by donor cues 1–2. Exact rational Bayes oracle: hybrid action must differ from *both* full receiver and full donor actions. The trained model must correctly classify both receiver and direct hybrid for eligibility. Report the attempted and eligible denominators.
- The model has seen the same finite six-cue *task grammar* and sometimes the same individual cue pattern in training. This is a heldout **pair** test, not unseen task/generalization to a new distribution.

## Frozen interventions and comparators

At visit 2, for receiver recurrent state \(h_r\), donor-prefix state \(h_d\), and \(\Delta=h_d-h_r\):

1. **Sham**: \(h_r\).
2. **Full donor-prefix state**: \(h_d\). Its native continuation must agree with direct hybrid evaluation to \(3\cdot10^{-5}\).
3. **GG-01 diagnostic**: rank-one patch along the frozen linear posterior-decoding direction, calibrated outside the GG-03 reporting panel.
4. **GG-02 learned rank-k**: \(h_r+U_kU_k^\top\Delta\), using the validation-selected rank.
5. **Matched-rank PCA**: same formula with a basis fitted from GG-02 calibration state displacements, without counterfactual labels. PCA is co-primary.
6. **Target-blind local output-gradient**: calculate \(g=\nabla_h[\ell_{\rm protect}-\ell_{\rm exploit}]\) from the *receiver suffix continuation* at its own unpatched visit-2 state. Use \(h_r+\hat g\hat g^\top\Delta\) for nonzero \(g\), with a zero patch for degenerate \(g\). The gradient uses model logits, inputs and weights, **never the hybrid oracle target**.
7. **Eight matched-norm random directions**: seeded Gaussian directions scaled per example to the learned basis's original patch norm, preserving the previous GG-02 control.

All patches use the same frozen receiver suffix. A counterfactual's true action is taken from the exact rational oracle. Report hybrid-action hit, mean target logit margin, patch norm, and denominator, for both seeds. No learning uses the new reporting panel.

## Exact legal-prefix-state support

For each prior in \(\{1/4,1/2,3/4\}\) and reward ratio in \(\{1/2,1,2\}\), enumerate all **four** two-cue prefixes \(00,01,10,11\), and capture their native visit-2 recurrent state on the frozen model. These **36 exact legal states** exhaust the finite prefix grammar and contain all possible clean patched states for this toy experiment.

For any patched state \(h'\) in context \(c\), let
\[
d_c(h')=\min_{x\in\{00,01,10,11\}}\|h'-h(c,x)\|_2.
\]
Let \(s_c\) be the *smallest positive pairwise Euclidean separation* between distinct legal prefix states for that context, and define \(r_c(h')=d_c(h')/s_c\). Record median, p95 and share with \(r_c\le 0.1\). When any \(s_c\le10^{-8}\), report the context as degenerate and fail the support gate rather than dividing by an unidentifiable scale. Exact full-state and sham controls must have \(d_c\le10^{-4}\) on every example.

A support rate is a **geometric plausibility control**, not a proof that an agent naturally visits every point of a continuous manifold between legal states. Exact support has only 36 points in this protocol.

## Bayes-invariant order-swap negative control

Generate an independent **256-example** panel using Python RNG seed **34002**. For each example, sample a registered prior, ratio, and four-cue suffix; alternate the receiver prefix between \(01\) and \(10\), with the other ordering as donor. Both prefixes have identical rational Bayesian sufficient statistic because they contain one positive and one negative cue, so the exact hybrid action is unchanged.

Filter to examples where the frozen model classifies both clean sequence orders correctly, and keep the denominator. Evaluate full-state, learned subspace, PCA, output-gradient and sham patches; report **action violations** against the unchanged exact target and the absolute final logit-margin drift versus the full-state swap. A directional patch that preserves only the action but severely distorts logits remains diagnostic rather than accepted.

## Registered interpretation gates

- **Instrumentation**: exact-oracle, disjoint-pair, full-state equivalence, independent panel generation, and state-support identity tests pass.
- **Applicability**: at least 32 eligible oracle-discriminating examples per seed and 32 correctly classified equal-statistic controls per seed.
- **Special learned-subspace support**: on the new frozen reporting panel, the learned selected-rank hit rate exceeds *both* PCA and output-gradient hit by at least 0.05 on each seed, and at least 95% of learned patches have \(r_c\le0.1\). The Bayes-invariant order-swap patch must introduce at most 1% action violations per seed.
- **If the main support gate fails**, retain the exact rates and denominators, classify the result as a **mechanistic-specificity negative**, and do not weaken thresholds post hoc.
- Even a pass is specific to an explicitly supervised six-bit Bayesian task. A separate experiment, registered later, must test actual optimizer updates and independently measured reward uncertainty before making claims about training awareness.

## Provenance and reproducibility

Save model weights, fitted bases, hashes of the two GG-03 panel definitions, all controls, source revision and a machine-readable report as GitHub Actions artifacts. The CI runs source/oracle tests, two-seed CPU replay, and an explicit schema and result-bound check. Reporting code may be repaired after tests, but any post-results changes must be labeled exploratory and cannot overwrite the registered frozen outcome.
