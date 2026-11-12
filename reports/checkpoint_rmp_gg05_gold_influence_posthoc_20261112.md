# GG-05P post-hoc: two historical gold examples control every depth-reversal

**November 12, 2026. Classification: EXPLORATORY POST-HOC INFLUENCE AUDIT.**
The original [GG-05 preregistration](../docs/rmp_goal_guarding_gradient_conflict_gg05.md), its exact source tests, and its [negative fresh-panel result](checkpoint_rmp_gg05_gradient_conflict_20261112.md) remain unchanged. After reading GG-05's results, a separate inspection identified an unusually concentrated source of baseline cross entropy in the previously seen six-cue gold panels. This note tests the *causal scoring sensitivity* of that specific historical result and documents the complete finite deletion census.

## Question and exact scoring identity

For heldout four-cue report query \(x\), the native recurrent model's weights are frozen at checkpoint \(\theta\) before computing the truthful/incorrect report gradients \(g_C,g_X\), globally clipped at norm 1. The experiment physically performs two independent one-step SGD interventions at learning rate \(\eta=0.05\), producing \(\theta_C=\theta-\eta g_C\) and \(\theta_X=\theta-\eta g_X\).

Let the old six-cue gold panel contain \(N=128\) examples. Define the per-sample, action-conditional physical loss difference
\[
d_i(x)=\ell_i(\theta_X)-\ell_i(\theta_C).
\]
Its original wrong-minus-truthful physical gold gap is
\[
 B_6(x)=N^{-1}\sum_{i=1}^{N}d_i(x).
\]
For any pair of gold scoring indices \(i\ne j\), **without retraining or touching either of the two task-model successors**, their two-deletion gold gap is
\[
\boxed{
B_6^{(-i,-j)}(x)=
\frac{N B_6(x)-d_i(x)-d_j(x)}{N-2}.
}
\tag{GG-05P/1}
\]

The source query roster, clipping, real SGD interventions, original four-cue gold panel and calibrated model parameters stay fixed. Only which six-cue gold examples contribute to the final loss average changes. This algebraic identity permits enumeration of every \(\binom{128}{2}=8128\) two-example scoring deletion using one source-bound matrix of 48×128 native per-example post-SGD loss differences.

The registered GG-05 sign ambiguity threshold \(10^{-7}\) is preserved. Since the historical four-cue gap is strictly positive on **48/48** source queries for either model seed, cross-depth reversal count here is the number of original source queries for which the recomputed six-cue gap is strictly negative.

## Native reproduction

The repository-native [GG-05P module](../research_gym/interp/goal_guarding_gold_influence_posthoc.py) loads **the actual GG-05 saved native \`EvaluationBeliefLooper\` checkpoint** and its SHA-256 source-bound physical-update ledger. It checks each query ID against GG-04B's original heldout 48-query roster. It recomputes the clipped report gradient from the native model, runs the real PyTorch SGD update independently for truthful and flipped report labels, and measures 128 separate exact task-gold cross-entropies after each step.

Every result is cross-checked against the original recorded whole-panel physical CE gap, allowing a declared \(10^{-5}\) tolerance for differences in float32 reduction order. The original source task-model weight digest must remain unchanged. The independent finite deletion mask implementation and the sum-subtraction enumerator agree in focused tests, including an exact synthetic case with a uniquely pivotal pair.

The [post-hoc native CI run 43123613707](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43123613707) passed **13 unit tests**, the original two-seed GG-05 experiment and the full native leave-two-out enumeration. Its first post-hoc receipt validation failed because the output-path lookup used the checkpoint subfolder rather than the artifact root. That file-path check was corrected; physical experiment results were unaffected. GitHub Actions retains the first run as a technical failure, and the final verified result is reported on the subsequent run.

## Exact forensic result on the historical six-cue panels

| Measurement | Native seed 11 | Native seed 17 |
|---|---:|---:|
| Gold-panel examples | 128 | 128 |
| Pre-update misclassified gold examples | 0 | 2 |
| Baseline old six-cue CE | 0.00398557 | 0.04975958 |
| Two highest-loss gold indices | **10, 53** | **10, 78** |
| Share of total baseline CE from those two | **83.7349%** | **99.8374%** |
| Original cross-depth sign reversals | **6/48** | **35/48** |
| Reversals after dropping highest-loss example only | 1/48 | 31/48 |
| Reversals after dropping second-highest only | 2/48 | 31/48 |
| Reversals after dropping both | **0/48** | **0/48** |
| Two-example deletion pairs eliminating all reversals | **1/8,128** | **1/8,128** |

The complete census finds exactly **one** two-example deletion that removes every historical cross-depth reversal for each model: indices \(\{10,53\}\) for seed 11, and \(\{10,78\}\) for seed 17. The unique pair is the two highest-loss examples of that model's historical six-cue scoring panel.

The most influential examples remain synthetic Bayes classification cases:

- **Seed 11, index 10:** observations \`100111\`, gold label 1, predicted 1, baseline CE 0.258719. **Index 53:** \`001111\`, gold label 1, predicted 1, baseline CE 0.168456. Both are correctly classified; their incomplete confidence drives most of the loss.
- **Seed 17, index 10:** \`100111\`, gold label 1, predicted 0, baseline CE 3.851051. **Index 78:** \`010100\`, gold label 0, predicted 1, baseline CE 2.507819. They are the old six-cue panel's **only** two misclassified examples and jointly contribute more than 99.8% of its total CE.

Every other pair among the 8128 leaves at least one original negative six-cue report-update gap. For seed 17, deleting either pivotal example in isolation preserves 31/48 cross-depth reversals; the interaction of removing both is essential to eliminate them. The 1/8128 figure is a **descriptive combinatorial frequency over a fixed, already observed panel**. It is not a randomized p-value or a population probability.

## Interpretation and limits

This post-hoc result isolates the gold scoring support as the immediate cause of the historical reversal's fragility. The two physical SGD successors are identical before and after the scoring deletion, as are all 48 report inputs. Reversals disappear solely because the protected objective averages a different set of six-cue examples. The original observation of 35/48 reversals on seed 17 remains correct for its original panel.

It is compatible with GG-05's independent *fresh-panel* finding of **zero reversals on both seeds**, and explains why an average CE objective can be extremely sensitive to a few difficult or misclassified samples. It does not prove that the sampled historical panel is statistically invalid. A tail-focused safety objective may intentionally emphasize rare hard cases. If such emphasis is desired, its weighting needs explicit registration and coverage guarantees.

**Applied-math implication:** an empirical gold objective \(L=\sum_i w_i\ell_i\) has a report-gradient influence gap \(B=\sum_iw_i d_i\). Any claimed reward-hacking or goal-guarding incentive should identify sensitivity of the *sign of this gap* to the gold sampling weights, including a finite leave-k-out instability or a predeclared worst-case tail weighting. Otherwise an apparent preference reversal can arise from the evaluators' few high-loss examples rather than a stable actor-conditioned strategy.

The current models do not represent or reason about their actual SGD updates. GG-05P contains no new learned reporting policy or spontaneous simulator-awareness test. Its evidence concerns physically measured gradient consequences and gold sampling sensitivity in a small supervised recurrent classifier.
