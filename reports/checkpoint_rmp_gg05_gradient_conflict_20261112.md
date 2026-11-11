# GG-05 closeout: cross-depth sign reversal is gold-panel-dependent

**November 12, 2026 | registered verdict: NO FRESH-PANEL REVERSAL AT ETA 0.05.**
The [GG-05 protocol](../docs/rmp_goal_guarding_gradient_conflict_gg05.md) was frozen before evaluating the new gold scoring panels. It reuses 48 **previously inspected** GG-04B query/context combinations, and adds independently seeded protected scoring panels at four and six recurrent observations. It measures physical PyTorch SGD for truthful and flipped reports, first-order gold gradients and the exact finite-step residual. All claims concern small, explicitly supervised native tied recurrent models.

## Source, test result, and reproducibility

- Base LoopedTransformers main: \`15512fb3b85c988442372c242317e82fe726372e\`; two GG-01 task-model seeds **11,17**, each trained 400 AdamW steps. The model remains at its frozen baseline for each independent post-report SGD intervention.
- [GitHub Actions run 43122006832](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43122006832) completed successfully. **Eight focused tests passed in 2.57 seconds**, including an independent backward/SGD reference and a central finite-difference check of first-order gold loss variation. Both full neural runs and their receipt schema/gradient decomposition checks passed.
- [Complete 48-query decompositions, native model weights and JSON receipts](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43122006832/artifacts/13901786108). Each seed executed **288 real SGD steps**, with gradients clipped to norm 1, at \(\eta\in\{0.0125,0.025,0.05\}\); every physical step was evaluated against four gold panels (1152 gold loss evaluations per seed). Source weights are unchanged after the audit.
- **New scoring panels**: \`make_panel(128,4,69002)\` and \`make_panel(128,6,68002)\`. **Historical comparison panels**: \`make_panel(128,4,69001)\` and \`make_panel(128,6,68001)\`. All label targets are computed independently from the exact rational Bayesian task rule. The machine-readable receipt includes source/model hashes, query ID hashes, panel hashes, report-gradient norms, first-order predictions, exact post-update CE and finite residuals.
- The new gold panels are independently drawn, **within the same finite task grammar**. No new trained agent, independent task family, unknown simulator or learned optimizer self-model was evaluated.

## Predeclared gradient/finite-step decomposition

For every query \(x\) and each gold panel \(k\), let \(\widetilde g_C,\widetilde g_X\) be the full-parameter clipped native report gradients at fixed baseline \(\theta\). The experiment computes

\[
 B_k=L_k(\theta-\eta\widetilde g_X)-L_k(\theta-\eta\widetilde g_C),\qquad
 A_k=-\eta\langle\nabla L_k,\widetilde g_X-\widetilde g_C\rangle,\qquad
 R_k=B_k-A_k.
\]

Here \(B_k\) is a physically measured gap in the independently licensed gold loss. The first-order prediction \(A_k\) uses *autograd at the same frozen baseline*. The residual \(R_k\) is the integrated difference between directional gold-loss curvatures along the two finite update paths. All signs below follow the registered \(\pm 10^{-7}\) ambiguity band. The primary comparison uses \(\eta=0.05\).

**Elementary sign guard:** when \(|R_k|<|A_k|\) and \(A_k\ne0\), \(B_k\) has the sign of \(A_k\). Consequently a measured sign flip against its first-order prediction requires a residual at least as large in magnitude as that prediction. This exact inequality explains why the finite-step residual can suppress apparent first-order cross-depth conflict without any additional learned reward-hacking mechanism.

## Primary results on independently drawn gold panels

| Registered metric | Seed 11 | Seed 17 |
|---|---:|---:|
| **Measured four/six depth sign reversals** at eta 0.05 | **0/48** | **0/48** |
| Opposed **first-order** four/six gold gap signs | 20/48 | 7/48 |
| Measured reversals fully explained by first order | 0 | 0 |
| Measured reversals requiring curvature influence | 0 | 0 |
| Fresh four/six gold-gradient cosine | +0.04234 | +0.55220 |
| Finite versus first-order sign agreement, four cue | 48/48 | 48/48 |
| Finite versus first-order sign agreement, six cue | 28/48 | 41/48 |
| Six-cue finite-gap sign changing over registered eta grid | **14/48** | **3/48** |
| Median absolute finite-step residual, six cue | 0.000853 | 0.00000883 |
| Median absolute finite-step residual, four cue | 0.00000602 | 0.00000741 |
| Median residual ratio \(R(0.025)/R(0.05)\), six cue | 0.200 | 0.232 |

The **two-seed first-order-conflict support gate is FALSE** because both fresh panels have zero measured sign reversals at the primary step size. The **curvature-sensitive measured-reversal gate is also FALSE**, because it requires at least eight actual fresh-panel reversals that first order fails to predict.

The first-order gradients nevertheless exhibit opposed signs on 20 and 7 of the new-panel query pairs. Their nonreversing *physical* results demonstrate that curvature/residuals can overturn one or more of the first-order predictions. In the six-cue fresh panels, finite and linear signs agree on 28/48 and 41/48 respectively; 14 and 3 of those finite six-cue signs vary over the frozen learning-rate grid. This is empirical learning-rate sensitivity within the registered experiment.

The residual half-step/full-step median ratios are around 0.20 to 0.23 on these panels, close to the \(1/4\) scaling expected when a small-step quadratic contribution dominates. These are descriptive finite measurements and do not establish a constant Hessian.

## Historical panels: original effect reproduced and decomposed

| Registered historical comparison | Seed 11 | Seed 17 |
|---|---:|---:|
| **Measured cross-depth sign reversals** | **6/48 (12.5%)** | **35/48 (72.92%)** |
| Opposed **first-order** four/six gold gap signs | 26/48 | 38/48 |
| Measured reversals whose both signs match first order | **6/6** | **35/35** |
| Measured reversals requiring residual sign corrections | 0 | 0 |
| Old four/six gold-gradient cosine | +0.26405 | +0.40648 |
| Old six-cue finite vs first-order sign agreement | 28/48 | 44/48 |
| Median critical finite gold-mix weight on reversal cases | 0.19149 | 0.006286 |

The old GG-04B sign reversal is reproduced on its exact scoring panels, including seed 17's 35/48. **All historically measured reversals are directionally present in the initial first-order gradient projections.** The finite residual still changes magnitudes, removes some predicted reversals, and produces learning-rate-dependent sign changes outside the measured reversal set.

For the 35 historical seed-17 reversals, the median critical mixture weight for the *convexly mixed* protected objective is about **0.006286**, derived from \(\lambda^*=B_4/(B_4-B_6)\). This indicates that the score's relative depth weights can alter incentives sharply for that particular pair of legacy gold panels. The numerical mixing threshold depends on the two gold losses' scales and their sampled composition.

## Scientific interpretation

GG-04B's 72.9% seed-17 cross-depth discrepancy was real for its frozen scoring panels, and the GG-05 physical first-order audit reproduces and explains its direction. The effect **did not generalize** to the independently seeded new gold panels at the preregistered 0.05 SGD step, despite reusing exactly the same 48 source queries and report-gradient interventions.

Thus **the choice of protected gold-scoring distribution is a first-class part of the causal intervention**. The gold gradient depends on the input distribution, task depth and model checkpoint. Changing the gold panel can change the signed projection \(\langle\nabla L_k,g_X-g_C\rangle\); the nonlinear update residual can then push physical effects across the first-order boundary.

A positive gold-gradient cosine between the four- and six-cue objectives does not rule out negative projections for selected report directions. Conversely, an opposite first-order projection is not sufficient for a physical sign reversal at a finite learning rate. The exact source-bound decomposition resolves those two questions on the declared finite panels.

GG-05 supplies experimental mechanics for a later self-update or policy-gradient meta-learning study, while providing **no evidence that the native task model or separate GG-04B reporting controller spontaneously represents, understands or alters its real optimizer**. The evidence remains a small supervised task and registered real-SGD audit. Future work should increase gold-distribution diversity and reproduce the sign statistics across separately generated panels before extrapolating beyond this particular model/seed pair.
