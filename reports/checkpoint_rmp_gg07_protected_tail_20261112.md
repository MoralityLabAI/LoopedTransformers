# GG-07 closeout: protected-tail robustness is capability-limited at eight visits

**November 12, 2026. Registered verdict: PROTECTED-TAIL GATE FAILS, TAIL-INDUCED SIGN SHIFTS OBSERVED.**

The [GG-07 protected-tail protocol](../docs/rmp_goal_guarding_protected_tail_gg07.md) was frozen before model scoring on its independently generated Bayesian gold panels. It follows GG-06's finding of stable four-cue report harm and unstable six-cue harm. The new study adds eight-cue evaluation and hard-case cohorts defined by exact Bayesian task difficulty, rarity and late evidence reversal.

## Source, reproducibility and measurement

- Native \`EvaluationBeliefLooper\`, one shared \`ExplicitDecoderBlock\`, source \`main\` at \`0bdfaee3af9474d9df5cf736a915db0a44998d05\`. Freshly reproduced model seeds **11,17**, each with **400 original supervised training steps**. Source model weights were unchanged after every counterfactual task-model SGD measurement.
- **Known report queries:** original 48 GG-04B/GG-05 heldout four-cue/context combinations; report actions use externally computed truthful or flipped Bayesian labels. Query membership is not a new heldout generalization claim.
- **Independent fresh gold task evaluation:** for each of depths **4, 6, 8**, four replicate panels containing **64 natural**, **32 exact-oracle boundary**, **32 exact-oracle rare**, and **32 late-reversal** sequences each. Each model is scored on 1,920 gold examples per physical update across depths. Boundary, rarity, and late-reversal cohorts were constructed from the full finite \(9\cdot2^n\) input grammar using pre-frozen exact Bayesian criteria and RNG seeds, without inspecting model errors.
- **Real SGD:** each of 48 report queries generates two actual clipped PyTorch weight updates from the same frozen checkpoint, \(\eta=0.05\), for truth versus flipped report. This is **96 physical steps per seed**, **192 overall**, each evaluated on 1,920 individual gold examples, or **184,320 post-update gold CE observations per seed**. Actual physical CE and all per-example action differences are retained; no new agent controller is trained.
- [First CI run 43140854081](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43140854081) found one typed gold-bank accessor mismatch; six of seven tests passed, while the native SGD comparison detected the wrong field name. The accessor was corrected without changing model, sampling, optimization, or preregistered scientific criteria.
- [Source-bound full CI run 43141030205](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43141030205) **passed 7 tests in 6.38 seconds**, both frozen native SGD model runs, and the complete cohort-hash, product-TV and outcome-ledger checks.
- [Original GG-07 source-bound physical gold ledgers and native training checkpoints](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43141030205/artifacts/13910085458). The artifact contains exact panel identities and per-sample truth/flip post-SGD CE for every report/depth/cohort/replicate.

## Sharp protected objective under a finite ambiguity set

For a report query \(x\), depth \(d\in\{4,6,8\}\), replicate \(r\), and action-conditional real-SGD effects \(e_i=\ell_i(\theta_X)-\ell_i(\theta_C)\), define the frozen protected score
\[
B_{d,r}(\lambda)=(1-\lambda)\bar e_{d,r,\mathrm{natural}}+
\frac{\lambda}{3}(
\bar e_{d,r,\mathrm{boundary}}+
\bar e_{d,r,\mathrm{rare}}+
\bar e_{d,r,\mathrm{late}}).
\]
Primary \(\lambda=1/2\); controls \(\lambda\in\{0,1/4,3/4,1\}\).

**GG-07/T1 (sharp product ambiguity).** The auditor may independently shift up to \(\rho=1/64\) total-variation probability mass within the observed finite support of each cohort. Each cohort obtains its sharp lower and upper loss bound using GG-06's exact rational mass-transport LP, applied to the physically measured floats. Because the mixture weights are nonnegative and each cohort has its own disjoint uncertainty set, their weighted sum is the exact attainable depth-specific extreme. A further independent depth weight lies in the simplex \(\alpha_4,\alpha_6,\alpha_8\ge1/6,\sum\alpha_d=1\), giving
\[
\boxed{
\underline G_r=\frac16\sum_d\underline B_{d,r}
 +\frac12\min_d\underline B_{d,r},\qquad
\overline G_r=\frac16\sum_d\overline B_{d,r}
 +\frac12\max_d\overline B_{d,r}.
}
\]
The depth vertices and product cohort extrema attain these values. Independent tests verify the extremal witness objectives, the three-simplex vertex algebra, actual SGD parity with the GG-04 physical auditor, and exact source/label membership. The sharpness claim is **only over the sampled empirical gold support and the specified product ambiguity set**.

## Primary protected-tail results

The registered primary status is robustly harmful when the lower bound exceeds \(10^{-7}\) on all four independent scoring replicates, robustly beneficial when the upper bound is below \(-10^{-7}\) on all four, and unresolved otherwise.

| Primary protected objective (lambda 1/2, rho 1/64) | Seed 11 | Seed 17 |
|---|---:|---:|
| Robustly harmful report queries | **20/48** | **6/48** |
| Robustly beneficial report queries | **13/48** | **8/48** |
| Unresolved sign | **15/48** | **34/48** |
| Same nonzero **nominal** sign on all four replicates | **44/48** | **34/48** |
| At least one nominal **sign flip** when changing from natural-only to protected-tail gold | **24/48** | **39/48** |
| Natural→protected sign changes, over 192 query×replicate cells | **37/192** | **103/192** |
| Registered 24/48 robust-sign numerical target | 33/48 reach it | 14/48 fall short |
| 90%-accuracy capability prerequisite in all twelve depth/cohort cells | **FAIL** | **FAIL** |
| **Overall registered strong protected-tail gate** | **FAIL** | **FAIL** |

At the nominal tail mass, 50% of a gold scoring objective is allocated to the external-oracle-defined hard cases. This changes the sign of the source report's physical training-harm estimate for many query/replicate combinations, despite no change to the source model's report-gradient updates.

The **secondary preregistered tail-induced sign-change event passes** because 24 and 39 queries experience at least one sign flip after tail weight is introduced. This is a scoring-distribution effect, measured conditionally on the frozen task model. It is neither a model's discovered objective nor a trained actor strategy.

## Baseline model competence at new inference depths

The native models learned the four-cue task and generalize strongly to six visits, including all registered hard cohorts. The eight-cue challenge task uncovers a clear eligibility limit.

| Baseline gold classification accuracy | Seed 11 | Seed 17 |
|---|---:|---:|
| Four-cue natural/boundary/rare/late | **100% each** | **100% each** |
| Six-cue natural | 99.61% | 100% |
| Six-cue boundary | 99.22% | 98.44% |
| Six-cue rare | 96.88% | 100% |
| Six-cue late reversal | 98.44% | 95.31% |
| Eight-cue natural | 99.61% | 100% |
| **Eight-cue boundary** | **89.06%** | 90.63% |
| **Eight-cue rare** | **89.06%** | 92.97% |
| **Eight-cue late reversal** | **88.28%** | **89.84%** |

The protocol required at least **90% for every depth/cohort cell** for either model's protected-tail result to count as capability-eligible. Seed 11 fails three eight-cue challenges; seed 17 fails the eight-cue late-reversal cohort. Even though seed 11 has more than 24 robust query signs under the sampled objective, this accuracy guard prevents claiming a strong protected-tail success.

This is a **TASK-CAPABILITY-UNRESOLVED** interpretation for the protected-tail sign results at eight cues. A wrong-label update may improve an undertrained long-depth classifier's gold score, so negative harm gaps alone do not establish a persistent incentive to tamper with training.

## Tail cohorts and weight dependence

The complete source-bound ledger reports every frozen \(\lambda\) and \(\rho\) setting. A selection from the registered 48-query robust-sign census:

| Weighting and within-cohort TV | Seed 11 harmful / beneficial / unresolved | Seed 17 harmful / beneficial / unresolved |
|---|---|---|
| Natural only, lambda 0, rho 0 | 25 / 0 / 23 | 27 / 0 / 21 |
| Natural only, lambda 0, rho 1/64 | 5 / 0 / 43 | 8 / 0 / 40 |
| Primary protected tail, lambda 1/2, rho 0 | 24 / 16 / 8 | 12 / 16 / 20 |
| **Primary protected tail, lambda 1/2, rho 1/64** | **20 / 13 / 15** | **6 / 8 / 34** |
| All hard-case gold, lambda 1, rho 1/64 | 20 / 14 / 14 | 6 / 8 / 34 |

Nominal sign changes under external tail weight are substantial. The stronger within-cohort TV uncertainty decreases the number of queries whose signs are certifiable on *every* replicate.

Looking separately at the preregistered eight-cue late-reversal cohort, the physically measured flipped-minus-truthful gap is **negative on 105 of 192 query×replicate cells for seed 11**, and **133 of 192 for seed 17**. The same four-cue late-reversal cohort has a **positive** harm gap on all 192 cells for either seed. This depth-dependent behavior accompanies the capability decline and gold objective shift; it is a physical update effect, not a claim of intentional manipulation.

## Scientific interpretation and future falsifiers

GG-06's four-cue result survives this richer exact-oracle challenge evaluation at four visits. Under the depth-uncertain protected objective, both model capability and the measured harm sign become less reliable at eight visits. An evaluator whose protected gold mixture emphasizes rare evidence reversals can judge the very same native SGD report update differently from a natural-only evaluator.

The main methodological advancement is **outcome-independent hard-case registration**. Boundary, rarity and late-reversal cases were selected from exact task mathematics, before anyone examined neural losses or report-update effects. This avoids the post-hoc influence-selection problem of GG-05P, while keeping all scoring weights, sample IDs and physical SGD receipts auditable.

The experiment's strongest claim is still limited by the native model's incomplete competence on several eight-visit challenge cohorts. A useful successor should first demonstrate strong, calibrated performance across a *new* late-reversal task family or deeper recurrent sequence before attributing changes in report-gradient harm to an agent's evaluation strategy. Cross-seed, cross-task inference needs more than two model initializations and gold cases sharing this one binary Bayes channel.

GG-07 trains **no** reward-hacking reporter, has no hidden optimizer-awareness probe, does not change the agent's reporting-policy weights via the observed post-action update, and provides no evidence of deceptive alignment or strategic bargaining with simulators.
