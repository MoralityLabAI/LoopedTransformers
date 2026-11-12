# GG-10 closeout: Markov gradient projections explain nominal signs while protected ambiguity persists

**November 12, 2026. Registered verdict: FIRST-ORDER-SUPPORTED; RESIDUAL-SENSITIVE SEED 11; UNRESOLVED-STATUS REPLICATION PASSED.**

The [GG-10 protocol](../docs/rmp_goal_guarding_markov_gradient_gg10.md) preceded analysis of new independently drawn gold panels. It follows GG-09's positive HMM competency transfer and negative protected-harm-sign transfer. GG-10 keeps the same exact hidden Markov grammar and frozen task-model endpoints, decomposes the *actual physical SGD* loss gaps into first-order autograd projections and finite-step residuals, and separates gold scorer uncertainty into within-cohort TV reweighting and a constrained depth mixture.

## Verification and source-provenance correction

- **Models:** the original GG-09 \`static_control\` and \`markov_transfer\` endpoints for seeds **11** and **17**. The immutable original source is [GG-09 GitHub Actions 43183115898](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43183115898/artifacts/13927786163). Every restored checkpoint is validated against the exact original GG-09 SHA-256 weight hash before computing any GG-10 gradients. Static and Markov endpoints share the same original GG-08 model source for each seed.
- **Technical source route:** GG-10 initially attempted to retrain the full GG-01→GG-08→GG-09 model sequence independently, as specified in its registration. That attempt failed the exact archived checkpoint hash check even after aligning its pool-construction order with the GG-09 runner. The expected source hash remained a hard requirement, and those runs did **not** produce scientific findings. A subsequent technical recovery switched CI to read-only downloading the original GG-09 archived checkpoints, verifying their protocol, seed, arm, and SHA. The restored original source is stronger lineage evidence for the same-model mechanism audit, but **GG-10 does not claim its own independent retraining matched the original weight bytes**. The model endpoint, scoring panels, oracle, learning rate and registered science gates were unchanged.
- **Gold:** original GG-09 HMM scoring panels, labelled **historical post-hoc mechanism reconstruction**, plus fresh HMM scoring panels sampled independently using preregistered natural seed base 105000 and challenge seed base 106000. Both use four replicates at each depth 4/6/8, each with natural 64, boundary 32, rare 32 and late-reversal 32 cases. All scoring examples come from the GG-09 continuation-heldout finite input/context support with exact rational Markov gold labels.
- **Reports:** the same **48 GG-09 HMM-labelled heldout four-cue source reports**, already known from GG-09 and retained unchanged.
- **Physical changes:** for every source report, separately compute full-model norm-1-clipped truthful and flipped report gradients; apply **one real PyTorch SGD step with LR 0.05** from the *same verified frozen model weights* for each. There are **384 real physical SGD steps** over both model seeds and the two control/Markov arms. Each physical successor is scored on all six historical/fresh gold banks, totaling **1,474,560 individual post-SGD gold CE evaluations**. All source weights remain unchanged after the cloned updates.
- **Independent autograd:** 96 separate whole-model cohort-mean gold backward gradients per model endpoint, retaining first-order projection and physical residual for every source report, depth, cohort and replicate. Reported numerical quantities were accumulated using PyTorch gradients and native CPU floating-point; the score bounds are exact only conditional on these measured values.
- [Original complete GG-10 source-bound CI 43195626686](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43195626686) passed the nine focused tests, archived checkpoint verification, complete physical SGD experiment and all receipt checks. [Final complete experiment CI 43196328428](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43196328428) additionally verifies the registered gold-cohort accuracy readout and **passed 9 tests in 4.10 seconds**. [Final model states, complete truth/flip per-example outcomes, gold cohort gradients and protected interval-width receipts](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43196328428/artifacts/13933651453).

## Task competence persists on newly sampled HMM gold

The new Markov scoring panels are not a repeat of the earlier gold samples. They remain inside the same finite HMM task and heldout input-support grammar. Minimum accuracy over all twelve depth×cohort cells:

| Scorer and native endpoint | Seed 11 | Seed 17 |
|---|---:|---:|
| GG-09 historical gold, Markov continuation | **96.09%** | **94.53%** |
| GG-10 fresh gold, Markov continuation | **96.88%** | **92.97%** |
| GG-09 historical gold, fixed-world static control under Markov labels | 39.06% | 39.06% |
| GG-10 fresh gold, fixed-world static control under Markov labels | 39.06% | 39.06% |

Both Markov-trained models clear the original 90%-per-cohort competency threshold even on the fresh scoring samples. The protected harm-sign instability therefore coexists with high accuracy on the independently defined new-grammar task. A supervised exact-action classifier's ordinary accuracy is not a guarantee that its one-step wrong-report SGD damage has a stable sign under a family of protected gold scorers.

## Exact first-order/finite-step comparison

For report x and gold cohort \(G_{d,r,c}\), the contrast between the real weight updates is

\[
B_{drc}=L_{drc}(\theta-\eta g_X)-L_{drc}(\theta-\eta g_C),\quad
A_{drc}=-\eta\langle\nabla L_{drc}(\theta),g_X-g_C\rangle,\quad
R_{drc}=B_{drc}-A_{drc},
\qquad \eta=0.05.
\]

The registered nominal comparison uses protected tail mass \(\lambda=1/2\) and **uniform depth weight \(1/3\)**, independently averaging B, A and R from cohort means. TV and depth uncertainty are applied to the **physical per-example B values only** for the separate robust sign certificate.

| Nominal protected-score diagnostic | Seed 11 Markov | Seed 17 Markov |
|---|---:|---:|
| First-order sign agreement, historical gold | 84.90% | 93.23% |
| **First-order sign agreement, fresh gold** | **144/192 = 75.00%** | **164/192 = 85.42%** |
| Fresh-panel sign disagreement with first order | 25.00% | 14.58% |
| Historical median absolute finite-step residual | 0.002492 | 0.005662 |
| Fresh median absolute finite-step residual | 0.002752 | 0.005115 |
| First-order explanatory threshold | at least 75% | at least 75% |
| **Registered first-order-supported explanatory gate** | **PASS** | **PASS** |
| **Registered 25% residual-sensitive event** | **PASS** | Fail |

Both Markov models meet the frozen first-order sign explanation gate on the 192 fresh query×replicate nominal scoring observations. Seed 11 meets the residual-sensitive event at exactly the registered 25% threshold. These measurements indicate that actual source/model gold-gradient direction matters for most nominal update effects, while finite-step nonlinear changes remain important in a subset. They do **not** identify a strategic model preference or internal intent.

New gold-gradient cosines between protected mean depth objectives:

| Cross-depth gold gradient cosine | Seed 11 Markov | Seed 17 Markov |
|---|---:|---:|
| Four versus six visits | +0.2531 | +0.3374 |
| Four versus eight visits | +0.0451 | +0.3040 |
| **Six versus eight visits** | **−0.3330** | **+0.7543** |

Thus one Markov-trained seed exhibits genuine *global* cross-depth gold-gradient opposition between six and eight visits on this new scoring set. The other has positive alignment while still exhibiting protected-sign ambiguity. Global gradient cosine is context and panel dependent, and its sign alone does not determine a particular report update's physical effect.

## Markov robust-status replication

The unchanged GG-07 scorer uses the primary protected half-tail objective, independent within-cohort empirical TV at \(1/64\), and constrained depth weights \(\alpha_d\ge1/6\). A robust status requires a strict sign across all four scoring replicates.

| Physical SGD protected-sign classification | Seed 11 Markov | Seed 17 Markov |
|---|---:|---:|
| **Historical robust harm / benefit / unresolved** | 10 / 1 / **37** | 19 / 1 / **28** |
| **Fresh robust harm / benefit / unresolved** | 2 / 0 / **46** | 6 / 0 / **42** |
| Original unresolved query identities remaining unresolved on fresh gold | **35/37** | **28/28** |
| Registered unresolved-status replication threshold | at least 16 | at least 16 |
| **Registered replication event** | **PASS** | **PASS** |

The static-control endpoints have different behavior on those same HMM gold labels. Their historical sign counts exactly reproduce GG-09 (seed 11: 13 harmful, 35 beneficial, zero unresolved; seed 17: 13 harmful, 33 beneficial, 2 unresolved). Fresh static-control counts are **13 harmful / 34 beneficial / 1 unresolved** and **13 harmful / 35 beneficial / 0 unresolved** respectively. The static controls have only ~39% minimum Markov decision accuracy, so their robust beneficial wrong-report effects under the new HMM gold should not be interpreted as training safety or correct task alignment.

The Markov models have high task accuracy and **more uncertain** protected-SGD harm signs. This failure replicates on newly sampled gold panels. The first-order explanatory and unresolved replication gates pass simultaneously.

## Exact scorer-ambiguity width and diagnostic classification

For a depth-specific protected lower/upper interval \(L_d,U_d\) obtained by summing independent GG-07 cohort TV extremes, the full composite width at one scoring replicate is exactly

\[
W=U-L=
\underbrace{\frac16\sum_d(U_d-L_d)}_{W_{\text{cohort TV}}}
+
\underbrace{\frac12(\max_d U_d-\min_d L_d)}_{W_{\text{extra depth selection}}}.
\]

The code verifies this identity against the original GG-07 complete protected interval, retaining all per-cohort positive width addends and the actual selected lower- and upper-worst depths. The second term includes the span between depth-specific bounds; it is **the additional uncertainty induced by choosing depth weights**, including within-depth interval effects, rather than an independent causal estimate of curriculum failure.

| Median width among fresh unresolved query×replicate cases | Seed 11 Markov | Seed 17 Markov |
|---|---:|---:|
| \(W_{\text{cohort TV}}\) | 0.003248 | 0.005183 |
| \(W_{\text{extra depth selection}}\) | 0.010684 | 0.011769 |
| Full interval width \(W\) | 0.013812 | 0.016784 |
| **Median individual depth-selection width fraction** | **75.2%** | **71.2%** |

The ordering of nominal, depth-mixture and cohort-TV ambiguity creates an **exclusive, descriptive** diagnostic for unresolved queries:

| Why protected sign is unresolved | Historical seed 11 / seed 17 | Fresh seed 11 / seed 17 |
|---|---:|---:|
| Nominal physical score signs vary among replicates | 12 / 6 | **29 / 22** |
| Signs agree nominally; constrained-depth score is already unresolved at rho 0 | 15 / 9 | **6 / 6** |
| Constrained-depth rho 0 is robust; adding cohort TV creates ambiguity | 10 / 13 | **11 / 14** |
| Other numerical categories | 0 / 0 | 0 / 0 |

For the Markov-trained models, the **boundary** cohort supplies the largest absolute *weighted first-order magnitude* for **26/46** unresolved fresh queries in seed 11 and **36/42** in seed 17. This is a ranking of first-order magnitudes; it does **not** prove that the boundary cohort alone causes the protected interval to straddle zero. The full per-cohort TV width addends, including the natural cohort's own width, remain available in the attached raw ledger.

## Decision and remaining limitations

GG-10 establishes a source-bound explanation of why GG-09's HMM training-harm certificates remain unresolved after successful task learning:

- The signed physical SGD effects are mostly aligned with the initial gold-gradient projection under **nominal** protected scoring, with a measurable finite-step residual component.
- Independent gold sampling changes the nominal sign for many source report queries.
- The protected scorer's composite weight ambiguity contributes substantial interval width, with an important extra term from uncertain depth selection.
- On the fresh gold panels, 35 of the original 37 unresolved seed-11 inputs and all 28 unresolved seed-17 inputs remain unresolved.

The procedure's strongest limits remain explicit. All task-model endpoints are small native recurrent supervised Bayes models, restored from the same GG-09 archived checkpoints. Source report inputs are the original known 48 Markov holdout identities, and the fresh gold draws come from the same finite HMM grammar. Neither the source model nor a separate reporter learns to predict its own optimizer; the real SGD updates modify independent clones used for measurement. No conclusion about spontaneous goal guarding, training concealment, general human values or out-of-grammar alignment follows.

The next useful control is a **fixed protected-depth objective**, registered independently of any report outcomes, alongside a larger rare-case scoring panel. This could distinguish how much of GG-10's ambiguity remains after removing the uncertainty over depth weights while retaining the HMM task and real SGD interventions.
