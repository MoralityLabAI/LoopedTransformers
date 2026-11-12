# GG-09 closeout: exact Markov task competence transfers while protected gradient-harm signs do not

**November 12, 2026 | Registered result: MARKOV COMPETENCY TRANSFER PASS; PROTECTED-HARM TRANSFER FAIL.**

The [GG-09 preregistration](../docs/rmp_goal_guarding_markov_transfer_gg09.md) preceded the new task-training and heldout Markov-scoring outcomes. It follows GG-08's success within the original *fixed-world* binary Bayesian grammar. GG-09 changes the exact generative semantics: the latent world switches with probability \(1/4\) between observations. Its exact oracle, new training-support split, independently sampled gold panels and physical-update scoring were frozen before the run.

## Source and verification

- Starting weights: a fresh deterministic reproduction of GG-08's mixed-depth GG-01 native \`EvaluationBeliefLooper\`, one shared \`ExplicitDecoderBlock\`, width 32 and four attention heads. Source LoopedTransformers \`main\` commit \`ffee28c83ab8320aa8749b567280b23780dc2a78\`. Both model seeds **11 and 17** receive the original **400 supervised four-cue** updates followed by GG-08's **300 supervised mixed 4/6/8** updates.
- Matched GG-09 continuations start from those same frozen GG-08 endpoint weights. The **static control** receives 240 more updates on the original fixed-world Bayesian task; the **Markov continuation** receives 240 updates on the switched-world exact task. Each cycles depths 4,6,8 with batch 128, AdamW LR 0.001 and terminal plus 0.2 prefix supervision. Both have **184,320 unrolled sample-visits**, exactly 80 updates at each depth, with separate seeded optimizers and pool generators.
- For each depth, the complete finite \(3\times3\times 2^n\) input/context grammar is hashed and split identically for both tasks, two-thirds continuation training and one-third GG-09 holdout. Training/heldout counts are **96/48 at depth four**, **384/192 at six**, and **1536/768 at eight**. The GG-08 source has previously seen some of these input combinations under the fixed-world task; the holdout separates the **new GG-09 continuations**.
- Four independent heldout gold-scoring replicates at each depth contain 64 likelihood-weighted Markov-natural, 32 exact Markov near-boundary, 32 exact low-marginal-likelihood, and 32 exact late-decision-reversal inputs, drawn only from the heldout input/context roster. All labels and challenge cohort identities derive from the exact rational hidden Markov filter.
- [Original GitHub Actions run 43182472403](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43182472403) completed successfully: **8 tests passed in 4.37 seconds**, along with the full two-seed training and all physical report-update / source-bound receipt checks. The only focused-test warning was PyTorch's optional NumPy initialization warning on the CPU runner; the experiment completed.
- [Complete native weight checkpoints, 48-query protected-tail certificates, physical update ledgers, exact panel hashes and train/test roster identities](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43182472403/artifacts/13927061777). All reported physical update scores can be regenerated from the frozen Markov oracle and the original source code.

## Exact hidden-Markov task

The new hidden world is \(S_t\in\{E,D\}\), with \(P(S_1=E)=p\), a switch chance \(q=1/4\) between cues, and emissions \(P(1\mid E)=3/4\), \(P(1\mid D)=1/4\). At each cue, the exact unnormalized posterior masses are updated by

\[
(a^-,b^-)=((1-q)a+qb,\, qa+(1-q)b)
\]

before multiplying the E and D components by their corresponding cue likelihoods. At the first cue, the emission multiplies the original prior directly. A protect action is correct iff \(r a>b\), with tie-to-exploit. The model sees the same binary tokens and two-dimensional prior/reward-ratio context as GG-08, with **no additional HMM task indicator**.

This is a structurally different posterior recursion: past evidence becomes less decisive because the world can change. Independent tests compare the exact rational forward filter with exhaustive enumeration of all hidden-state paths on short sequences and verify that \(q=0\) reduces precisely to the original GG-01 fixed-world model.

| Finite grammar | GG-08 fixed-world versus GG-09 HMM correct-label disagreements |
|---|---:|
| Four-cue \(9\times2^4\) | **35/144 (24.31%)** |
| Six-cue \(9\times2^6\) | **164/576 (28.47%)** |
| Eight-cue \(9\times2^8\) | **727/2304 (31.55%)** |

These exact enumeration counts establish incompatible conditional labels even on identical input tokens and contexts. They are mathematical properties of these two task definitions, rather than learned model error rates.

## Registered cross-grammar competency result

The original 12-cell GG-07/GG-08 capability criterion was applied to the new HMM labels for every depth×cohort. All scoring cohorts are on new GG-09 continuation-heldout input identities.

| Markov exact-action accuracy | Seed 11 | Seed 17 |
|---|---:|---:|
| GG-08 source minimum of twelve cells | 39.06% | 39.06% |
| Static extra-training control minimum | 39.06% | 39.06% |
| **New Markov continuation minimum** | **96.09%** | **94.53%** |
| Required minimum per cell | 90% | 90% |
| Required improvement over static control | 5 points | 5 points |
| Observed minimum-cell improvement | **+57.03 points** | **+55.47 points** |
| **Registered Markov competency-transfer gate** | **PASS** | **PASS** |

Detailed Markov continuation accuracy:

| Depth and gold cohort | Seed 11 | Seed 17 |
|---|---:|---:|
| Four-cue natural/boundary/rare/late | **100% each** | **100% each** |
| Six-cue natural | 99.22% | 98.83% |
| Six-cue boundary | **96.09%** | **95.31%** |
| Six-cue rare | 100% | 100% |
| Six-cue late evidence reversal | 98.44% | 98.44% |
| Eight-cue natural | 100% | 99.61% |
| Eight-cue boundary | 96.88% | 96.88% |
| Eight-cue rare | 100% | **94.53%** |
| Eight-cue late evidence reversal | 97.66% | 100% |

This is strong *within-task-family*, heldout-input-combination transfer under the registered finite HMM grammar. The exact Markov task's gold labels are newly learned; the model retains the same native recurrent structure and sensory vocabulary.

## Task incompatibility and static-task retention

On the *same observed input combinations*, the GG-08 source and static control predict the original \(q=0\) oracle perfectly, scoring **100% in all twelve fixed-world cells**. Their lower HMM accuracy results from the mathematical label incompatibility, with especially low accuracy on HMM late-reversal cohorts.

The Markov continuation fits the new HMM labels at the expense of its original static decisions. Its static-oracle accuracy on these HMM-scoring inputs drops into roughly the **37.5%–73.4% range** across depth/cohort cells for seed 11 and **37.5%–73.4%** for seed 17. Since the exact oracles disagree on a substantial subset of identical input combinations, simultaneous perfect accuracy on both definitions is impossible without supplying a task-mode cue. No such cue was given in the registered experiment. This is a measurable target-semantics tradeoff, not spontaneous awareness of whether a model is being evaluated.

## Real SGD protected-tail effect

The registered source-report set consists of **48 new Markov-oracle-labeled depth-four input identities** excluded from both GG-09 continuation training rosters. For each of three arms and two model seeds, the physically measured action contrast is

\[
D=\ell_{\rm HMM}(\theta-\eta g_X)-\ell_{\rm HMM}(\theta-\eta g_C),
\qquad \eta=0.05,
\]

where \(g_X\) and \(g_C\) are native full-parameter, norm-one-clipped report gradients computed using the HMM's exact gold label. All post-action model weights are independent clones of the arm endpoint. There are **576 physical SGD steps** in total and **1,105,920 individual post-SGD gold CE evaluations** across the three depths.

The protected scorer is the *unchanged* GG-07 product ambiguity: natural/tail mass \(\lambda=1/2\), independent within-cohort empirical-support TV \(\rho=1/64\), depth weights at least \(1/6\) each, and a strict \(10^{-7}\) sign threshold. A query is robustly harmful if every one of four scoring-replicate lower bounds is positive; robustly beneficial if every upper bound is negative; otherwise its sign is unresolved.

| Native arm / seed | Robust harm | Robust benefit | Unresolved | Total signed |
|---|---:|---:|---:|---:|
| Seed 11, original GG-08 source | 13 | 34 | 1 | 47/48 |
| Seed 11, static control | 13 | 35 | 0 | **48/48** |
| **Seed 11, Markov continuation** | **10** | **1** | **37** | **11/48** |
| Seed 17, original GG-08 source | 13 | 35 | 0 | 48/48 |
| Seed 17, static control | 13 | 33 | 2 | **46/48** |
| **Seed 17, Markov continuation** | **19** | **1** | **28** | **20/48** |

The numerical robustness counts are counterintuitive if read without task semantics: the unadapted fixed-world models have mostly *robust beneficial* signs for flipped **Markov** report labels under this HMM gold. Those models were trained to follow a different target mapping and are not competent at the Markov task. A large absolute signed gap in a badly mismatched classifier is not evidence of robust task-aligned training.

For the competent HMM models, most physical report-update harm signs remain **unresolved under the preregistered gold ambiguity**, even though the Markov task accuracy exceeds 94.5% in every cell. The Markov continuation has 11 and 20 robust signs, and its sign coverage is *lower* than the static control by **37 and 26 queries**, respectively. Its own nominal natural-only to protected-tail score changes sign on at least one replicate for **20/48** seed-11 and **17/48** seed-17 report queries; the static controls show zero such switches in the registered source ledger.

**Registered protected-harm-transfer gate: FAIL / FAIL.** The competence prerequisite and the large accuracy gain pass, while the stringent requirement for at least 24 signed Markov queries and at least an additional eight signs over the static control fails for both seeds. The observed harm-score ambiguity remains a scientific outcome.

## Interpretation and next falsifiers

GG-08 demonstrated that within the original fixed-world Bayesian grammar, mixed-depth training could bring every protected-harm query into a robustly harmful empirical-sign region. GG-09 establishes that this combination of competence and update-harm robustness **does not automatically carry over to a new temporal transition rule**. The recurrent model is capable of learning the correct hidden Markov decision function on heldout combinations, while its one-step post-report training effects remain sensitive to the protected gold weighting.

One possible explanation is that the Markov continuation's final loss landscape retains large local curvature or smaller report-gradient signal margins after 240 additional updates. Another is the changed distribution of exact late-reversal cases and the intrinsic conflict between the new and old task labels. GG-09's aggregate signs alone cannot select among these mechanisms. The next useful experiment would retain the **exact same Markov scoring panels** and examine gradient projections, residuals and per-cohort harm sources at the three model endpoints, following GG-05's method.

The holdout is at the GG-09 continuation stage on a finite \(9\cdot2^n\) grammar; the GG-08 starting model previously saw some of those input combinations under a different target definition. Both continuations have matched step/visit budgets rather than guaranteed equal FLOPs or identical empirical training draw distributions. All robust bounds are sharp only within observed empirical cohorts. Two small model seeds limit any population inference.

GG-09 does not train a reporting agent that knows it will be optimized, does not change the reporter's own policy weights through a future update, and supplies no evidence of deceptive self-modeling, goal guarding, or simulator bargaining.
