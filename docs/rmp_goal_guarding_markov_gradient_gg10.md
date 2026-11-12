# RMP-GG-10: Gradient and scoring-ambiguity anatomy of GG-09 Markov transfer

**Registered November 12, 2026 before any GG-10 measurements.**
Base: LoopedTransformers main \`8299198f925dcbf5e875d8a241be317bb42e35cf\`.
The original [GG-09](../reports/checkpoint_rmp_gg09_markov_transfer_20261112.md) run has already been read: two Markov-trained recurrent models achieved >94.5% in every heldout Markov cohort, but had **37/48** and **28/48** unresolved protected SGD-harm signs. This follow-up explicitly distinguishes *post-hoc mechanistic reconstruction of known panels* from *independent newly sampled gold scoring panels*.

## Frozen source and cohorts

Reproduce GG-09's native tied \`EvaluationBeliefLooper\` from source seed 11 and 17 using 400 GG-01 AdamW updates, 300 GG-08 mixed-depth continuation updates and the same 240-step GG-09 \`static_control\` or \`markov_transfer\` continuation. Keep all hyperparameters and RNG schedules unchanged; both continuation models start from the same GG-08 weights per seed. Independently verify their endpoint weight hashes against the original source-bound GG-09 receipt:

| Model seed | Static endpoint SHA-256 | Markov endpoint SHA-256 |
|---|---|---|
| 11 | \`35be2252711351d11364a625dc2ef01bb61c792ad72392225863e750af0e3e33\` | \`31e31149171ba6452acc27abd552c83624922f5d8b051651aed3142835ad1ad7\` |
| 17 | \`2a335c1c5b0075cabb8baf24f41635bbc7723ec8fcf8be6a91514bfa994cf88a\` | \`0c11e22c64e0f50bf5ee732e004adeb7200584bf8fee1cdd4dabe1d38427011b\` |

These references came from the completed [GG-09 artifacts](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43183115898/artifacts/13927786163). The endpoints are known and their exact reproduction is an instrumentation gate.

The source reports are the **same 48 heldout Markov-labelled four-cue input/context combinations** from GG-09, unchanged and labelled by the external rational HMM oracle. The gold panel cohorts are natural 64, boundary 32, rare 32 and late-reversal 32, over depths 4/6/8 with 4 replicates each. No new reporter is trained.

Two separately labelled panel families:

- **Historical reconstruction:** the original frozen GG-09 \`markov_gold_bank(depth)\` with old RNG seeds (natural \`95000+100*depth+rep\`; challenges \`96000+1000*depth+10*rep+cohortOffset\`). These results *are post hoc* and must reproduce GG-09's robust-sign census: static seed 11: harm13/benefit35/unresolved0; static seed17: harm13/benefit33/unresolved2; Markov seed11: harm10/benefit1/unresolved37; Markov seed17: harm19/benefit1/unresolved28.
- **Fresh scoring family:** identical exact HMM task grammar and heldout input support with new seeds: natural \`105000+100*depth+rep\`; boundary/rare/late \`106000+1000*depth+10*rep+offset\`, offsets 1/2/3. All cohort pools remain determined only by exact HMM math and the GG-09 heldout ID split. Four independent replicated gold panels at each depth, with hashes and full membership frozen before evaluation. This tests stability of the already-seen Markov failure over new evaluator composition, *not generalization to a new task grammar*.

Both model arms are scored on both panel families under identical physical report gradients and SGD successor weights.

## Native gradient decomposition

For a report query \(x\), let \(g_C\) and \(g_X\) be the actual full-model truthful and flipped report gradients, each clipped to norm 1. On a frozen native model checkpoint \(\theta\), physically apply one PyTorch SGD step from identical weights at \(\eta=0.05\) for each report action. For depth \(d\), replicate \(r\), cohort \(c\), define
\[
 B_{drc}(x)=L_{drc}(\theta-\eta g_X)-L_{drc}(\theta-\eta g_C).
\]
Compute the exact autograd **mean gold cohort** gradient at baseline \(G_{drc}=\nabla_\theta L_{drc}(\theta)\) and the first-order term
\[
 A_{drc}(x)=-\eta\langle G_{drc},g_X-g_C\rangle,\qquad
 R_{drc}(x)=B_{drc}(x)-A_{drc}(x).
\]

The equality \(B=A+R\) is exact up to the machine precision of physical loss and gradient computation. \(R\) is the difference of the two integrated Hessian contributions along the two report-update paths, and is *not* assumed small. Gold cohort labels are calculated by the independent rational Markov oracle, never by an action/report-target model.

At the registered **primary nominal protected scoring weights** \(\lambda=1/2,\rho=0\) and uniform depth weights \(1/3\), compute
\[
 A_r=\frac13\sum_{d\in\{4,6,8\}}
 [\tfrac12A_{dr,N}+\tfrac16(A_{dr,B}+A_{dr,R}+A_{dr,L})],
\]
and \(B_r,R_r\) by the same linear combination. Each quantity is retained per source query/replicate, together with a first-order-versus-physical sign comparison at ambiguity tolerance \(\varepsilon=10^{-7}\). The *nominal first-order agreement fraction* counts an unambiguous physical sign explained by the corresponding first-order projection. Report physical zeros, predicted zeros and residual-driven disagreements explicitly.

For the primary GG-09 **robust** \(\rho=1/64\) score, use GG-07's complete physical per-example TV cohort bounds and constrained three-depth scorer *unchanged*. The first-order comparison concerns **nominal cohort means**, not a fictitious finite-TV bound on mean gold gradients. The physical robust scorer remains the only source of signed uncertainty certificates.

## Exact scorer-ambiguity width algebra

Let \(\underline B_{dr},\overline B_{dr}\) be the primary protected depth-specific lower/upper bounds from the GG-07 cohort TV LP, at lambda \(1/2\) and rho \(1/64\). Its composite width obeys the exact identity
\[
\overline G_r-\underline G_r
=\underbrace{\tfrac16\sum_d(\overline B_{dr}-\underline B_{dr})}_{W_{\rm cohort}}
+\underbrace{\tfrac12(\max_d\overline B_{dr}-\min_d\underline B_{dr})}_{W_{\rm depth}}.
\]
Within-depth width is the nonnegative fixed protected weighted sum of the four cohort TV widths. Record the separate natural/boundary/rare/late contributions to \(W_{\rm cohort}\), the additional depth-selection width \(W_{\rm depth}\), and the selected worst-case lower and upper depth for each of four replicates. This is an algebraic decomposition of the *auditor's optimization set*; it is not an attribution of model intent.

For each report query classify historical unresolved status into exclusive reasons, in order:

1. **Panel sign heterogeneity:** uniform-depth nominal \(B_r\) has opposing or zero signs across the four replicates.
2. **Depth-mixture ambiguity:** nominal \(B_r\) has the same nonzero sign across replicates, but \(\rho=0\) constrained-depth robust status is unresolved.
3. **Cohort TV ambiguity:** \(\rho=0\) constrained-depth status is robust, while \(\rho=1/64\) is unresolved.
4. **Other numerical boundary:** a remaining unresolved case. Retain every individual original bound and epsilon threshold.

For an unresolved case, record the cohort with the largest absolute **nominal first-order magnitude** averaged over all depths and replicates, as a diagnostic, and the number of replicates with a residual-driven sign disagreement. Do not equate largest-magnitude cohort with causally necessary source of ambiguity.

## Registration of outcomes and gates

For each seed and arm:

- Original and new 12-cohort task accuracy (Markov exact gold), original and fresh 48-query robust-status census under \((\lambda,\rho)=(1/2,1/64)\), and overlap of the unresolved source-query ID sets.
- Original and fresh nominal first-order sign agreement and residual-driven sign discordance, over the **192 source query × scoring replicate** pairs, plus median/maximum absolute residual and gold-gradient cosine between three depths as a context summary.
- Counts of historical-unresolved queries by the exclusive ambiguity reasons, fraction whose robust status remains unresolved on fresh scoring panels, and cohort/depth ambiguity width contributions.
- A shadow diagnostic for the static control under *incompatible* Markov gold, with no claim of valid goal-guarding harm incentives.
- Source weight checksum, all oracle/panel input hashes, physical clipped report gradients, cohort physical \(B,A,R\), all GG-07 protected bounds, and final model checkpoints uploaded as CI artifacts.

**Instrumentation gate:** exact source SHA, original GG-09 numeric census, independent rational heldout/cohort labels, no-update weights, native SGD versus independent PyTorch reference on a small panel, exact width decomposition, and all focused tests must pass.

**First-order-supported explanatory gate:** for both *Markov-trained* seeds on the **new** panels, at least 75% of unambiguous nominal protected query×replicate physical signs agree with first order, with at least 96 unambiguous outcomes per seed.

**Residual-sensitive event:** at least 25% sign disagreements on those fresh, unambiguous protected outcomes in either Markov seed. Report it independently of the first-order gate.

**Unresolved-status replication event:** at least 16 of 48 original Markov unresolved query IDs remain unresolved under new gold panels in **each** seed. This is a prospective new-scoring-panel check of a known source-model failure, not a test of an unseen task.

A failed scientific gate stays visible and cannot be repaired by changing the scorer, numerical tolerance or model endpoint. GG-10 is a mechanism audit on small supervised recurrent classifiers and does not establish autonomous training awareness, future-reward self-modeling or deceptive goal guarding.
