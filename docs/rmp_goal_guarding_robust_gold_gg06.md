# RMP-GG-06: Distributionally robust gold scoring for physical SGD update harm

**Registered November 12, 2026, before running GG-06 and inspecting its fresh scoring panels.**
Source baseline: LoopedTransformers main commit \`7c437c4b1e66574bfe453e12992737fc59104fe1\`, after GG-05/05P. Earlier GG-04–05 saw sensitive gradient-sign reversals and a post-hoc finding that two historical scoring examples accounted for almost all six-cue CE. Those results motivated this study; they are not prospective evidence for it.

## Research question

The same report-labelled physical SGD update can have different measured effects when an independent gold auditor changes which examples it scores. We need to distinguish stable report harm from a direction that only looks harmful under a particular gold sample or concentration of high-loss cases.

Let \(x\) be one of the frozen 48 GG-04B heldout four-cue input/context combinations. Let \(g_C(x),g_X(x)\) be independently computed native full-parameter report-loss gradients, globally clipped at norm one. Two **real PyTorch SGD** updates are made from the same native recurrent model checkpoint:
\[
\theta_a(x)=\theta-0.05g_a(x), \qquad a\in\{C,X\}.
\]
For gold example \(i\) of depth \(k\in\{4,6\}\), let
\[
d_{x,k,i}=\ell_{k,i}(\theta_X(x))-\ell_{k,i}(\theta_C(x)).
\]
A gold weighting \(w\) scores the sign of \(B_{x,k}(w)=\sum_i w_i d_{x,k,i}\). Positive means the false-label report damages that weighted gold task more than the truthful report. No actor is trained in this study. The label generator remains the *external exact-rational Bayesian rule*.

## Frozen models, queries, and scoring populations

- Native GG-01 \`EvaluationBeliefLooper\`, tied \`ExplicitDecoderBlock\`, width 32, four heads, four-cue training: reinitialize and train seeds **11, 17** using the unchanged \`train_cell(seed,steps=400,batch=128)\`. Retain final weights, seed, checkpoint hash, and training receipt. Model weights remain frozen across the report-outcome audit.
- Source-report roster: **48** previously inspected queries \`query_splits()['heldout']\`. The prior GG-04B and GG-05 source query roster has been seen, so this is a known-query mechanistic follow-up, not independent unseen-query validation. Confirm exact-rational gold labels and query identity digest.
- Independent *fresh* gold panels, all sampled before any new model scores:
  - Four-cue: **8 panels of 128 examples**, \`make_panel(128,4,seed)\` for seed **69020 through 69027** inclusive.
  - Six-cue: **8 panels of 128 examples**, \`make_panel(128,6,seed)\` for seed **68020 through 68027** inclusive.
  - Each panel is generated independently by its declared Python RNG seed. Paired indices 0..7 are used for descriptive four/six depth comparisons; separate seeds give separate samples and do not produce samplewise cross-depth coupling.
- The already-seen 69001/68001 and 69002/68002 panels remain contextual prior results only. The primary GG-06 summary uses only 16 new scoring panels. For each depth, the union has 1,024 gold examples. The empirical union retains duplicates as separately sampled observations with uniform nominal weights.

## Physical measurement and provenance

For each source query, compute truth and flip gradients at the same frozen task-model weights, with the GG-05 full-parameter clip of 1.0. From fresh copies of the source weights, execute **one actual PyTorch SGD step at LR 0.05, momentum 0, weight decay 0** for each submitted report. Preserve all 1,024 per-sample cross-entropies after each action for each depth. There are **96 independent real SGD steps per seed**, 192 total. The auditor measures before/after losses on every gold sample; neither gold target nor any computed report-harm derivative is sent to a policy learner.

For each seed, report task-model training weight hash, exact panel hashes, report/query hashes, per-example baseline gold CE, per-example false-minus-true post-SGD gold CE differences, all clipped gradient norms, optimizer identity controls, all 8 panel means by depth and derived robust intervals. A source digest check must establish that the original task-model weights remained unchanged throughout the native replays. Model training and physical SGD are deterministic under the frozen seed/software protocol; save source artifacts from CI.

## Sharp finite-support total-variation ambiguity certificate (GG-06/T1)

For any one fixed scoring panel of \(n=128\) physically measured numbers \(d_1,\ldots,d_n\), nominal weights are \(u_i=1/n\). Define the **empirical-support-only** ambiguity set
\[
\mathcal W_\rho=\left\{w_i\ge0,\;\sum_i w_i=1,\;
\tfrac12\sum_i|w_i-u_i|\le\rho\right\}.
\]
The preregistered radii are
\[
\rho\in\{0,\tfrac1{64},\tfrac1{16}\}.
\]
Compute the **sharp** finite robust interval
\[
\boxed{\underline B_\rho=\min_{w\in\mathcal W_\rho}\sum_i w_id_i,\qquad
\overline B_\rho=\max_{w\in\mathcal W_\rho}\sum_iw_id_i.}
\]
For \(\rho\le1/16\) and \(n=128\), the minimum is obtained by moving at most total mass \(\rho\) from the largest \(d_i\) values, greedily exhausting their nominal masses, to a smallest \(d_j\) value; for the maximum reverse largest and smallest. A witness weight vector, its normalization, TV radius and attained objective are checked. This is an exact finite-dimensional linear optimization argument conditional on the **floating-point measured** values of \(d_i\); it is not a numerical certificate about neural parameter rounding. An independent small-n rational-grid oracle tests the greedy formula.

Also compute the same \(\rho\) interval on the **pooled 1,024-example gold support** per depth for each source query. The method does not cover genuinely novel out-of-support events. It explicitly permits concentrating more weight on already observed hard cases.

A query/depth is **robust harmful** at a radius if *every one* of its eight panel-specific lower bounds is strictly positive by more than \(10^{-7}\). It is **robust beneficial** if every one of the eight upper bounds is below \(-10^{-7}\). Otherwise classify the sign as **unresolved**. Keep the per-panel nominal signs and disagreement count separate from robust bounds. The primary robustness radius is \(1/16\); \(0\) and \(1/64\) are registered sensitivity controls.

## Auditor-chosen hard-case emphasis

For every gold panel, compute **baseline** per-example cross-entropy before any report-conditioned update. Freeze the top \(\lceil0.05n\rceil=7\) highest-loss scoring indices (ties by smaller original index), and define a prespecified family of gold weights
\[
w_i(\tau)\propto
\begin{cases}
\tau,&i\text{ in baseline hardest seven},\\
1,&\text{otherwise},
\end{cases}
\quad \tau\in\{1,4,16\}.
\]
These weights are evaluator-defined from **pre-update gold loss**; they cannot depend on whether a report subsequently damages the target. Record nominal, 4× and 16× weighted report-harm gaps, signed agreement and the fraction of gold mass on those seven cases. Report results per panel and pooled across eight fresh panels at each depth. This deliberately tests a tail-sensitive objective and does not claim that tail weights are calibrated probabilities of deployment harms.

## Registered readouts and gates

For each of the two seeds and each depth, report:
1. Across 48 queries, the number whose nominal wrong-minus-true loss gap has the same nonzero sign on *all eight* independent panels, and the number with any panel sign disagreement.
2. At \(\rho=0,1/64,1/16\), the counts of **robust harmful**, **robust beneficial** and **unresolved** queries across all eight panels, plus a 1,024-example pooled certificate.
3. At tail factors 1,4,16, report sign unanimity and the number of query/panel sign changes relative to the uniform gold baseline.
4. Across the eight independent 4-vs-6 panel-index pairs, the distribution of **actual physical gradient sign reversals** over 48 queries, and the fraction of queries with any and with all eight reversals. This pairs *panel indices* only, without asserting a joint task-example coupling.
5. The min/max signed report-harm effects and the identities of the baseline hardest seven examples for every panel, in the raw source-bound receipt.

**Strong nominal sampling stability** requires at least **36/48 queries** per seed/depth with the same nonzero sign on all eight fresh panels. **Strong TV robustness** requires at least **24/48 queries** per seed/depth classified either robust harmful or robust beneficial across all eight panels at \(\rho=1/16\). The two-seed result supports either strong gate only if both seeds meet it at **both depths**. A zero or negative result is retained with denominators unchanged. In particular, we expect a robust certificate may fail on the six-cue task when rare high-loss cases dominate the gradient.

These are **finite sampled-support sensitivity tests**, not population guarantees or full distribution-shift robustness. They do not test spontaneous training awareness. No policy or training strategy is selected using any GG-06 gold panel result.

## Code, independent controls, and CI

- Native analysis: \`research_gym/interp/goal_guarding_robust_gold.py\`, using the existing exact Bayes task model, physical report gradients, and source-bound hashes.
- Independent tests: a small rational-grid exhaustive LP comparison for the TV bound, witness validation, a hand-constructed high-loss tail-weight control, an independent PyTorch reference SGD, and empirical sign-classification checks.
- Frozen CPU GitHub Actions: run tests, execute both native seeds, validate source/ledger/schema and original-parameter invariance, upload full physical-update arrays and a reproducible JSON closeout. CI success indicates valid **measurement**, not a positive scientific gate.
