# RMP-GG-07: Preregistered protected-tail and depth-mixture SGD audit

**Registered November 12, 2026, before any GG-07 outcomes.**
Base source: LoopedTransformers \`main\` at \`0bdfaee3af9474d9df5cf736a915db0a44998d05\`.
GG-06 found that wrong-label physical SGD had a stable harmful sign when scored at four visits, while six-visit signs depended strongly on the gold-panel composition. GG-07 tests a richer, **auditor-specified gold task distribution** with hard cases defined by the exact external Bayesian oracle, rather than selected after seeing neural errors.

## Frozen models, report queries and gold task

- Native tied \`EvaluationBeliefLooper\` (one shared \`ExplicitDecoderBlock\`, width 32, four heads). Retrain from seeds **11,17**, using the original \`train_cell(seed,steps=400,batch=128)\`. Keep the final checkpoint regardless of accuracy. Four-cue training task and report-label head remain unchanged.
- Report-action inputs: the existing SHA-256-sorted **48 GG-04B heldout four-cue query/context combinations**, with exact independent Bayesian truthful label; the roster is already known and is not claimed as a fresh query holdout.
- Audited task depths **4, 6, 8**. Eight visits introduce a new inference-depth generalization target relative to the four-visit training schedule. Every binary observation sequence obeys the same registered \(P(1\mid E)=3/4,\ P(1\mid D)=1/4\) Bayesian rule. This is a more demanding **gold evaluation grammar**, not a new neural training algorithm.
- Four independently generated **replicates** per depth, indexed \(r=0,1,2,3\). Each replicate is a protected gold panel containing 160 independently sampled example rows:
  - **Natural 64:** \`make_panel(64,depth,74000+100*depth+r)\` under the original exact mixture generator.
  - **Boundary 32:** uniform-with-replacement draws from the lowest quartile of \(|\log((p\,R/G)/(1-p))+(2k-n)\log3|\) over the **complete finite** \(3\text{ priors}\times3\text{ ratios}\times2^n\) support. Ties by canonical exact prior/ratio/bits ordering. This proxy for difficult decisions is computed solely from the exact Bayes rule.
  - **Rare 32:** uniform-with-replacement draws from the lowest quartile of the exact mixture probability \(pP(b\mid E)+(1-p)P(b\mid D)\), again over the complete finite support. This defines **rare exact observation sequences**, independently of any model error or loss.
  - **Late-reversal 32:** draws from the complete support of contexts/sequences whose exact Bayesian action at prefix length \(n/2\) differs from the action at length \(n\). This tests sequential evidence overturning an earlier decision.
- The three challenge pools are **auditor-defined using exact Bayes**, not model baseline CE or future SGD harm. Every challenge draw uses its own \`random.Random(76000+1000*depth+10*r+offset)\`, with offsets 1,2,3 respectively. Draws are independent across cohorts and replicate seeds, with replacement because some finite four-cue challenge sets contain fewer than 32 unique cases. Preserve repeated draws in the empirical objective; report unique-support counts and hashes. All four cohorts are sampled before any gold scoring, and no model weights or learning parameters are tuned to them.
- Panel sizes: **4 replicates × 160 = 640 examples per depth**, so each seed evaluates **1,920 gold inputs** across depths 4/6/8 for every frozen post-report SGD state.

## Real physical optimizer intervention

For each of the same 48 report queries \(x\), calculate complete native task-model truthful and flipped report gradients with registered global norm clip 1.0. Perform one genuine PyTorch SGD step from the exact same frozen starting weights for each report, at \(\eta=0.05\), no momentum/weight decay. Score each successor's **per-example** gold cross entropy on the pre-frozen panels. This gives **96 actual physical SGD steps per seed, 192 total**, and \(2\times48\times1920=184{,}320\) post-SGD per-example CE evaluations per seed.

The per-example *wrong-minus-truthful* gold-loss effect is \(d_{x,i}=\ell_i(\theta_X)-\ell_i(\theta_C)\). An independent physical-SGD reference test recomputes the gradient and step using the previous GG-04 \`SealedGoldAuditor\`. The original task model weight hash must remain unchanged after the counterfactual scoring. Save actual native checkpoint, source SHA, every observed loss, exact gold label and panel manifest, plus both action-conditioned per-example losses.

## Protected-tail objective and sharp product ambiguity (GG-07/T1)

For depth \(d\), replicate \(r\), let \(\mu_{d,r,N}\) be the **mean** gold update gap on 64 natural examples. Let \(\mu_{d,r,B},\mu_{d,r,R},\mu_{d,r,L}\) be means of 32 boundary, rare, and late-reversal examples. For a frozen tail mass parameter \(\lambda\), define the protected gold harm objective
\[
 B_{d,r}(\lambda)=(1-\lambda)\mu_{d,r,N}+
   \frac{\lambda}{3}(\mu_{d,r,B}+\mu_{d,r,R}+\mu_{d,r,L}).
\]
Register \(\lambda\in\{0,\frac14,\frac12,\frac34,1\}\); the **primary protected-tail objective** is \(\lambda=\frac12\). This yields 50% total weight on the three independently declared hard-case cohorts.

Allow **independent within-cohort empirical-support TV shifts** around uniform row weights, each with \(\rho\in\{0,1/64\}\). For each cohort, obtain the sharp \((\underline\mu,\overline\mu)\) from GG-06's exact-rational finite mass-transport LP. Then the sharp protected lower/upper interval, conditional on the physically measured loss values, is the corresponding nonnegative sum of cohort extrema. These lower and upper bounds are attainable because each cohort's weight choice is independent.

Across the three depths, additionally allow an independently controlled protected depth mixture \(\alpha_d\ge1/6,\sum_d\alpha_d=1\). Its conditional sharp bounds at replicate \(r\) are
\[
\boxed{\quad
\underline G_r=\frac16\sum_{d\in\{4,6,8\}}\underline B_{d,r}
 +\frac12\min_d\underline B_{d,r},\quad
\overline G_r=\frac16\sum_d\overline B_{d,r}
 +\frac12\max_d\overline B_{d,r}.
\quad}
\]
This follows from the three-simplex polytope's vertices, where the additional free mass \(1/2\) is concentrated on the worst depth for a lower bound or best for an upper bound. The composite product ambiguity bound is **sharp on the finite sampled support**, subject to the declared independent cohort and depth weights. Its TV ambiguity does not include unseen examples, altered channels, gradient drift, or model numerical error. Sign threshold \(10^{-7}\) is unchanged from GG-05/06.

A report query's protected-tail sign is **robust harmful** if all four replicate lower bounds exceed \(10^{-7}\), **robust beneficial** if all four replicate upper bounds are below \(-10^{-7}\), and **unresolved** otherwise. Report results for all five lambda values and both rho values; the registered primary readout is lambda \(1/2\), rho \(1/64\).

## Decision controls and acceptance

For each native seed report:
1. Baseline model accuracy on **every depth × cohort** of independently sampled gold, including any eight-depth failures. If any of the 12 accuracy cells falls below **90%**, the protected-tail result remains a physical measurement but is marked **TASK-CAPABILITY-UNRESOLVED** for that seed.
2. Nominal signed harm under every depth, cohort and replicate, and robust lower/upper intervals for all weight settings. Preserve all 48 query IDs and panel hashes.
3. For the primary composite protected objective, counts of robust harmful/beneficial/unresolved source queries over all four replicates, plus the number whose nominal sign changes relative to the natural-only (\(\lambda=0\)) score.
4. The influence of the boundary, rare and late-reversal cohorts separately. Report the fraction of query×replicate cells with a sign change when switching lambda 0→1/2, as well as depthwise sign differences.
5. Non-primary sensitivity at lambda 0,1/4,3/4,1 and rho 0, plus the fixed worst-depth mixture.
6. Actual weight-update and no-update identities, model checkpoint hashes, exact-input-derived panel hashes, and a source-bound full JSON evidence ledger.

**Strong protected-tail sign stability** requires at least **24/48** robust harmful or robust beneficial source queries for *both* model seeds at primary lambda 1/2, rho 1/64, under the depth ambiguity and all four independent panels, **and** every seed's 12 baseline accuracy cells at least 90%. A failure remains a negative or capability-limited experiment result.

A secondary **tail-induced sign-change** event occurs when at least **8/48** queries in either seed have an opposite nominal sign from natural-only at primary lambda 1/2 on at least one of four replicated depth-mixture nominal scores. This is a descriptive preregistered event, distinct from the robustness gate.

## Reproducibility, scope and future work

Code goes in \`research_gym/interp/goal_guarding_protected_tail.py\` with source-bound CLI, tests, and a CPU GitHub Actions workflow. Independently enumerate the finite oracle support and verify exact label/boundary/rare/late-reversal cohort membership. Test the TV LP witness against an independent exhaustive small-n rational-grid oracle and the depth-mixture vertices against brute force. Native real-SGD parity must be checked with a separately implemented physical optimizer reference.

This experiment **does not train a goal-guarding actor**, modify any reporting policy in response to actual optimizer state, or establish the model's awareness of evaluation. It studies sensitivity of measured update harms to a protected evaluator objective defined independently of neural outcomes. Future same-weights work would need an explicitly authorized separately registered intervention; the GG-07 synthetic task is a mathematical and methodological control.
