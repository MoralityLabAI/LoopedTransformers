# LS-NEURON-14: certificate-aware discrete search

LS14 failed its locked primary repair gate. With single-code moves, interval-aware
selection produced **4/24 exact certificates**, while point-loss selection produced
**7/24**. Both paired-move arms produced **9/24**. The secondary gate also failed.
The empirical comparison remains **PILOT_ONLY**; the FSI-01 reports classify both
population comparisons as **inconclusive**.

All 24 starting models had exact point-violation witnesses. Among the 96 final
arm/seed endpoints, the unchanged rational verifier returned 29 certificates and
67 point violations, with zero unresolved cases. Those 29 certified endpoints
contain 20 distinct decoded weight sets. Each comparison still has 24 independent
initialization seeds.

The numerical interval threshold and the exact verdict agreed on all 96 final
endpoints. This agreement supports the score's threshold decisions on this cohort;
the cause of the search outcome remains open.

## Outcome tables

Source: [audit_results.json](audit_results.json). All reported new repairs began
with a proven violation of the fixed-region invariant. The disturbance budget
stayed at 1/32 throughout the study.

| Search arm | New exact certificates | Certificate rate | T8 final-state accuracy | T128 final-state accuracy | All prefixes correct through T128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Point loss, single moves | 7/24 | 29.17% | 93.59% | 78.66% | 72.22% |
| Interval score, single moves | 4/24 | 16.67% | 78.29% | 57.98% | 45.72% |
| Point loss, paired moves | 9/24 | 37.50% | 97.10% | 86.26% | 82.42% |
| Interval score, paired moves | 9/24 | 37.50% | 87.32% | 69.24% | 60.21% |

The numerical columns average the inherited 256-sequence evaluations within each
seed. The untouched initial models averaged 28.24% at T8 and 22.15% at T128, with
0% correct on every prefix through T128. Rollouts used FP32 arithmetic and no added
state disturbance. Exact certificates apply to the ideal real recurrence and its
declared disturbance budget.

The primary gate required at least 12 new repairs out of 24, together with at
least six more repairs than the matched point arm. The interval-single arm
obtained four, with an advantage of minus three. The interval-paired arm obtained
nine, with an advantage of zero. Both parts of each gate failed.

| Matched comparison | Interval-only repairs | Point-only repairs | Ties | Paired difference | Exact two-sided p | Bonferroni p, family of two |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single moves, primary | 0 | 3 | 21 | -12.5 percentage points | 1/4 | 1/2 |
| Paired moves, secondary | 2 | 2 | 20 | 0 percentage points | 1 | 1 |

For single moves, seeds 447, 452, and 459 certified only under point selection.
The four interval-single successes were 448, 449, 460, and 461. In the paired
comparison, interval selection alone succeeded on 459 and 463; point selection
alone succeeded on 452 and 462. Complete paired outcomes appear in
[paired_single.jsonl](evidence/results/paired_single.jsonl) and
[paired_paired.jsonl](evidence/results/paired_paired.jsonl).

The official FSI reports use the allocated alpha of 0.025. Their marginal
Clopper-Pearson intervals therefore have 97.5% coverage under the stated iid
seed-draw assumptions: 11.02% to 53.96% for 7/24, 3.84% to 40.30% for 4/24, and
16.83% to 62.14% for each 9/24 rate. These marginal intervals have their individual
coverage level; a joint four-rate coverage claim would need a separate adjustment.

For the declared 25-point population effect floor, the FSI one-sided Hoeffding
radius is 0.5544426 at 24 seeds and two comparisons. The single-move lower bound
is -0.6794426, and the paired-move lower bound is -0.5544426. Both miss the floor.
The observed negative primary difference describes this cohort. Population harm
and population equivalence remain unresolved by these data. See
[statistics.json](evidence/results/statistics.json) and the independent
[postrun statistical review](evidence/postrun_review.md).

## What the experiment changed

The parent is LS13 at commit
[`88aef2c`](https://github.com/MoralityLabAI/LoopedTransformers/commit/88aef2c14c6b9d75775c2379c2ee5cc6ce5484a9).
LS14 kept its eight-GELU scalar mod4 cell, Q4 code limits, and continuous
least-squares fitter. Fresh seeds 445 through 468 received the same 15% tensor-RMS
perturbation recipe. One saved initialization supplied all four arms for each seed.

Each pair of arms differs in the score used to rank proposals and accept fitted
candidates. Point selection minimizes mean squared error plus the inherited
squared excess penalty. Interval selection minimizes the largest numerical
enclosure of transition error over the region. Both selection rules use the same
17 endpoints per state/input case, while the shared continuous fitter and scale
profiler retain LS13's nine-point objective.

This intervention combines a change from average loss to worst-case loss with an
interval-curvature allowance. Its interpretation concerns that combined change.
A point-maximum-only control would separate the curvature term from the change
in aggregation.

The model has 54 decoded recurrence slots. Its storage convention uses 52 integer
codes in [-7,7], four positive tensor scales, and two FP32 scalar offsets. The
precision claim applies to the integer codes. Paired moves use the known
reset/ramp architecture, including paired incoming directions, bias gaps, and
opposite output weights.

## Bound used for selection

For state p, the center is c_p=p-2 and the region radius is 1/8. The input
(a,b,n) ranges over all eight binary triples, with target state (p+a+b) mod4.
The nuisance bit n remains in the complete case set.

With z=(h,a,b,n), the recurrence is

\[
F(h,x)=h+s\cdot z+b_s+
\frac15\left(\sum_{i=1}^{8}v_i g(u_i\cdot z+b_i)+b_d\right).
\]

Write g(z)=z Phi(z). Its derivatives satisfy

\[
g''(z)=(2-z^2)\phi(z),\qquad
g'''(z)=z(z^2-4)\phi(z).
\]

The maximum of |g''| over a preactivation interval occurs among its endpoints
and any contained points -2, 0, and 2. Split each state region into 16 segments
of width Delta=1/64. For a segment [l,r], the interpolation bound is

\[
B_{l,r}=
\max\{|F(l,x)-c_t|,|F(r,x)-c_t|\}
+\frac{(r-l)^2}{40}
\sum_i |v_i|u_{ih}^{2}
\max_{z\in Z_i([l,r])}|g''(z)|.
\]

The coefficient 1/40 combines the recurrence's beta=1/5 with the interpolation
remainder factor 1/8. The selected score takes the maximum over every segment
and all 32 state/input cases. The mathematical enclosure follows from the second
derivative remainder. The implementation evaluates it in Float64 without directed
rounding, so it remains a numerical search score. The final rational verifier
supplies each certificate.

Summing absolute neuron curvature discards cancellation between neurons. In this
cohort, every final score at or below 3/32 certified, and every final score above
3/32 had an exact violation witness. The closest certified single-point endpoint
had score 0.0904591; the smallest score among failed single endpoints was 0.107782.
The fixed threshold was 0.09375. The complete per-arm diagnostic ranges appear in
[diagnostics.json](diagnostics.json).

## Equal search opportunities

Each run begins with the same initial least-squares fit, then performs 16 rounds
with current and original nonlinear-scale anchors. An anchor profiles every
fixed proposal slot and fits the no-op plus four distinct selected codebooks.
The incumbent remains eligible. Selection after the common initial fit requires
a score decrease greater than 1e-12.

Every arm/seed receives exactly 161 fitting calls, capped at 40 function
evaluations each. The single family uses 105 profile slots per anchor; the paired
family uses 177. Illegal integer moves become no-op, and duplicates stay in the
profiling count. Stable ordering resolves equal scores.

| Arm | Profile slots across 24 seeds | Fitting calls | Actual function evaluations | Mean final code Hamming distance |
| --- | ---: | ---: | ---: | ---: |
| Point, single | 80,640 | 3,864 | 82,026 | 3.04 |
| Interval, single | 80,640 | 3,864 | 86,544 | 2.83 |
| Point, paired | 135,936 | 3,864 | 65,231 | 4.88 |
| Interval, paired | 135,936 | 3,864 | 69,398 | 5.04 |

The caps match within each registered comparison. Actual solver evaluations
depend on convergence. Paired proposals have a larger slot budget and use
architectural information, so the single-versus-paired totals have a different
interpretive scope from the two matched comparisons.

All proposals and fitted candidates pass through the actual FP32 decoder before
selection scoring. Continuous fitting still minimizes the point objective. That
choice isolates the ranking/acceptance intervention while leaving a possible
interaction between the fitter and the minimax score for later study.

## Exact result and audit scope

A successful certificate proves

\[
F(R_p,x)\subseteq
[c_{(p+a+b)\bmod4}-3/32,\ c_{(p+a+b)\bmod4}+3/32].
\]

Any real disturbance bounded by 1/32 then returns the state to the target
radius-1/8 region. Starting at h=-2, induction gives correct region membership
for every finite input sequence under this exact-real recurrence. Nearest-center
readout has margin 3/8 beyond the certified region boundary.

The verifier uses exact rational enclosures of GELU, with interpolation and
ReLU-tail bounds inherited unchanged from LS12. Stored FP32 weights become exact
dyadic rationals; beta is exactly 1/5. The proof scope leaves hardware rounding
and native transformer transfer for separate validation. A Python rational
certificate also has a separate trust basis from a Lean/kernel-checked theorem.

The audit passed with these counts:

| Check | Count |
| --- | ---: |
| Independent perturbation seeds | 24 |
| Final arm/seed endpoints | 96 |
| Initial files plus search receipts | 120 |
| Exact verifier calls, including the positive control | 121 |
| Final certificate / point violation / unresolved | 29 / 67 / 0 |
| Profile slots replayed | 433,152 |
| Fitted candidate states checked | 15,456 |
| Accepted discrete code edits | 365 |
| Accepted updates including scale-only changes | 407 |
| Numerical horizon conditions | 240 |

Audit replay covers source hashes, shared initial states, every profile ranking,
deterministic shortlists, candidate fit losses, legal named moves, and the
accepted winner. A separate direct-tensor implementation checks the numerical
scores. Final models also receive a 257-point region diagnostic, both rollout
horizons, and the unchanged exact verifier. The stored SciPy fit outputs and their
selection consequences were checked; solver optimization trajectories remain
outside the replay scope.

Full rational certificates and numerical correctness vectors are in
`certificates.json` within the companion archive, or in
[certificates.json.gz](certificates.json.gz) in the repository. The compressed file
expands to SHA-256
`767519e2c70424e4d20b72cb8aa62fea279305894ce5b4bc39ba6ab8a815a60c`.

## Chronology and reproduction

The protocol, implementation, auditor, evidence registrations, and dependency
sources were committed at
[`0e8b42a`](https://github.com/MoralityLabAI/LoopedTransformers/commit/0e8b42abd1b3771e38d47fb2109620e4badc6108)
before generating the fresh cohort. The source lock SHA-256 is
`aef20a7da64917eb385b502c88877a7aa0d50fd9f08887b8787d7b4d01e21389`.
All 96 searches finished before certificate and rollout scoring began.
This is prospective source chronology for a development pilot.

The official EvalMath preflight returned REDESIGN_OR_PILOT_ONLY using historical
LS13 context. The current study had no measured development effect for the new
ranking rule or an empirical matched-sham calibration. Its supplied floor of one
was the universal support bound for binary paired differences. The preserved
[preflight scope](evidence/preflight_scope.json) explains that conservative input.

The supplied LS13 archive was verified against SHA-256
`02b988671c37e16391c25c44ac560f814e2709b6d1146ec2963d817961b0f316`.
Implementation checks reproduced all 12 old 15% initializations exactly, matched
the inherited unique proposal families, checked the constructive positive
control, and replayed two complete development-seed searches. Fresh seeds stayed
untouched until the remote lock.

Python, NumPy, SciPy, and CPU Torch versions appear in [environment.json](environment.json).
Execution used four workers with one numerical thread each. Search completed in
about 218 seconds of wall time; the full audit took about 142 seconds. The work
used the available CPU environment and incurred no paid GPU allocation.

From this directory, with the pinned packages installed:

```bash
sha256sum -c lock.sha256
python3 -m zipfile -e receipts.zip .
python3 audit14.py --workers 4
python3 evidence/stats14.py --certificates certificates.json --outdir evidence/results
```

The companion ZIP already contains the expanded `initial/` and `results/`
directories. For a fresh search replay, check out the source-lock commit in a
separate working directory and run `python3 search14.py --workers 4`; the script
requires those output directories to be absent. Every stored raw receipt is
listed in [receipt_manifest.json](receipt_manifest.json). The repository's
`receipts.zip` has SHA-256
`2b3dc70a46aefe327057c4be7ea26f9940dca9630b28d16a439fdee340ca4e44`.

## Proposed LS15 question

A next experiment can fit the linear scale block directly to a worst-case
objective. Hold the codebook, decoded up weights, and decoded up biases fixed.
At endpoint e, define A_e from the skip codes, D_e from the GELU activations and
down codes, and r_e=h_e-c_e. Let kappa_j be the segment's curvature coefficient
using down codes, before applying the positive down scale q. Then solve

\[
\begin{aligned}
\min_{a,q,o,t}\quad &t\\
\text{subject to}\quad
&\sigma(r_e+aA_e+qD_e+o)+q\kappa_j\le t,
&&\sigma\in\{-1,+1\},\\
&10^{-6}\le a,q\le128,\quad -8\le o\le8,\quad t\ge0.
\end{aligned}
\]

Both endpoints of every segment receive both sign constraints. With exact
coefficients this is a convex linear program for the conditional enclosure.
Numerical solving and FP32 decode still require decoded rescoring followed by
the rational verifier.

A matched point-minimax arm sets kappa_j to zero, while an interval-minimax arm
uses its local-curvature value. Giving both arms the same nonlinear update and
solve budget would isolate the padding term. A concurrent least-squares control
would address the separate hypothesis about fitter/selection mismatch. These
are proposed tests; LS14 contains the original locked cohort only. The complete
derivation is in [bound_postrun.md](bound_postrun.md).
