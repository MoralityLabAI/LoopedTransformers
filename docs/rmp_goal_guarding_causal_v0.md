# RMP-GG-01: Causal evaluation-belief pilot in a tied looped transformer

**Registered 2026-11-11, before the experiment code or trained results were committed.**
This pilot transfers the frozen finite Bayesian decision model from
[EvalMath RHT-20–25](https://github.com/MoralityLabAI/EvalMath/pull/29)
into the actual native \`ExplicitDecoderBlock\` of LoopedTransformers. It is an intentionally supervised **positive control** for recurrent-state measurement and interventions. It cannot show spontaneously acquired goal guarding, gradient awareness or deceptive intent.

## Question

Does a small shared-weight recurrent decoder **learn a causally useful internal coordinate**
for evaluation-versus-deployment odds from noisy observations, such that an
intervention on a frozen probe-defined direction changes its final choice
toward an independently calculated counterfactual?

The task label is computed outside the model from the exact rational rule
\[
 C\iff pR\prod_t P(o_t\mid E)>(1-p)G\prod_t P(o_t\mid D),
\]
with ties assigned EXPLOIT. Each observation is a binary channel with
\(P(1\mid E)=3/4,\ P(1\mid D)=1/4\); the hidden type is sampled once per sequence.
The model receives the observation tokens and \(\log(p/(1-p)),\log(R/G)\),
but never receives the oracle posterior or terminal label except through
supervised training. The **independent gold target** is the exact-rational
Bayes-optimal protect/exploit action for this controlled task. It makes no
claim about general human values.

## Frozen training contract

- Two seeds: **11, 17**; CPU only, one torch thread, no external checkpoints.
- Native shared \`ExplicitDecoderBlock\` applied once per observation;
  two positions per visit (evidence token and recurrent memory token), width
  32, 4 heads, pre-norm, alpha 1.0 and beta 0.5. One physical block is reused
  at every visit. New evidence arrives only on its designated visit.
- Four observations per training example; prior selected uniformly from
  \(\{1/4,1/2,3/4\}\), reward/gain ratio uniformly from \(\{1/2,1,2\}\).
- Deterministic training stream with independent Python RNG seeded from
  model seed plus 100; cache 16,384 examples, sample with independent torch
  generator seeded from model seed plus 1000.
- **400 optimizer updates**, batch 128, AdamW LR 0.003, weight decay 0.001,
  gradient clip 1.0. Terminal action cross entropy plus 0.2 times mean
  intermediate prefix-action cross entropy. This auxiliary supervision
  encourages an iterative belief calculation and must be reported.
- Random-init baseline and a label-shuffled control using an independent
  fixed permutation of training targets, preserving label marginals.
- Save initialization and final weights as local/CI artifacts with hashes.
  No checkpoint selection using validation outcomes.
- Development/calibration and evaluation are independent seeded streams.
  The four-cue dataset has only 144 distinct observations/context pairs,
  so it tests generalization to new samples from the same finite support.
  Six-cue evaluation is the **depth-extrapolation** test.

## Readouts

- Validate exact Bayesian labels with independent rational enumeration.
- Report final action accuracy on 4-cue and 6-cue panels of 4,096 examples,
  both for trained and random-initialized models; save calibration controls.
- For a frozen cutoff \(k=2\), take the hidden recurrent state after
  processing the first two cues. On 1,024 calibration examples, ridge-fit
  one linear coordinate to the exact **log posterior odds**
  \(\log[p/(1-p)]+(2n_1-k)\log 3\).
  Ridge penalty \(10^{-3}I\), with centering; calibration labels are never
  used as action labels during this fitting. Evaluate R² on a distinct
  1,024-example panel and on six-cue samples at the same cutoff.
- Construct the source/receiver intervention manifest *from the rational
  oracle*, before scoring neural outcomes. Within each pair, prior and ratio
  match, donor prefix replaces receiver prefix of length 2, while the
  receiver suffix stays fixed. Retain pairs where the hybrid oracle action
  differs from both receiver and donor **full-sequence** oracle actions.
  Cap at 256 pairs with fixed RNG seed 21111.
- Eligibility additionally requires that the *frozen trained model* predicts
  both receiver and explicitly re-evaluated hybrid correctly. Preserve
  the attempted count and eligible denominator; this condition is reported
  and the analysis is conditional on it.
- Interventions at visit 2: receiver-state sham; full donor-prefix state
  replacement as a **structural equivalence check**; and rank-one
  replacement of the donor-minus-receiver displacement along the
  calibration-fitted normalized linear-probe direction. The native tied
  suffix then runs on **receiver** future cues.
- Controls: eight seeded random directions with the same per-example
  perturbation norm as the rank-one intervention; donor-full-sequence
  copy; and the sham. Report downstream exact hybrid-action hit rate,
  signed hybrid-target logit-margin shift and rank-one fraction of the
  full counterfactual margin shift. Full-state replacement equals direct
  hybrid evaluation by construction of the shared recurrent transition;
  it is a *sanity check*, never considered evidence for a discovered feature.

## Decision gates and limits

A pilot is *instrumented* if the exact oracle, state replacement identity,
source/receiver suffix guard and independent test suite pass. Its learned
behavior is *eligible for causal interpretation* only if both seeds achieve
at least 0.90 held-out 6-cue action accuracy, calibration-to-validation
probe R² at least 0.50, and at least 32 oracle-discriminating eligible
counterfactuals per seed.

A rank-one direction is *causally supported* only if its hybrid-action hit
rate exceeds sham by 0.10 and exceeds **every** random-direction control
by 0.05, on each seed's frozen eligible panel. Negative or partial results
stay recorded, including direction probe failure and collapsed denominators.

Even if all gates pass, this is a supervised test of Bayesian belief tracking.
Any subsequent claim about spontaneously emerging training awareness needs
a separate held-out RL experiment with randomized *actual* optimizer
updates, no explicit posterior supervision, independently audited reward,
training/evaluation disclosure crossed with matched cue tokens, and an
unexposed deployment-like environment. No such claim is authorized by GG-01.
