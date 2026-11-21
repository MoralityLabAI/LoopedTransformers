# Inkling 2 Mini literature bridge to recurrence monitorability and architecture search

2026-11-21. Status: **development design; source-only literature review**.
Companion to [Recurrence Monitorability Program v0](recurrence_monitorability_program_v0.md)
and [Loop Schedule Architecture Discovery](loop_schedule_architecture_discovery_v1.md).
This file does not revise any completed or sealed LSAD/RMP/LS-SAE/LOOPSCOPE verdict.

## Citations followed

1. Thinking Machines, [Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/),
   November 20, 2026. 24 stored layers, middle 12 looped three times, 48 effective applications;
   27/28 evaluations favor middle over last-layer recurrence.
2. Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819):
   position-wise adaptive halting and iterative refinement, subject to its stated assumptions.
3. Chen et al. (Kimi Team), [Attention Residuals](https://arxiv.org/abs/2603.15031):
   content-dependent depth-wise selection and block-level aggregation; state access is richer
   than a simple running residual sum.
4. Yang et al., [Looped Transformers Are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424):
   repeated decoder can imitate an iterative regression algorithm with prompt reinjection;
   extrapolation beyond trained loops is task- and regime-dependent.
5. Zhu et al., [Scaling Latent Reasoning via Looped Language Models (Ouro)](https://arxiv.org/abs/2510.25741):
   entropy-regularized exit distribution and additional exit-controller training.
6. Bae et al., [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524):
   token-specific recurrence budgets, selective KV caching, causal routing trade-offs.
7. Goodfire, [Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/):
   block-boundary writing, cross-loop reader decomposition, outlier coordinate 2027 on
   the FIXED 3-pass release, and tuned/coda lens results.
8. Chen Yueh-Han et al., [Reasoning Models Struggle to Control their Chains of Thought](https://arxiv.org/abs/2603.05706):
   monitorability depends on what the monitor can actually infer, not just trace readability.
9. Korbak et al., [Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473):
   monitorability can change under architecture/training decisions.
10. Thinking Machines, [variable-loop research model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/):
    pass counts 1–5 midtrained in ratio 1:2:6:2:1, followed by post-training at 3.
    Passes 7 and 10 extrapolate outside the trained range.

## Structural lesson for the existing gym

Inkling routes information over **block sums**. Its core block receives a
state-dependent cross-loop message from the previous pass; near the end
of a pass, high-magnitude coordinates can influence the *next pass's*
reader scores. In Goodfire's 400-document FineWeb sample, fixed-pass
coordinate 2027 accounts for approximately 46% of the squared norm
of a normalized activation vector. The same coordinate contributes
6–19% of selected cross-loop reader score gaps. These observations
are correlations in a specific released checkpoint, not a certified
circuit or a stable coordinate index across models.

For two source blocks, a useful pre-intervention readout is

    beta_3 = sigmoid((u · (RMSNorm(v_3)-RMSNorm(v_2))) / sqrt(d)).

Record the score gap, every coordinate contribution, the normalized
vectors, and the reader weights BEFORE the perturbation; compare with
full recomputation after intervention. Intervening on a single coordinate
changes the normalizer, so its causal effect is generally different
from deleting its additive contribution from a fixed normalized vector.

### Experiment A: a trained positive control first

The existing RMP document records LSAD v1's chance-level results.
Do not use those models to infer a negative depth effect. Begin with
a checkpoint that demonstrates an independently verified algorithm:
native Inkling research weights if a 32GB+ host is available, or a
compact trained pointer-chase checkpoint with frozen evidence of
multi-hop success. Report exact checkpoint identity and training depth
distribution. The 5080 does not have sufficient memory for the native
20GB variable-loop checkpoint without a supported external offload path.

Cross pass counts [1,2,3,5,7,10] on the SAME task roots, with separate:
- direct final-answer/no-CoT evaluation on single-hop versus 2/4/8/16-hop chains;
- visible reasoning quality at matched generated-token budgets;
- output-correctness and regression after an earlier correct answer;
- observed prefill+decode time, memory and energy where available;
- fixed 3-pass and constant-cost controls, including a task-agnostic exit;
- inference-only perturbations of positional depth tags and input reinjection.

Report prefix task-state correctness alongside final success; an optimizer
converging in state norm is not equivalent to serial symbolic progress.
Treat architectural passes, RLT student visits, autoregressive tokens and
environment verifier calls as distinct quantities.

### Experiment B: middle versus terminal recurrence

Train the same-size core blocks on identical source task families with
identical training-token exposure. Compare (i) middle-12 looping,
(ii) last-12 looping, (iii) untied fixed-depth, and (iv) tied original
architecture as controls. Match *applied* layer count at each test depth
and separately report unique-parameter counts, state-storage costs and
time-to-quality. This is a training intervention. Comparing the
pretrained native Inkling middle-loop to an untrained final-loop splice
would confound architecture with weight specialization.

### Experiment C: block routing and coda lens

Collect the four cross-loop readers at block starts (query/key/value/MoE),
block outputs after layers 12 and 18, and coda states. Train a shared
tuned lens, a coda lens, and fixed decoder as competing methods on
monitor-TRAIN; calibrate lens choices and thresholds on a separate
monitor-CAL. Evaluate first-hop through last-hop decoding accuracy and
causal repair on untouched roots. Lens CE and token readability alone
do not establish mechanism.

Search for outlier routing features independently in the variable-loop
checkpoint; keep fixed-three-pass coordinate 2027 as a hypothesis only.
Use dose-matched random-coordinate and subspace controls. Record clean
task accuracy, corruption, repair, state displacement and monitor
false-positive rates.

### Experiment D: exit controllers

Compare fixed depth with (a) task-level learned gate (Ouro),
(b) per-position ACT from Universal Transformers, and
(c) causal token-level routing (Mixture-of-Recursions).
Use a frozen cost budget and separate prompt-reinjection control as
suggested by Yang et al. Selection and learning take place entirely
on development roots. Confirm once on untouched task families.
The exit gate's ground-truth label is **first verifier-valid depth**,
not the number of tokens in the generated explanation; an oracle
look-ahead exit is an analysis bound.

## Data contract, interpretation and controls

- Independent unit: source task root / checkpoint population as registered.
  Tokens, loop passes, directions and lens features are within-root repeats.
- Primary positive-control metric: final independently verified task success
  at fixed measured effective inference cost. Set a meaningful effect above
  development sham variation with EvalMath preflight before running a GPU sweep.
- Preserve train/calibration/heldout/OOD custody, registration SHA and all
  failed model candidates. Holm-correct prespecified comparison family.
- Monitor access arms: tool outputs only; tool + written explanation;
  tool + sparse activations; tool + full monitored boundary data.
  Calibrate each monitor at equal false-positive rate on development.
- Compare a visible CoT compression metric on *all* attempted tasks as
  well as the intersection correct at every depth. Report that
  intersection's selection bias explicitly.
- The linked Inkling model card notes RL at depth 3 only. Differences at
  other depths are joint effects of architecture and training exposure.

Next required work: source-only adapter interface and a positive-control
checkpoint audit before selecting candidate interventions. This registration
is proposed, not frozen and not evidence of a mechanism.
