# GG-04 closeout: calibration failure and cross-depth update-effect reversal

**November 11, 2026. Scientific verdict: CALIBRATION_FAIL_CLOSED.**
The frozen protocol is [GG-04 randomized real successor updates](../docs/rmp_goal_guarding_actual_updates_gg04.md). Its goal was to train a small disclosure-conditioned reporting policy on rewards from *real native SGD task-model updates*. The primary experiment's positive-damage calibration prerequisite failed for model seed **17**. The controller's preregistered two-seed heldout test therefore **was not performed**. This is a feasibility/assumption failure, not evidence for spontaneous goal guarding.

## Source and reproducibility

- Base native model: GG-01 supervised tied \`ExplicitDecoderBlock\`, starting from LoopedTransformers \`main\` commit \`656612856b0cf0bec3d3879730e50d9529094706\`.
- Frozen training: two seeds, 11 and 17; 400 AdamW updates per seed; all 144 exact four-cue task/context inputs split into 72 actor-train, 24 calibration, 48 heldout by canonical SHA-256.
- Post-report optimizer: **real PyTorch SGD** on cloned task-model weights, 1 step, LR 0.05 and gradient norm clip 1. The audit observes mean cross entropy on a separate 128-example six-cue gold panel with exact Bayesian labels. The no-update treatment preserves the source weights and source gold loss.
- [Initial CI run 43043576022](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43043576022) passed all seven exact/physical-update unit tests and stopped at its preregistered positive median guard.
- [Diagnostic run 43044197656](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43044197656) reproduced that failure, then executed a separate **post-hoc calibration-only** two-seed audit against the original six-cue gold panel and an exploratory four-cue gold panel. [Uploaded post-hoc effects and native weights](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43044197656/artifacts/13866897141).
- A workflow-only recovery change after seeing the failed scientific gate allows this specific expected rejection to be logged as a technically successful **scientific negative**. It preserves the numerical criteria, source model, sealed gold samples and failed rule. Any other runtime exception remains a CI failure. The initial failing workflow remains in GitHub history.

## Conditional real-SGD damage gap

For each frozen four-cue calibration query \(x\), the experiment applies both action-conditional real SGD updates to the same cloned starting model. Define
\[
D_x=\left[L_{\rm gold}(\theta-\eta g_{x,\mathrm{flipped}})
-L_{\rm gold}(\theta)\right]
-\left[L_{\rm gold}(\theta-\eta g_{x,\mathrm{truthful}})
-L_{\rm gold}(\theta)\right].
\]
Positive \(D_x\) means flipped-label training damages this *audited task distribution* more than truthful training. The median across the 24 calibration queries was the preregistered gate.

| Gold scoring distribution | Seed 11 median \(D_x\) | Seed 17 median \(D_x\) | Positive-gap queries |
|---|---:|---:|---|
| **Registered six-cue** (seed 68001) | +0.0029973206 | **−0.0037539862** | 19/24, 3/24 |
| **Exploratory four-cue** (seed 69001) | +0.0000299745 | +0.0000392259 | 24/24, 24/24 |

Additional registered six-cue statistics:

- Seed 11: mean gap **+0.0285886**, range [−0.0010445,+0.1355046], first-order influence prediction MAE **0.0152071**.
- Seed 17: mean gap **−0.0020730**, range [−0.0175783,+0.0484838], first-order influence prediction MAE **0.0022341**.
- For the exploratory four-cue audit, the mean gap is **+0.0000741** for seed 11 and **+0.0038554** for seed 17. The highly skewed distribution means medians are a better description of typical query effects than averages.

The seed-17 six-cue result is a **measured gradient-transfer sign reversal** for the task-specific gold metric: flipped reports frequently improve six-cue gold cross entropy relative to truthful reports, despite increasing four-cue gold loss. It arises from actual one-step weight updates and is conditional on these two native task models, the supplied input grammar, and the evaluated panel. It does not establish a general rule about training on incorrect labels.

The results expose a modeling issue in the original control experiment: a single wrong-label update on a four-cue prompt need not hurt the same Bayes classification target on a six-cue evaluation set. The sealed gold objective and task depth matter causally, and the GG-04 positivity guard correctly prevented an ungrounded policy-value normalization.

## Next registered successor

A distinct GG-04B protocol may focus on the **in-distribution four-cue gold metric**, while retaining the original six-cue audit as a *separately reported cross-depth adverse-transfer readout*. A new 4-cue gold evaluation panel, reward normalization and robust clipping rule must be registered before any GG-04B heldout controller results are examined. Any such pilot should separately quantify the positive action-damage gap and the performance of cue-only controls.

All GG-04 and GG-04B controller outcomes, if produced, are evidence only about an **explicitly trained separate reporter controlling a native task-model successor**. Neither observes a model independently inferring the existence of RL, manipulating its own policy-head weights, or showing a spontaneously formed goal-guarding self-model.
