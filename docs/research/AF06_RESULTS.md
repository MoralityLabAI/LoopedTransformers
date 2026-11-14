# AF-06 final scored research result | November 14, 2026

**Registered primary width-2 causal-advantage hypothesis: FAIL.** The frozen scientific source is [36b83eb](https://github.com/MoralityLabAI/LoopedTransformers/commit/36b83eb83bc116dd4e938213edeba3057ffa9016), built on AF-05 [PR #45](https://github.com/MoralityLabAI/LoopedTransformers/pull/45). [AF-06 PR #46](https://github.com/MoralityLabAI/LoopedTransformers/pull/46) remains draft and unmerged. Preregistered protocol SHA-256: `37d2404bb1bcf791930aa099ec2c82dc7232d75c682abfaf4b87e5756ef172f4`.

**Engineering:** The original [AF-06 GitHub Actions run #43599882702](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43599882702) passed preflight, six training jobs, and six independent evaluation/replay jobs. Its final aggregation job failed before reading evidence because that runner omitted NumPy. The separately frozen-source [aggregation repair #43600726497](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43600726497) passed using exactly those six evaluation artifacts. This failure and repair are part of the engineering record.

## What AF-06 changed

Unlike AF-05, AF-06 first trains **one shared status-supervised 480-update checkpoint** per seed/width pair. Its `status`, `ramp`, and `abrupt` continuations copy that checkpoint and each consume 160 matched additional updates. Both actual training depths, 2 and 3, are sampled at every update, with 4 or 6 upstream native-cell calls respectively. The causal curriculum increases addressed interventions from zero to the entire 32-example minibatch, while the abrupt arm uses them from its first update. Each patched outcome is supervised by the exact Horn closure oracle, with irreversible downstream evidence never exposed to the recurrent input.

All arms preserve the same recurrent-cell and carried-state architecture: 15 atom and 14 rule tokens, each with 16 working coordinates, compressed to one or two carried coordinates per token. The original native TypedLoop remains source-hash pinned and trainable. The experiment is a synthetic proof-state study whose generated Lean theorems certify the Boolean oracle; the neural model is evaluated separately.

## Eligibility and scientific interpretation

Two 58-coordinate seeds, **17 and 29**, each reached perfect goal and joint-Q development accuracy at the shared warm-up checkpoint and retained it through every continuation. Seed 43 did not qualify. The width-1 stress runs were also ineligible. Thus AF-06 produced two genuinely eligible causal-vs-status comparisons, improving upon the AF-05 engineering/scientific limitation in which all primary-width causal arms were ineligible.

On depth4, both eligible seeds had **0.000** addressed informative-goal patch gain. On depth5, seed17 gained **0.09375** while seed29 gained **0.000**. The ramp models failed the required 0.90 untouched-subgoal and 0.90 irrelevant-patch goal-retention thresholds, with depth4/5 clean goal accuracy also low. Accurate training-depth Q state therefore remained insufficient for depth-stable causal generalization. Locality failure exposes a mechanism issue beyond ordinary goal-classification accuracy.

For seed17 width2, ramp held-depth5 untouched Q accuracy is **0.500** and irrelevant-patch goal retention is **0.4375**; its held-depth4 counterparts are **0.6015625** and **0.546875**. For seed29 width2, depth4 untouched Q accuracy is **0.500**, with irrelevant-patch retention **0.625**; at depth5 these are **0.750** and **0.46875**. These are registered negative controls, not optional post hoc interpretation.

## Full frozen seed matrix

| Seed | Width | Warm-up dev goal/Q exact | Status dev goal/Q exact | Ramp dev goal/Q exact | Depth4 status/ramp/abrupt | Depth5 status/ramp/abrupt | Gate |
|---:|---:|---|---|---|---|---|---|
| 17 | 1 | 0.509/0.277 | 0.509/0.277 | 0.491/0.277 | 0.500/0.500/0.500 | 0.500/0.500/0.500 | inconclusive |
| 17 | 2 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.406/0.406/0.328 | 0.500/0.594/0.359 | fail |
| 29 | 1 | 0.839/0.679 | 0.737/0.210 | 0.866/0.705 | 0.562/0.594/0.594 | 0.562/0.938/0.938 | inconclusive |
| 29 | 2 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.438/0.438/0.500 | 0.594/0.594/0.500 | fail |
| 43 | 1 | 0.491/0.277 | 0.491/0.205 | 0.759/0.455 | 0.500/0.562/0.500 | 0.500/0.562/0.500 | inconclusive |
| 43 | 2 | 0.759/0.562 | 0.830/0.679 | 0.491/0.277 | 0.594/0.500/0.500 | 0.594/0.500/0.500 | inconclusive |

## Evidence and provenance

- **6 shared warm-up checkpoints plus 18 continuation models** frozen across seeds 17, 29, 43 and per-token code widths 1, 2.
- **8,928 clean held-out rows and 17,856 addressed-donor counterfactual rows**, with full raw logits, actual transported codes and whole-code/direct-donor replay.
- Independent scorer reconstructed all status and goal targets from the original finite least-Horn fixed point, verified model exports and sampled source-bound torch forward passes for each heldout family.
- **4,176 closed Boolean-Horn certificates** covered the 1,392 inputs across AF-05's unchanged corpus splits. Kernel checking supports the oracle, not novel Lean proof search from the student.
- The six held-out evaluation jobs passed. The aggregate repair succeeded on the **same frozen source** and its results JSON SHA-256 is `0dd34f1067ea2d5e3e2aa410b3bb3dd26eef1406298de2069a780a244501a254`.
- The scored verdict uses preregistered thresholds and preserves all negative/ineligible outcomes.

A future experiment should examine whether downstream Q-state locality can survive graph depth by training on explicit subgoal-to-goal dependency transport, while measuring change to the untouched Q at every recurrence step. Such a revision would require fresh preregistration.
