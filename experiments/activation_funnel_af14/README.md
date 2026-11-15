# AF-14: bounded progress and stable closure in a small TypedLoop

AF-12's synchronous-wave objective damaged deep Q transport on seed 17. AF-13
allowed the model to choose its own arrival times, but still lost exact Q
accuracy on depth-5 graphs. AF-14 tests an intermediate constraint: each
reachable proof atom must be correctly classified **by a certified upper
bound on its proof depth** and remain correct on later recurrent calls.

The finite Horn least-fixed-point oracle supplies the minimum proof depth
`d(v)` of each reachable atom from the initial facts. Every training example
contains only the original initial facts and graph incidence. A fact (depth 0)
is expected by the first neural call, and another reachable atom of proof depth
`d` must be true from step `2*d` onward. It may become true at an earlier step.
No exact-wave target is imposed. Unreachable atoms are penalized as before.

The registered arms are `relay_final` (terminal atom truth), `async_sound`
(AF-13's unclocked factual soundness, monotonicity, and premise support), and
`bounded_progress` (the identical unclocked loss plus 0.35 times mean
`softplus(0.25 - true_positive_logit_margin)` across each atom's bounded
postdeadline suffix). All arms retain the same original AF-13 clean and donor
patch losses, parent qualified AF-06 seed 17/29 checkpoints, 160 fixed update
steps, example roster, optimizer settings and call counts. The bounded arm has
extra backward arithmetic, so this is a **forward-call-matched** comparison,
not a wall-time/FLOP-matched study.

The frozen pre-training sources and AF-01 through AF-13 conclusions remain
unchanged. New physical layout IDs 26 through 31 are disjoint from AF-13's
20–25 and earlier layouts. Depth 4/5 and their longer recurrence horizons are
held out until the fixed checkpoints are written. The generated closed Lean
Horn theorems certify the oracle rather than the student model.

The primary scientific gate compares `bounded_progress` to `async_sound` on
heldout `fresh_depth4` and `fresh_depth5`, conditioned on both arms passing
90% original train/dev goal and exact joint-Q accuracy. For each eligible
seed and depth it requires at least +0.05 joint-Q and final relay-atom
improvements, at least 85% persistent bounded-deadline satisfaction, no more
than 3-point loss of clean/extended-goal and extended-Q accuracy or informative goal-flip
accuracy, irrelevant-patch goal retention ≥90%, and no added false-goal
attempts. An eligible failure yields FAIL; missing eligibility yields
INCONCLUSIVE. Every seed and failure is preserved.

## Reproduction

From the repository root, with Python 3.12, Torch 2.5.1+cpu, NumPy and SciPy
matching the pinned workflow:

```sh
python -m unittest discover -s experiments/activation_funnel_af14/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af14/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af14.experiment train --seed 17 --source-zip path/to/af06-train-s17-w2.zip --out artifacts/af14/s17/training
python -m experiments.activation_funnel_af14.experiment evaluate --seed 17 --train artifacts/af14/s17/training --out artifacts/af14/s17/evaluation
python -m experiments.activation_funnel_af14.verify --seed 17 --training artifacts/af14/s17/training --evaluation artifacts/af14/s17/evaluation --out artifacts/af14/s17/evaluation/independent_audit.json
```

The independent verifier computes minimal proof depths with a separate
synchronous frontier evaluator. It recomputes the deadline and stable-after-
deadline scores from every retained raw stage logit, then source-binds sampled
checkpoint forward replays and donor intervention logits. Positive claims
require the preregistered complete two-seed aggregate and hosted checks.
