# AF-06: staged causal interchange for compact proof state

AF-05's direct causal-interchange training failed to establish development eligibility for the 58-coordinate native models. AF-06 starts with a **shared, saved status-supervised checkpoint**, then compares three equal-budget continuations to test whether causal gradients can preserve correctness on unseen depth-4 and depth-5 Horn graphs.

The original source, equations and negative results for AF-01 through AF-05 are unchanged. AF-06 is stacked on AF-05 PR #45. Its [PROTOCOL.json](PROTOCOL.json) freezes the data, interventions, training schedule, and scientific gates before scored training. It uses the AF-05 graph/target oracle and the hash-pinned original `TypedLoop` implementation.

## Training

Each of the three seeds (17, 29, 43) and two per-token code widths (1 and 2 floats; 29 and 58 total) trains a 480-update status-supervised warm-up. Each continuation copies **exactly the same** source tensors, trains for 160 additional updates, and receives identical training examples/donor choices. Depth-2 and depth-3 examples are balanced at every update. The model performs 4 or 6 upstream recurrent steps according to the actual depth, followed by 3 downstream steps. All three continuation arms run one clean forward and exactly one recipient/donor prefix plus patched suffix for every example:

- `status`: irrelevant-Q patch throughout continuation.
- `ramp`: number of addressed-patch examples increases monotonically from 0 to 32 per minibatch, using exact Horn counterfactual labels.
- `abrupt`: addressed-patch supervision from the first continuation update.

The shared status loss is clean goal CE + Q/pre/post CE + commuting readout MSE + patched goal/Q CE. All three arms receive the same loss weights and recurrence-call count; actual wall time is recorded. This is a forward-call matched study, with backward computation potentially differing because the patch selection changes the gradient path.

## Scientific gate

Source warm-up and both compared continuation models must first qualify at **80% goal and joint Q accuracy**, separately on train and dev. The primary width-2 comparison requires ramp minus status informative goal-flip patch accuracy of at least 0.05 in **both** depth-4 and depth-5 heldouts, while preserving original task and untouched-Q accuracy and the irrelevant-Q negative control. Ineligible seeds stay in the record and contribute an inconclusive verdict. The abrupt-causal arm provides an additional measured schedule control.

The held-out `held_composition` and `held_layout` families remain diagnostics. Every held-out record is independently checked against the finite Horn-closure oracle and actual donated code, with whole-code/direct-donor replay. Lean checks the 4,176 generated closed Boolean oracle theorems, rather than certifying student-generated proofs.

## Local reproduction

From repository root with Python 3.12, Torch 2.5.1+cpu and NumPy:

```bash
python -m unittest discover -s experiments/activation_funnel_af06/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af06/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af06.experiment manifest --out /tmp/af06-corpus.json
cmp /tmp/af06-corpus.json experiments/activation_funnel_af06/corpus_manifest.json
python -m experiments.activation_funnel_af06.experiment train --seed 17 --width 2 --out /tmp/af06-train
python -m experiments.activation_funnel_af06.experiment evaluate --seed 17 --width 2 --training /tmp/af06-train --out /tmp/af06-eval
python -m experiments.activation_funnel_af06.verify --train /tmp/af06-train --evaluation /tmp/af06-eval --seed 17 --width 2 --out /tmp/af06-eval/independent_audit.json
```

Each output directory must be fresh. The training subprocess never accesses heldout data. The separately registered GitHub Actions workflow builds the full 3×2 matrix, retains source/checkpoint hashes, conducts original-label and intervention replay and publishes aggregate verdicts independently of engineering CI status.
