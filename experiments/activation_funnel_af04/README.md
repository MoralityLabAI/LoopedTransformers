# AF-04: learning latent Horn proof state with causal interchange

AF-04 is a registered extension of the AF-03 intervention-training experiment, using the **unchanged original `TypedLoop` graph-attention cell** imported through AF-02's source-hash verification. The source weights are trained afresh for each AF-04 arm and are initially identical within each seed/width pairing.

The model receives initial fact truth values and a directed Horn rule graph. Intermediate subgoal truth values are **derived**, not supplied. An upstream phase applies five recurrent updates to infer two independent subgoals. A downstream phase drops the upstream rule edges, reveals only the gate from those subgoals to the final goal, and applies three more updates. The recurrent state must carry the inferred subgoals into this second stage.

A donor and recipient use the same graph skeleton, goal gate, and untouched-branch facts. The donor changes the truth of a selected derived subgoal. A code patch replaces that subgoal's *carried latent activation* after the upstream phase; the oracle recalculates the goal with the donor subgoal and the recipient's other subgoal. The negative control patches the untouched subgoal. Whole-code replacement must exactly reproduce the donor's final logits, which is independently checked from the stored prediction table.

The architecture uses 11 atom tokens and 10 rule tokens with 16 hidden dimensions each, encoded to one or two float coordinates per token at every native cell call. Its full working state is 336 floats, and its 21- or 42-float carried state is an actual recurrent bottleneck. Training always updates the native cell and its input/output projections.

## Frozen cases

| Split | Cases | Changed-subgoal donor pairs | Content |
|---|---:|---:|---|
| train | 336 | 672 | Seven graph compositions, 3 physical permutations, depth 2 |
| dev | 112 | 224 | Same compositions, disjoint permutation 3 |
| held_composition | 16 | 32 | Both subgoals OR, downstream goal AND; permutation 4 |
| held_layout | 112 | 224 | Seen compositions under permutation 5 |
| held_depth | 128 | 256 | All compositions, dependency depth 3; permutation 6 |

The complete canonical input manifest is frozen in `corpus_manifest.json`. Goal and subgoal labels come from an independent finite least-fixed-point Horn oracle. `gym.lean_certificate(split)` generates a Boolean-Horn theorem for each case and each queried node. Lean checks its closed formula with `by decide`. This certifies the synthetic oracle, not the model's theorem-proving ability.

## Registered experiment

Three seeds (`17,29,43`) × two per-token code widths (`1,2`) × three objectives (`task,status,causal`) = 18 models. A seed-width job fits all three objectives with the same initial weights and sampled inputs. Every arm performs equal native-cell forward calls per step. The primary comparison uses width 2; width 1 is a stress test.

`status` receives supervised intermediate truth and an equal-compute negative-control code patch. `causal` receives those same losses plus supervision at the addressed code interchange using the independent counterfactual oracle. See [PROTOCOL.json](PROTOCOL.json) for exact loss weights, qualification criteria, and statistical gates.

The principal gate requires both status and causal models to meet 80% goal and exact-subgoal accuracy on training and development, followed by at least five percentage points of improvement in informative goal-flip patch accuracy in both held-out composition and depth families, with explicit locality and clean-goal retention requirements. The report must preserve negative and ineligible runs.

## Commands

From the repository root with Python 3.12 and Torch 2.5.1+cpu:

```bash
python -m unittest discover -s experiments/activation_funnel_af04/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af04/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af04.experiment manifest --out /tmp/af04-manifest.json
cmp /tmp/af04-manifest.json experiments/activation_funnel_af04/corpus_manifest.json
python -m experiments.activation_funnel_af04.experiment lean --split held_depth --out /tmp/af04-depth.lean
lean /tmp/af04-depth.lean  # Lean 4.19.0
python -m experiments.activation_funnel_af04.experiment train --seed 17 --width 2 --out /tmp/af04-train
python -m experiments.activation_funnel_af04.experiment evaluate --seed 17 --width 2 --training /tmp/af04-train --out /tmp/af04-eval
python -m experiments.activation_funnel_af04.verify --seed 17 --width 2 --train /tmp/af04-train --evaluation /tmp/af04-eval --out /tmp/af04-eval/independent_audit.json
```

Training never consumes the held-out examples. Each job retains three Torch model checkpoints and three named-array NumPy exports, all canonical train/dev logits, complete held-out logits, donor-pair code values and exact counterfactual targets, plus the independent audit. Scientific verdicts are evaluated separately from evidence integrity and CI success.
