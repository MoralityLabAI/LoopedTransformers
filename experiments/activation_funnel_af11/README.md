# AF-11: Boolean-influence training through a causal cut

This experiment follows AF-10's frozen-model structural-causal-cut diagnostic. It trains the **original hash-verified TypedLoop cell**, beginning from the original, qualified AF-06 58-coordinate model. Both subgoal Q codes are latched during the downstream recurrence. The new objective teaches the goal output to respond to donor-swapped Q statuses according to the exact Boolean influence of one subgoal on the other.

## Exact equation

Let `q` be recipient truth, `d` be donor truth with a single selected subgoal `j` changed. For a final AND gate, `I_j = q_(1-j)`, while for OR `I_j = 1 - q_(1-j)`. Then

\[
G(q_j\leftarrow d_j)-G(q)=(d_j-q_j)I_j.
\]

The model sees the public graph and initial facts, never the target Q truth, influence value, or counterfactual label. These come from the unmodified AF-05 least-fixed-point Horn oracle in the training loss and independent scorer. Lean verifies the four finite Boolean influence identities in `BooleanInfluence.lean`. AF-10 separately verifies protected-code invariance under an arbitrary recurrent map.

## Comparators

The experiment uses seeds 17/29, both exactly source-bound to originally qualified AF-06 warm-up model exports. A **shared** 160-update causal-cut adaptation trains on original AF-05 `train`, then four continuation arms each receive 160 further AdamW updates with matched cases/donors/native cell calls:

- `status_cut`: clean goal, intermediate subgoal, irrelevant-patch controls, and addressed-Q status supervision.
- `counterfactual_cut`: adds exact addressed goal/subgoal counterfactual supervision; the main strong baseline.
- `influence_cut`: same as the strong baseline with signed Boolean finite-difference goal supervision.
- `influence_ordinary`: same influence objective, with AF-10's cut disabled.

The influence loss adds backward/elementwise computation; call parity is preserved without claiming full FLOP parity. Both model parameters and carried-code decoder are trainable.

## New prospective evaluation

Frozen input-feature hashes bind `fresh_depth3` (128 cases), `fresh_comp` (16), `fresh_depth4` (256) and `fresh_depth5` (256). Each uses new layout indices `8..13`; all original AF-05 training/development and previous heldout layout hashes are disjoint from them. Graph logical AND/OR templates remain identical. Each 656-case heldout roster has two exact opposite-subgoal donor counterfactuals, giving 1,312 rows per arm and seed. The primary gate compares `influence_cut` to `counterfactual_cut` in both new depths after separate train/dev qualification. Upstream Q correctness, conditional flip performance, irrelevant-patch retention and false-positive goals are reported separately.

All model checkpoints and named NumPy array exports are saved at fixed final steps. `verify.py` independently recomputes every Boolean/Horn target, signed influence and raw-logit metric, and replays sampled native forwards. The original AF-06 data and parent AF-01..10 science are unchanged.

## Commands

From the repository root after installing Python 3.12, Torch CPU 2.5.1, NumPy and SciPy:

```bash
python -m unittest discover -s experiments/activation_funnel_af11/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af11/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af11.experiment train --source-zip /path/to/AF06_original_s17.zip --seed 17 --out artifacts/af11/s17/training
python -m experiments.activation_funnel_af11.experiment evaluate --train artifacts/af11/s17/training --seed 17 --out artifacts/af11/s17/evaluation
python -m experiments.activation_funnel_af11.verify --seed 17 --training artifacts/af11/s17/training --evaluation artifacts/af11/s17/evaluation --out artifacts/af11/s17/evaluation/independent_audit.json
python -m experiments.activation_funnel_af11.aggregate --root artifacts/af11 --out artifacts/af11/aggregate/results.json
```

The GitHub CI downloads the exact original AF-06 source artifact IDs and checks their pinned SHA-256. Training jobs never read heldout data for optimizer selection; their outputs are exported before independent evaluation. Scientific failure is retained even when the engineering audit passes. AF-11 is an experiment on small structured Horn dependency graphs, not a demonstration of independent Lean theorem discovery.
