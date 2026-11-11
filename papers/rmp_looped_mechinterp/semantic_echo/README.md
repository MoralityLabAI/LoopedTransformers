# Semantic Echo

SEM-ECHO-01 studies whether a looped model preserves a specified symbolic transition across recurrent visits. The first implementation targets the fixed-depth permutation-pointer models used in LOOPSCOPE Q4–Q9.

## Documents

- [Mathematics and experimental strategies](theory.md) develops the connection between reversible semantic dynamics, information recovery, proof continuations, and recurrent computation. It includes the invariant-region theorem and a finite-trace phase-error bound.
- [Development protocol](protocol.md) fixes the initial readout, donor selection, outcome measures, and interpretation. The 24 existing Q4/Q5 checkpoints form a development set whose extrapolation outcomes are already known.

The candidate symbolic state in the first runner is the model's own predicted cursor. All token positions are retained in the recurrent state. Same-input states at different visits are eligible for comparison when the exact permutation oracle gives them the same cursor and both endpoint readouts agree with it. Future outcomes enter the score after that selection.

## Run the controls

From the repository root, with NumPy available:

```bash
python -m unittest discover -s tests -p 'test_semantic_echo.py' -v
python -m research_gym.scripts.semantic_echo_controls \
  --out experiments/recurrence_monitorability_v0/semantic_echo_controls
```

The committed [control report](../../../experiments/recurrence_monitorability_v0/semantic_echo_controls/report.md) records exact and analytic controls. These outputs are labeled `CONTROL_ONLY`.

## Validation at introduction

The [validation receipt](../../../experiments/recurrence_monitorability_v0/semantic_echo_controls/validation.md) records 35 passing tests, all seven constructed controls, and a native CPU PyTorch smoke run. Its machine-readable companion includes source hashes and the measured tolerances.

The untrained smoke exercises an important failure case. Eleven pairs pass the endpoint gate and agree throughout their four-step futures, while every pair shares an oracle error. The primary disagreement score therefore appears alongside oracle agreement, complete-future correctness, and coverage. Each independently sampled graph/start/prompt item receives equal weight in the primary mean.

## Check the neural adapter

The checkpoint adapter also requires PyTorch, already used by the LOOPSCOPE models:

```bash
python -m unittest discover -s tests -p 'test_loopscope_semantic_echo.py' -v
python -m research_gym.scripts.loopscope_semantic_echo \
  --smoke --device cpu --out /tmp/semantic_echo_smoke
```

The smoke constructs a small randomly initialized network and labels its output `UNTRAINED_SMOKE`. It checks native-forward parity and intervention execution. Scientific evaluation uses the frozen trained checkpoints.

## Evaluate the existing checkpoints

Run each checkpoint set separately so its receipt retains the original Q4/Q5 provenance:

```bash
python -m research_gym.scripts.loopscope_semantic_echo \
  --checkpoint-dir experiments/recurrence_monitorability_v0/loopscope_toy/ckpt \
  --out experiments/recurrence_monitorability_v0/semantic_echo_q4 \
  --items 64 --max-loops 32 --horizon 4 --max-pairs 16 \
  --seed 940117 --device cuda --dtype float32

python -m research_gym.scripts.loopscope_semantic_echo \
  --checkpoint-dir experiments/recurrence_monitorability_v0/loopscope_toy_q5/ckpt \
  --out experiments/recurrence_monitorability_v0/semantic_echo_q5 \
  --items 64 --max-loops 32 --horizon 4 --max-pairs 16 \
  --seed 940117 --device cuda --dtype float32
```

An explicit `--checkpoint PATH` can be repeated to select individual models. The `--max-pairs` cap applies to rerun interventions across a checkpoint; the continuation analysis considers every eligible pair. A full-state donor replay checks execution. Comparing the futures of equal-semantic states tests the proposed quotient. Partial swaps examine how cursor and edge positions contribute.

The adapter accepts the fixed-depth training contract, which has no requested-hop token. Checkpoints trained with a requested depth and a hold-after-completion rule require a different symbolic oracle. Missing checkpoint directories and empty selections produce an error.

## Interpretation and next measurements

A disagreement after equal-semantic endpoints witnesses failure of the selected own-head quotient on those observed states. A successful comparison supports that candidate over the tested continuation horizon. Alternative internal algorithms can require another symbolic state definition.

The control and smoke receipts validate the assay. The [trained Q4/Q5 run](../../../experiments/recurrence_monitorability_v0/semantic_echo_q4q5_summary/report.md) supplies the model measurements: all seven Q9 extrapolation failures show endpoint-gated continuation disagreement, two Q9 passes degrade past depth 16 and show it too, and on this set the primary score tracks deep own-head inaccuracy (Spearman 0.92). New training seeds are required for a prospective test of extrapolation prediction after the protocol is frozen.

The next extension fits a shared semantic decoder on separate graphs and tests causal phase shifts. Further experiments measure recovery under noise and bounded proof-continuation loss. Their mathematical assumptions and controls are specified in the accompanying documents.
