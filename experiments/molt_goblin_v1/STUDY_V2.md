# MOLT-GOBLIN-02: budget panels and projected-prefix cache

The existing pilot is preserved. study.py adds a development driver with final-only
and all-loop supervision and a single-pass control that reinvests forward block calls
into extra optimizer updates. The recurrent arms share initialization, processed data,
optimizer and update counts; the reinvested control shares forward block calls only.
Actual FLOPs and process-tree peak RSS remain missing. Extra readout losses also cost
work and are counted separately as supervised token-head visits.

append_only computes each necessary prefix K/V projection once. Its test compares
forward values and undetached parameter gradients against the sequential reference.
For length 16 the projected prefix token count falls from 120 to 15 per sequence,
for the K and V projections individually. This is an operation count, not measured
speedup. Tensor concatenation and dense attention still take work. This is a single
autograd-graph reference; it does not implement cross-update reuse or native MoLT.

```sh
python -m pytest -q experiments/molt_goblin_v1/test_study.py
python -m experiments.molt_goblin_v1.study --steps 32 --loops 4 --seeds 31 37 --threads 2 --out /tmp/molt-study.json
```

Use an external resource cap. CUDA is optional; no downloads. The run preserves
per-sequence outcomes at fixed development input seeds, position-wise accuracy, actual
loop count, head visits, wall time and process CPU time. Model-seed and shared task
dependence must be respected in downstream analysis. This module does not establish
equal measured time or FLOPs. It does not use a sealed evaluation set or authorize
promotion. A substantive study requires competence calibration, frozen comparisons,
independent roots and EvalMath preflight before confirmation.
