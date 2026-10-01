# Checkpoint: LSAD v1 Closeout Addendum 1 (Chance-Level Audit)

Date: `2026-10-01`

## Scope

This is a post-outcome, unregistered audit of the sealed A1 and A2 artifacts.
It reruns no training and changes no registered decision. LSAD v1 still closes
with zero qualifying architectures, which the registration declared a valid
null result. The addendum narrows what that null can be taken to show.

## Finding

No LSAD v1 cell learned its tasks. On the macro score, all 27 completed cells
fall below a predictor that emits one fixed token per family, and no cell's
training loss is below the entropy of the training targets. The A2
candidate-versus-control deltas are differences between near-constant
predictors.

## Evidence

The scored subset is the same 768 calibration examples in every cell: 256 per
family, depths 5 and 6. Training used depths 1 to 4 only.

| Family | Uniform chance | Best constant token | A2 range, nine cells |
| --- | ---: | ---: | ---: |
| Pointer chase | 0.1250 | 0.1602 | 0.117 to 0.164 |
| Modular recurrence | 0.0588 | 0.0898 | 0.027 to 0.090 |
| Rewrite normalization | 0.1667 | 0.2344 | 0.168 to 0.195 |
| Macro | 0.1168 | 0.1615 | 0.1211 to 0.1328 |

| Stage | Cells | Examples seen | Final training loss | Correct of 768 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 18 | 1,024 to 1,376 | 3.689 to 9.119 | 71 to 108 |
| A2 | 9 | 4,096 | 2.263 to 2.393 | 93 to 102 |

- A scorer at uniform chance expects 89.7 correct with standard deviation
  8.8 items. Every cell is within 2.2 standard deviations of that; every A2
  cell is within 1.4.
- Per family, 4 of the 81 family-cells exceed the best-constant count, by one
  to three items of 256. No cell does so on the macro score.
- The mean entropy of the training targets is 2.2325 nats. A2 final training
  losses exceed it by 0.031 to 0.160. Each is the loss of the last optimizer
  step, a 32-example batch, so it is a noisy estimate; none of the nine falls
  below the entropy.
- Every A2 prediction lies inside its family's answer range. Each cell emits at
  most five distinct tokens per family, and in 15 of the 27 A2 family-cells one
  token accounts for at least 90% of predictions.
- A2 prediction profiles follow the seed more than the schedule. At seed 419
  all three schedules emit a single token for pointer chase and a single token
  for modular recurrence.
- The A2 net-item deltas of `-5, -8, +1` and `-2, -3, -2` are smaller than one
  uniform-chance standard deviation. The A1 advantage of `+0.026042` is 20
  items between two cells whose training losses were 7.14 and 7.56.

## What stands

- The registered outcome and transition hashes.
- The arithmetic of `|mean delta| <= 0.005209`.
- The procedural lesson that a one-seed screen promotes item-scale noise.

## What is narrowed

1. The bound is not an effect bound on dispatch order. It bounds the gap
   between models that had not begun to compute the tasks. Whether schedule
   order matters for a model that solves them is untested.
2. Near-equality of candidates and controls does not support matching
   fidelity. Any two models in this regime would agree to within item noise,
   matched or not.
3. The closeout sentence describing "a second clean negative on a proposed
   loop-design degree of freedom" should read as not yet tested for LSAD v1.
4. The binding constraint was exposure, not scale. A2 saw 4,096 examples.
   Transfer to S1 or S2 at the same token-visit budget would be expected to
   reproduce this result.
5. The calibration split holds out depth by construction, so calibration
   accuracy alone could not have distinguished a model that learned depths 1
   to 4 from one that learned nothing. No train-depth evaluation was recorded.
   The training-loss and prediction-profile evidence above is what rules out
   learning.

## Recommendation for any successor

Add a learning gate that must pass before candidates and controls are compared:
training loss below target entropy by a registered margin, in-distribution
exact match above the best-constant baseline by a registered margin, and an
in-distribution evaluation split alongside the depth-held-out one. Include an
intentionally mismatched positive control, as the A2 checkpoint already noted.
`docs/recurrence_monitorability_program_v0.md` adopts this as its gate `G0`.

## Reproduction

```powershell
python -m research_gym.scripts.audit_lsad_v1_chance_level
```

The script verifies the train shard against the dataset manifest and each
prediction file against the SHA-256 recorded in its cell result, then writes
`experiments/loop_schedule_architecture_discovery_v1/chance_level_audit.json`.

## Provenance

- Audit JSON canonical SHA-256:
  `770b7ed7aec058a55bc0e4187ceab853bc19a6adc5a8bf13ef6f3be9b8d4564f`.
- Train shard SHA-256:
  `8fa0a7c3a98b40db12c764a100481abc518fb19d0a749f571ae7eec499dd9616`.
- Analysis: `research_gym/analysis/lsad_chance_level.py`.
- Test: `tests/test_lsad_chance_level_audit.py`. On the audit host pytest was
  not installed; the three test functions were called directly and passed.
