# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | -5.175408 | [-11.14286, -1.18610] |
| 8 | affine | shared | False | 8 | 2.872691 | [0.25640, 6.34714] |
| 8 | mlp | separate | False | 8 | 0.869303 | [-0.80061, 1.95795] |
| 8 | mlp | shared | False | 8 | 0.136055 | [-1.81465, 1.83015] |
| 16 | affine | separate | False | 16 | 4.693362 | [1.24207, 8.14465] |
| 16 | affine | shared | False | 16 | 1.602987 | [-3.21256, 4.97675] |
| 16 | mlp | separate | False | 16 | 1.723440 | [0.78937, 2.65751] |
| 16 | mlp | shared | False | 16 | 2.854415 | [1.65923, 4.02412] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
