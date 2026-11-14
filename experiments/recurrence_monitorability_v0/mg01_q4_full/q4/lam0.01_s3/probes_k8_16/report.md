# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | -1.862951 | [-5.44079, 0.87574] |
| 8 | affine | shared | False | 8 | -1.582206 | [-2.82994, -0.62635] |
| 8 | mlp | separate | False | 8 | -0.437506 | [-1.81459, 1.26655] |
| 8 | mlp | shared | False | 8 | 0.016765 | [-0.80186, 0.83539] |
| 16 | affine | separate | False | 16 | -6.891240 | [-9.17979, -4.60269] |
| 16 | affine | shared | False | 16 | -3.553428 | [-8.32365, -1.01004] |
| 16 | mlp | separate | False | 16 | -1.718023 | [-4.17533, 0.74669] |
| 16 | mlp | shared | False | 16 | -1.571123 | [-2.03624, -1.10601] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
