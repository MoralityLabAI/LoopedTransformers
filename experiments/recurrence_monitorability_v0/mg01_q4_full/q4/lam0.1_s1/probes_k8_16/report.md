# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 5.040916 | [0.50326, 10.05671] |
| 8 | affine | shared | False | 8 | 0.885682 | [-1.48820, 3.11590] |
| 8 | mlp | separate | False | 8 | 2.586010 | [2.11816, 3.08245] |
| 8 | mlp | shared | False | 8 | -0.621593 | [-2.22194, 0.63198] |
| 16 | affine | separate | False | 16 | 11.091503 | [6.75061, 15.33176] |
| 16 | affine | shared | False | 16 | 7.229946 | [6.30283, 8.27828] |
| 16 | mlp | separate | False | 16 | -0.782826 | [-1.63984, 0.07419] |
| 16 | mlp | shared | False | 16 | -0.016523 | [-1.82887, 2.18768] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
