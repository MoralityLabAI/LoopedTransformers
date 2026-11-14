# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 15.644682 | [10.16012, 21.55615] |
| 8 | affine | shared | False | 8 | 6.621174 | [2.71394, 10.39731] |
| 8 | mlp | separate | False | 8 | 0.575895 | [-0.42755, 1.81260] |
| 8 | mlp | shared | False | 8 | -1.678758 | [-2.72095, -0.83964] |
| 16 | affine | separate | False | 16 | 20.259746 | [14.90053, 26.95505] |
| 16 | affine | shared | False | 16 | 5.977603 | [0.18699, 13.89795] |
| 16 | mlp | separate | False | 16 | 2.340493 | [0.46768, 4.29135] |
| 16 | mlp | shared | False | 16 | -0.061493 | [-1.61426, 1.02938] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
