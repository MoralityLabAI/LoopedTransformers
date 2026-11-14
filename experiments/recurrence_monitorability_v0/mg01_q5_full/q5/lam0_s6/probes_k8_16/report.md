# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 4.556404 | [-0.60656, 9.71937] |
| 8 | affine | shared | False | 8 | 0.869410 | [-2.38896, 3.28747] |
| 8 | mlp | separate | False | 8 | 2.050070 | [0.10286, 3.99728] |
| 8 | mlp | shared | False | 8 | 1.394144 | [0.44201, 2.61003] |
| 16 | affine | separate | False | 16 | 7.780051 | [3.53745, 12.14930] |
| 16 | affine | shared | False | 16 | 1.100743 | [-2.92151, 5.20654] |
| 16 | mlp | separate | False | 16 | -1.515649 | [-3.33761, -0.04368] |
| 16 | mlp | shared | False | 16 | 1.180177 | [0.64043, 2.09505] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
