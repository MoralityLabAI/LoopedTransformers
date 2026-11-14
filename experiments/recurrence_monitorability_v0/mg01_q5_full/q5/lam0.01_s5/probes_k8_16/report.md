# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 1.618244 | [-0.24904, 3.48553] |
| 8 | affine | shared | False | 8 | 0.964497 | [-0.47532, 2.42483] |
| 8 | mlp | separate | False | 8 | 0.417341 | [-0.79092, 1.18424] |
| 8 | mlp | shared | False | 8 | 1.084967 | [0.67483, 1.68578] |
| 16 | affine | separate | False | 16 | 6.661059 | [2.77941, 9.69654] |
| 16 | affine | shared | False | 16 | 0.518827 | [-3.69648, 3.61730] |
| 16 | mlp | separate | False | 16 | 0.806801 | [-0.81633, 2.37031] |
| 16 | mlp | shared | False | 16 | -0.104346 | [-1.91577, 1.78632] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
