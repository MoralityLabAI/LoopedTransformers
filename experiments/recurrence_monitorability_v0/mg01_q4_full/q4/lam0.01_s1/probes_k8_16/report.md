# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 0.772502 | [-0.82007, 2.62743] |
| 8 | affine | shared | False | 8 | -0.659339 | [-3.06566, 1.74698] |
| 8 | mlp | separate | False | 8 | 1.739357 | [0.92710, 2.66881] |
| 8 | mlp | shared | False | 8 | 0.272390 | [-0.17915, 0.73285] |
| 16 | affine | separate | False | 16 | 8.679169 | [2.62821, 14.00773] |
| 16 | affine | shared | False | 16 | 1.492929 | [-0.45546, 3.53915] |
| 16 | mlp | separate | False | 16 | 0.919093 | [0.32972, 1.75031] |
| 16 | mlp | shared | False | 16 | 2.506255 | [0.67694, 4.56187] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
