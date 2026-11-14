# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 1.747463 | [-1.77772, 5.27265] |
| 8 | affine | shared | False | 8 | 0.135557 | [-1.37955, 1.02062] |
| 8 | mlp | separate | False | 8 | 1.121418 | [-1.16326, 2.97891] |
| 8 | mlp | shared | False | 8 | 0.611112 | [-0.41866, 1.89637] |
| 16 | affine | separate | False | 16 | 2.782617 | [1.70094, 4.10264] |
| 16 | affine | shared | False | 16 | 0.409409 | [-0.92572, 1.66084] |
| 16 | mlp | separate | False | 16 | 1.083021 | [-1.18641, 2.83501] |
| 16 | mlp | shared | False | 16 | 0.122740 | [-1.39255, 1.63803] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
