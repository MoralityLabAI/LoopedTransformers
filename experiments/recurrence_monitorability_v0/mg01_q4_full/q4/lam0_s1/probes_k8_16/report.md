# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 0.515125 | [-2.86207, 4.57832] |
| 8 | affine | shared | False | 8 | -2.223085 | [-4.22099, 0.17896] |
| 8 | mlp | separate | False | 8 | 1.942088 | [-0.06977, 3.95395] |
| 8 | mlp | shared | False | 8 | 0.036246 | [-0.44828, 0.84124] |
| 16 | affine | separate | False | 16 | 0.885951 | [-2.54945, 4.32135] |
| 16 | affine | shared | False | 16 | -2.386770 | [-5.08181, 1.56069] |
| 16 | mlp | separate | False | 16 | 0.284367 | [-0.46763, 1.30473] |
| 16 | mlp | shared | False | 16 | -0.745269 | [-1.52680, 0.03626] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
