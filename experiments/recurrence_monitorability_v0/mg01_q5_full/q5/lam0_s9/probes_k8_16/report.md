# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 2.601761 | [-0.17685, 4.46003] |
| 8 | affine | shared | False | 8 | 1.229251 | [-3.16422, 4.50229] |
| 8 | mlp | separate | False | 8 | 1.554127 | [1.10656, 2.24787] |
| 8 | mlp | shared | False | 8 | 1.015474 | [-0.05252, 2.08347] |
| 16 | affine | separate | False | 16 | 8.684854 | [4.86043, 12.50928] |
| 16 | affine | shared | False | 16 | 0.536681 | [-1.75534, 2.89482] |
| 16 | mlp | separate | False | 16 | 1.151066 | [-0.45081, 2.33933] |
| 16 | mlp | shared | False | 16 | 1.220017 | [-1.40578, 3.43643] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
