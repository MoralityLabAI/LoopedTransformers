# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 0.392302 | [-0.25584, 1.00003] |
| 8 | affine | shared | False | 8 | 3.134839 | [0.21608, 4.82420] |
| 8 | mlp | separate | False | 8 | 1.850694 | [0.69935, 3.27720] |
| 8 | mlp | shared | False | 8 | 1.193471 | [0.07174, 2.31520] |
| 16 | affine | separate | False | 16 | 10.440476 | [6.96089, 14.94816] |
| 16 | affine | shared | False | 16 | 1.320118 | [0.54445, 2.09578] |
| 16 | mlp | separate | False | 16 | 1.343164 | [0.00525, 2.82155] |
| 16 | mlp | shared | False | 16 | -0.082703 | [-0.55806, 0.37947] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
