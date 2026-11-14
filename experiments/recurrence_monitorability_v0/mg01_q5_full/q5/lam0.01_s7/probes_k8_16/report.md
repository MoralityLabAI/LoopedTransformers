# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 6.550936 | [4.06117, 9.07863] |
| 8 | affine | shared | False | 8 | 1.703030 | [-0.15886, 3.56492] |
| 8 | mlp | separate | False | 8 | 1.385358 | [0.20355, 2.51843] |
| 8 | mlp | shared | False | 8 | 0.571299 | [0.15415, 1.14769] |
| 16 | affine | separate | False | 16 | 10.795901 | [8.54871, 12.49254] |
| 16 | affine | shared | False | 16 | 3.065602 | [1.34921, 5.48565] |
| 16 | mlp | separate | False | 16 | 2.342724 | [0.14824, 5.06687] |
| 16 | mlp | shared | False | 16 | 0.735095 | [-0.65907, 1.94438] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
