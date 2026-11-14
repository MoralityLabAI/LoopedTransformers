# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 13.614991 | [7.12029, 20.43817] |
| 8 | affine | shared | False | 8 | 2.758168 | [0.78906, 5.05130] |
| 8 | mlp | separate | False | 8 | 1.263168 | [0.42613, 2.03889] |
| 8 | mlp | shared | False | 8 | 1.341809 | [0.13187, 2.25315] |
| 16 | affine | separate | False | 16 | 16.454973 | [15.41398, 18.06116] |
| 16 | affine | shared | False | 16 | 4.773591 | [2.44742, 6.63881] |
| 16 | mlp | separate | False | 16 | 3.470069 | [2.07363, 4.86651] |
| 16 | mlp | shared | False | 16 | -0.072820 | [-0.91637, 0.86008] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
