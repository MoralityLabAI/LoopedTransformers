# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | -0.870029 | [-3.01950, 0.97611] |
| 8 | affine | shared | False | 8 | 0.315240 | [-2.15466, 2.78514] |
| 8 | mlp | separate | False | 8 | -1.612524 | [-3.04699, -0.60559] |
| 8 | mlp | shared | False | 8 | 0.420457 | [-0.27049, 1.11232] |
| 16 | affine | separate | False | 16 | 11.101225 | [9.28349, 12.93801] |
| 16 | affine | shared | False | 16 | 5.385211 | [4.20638, 6.59003] |
| 16 | mlp | separate | False | 16 | -0.002008 | [-0.60329, 0.59928] |
| 16 | mlp | shared | False | 16 | 0.517544 | [-0.64189, 1.67698] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
