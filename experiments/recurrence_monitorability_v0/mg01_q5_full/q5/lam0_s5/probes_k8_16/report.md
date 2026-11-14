# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | -2.061928 | [-5.21774, -0.15461] |
| 8 | affine | shared | False | 8 | -2.322165 | [-5.27120, -0.57677] |
| 8 | mlp | separate | False | 8 | 1.353507 | [0.81443, 1.89258] |
| 8 | mlp | shared | False | 8 | -0.364135 | [-1.50266, 0.80798] |
| 16 | affine | separate | False | 16 | -6.320591 | [-12.53199, -1.37696] |
| 16 | affine | shared | False | 16 | -1.663252 | [-5.02715, 1.70065] |
| 16 | mlp | separate | False | 16 | -1.315985 | [-2.66988, 0.03791] |
| 16 | mlp | shared | False | 16 | 0.738635 | [-0.41899, 1.86446] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
