# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 4.410285 | [1.74290, 6.07091] |
| 8 | affine | shared | False | 8 | -2.314022 | [-3.67577, -0.18765] |
| 8 | mlp | separate | False | 8 | 0.359873 | [-1.23356, 1.33886] |
| 8 | mlp | shared | False | 8 | -1.179119 | [-2.58062, 0.22883] |
| 16 | affine | separate | False | 16 | -3.844045 | [-7.71243, -1.58653] |
| 16 | affine | shared | False | 16 | -4.834504 | [-6.50007, -3.16894] |
| 16 | mlp | separate | False | 16 | -2.569470 | [-3.26814, -1.87080] |
| 16 | mlp | shared | False | 16 | -1.977536 | [-2.67689, -1.33269] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
