# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | -4.942491 | [-9.68450, -0.25428] |
| 8 | affine | shared | False | 8 | -3.334562 | [-5.16923, -1.61895] |
| 8 | mlp | separate | False | 8 | -3.689811 | [-5.61124, -2.01093] |
| 8 | mlp | shared | False | 8 | -1.266949 | [-1.71361, -0.55590] |
| 16 | affine | separate | False | 16 | 1.630308 | [-3.64667, 6.90729] |
| 16 | affine | shared | False | 16 | 0.883484 | [-0.88500, 2.65196] |
| 16 | mlp | separate | False | 16 | -1.505441 | [-3.17097, -0.33754] |
| 16 | mlp | shared | False | 16 | -0.919230 | [-1.78392, -0.07730] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
