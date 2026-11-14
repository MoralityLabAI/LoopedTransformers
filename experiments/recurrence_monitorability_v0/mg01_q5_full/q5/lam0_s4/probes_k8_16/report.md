# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | -3.039072 | [-5.88117, -0.81875] |
| 8 | affine | shared | False | 8 | -2.712375 | [-5.57125, -0.29889] |
| 8 | mlp | separate | False | 8 | -0.572622 | [-1.14785, 0.15925] |
| 8 | mlp | shared | False | 8 | -0.491199 | [-1.22961, 0.46946] |
| 16 | affine | separate | False | 16 | 0.905608 | [-3.56152, 3.75352] |
| 16 | affine | shared | False | 16 | -1.642761 | [-4.05377, 0.05793] |
| 16 | mlp | separate | False | 16 | -0.591770 | [-3.26519, 1.23892] |
| 16 | mlp | shared | False | 16 | 0.915306 | [0.38435, 1.41297] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
