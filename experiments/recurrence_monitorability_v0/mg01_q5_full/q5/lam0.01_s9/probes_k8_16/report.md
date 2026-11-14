# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 1.189786 | [-3.13991, 3.61357] |
| 8 | affine | shared | False | 8 | 1.551455 | [-1.96247, 5.06538] |
| 8 | mlp | separate | False | 8 | -0.089832 | [-1.84008, 1.52510] |
| 8 | mlp | shared | False | 8 | 0.526559 | [-0.81534, 1.77439] |
| 16 | affine | separate | False | 16 | 5.518423 | [0.68299, 11.06670] |
| 16 | affine | shared | False | 16 | 1.706025 | [-2.62340, 4.70240] |
| 16 | mlp | separate | False | 16 | 0.240370 | [-3.19248, 2.36739] |
| 16 | mlp | shared | False | 16 | 0.730427 | [-2.89817, 3.22826] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
