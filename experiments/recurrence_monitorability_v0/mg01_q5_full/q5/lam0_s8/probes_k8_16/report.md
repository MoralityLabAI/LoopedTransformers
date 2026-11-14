# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 3.797656 | [-1.11837, 8.20076] |
| 8 | affine | shared | False | 8 | -0.989532 | [-3.32330, 1.14542] |
| 8 | mlp | separate | False | 8 | 1.688971 | [-0.39650, 3.50489] |
| 8 | mlp | shared | False | 8 | 0.400135 | [-0.32837, 1.05962] |
| 16 | affine | separate | False | 16 | 4.040201 | [1.45697, 5.75023] |
| 16 | affine | shared | False | 16 | 0.798023 | [-1.36317, 2.23760] |
| 16 | mlp | separate | False | 16 | 0.855839 | [-0.47687, 2.18855] |
| 16 | mlp | shared | False | 16 | 0.815671 | [0.51280, 1.31493] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
