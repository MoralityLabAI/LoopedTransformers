# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 5.403990 | [-0.83313, 11.64111] |
| 8 | affine | shared | False | 8 | 0.554774 | [-1.16232, 2.90954] |
| 8 | mlp | separate | False | 8 | 0.166179 | [-2.17960, 2.10508] |
| 8 | mlp | shared | False | 8 | 0.100608 | [-1.87393, 1.27001] |
| 16 | affine | separate | False | 16 | 11.411333 | [8.41765, 13.35696] |
| 16 | affine | shared | False | 16 | 1.233556 | [-1.54289, 4.01000] |
| 16 | mlp | separate | False | 16 | -0.197823 | [-0.62513, 0.22948] |
| 16 | mlp | shared | False | 16 | 0.655195 | [0.07125, 1.35261] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
