# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 1.783187 | [-4.42639, 7.99277] |
| 8 | affine | shared | False | 8 | -1.544017 | [-3.16362, 0.03361] |
| 8 | mlp | separate | False | 8 | -1.076144 | [-2.46320, 0.30756] |
| 8 | mlp | shared | False | 8 | -0.864118 | [-2.54023, 0.85477] |
| 16 | affine | separate | False | 16 | 12.189808 | [9.17473, 15.55400] |
| 16 | affine | shared | False | 16 | -2.642771 | [-6.77225, 2.43862] |
| 16 | mlp | separate | False | 16 | -1.304022 | [-4.14540, 0.64530] |
| 16 | mlp | shared | False | 16 | 0.208507 | [-1.01786, 1.56204] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
