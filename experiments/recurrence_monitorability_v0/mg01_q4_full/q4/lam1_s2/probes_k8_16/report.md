# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 0.793796 | [-3.73458, 5.32217] |
| 8 | affine | shared | False | 8 | -2.051788 | [-3.89866, -0.39389] |
| 8 | mlp | separate | False | 8 | -2.988144 | [-4.23336, -2.03686] |
| 8 | mlp | shared | False | 8 | -3.138633 | [-4.24613, -1.50175] |
| 16 | affine | separate | False | 16 | 10.108502 | [7.27672, 13.13301] |
| 16 | affine | shared | False | 16 | -1.817566 | [-4.04187, 0.13414] |
| 16 | mlp | separate | False | 16 | -0.573915 | [-1.94262, 0.79479] |
| 16 | mlp | shared | False | 16 | -1.600590 | [-3.45696, 0.00522] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
