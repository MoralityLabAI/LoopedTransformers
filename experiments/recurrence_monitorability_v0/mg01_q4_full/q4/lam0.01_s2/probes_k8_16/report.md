# RMP-MG-01 bounded-probe receipt

Status: `FROZEN_CHECKPOINT_DEVELOPMENT`.

Actor weights remained frozen. Test scores followed the saved selection receipt.
Groups are disjoint cycle types, including all graph renamings.

| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |
|---|---|---|---|---|---:|---|
| 8 | affine | separate | False | 8 | 9.610002 | [6.50176, 13.26063] |
| 8 | affine | shared | False | 8 | 3.625282 | [2.17441, 4.59617] |
| 8 | mlp | separate | False | 8 | 4.458889 | [3.88764, 5.20216] |
| 8 | mlp | shared | False | 8 | 1.803013 | [1.34302, 2.22695] |
| 16 | affine | separate | False | 16 | 15.847821 | [10.22315, 21.47249] |
| 16 | affine | shared | False | 16 | 2.153482 | [0.10008, 4.20688] |
| 16 | mlp | separate | False | 16 | 5.850635 | [3.71793, 8.00047] |
| 16 | mlp | shared | False | 16 | 2.690657 | [2.30894, 2.95119] |

Intervals are exploratory and conditional on this actor and selected structural families.
Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.
Q4/Q5 evaluation requires the original external checkpoints.
