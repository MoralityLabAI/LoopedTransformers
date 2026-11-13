# LOOPSCOPE-01 Q5: does early-loop contraction predict length extrapolation? (amendment 5)

Decision: **weak**. 9/12 valid arms classified correctly; 8 of 12 extrapolate.

Rule: predict extrapolation iff F1/F8 < 0.05 (depth-8 items only); extrapolation = held-out accuracy >= 0.9.
Supported if >= 90% correct with >= 2 arms in each class; weak if >= 75%; else not supported.

| arm | acc d8 | F1/F8 | predicted | held-out acc | extrapolates | probe 1-16 | final loss | TCA valid |
|---|---|---|---|---|---|---|---|---|
| lam0.01_s4 | 1.000 | 0.2207 | False | 0.990 | True | 0.997 | 7.25e-09 | False |
| lam0.01_s5 | 1.000 | 0.0354 | True | 1.000 | True | 1.000 | 8.78e-10 | True |
| lam0.01_s6 | 1.000 | 0.0329 | True | 1.000 | True | 1.000 | 2.76e-08 | True |
| lam0.01_s7 | 1.000 | 0.0388 | True | 1.000 | True | 1.000 | 4.74e-09 | False |
| lam0.01_s8 | 1.000 | 0.0602 | False | 1.000 | True | 1.000 | 5.17e-10 | False |
| lam0.01_s9 | 1.000 | 0.0260 | True | 1.000 | True | 1.000 | 7.64e-10 | False |
| lam0_s4 | 1.000 | 0.1085 | False | 0.299 | False | 0.862 | 1.75e-08 | False |
| lam0_s5 | 1.000 | 0.0533 | False | 0.441 | False | 0.568 | 4.19e-09 | False |
| lam0_s6 | 1.000 | 0.1031 | False | 0.876 | False | 0.914 | 4.89e-09 | False |
| lam0_s7 | 1.000 | 0.1811 | False | 0.303 | False | 0.757 | 4.89e-09 | True |
| lam0_s8 | 1.000 | 0.0438 | True | 0.981 | True | 0.894 | 7.68e-09 | False |
| lam0_s9 | 1.000 | 0.0503 | False | 1.000 | True | 0.946 | 1.30e-08 | True |

Spearman with held-out accuracy: F1/F8 -0.718; final training loss -0.384; probe (needs hop labels) +0.927.
training_failed: none
