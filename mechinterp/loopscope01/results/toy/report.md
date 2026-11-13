# LOOPSCOPE-01 Q4 frontier (complete)

Decision: **no trade-off**

Held-out depths 9-16. Probe = ridge probe on a separate split, mean over loops 1-16. Lens = model head at loop j. mean+-sd over seeds.

Validity: depth-8 accuracy >= 0.90 per seed (amendment 2). training_failed: none

| lam | valid seeds | acc d8 | acc d1-8 | acc held-out | probe 1-16 | probe d1-8 | probe held-out | lens 1-16 |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 1.000+-0.000 | 0.465+-0.278 | 0.460+-0.383 | 0.761+-0.259 | 0.762+-0.252 | 0.761+-0.268 | 0.466+-0.328 |
| 0.01 | 3 | 1.000+-0.000 | 0.998+-0.003 | 0.787+-0.301 | 0.916+-0.119 | 0.994+-0.008 | 0.837+-0.230 | 0.892+-0.153 |
| 0.1 | 3 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 |
| 1 | 3 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 | 1.000+-0.000 |

Differences vs lam=0 (point, 95% CI; bootstrap over seeds then items):

| lam | held-out acc diff | probe diff |
|---|---|---|
| 0.01 | +0.327 [-0.213, +0.819] | +0.154 [-0.121, +0.442] |
| 0.1 | +0.540 [+0.000, +0.832] | +0.239 [+0.034, +0.603] |
| 1 | +0.540 [+0.000, +0.832] | +0.239 [+0.034, +0.603] |

trade-off if some lam>0 has held-out accuracy below lam=0 by >0.05 with seed+item bootstrap 95% CI excluding 0 and probe decodability (mean over loops 1-K) above lam=0; no trade-off if every lam>0 is >= lam=0 on both; else mixed.
