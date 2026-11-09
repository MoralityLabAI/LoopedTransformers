# LOOPSCOPE-01 Q6: early-loop contraction vs test-time-loop gain on Lean-CGRT Gate-0 loopers (amendment 6)

Decision: **uninformative (primary outcome constant across checkpoints)**. Spearman(F1/F8, deep/best gain T32-T8) = +nan; one-sided p (negative) nan, (positive) nan; n = 18.
Within family: S-loop +nan (n=10), S-rd +nan (n=8).
Secondary Spearman with F1/F8 (nan = outcome constant): deep_best_T8 +nan, id_best_gain +nan, deep_greedy_gain +0.328, id_greedy_gain +0.103.

Exploratory (added after seeing the ceiling; not registered), best policy, gain T32-T8 at tighter budgets: test_mirror_deep/best/b1: rho +0.340, two-sided p 0.171, gain range [-0.339, +0.002]; test_mirror_id/best/b1: rho +0.241, two-sided p 0.339, gain range [-0.123, +0.003]; test_mirror_deep/best/b2: rho +0.082, two-sided p 0.757, gain range [-0.051, +0.000]; test_mirror_id/best/b2: rho -0.068, two-sided p 0.795, gain range [-0.020, +0.000].

| ckpt | F1/F8 | deep best T8 | deep best T32 | gain | id best gain | deep greedy gain | states |
|---|---|---|---|---|---|---|---|
| g0_Sloop_s17 | 1.5349 | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 | 69 |
| g0_Sloop_s29 | 1.1992 | 1.000 | 1.000 | +0.000 | +0.000 | -0.068 | 69 |
| g0_Sloop_s43 | 2.4090 | 1.000 | 1.000 | +0.000 | +0.000 | +0.002 | 69 |
| g0_Sloop_s53 | 2.0533 | 1.000 | 1.000 | +0.000 | +0.000 | -0.008 | 69 |
| g0_Sloop_s61 | 1.4914 | 1.000 | 1.000 | +0.000 | +0.000 | +0.005 | 69 |
| g0_Sloop_s71 | 1.7729 | 1.000 | 1.000 | +0.000 | +0.000 | -0.377 | 69 |
| g0_Sloop_s83 | 1.3505 | 1.000 | 1.000 | +0.000 | +0.000 | -0.175 | 69 |
| g0_Sloop_s97 | 1.2364 | 1.000 | 1.000 | +0.000 | +0.000 | -0.372 | 69 |
| g0_Sloop_s101 | 1.0663 | 1.000 | 1.000 | +0.000 | +0.000 | -0.043 | 69 |
| g0_Sloop_s113 | 1.5771 | 1.000 | 1.000 | +0.000 | +0.000 | -0.003 | 69 |
| g0_Srd_s43 | 1.4707 | 1.000 | 1.000 | +0.000 | +0.000 | +0.002 | 69 |
| g0_Srd_s53 | 1.2448 | 1.000 | 1.000 | +0.000 | +0.000 | +0.002 | 69 |
| g0_Srd_s61 | 1.5285 | 1.000 | 1.000 | +0.000 | +0.000 | -0.038 | 69 |
| g0_Srd_s71 | 2.1166 | 1.000 | 1.000 | +0.000 | +0.000 | +0.005 | 69 |
| g0_Srd_s83 | 0.9994 | 1.000 | 1.000 | +0.000 | +0.000 | -0.008 | 69 |
| g0_Srd_s97 | 1.6830 | 1.000 | 1.000 | +0.000 | +0.000 | -0.013 | 69 |
| g0_Srd_s101 | 1.4597 | 1.000 | 1.000 | +0.000 | +0.000 | +0.002 | 69 |
| g0_Srd_s113 | 2.0353 | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 | 69 |
