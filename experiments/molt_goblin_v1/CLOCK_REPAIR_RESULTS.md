# MOLT-14 measured finite repair development result

[CPU run 45922095011](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45922095011)
passed at 5b976d36c1a23ef99ac1deeec89531e6886ee593. Three focused tests
passed. The clock, native factor fit and repair worker completed under
the declared limits. There were 32 fresh roots, each with a changed
addition and a same-target compensated prefix edit.

| Arm | Changed donor suffix correct /32 | Same target preserved /32 |
| --- | ---: | ---: |
| Intact | 0 | 32 |
| Native swap | 17 | 32 |
| Native solve | 10 | 32 |
| Random solve | 0 | 32 |
| Observer solve | 9 | 32 |
| Finite carry sham | 10 | 32 |
| Donor residual ceiling | 32 | 32 |

Native and observer optimizers started at zero and minimized squared
legal margin error. The changed-target mean dose was 8.08619663 for the
native swap and each capped solver. Mean changed-target margin L2 was
11.76846 (swap), 12.57273 (native solve), 12.61270 (observer solve),
18.26322 (random solve). The privileged ceiling's mean dose was 15.28915.
The raw native swap did better than the zero-start native solver. Equal
budget alone does not guarantee nonlinear optimization quality.

All 64 finite carry shams matched, with maximum future-logit error
7.02993e-12. This is the predicted exact finite output equivalence,
not support for a donor-specific mechanism. The repair phase elapsed
3.67638 s and worker kernel RSS high water was 344,981,504 B.
No training, protected readout, confirmation or promotion.

Next: prespecify a decision-aligned objective and compare zero-start
versus raw-swap warm starts. Train a small amortized repair controller
on separate source roots, with frozen teacher/factors and fixed charts.
Its target is observed donor behavior, not autonomous task competence.
Keep ordinary teacher competence, causal intervention headroom, and
controller learning as distinct evidence classes.
