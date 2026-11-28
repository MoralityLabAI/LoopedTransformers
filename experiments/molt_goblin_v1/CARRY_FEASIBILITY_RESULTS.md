# MOLT-16 measured finite-dose feasibility

[CPU run 45922968521](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45922968521)
passed at 000d5abc5f753acbd5ea048a5fc0cfe9503dab2d. Three focused tests passed.
The clock, BPD fit, controller reproduction and feasibility worker all completed
under unchanged caps. No teacher updates in the feasibility phase.

| Chart | Changed strict winner witnessed | Conditional obstruction | Unresolved | Learned suffix correct |
| --- | ---: | ---: | ---: | ---: |
| Native | 25/32 | 7/32 | 0 | 23/32 |
| Observer | 28/32 | 4/32 | 0 | 28/32 |
| Random | 0/32 | 32/32 | 0 | 0/32 |

Best-cut-gap suffix counts were identical to the learned suffix counts on this
panel. Every same-target row was witnessed and every candidate preserved the
same-target suffix. The numerical obstructions concern the immediate legal
winner at the fixed dose in each chart, not the whole future trajectory.
There are 32 world roots with paired edits, and 192 chart/edit observations.
One fixed teacher and learned controller checkpoint per arm.

Feasibility worker time 10.4724 s, worker peak RSS 339,857,408 B.
The halfspace bounds use ordinary floating-point SVD, and the dual witnesses
are not yet verified with exact arithmetic. These counts are development-only.
No confirmation, promotion, hard RAM guarantee, or additional teacher training
benefit. Next: replay the same roster using rational exact checks on frozen
binary64 data, with the actual physical norm in each chart.
