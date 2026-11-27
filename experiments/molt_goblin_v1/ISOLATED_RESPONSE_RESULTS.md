# MOLT-11 measured isolated response-memory results

[CPU run 45859196843](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45859196843)
passed at source head 82fb468f854c32df18e5fcf45eba1b51b24e9be9.
Two focused tests passed. All eight isolated phases completed, including
the separate final parameter/prediction audit. One teacher, 256 native
TRAIN donor records, 16 student updates per arm, one CPU thread.

The teacher's validation exact score was 0.0735294118; the constant
baseline was 0.0588235294. The fixed competence gate failed. The engineering
comparison remained authorized; no task-learning advantage is established.

| Arm | RSS after load (B) | Kernel lifetime peak (B) | Training wall s | Parameter error |
| --- | ---: | ---: | ---: | ---: |
| Dense | 398753792 | 574038016 | 2.2493243 | 0 |
| Decode | 381005824 | 558411776 | 2.2300136 | 0 |
| Gram | 385888256 | 559316992 | 2.3224775 | 2.22045e-16 |
| Recompute | 386646016 | 555905024 | 2.8970341 | 5.55112e-17 |

All ordinary validation predictions agreed; reported total loss differences
were zero. Maximum response-loss error was 4.04161e-21. Decode saved
15,626,240 B of measured lifetime peak RSS relative to dense in this
fixed-order run, with identical final parameters. Recompute had the lowest
measured peak and the highest training time. Timing/RSS differences are
descriptive, single-run measurements, not population estimates.

Common native coordinate/task tensor payload: 174,080 B.
Dense response archive: 20,971,520 B, file 20,973,161 B.
Factor response: 1,310,720 B, frozen centered head: 2,097,152 B,
combined file 3,409,701 B. Preparation peaks were 390,705,152 B
(dense) and 343,695,360 B (factor). Gram adds 131,072 B. Recompute
retains 3,678,208 B of teacher parameter tensors and a 2,097,152 B
centered head. Separate processes loaded only their assigned representation.

The per-process virtual address-space and CPU time limits were hard;
the RSS guard was sampled and worker high water checked. This is not
a kernel-enforced process-tree physical RAM cap. Existing setup phases
were outside the isolation wrapper. No held cohort was opened.

Next: fix the competence bottleneck with a separately labeled typed
operand-clock teacher. Preserve the existing AUX-V2 failure; a structured
primitive-transition curriculum changes the task interface and cannot
be counted as a performance improvement of the original teacher.
