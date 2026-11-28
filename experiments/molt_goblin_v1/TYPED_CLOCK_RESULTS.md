# MOLT-12 competence result and engineering repair

First [CPU run 45915662450](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45915662450)
at source head 63d82ddd2000f6d58570925a6fe9cba7c5aa5ffe passed all three focused
tests and the fixed teacher training/evaluation phase. Both addressed teachers
solved 561/561 unrestricted primitive pairs and 128/128 complete traces at
each depth 1,2,4,8,12. There were 640 development sequence roots shared by the
two model initializations. Every primitive pair was in the training corpus;
this tests composition of learned transitions, not new arithmetic operations.

| Seed and arm | Primitive correct /561 | D1 terminal /128 | D2 | D4 | D8 | D12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 157 separable | 122 | 7 | 8 | 4 | 7 | 9 |
| 163 separable | 141 | 12 | 7 | 7 | 6 | 8 |
| 157 addressed | 561 | 128 | 128 | 128 | 128 | 128 |
| 163 addressed | 561 | 128 | 128 | 128 | 128 | 128 |

The clock phase elapsed 12.4433 s and sampled peak RSS 439,054,336 B.
The native BPD phase was killed by the child-process guard after 1.077 s,
before a factor-fit receipt. Its loader verifies the Git checkout using
subprocesses, which conflicts with the declared child-free worker contract.
Do not reinterpret the termination as a factor-quality or competence failure.

The repair moves verification of the pinned BPD commit and its tracked Python
source blobs into the supervisor before launching capped workers. The worker
rechecks the complete Python source inventory and SHA-256 manifest in process
before loading native BPD. This is a source identity contract under a trusted
supervisor and immutable checkout, not a sandbox for malicious Python. No
child-process exception, raised resource cap or altered teacher training
recipe. The amended workflow checks the worker-reported kernel RSS high water
against the same guard after exit. The predeclared addressed seed 157 remains
the only native-fit candidate. The original failed run remains in the record.

The addressed register interface exposes the finite operation/state address,
and the carry is a soft predicted distribution. The separable model did not
learn the primitive table adequately under this training recipe. None of
these results establishes an improvement on the original AUX-V2 token task.
The model and world dependencies are crossed; no confirmation or promotion.
