# RAM-01 live Lean results | 2026-11-14

[Regression CI 43557331256](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43557331256) completed successfully for every fixed model configuration and the native comparator. Its source-code revision is `518ccdfa355e0b33d647cc0d542262a1a627284a`. The checkpoints came from [successful CPU training run 43557076648](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43557076648), revision `fb1745362323db97c105209f9c7c6cc15429a43c`.

## Complete theorem certificates

The suite contains the previously published 16 FC-13 original MathCan theorems. This is a post-hoc regression experiment. It supplies no fresh-holdout claim and leaves the original FC-13 result unchanged.

| Policy | Parameters | Seed | Lean-certified proofs | Total search wall seconds | Lean subprocesses | Seconds per certificate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed native | 0 | — | 6/16 | 105.043 | 132 | 17.507 |
| flat64 | 173,252 | 17 | 6/16 | 127.701 | 127 | 21.284 |
| flat64 | 173,252 | 29 | 6/16 | 127.341 | 125 | 21.223 |
| loop32 | 82,692 | 17 | 6/16 | 91.487 | 125 | 15.248 |
| loop32 | 82,692 | 29 | 6/16 | 89.877 | 123 | 14.979 |
| loop64 | 198,148 | 17 | 6/16 | 103.255 | 140 | 17.209 |
| loop64 | 198,148 | 29 | 6/16 | 88.652 | 121 | 14.775 |
| loop128 | 527,364 | 17 | 6/16 | 75.822 | 128 | 12.637 |
| loop128 | 527,364 | 29 | 6/16 | 72.512 | 122 | 12.085 |

Every policy certified the same six declarations. All eight trained policies had zero model-only successes and zero native-only misses. The successful declarations were HookMultiTick.before_swap_exact_input, HookMultiTick.specified_credit_restores_debit, Settlement.exact_five_account_partition, Settlement.terminal_pool_floor_budget, Settlement.certificate_gap_identity, and TOEJordan.metric_derivative_vanishes_at_root.

The successful neural scripts used omega, simp_all, or a simp_all followed by omega. No newly parameterized exact/rw/rcases/induction script appeared in a completed regression proof. Those families retain real positive training transition examples.

Search wall times are measured outcomes from separate GitHub-hosted matrix workers. Each variant ran both seeds on its own worker; native used another worker. Hardware scheduling and worker variation were not controlled with repeated interleaved trials. These times cannot establish a robust speed advantage. The subprocess counts are preserved as a separate resource measure.

## Proof and budget checks

All controllers used the same original context-derived candidate grammar, twenty-second per-theorem deadline, twelve actual Lean subprocess calls, and at most three transitions. A successful scratch closure required another full-source Lean compilation. The final compiler call printed the selected theorem's axiom dependencies; only propext, Classical.choice and Quot.sound, or an empty dependency list, were accepted. Multiple remaining goals caused an explicit abstention.

Each accepted proof then underwent a fresh, untimed Lean recompilation. This second check could not change its score. The original proof oracle compiled 16/16 in each job. The preflight also checked a deliberately custom-axiom theorem: compilation returned zero, while the proof gate rejected it.

## Downloaded artifact audit

A separate local inspection verified all nine policy outputs against the downloaded compiler artifacts:

- 144 complete theorem-search trajectories.
- 1,143 saved compiler invocations, each with its generated Lean source and compiler log.
- 54 successful policy/theorem instances with final-source hashes matching their fresh replay sources.
- Zero source-hash or budget-accounting discrepancies.

This local inspection was a static source-byte/log/deadline audit. CI performed the actual Lean recompilations. There are six unique solved theorems, repeated across nine policies; the 54 successful instances are not 54 new theorems.

## Interpretation

CPU/RAM training and reproducible export are demonstrated at 83k–527k parameters, with the full training campaign below 520 MiB peak process RSS. The source-module development check improved first-choice executable-tactic counts over native ordering. Complete theorem coverage remained 6/16 for every policy.

The 83k recurrent model reproduces the same six successes as the larger models on this regression set. More capacity did not expand the solved set. Exact and rcases each have only one successful training example, and the controller still stops on branching goal states. Those constraints remain visible in the archived failures.
