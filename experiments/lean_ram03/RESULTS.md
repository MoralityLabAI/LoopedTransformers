# RAM-03 measured results | 2026-11-14 (Asia/Taipei)

RAM-03 tested genuine branch-state supervision, a goal-count auxiliary head and a persistent Lean search backend. RAM-02 remains preserved. All coverage below requires complete original-theorem compilation plus independent exact-source replay.

Primary prospective coverage: native 8/12, loop32 8/12, loop64 8/12, flat64 8/12. Primary post-hoc FC-13 coverage: native 14/16, frequency 15/16, loop32 15/16, loop64 14/16, flat64 15/16.

Secondary prospective coverage: native 8/12, loop64 8/12. Secondary post-hoc FC-13 coverage: native 14/16, loop64 14/16. The matched model-only and native-only sets below determine the scope of every gain.

## Coverage against contemporary native search

| Budget | Suite | Policy | Certified | Model-only vs native | Native-only misses | Logical evaluations | Lean processes | Seconds / certificate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | prospective | native | 8/12 | 0 | 0 | 252 | 28 | 11.701 |
| primary | prospective | frequency | 8/12 | 0 | 0 | 288 | 28 | 12.513 |
| primary | prospective | loop32 | 8/12 | 0 | 0 | 243 | 28 | 12.615 |
| primary | prospective | loop64 | 8/12 | 0 | 0 | 305 | 28 | 12.915 |
| primary | prospective | flat64 | 8/12 | 0 | 0 | 243 | 28 | 12.827 |
| primary | posthoc_fc13 | native | 14/16 | 0 | 0 | 144 | 44 | 8.310 |
| primary | posthoc_fc13 | frequency | 15/16 | 1 | 0 | 105 | 46 | 8.064 |
| primary | posthoc_fc13 | loop32 | 15/16 | 1 | 0 | 93 | 46 | 8.102 |
| primary | posthoc_fc13 | loop64 | 14/16 | 0 | 0 | 158 | 44 | 8.387 |
| primary | posthoc_fc13 | flat64 | 15/16 | 1 | 0 | 101 | 46 | 8.004 |
| secondary | prospective | native | 8/12 | 0 | 0 | 559 | 28 | 15.111 |
| secondary | prospective | loop64 | 8/12 | 0 | 0 | 612 | 28 | 15.882 |
| secondary | posthoc_fc13 | native | 14/16 | 0 | 0 | 268 | 44 | 8.402 |
| secondary | posthoc_fc13 | loop64 | 14/16 | 0 | 0 | 282 | 44 | 8.658 |

Primary uses 30 seconds, 48 logical evaluations and 8 successful transitions per path. Secondary uses 60 seconds, 128 evaluations and 16 transitions. Both budgets were registered before scoring. Every policy starts a fresh persistent probe process, with startup charged. Final compile and replay are separate processes charged to the same deadline. All sibling goals must close. Actual process counts and logical evaluations are different measures.

**primary/prospective/loop32:** model-only declarations: none. Native-only declarations: none.

**primary/prospective/loop64:** model-only declarations: none. Native-only declarations: none.

**primary/prospective/flat64:** model-only declarations: none. Native-only declarations: none.

**primary/posthoc_fc13/loop32:** model-only declarations: MathCan/Settlement.lean:marked_total_conserved. Native-only declarations: none.

**primary/posthoc_fc13/loop64:** model-only declarations: none. Native-only declarations: none.

**primary/posthoc_fc13/flat64:** model-only declarations: MathCan/Settlement.lean:marked_total_conserved. Native-only declarations: none.

**secondary/prospective/loop64:** model-only declarations: none. Native-only declarations: none.

**secondary/posthoc_fc13/loop64:** model-only declarations: none. Native-only declarations: none.

RAM-02 native coverage on the published FC-13 suite was 6/16. Changes in grammar, backend and budget mean an increase over that historical number is a system-level result, not evidence that a learned ranker beats native search. The matched RAM-03 native rows above determine learned-policy gains.

## Search limits observed

| Budget / suite / policy | Evaluation-budget stops | Pool exhausted | Deadline stops | Path-cap events | Goal-cap events | Multi-goal transitions | Unknown requests | Final compile + replay seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary / posthoc_fc13 / flat64 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 76.452 |
| primary / posthoc_fc13 / frequency | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 77.034 |
| primary / posthoc_fc13 / loop32 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 76.990 |
| primary / posthoc_fc13 / loop64 | 2 | 0 | 0 | 1 | 0 | 6 | 0 | 71.789 |
| primary / posthoc_fc13 / native | 2 | 0 | 0 | 0 | 0 | 4 | 0 | 71.634 |
| primary / prospective / flat64 | 4 | 0 | 0 | 3 | 0 | 5 | 0 | 45.675 |
| primary / prospective / frequency | 4 | 0 | 0 | 3 | 0 | 7 | 0 | 46.162 |
| primary / prospective / loop32 | 4 | 0 | 0 | 4 | 0 | 5 | 0 | 46.168 |
| primary / prospective / loop64 | 4 | 0 | 0 | 4 | 0 | 5 | 0 | 45.686 |
| primary / prospective / native | 4 | 0 | 0 | 0 | 0 | 2 | 0 | 45.299 |
| secondary / posthoc_fc13 / loop64 | 1 | 1 | 0 | 3 | 1 | 17 | 0 | 69.772 |
| secondary / posthoc_fc13 / native | 1 | 1 | 0 | 0 | 0 | 10 | 0 | 69.069 |
| secondary / prospective / loop64 | 3 | 1 | 0 | 0 | 0 | 8 | 0 | 39.868 |
| secondary / prospective / native | 3 | 1 | 0 | 0 | 0 | 6 | 0 | 40.226 |

Stops are per theorem; path/goal caps and transitions are search events and can repeat within one theorem. Unsolved cases are bounded-search failures, not impossibility results. A wall/verification budget stop is recorded without treating an incomplete compile as a proof. Full per-family attempted/progress/branch/reduction counts are in metrics.json; no timeout is used as a fabricated negative label.

## Certified branching example

Looped32's extra primary certificate is `MathCan/Settlement.lean:marked_total_conserved`. Flat64 found the same two-action proof; frequency also certified this declaration. Native exhausted both registered evaluation caps. The result is post-hoc FC-13 coverage, not a prospective or recurrence-specific gain.

```lean
induction actions generalizing s
all_goals simp_all [MathCan.HookState.endpoint]
```

Exact complete-source SHA256: `48ead689106eb70505ff8fdd1483b076dad3e913a3d51257a1c33326de6c2b39`. A fresh complete compile and independent byte-identical replay passed the axiom gate.

Native search independently certified `MathCan/V4Transient.lean:count_nonnegative` with the following complete action sequence, including continuation through multiple goals. The model policies receive the same branching grammar. This demonstrates functioning branch continuation; it is not a learned-only result.

```lean
induction xs
simp_all [MathCan.V4Transient.count]
simp_all [MathCan.V4Transient.count]
simp_all [MathCan.V4Transient.count, MathCan.V4Transient.nz]
omega
```

## Genuine branch supervision

The completed corpus contains 4,973 compiler-observed action transitions from 209 visible states at 27 original TRAIN theorem roots. A fresh Lean process independently replayed every observation before fitting. Acceptance and canonical complete-goal-list results agreed with zero discrepancies.

| Split | Theorems | States | Multi-goal states | Candidate rows | Multi-goal input rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 22 | 167 | 52 | 3939 | 1205 |
| development | 5 | 42 | 15 | 1034 | 375 |

All descendants of five named development theorems remain outside optimization. Exact cross-split state hashes are excluded from training; this run excluded 0 rows. The original 51-state/961-probe corpus was not overwritten or relabeled. New collection starts at theorem roots and never executes the original target reference proof or a teacher prefix.

Outcome labels are failed/unchanged, reduced goal count, other observed progress (including branch creation), and scratch all-goal closure. The auxiliary target is clipped net goal-count decrease. Only current active context/goal, pending conclusions, observed goal count, candidate and family are features. Future states, file/theorem IDs, original commands and action-prefix metadata are excluded and tested. Half of every effective training batch is drawn from actual multi-goal inputs.

## Training, seed stability and selection

Nine fits completed 4,700 updates in 190.893 summed fit seconds, including validation/checkpoint work. All losses remained finite and saved last checkpoints passed exact CPU output reload. The complete last/best optimizer/RNG checkpoints remain in gitignored storage.

| Architecture | Seed | Parameters | Updates run | Selected update | Train selection loss | Dev selection loss | Dev top1 usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loop32 | 17 | 85,061 | 500 | 100 | 0.3756 | 0.8583 | 35/42 |
| loop32 | 29 | 85,061 | 500 | 100 | 0.4213 | 0.8647 | 32/42 |
| loop32 | 43 | 85,061 | 500 | 100 | 0.4371 | 0.9016 | 34/42 |
| loop64 | 17 | 206,981 | 500 | 100 | 0.3678 | 0.9186 | 26/42 |
| loop64 | 29 | 206,981 | 500 | 100 | 0.3883 | 0.9502 | 31/42 |
| loop64 | 43 | 206,981 | 500 | 100 | 0.4074 | 0.8868 | 36/42 |
| flat64 | 17 | 182,085 | 500 | 100 | 0.3532 | 0.9149 | 24/42 |
| flat64 | 29 | 182,085 | 700 | 300 | 0.2157 | 1.0002 | 36/42 |
| flat64 | 43 | 182,085 | 500 | 100 | 0.3986 | 0.8954 | 29/42 |

Selection loss balances single/multi-goal development weighted CE plus 0.2 SmoothL1 goal-delta loss. Training uses AdamW(.002,.01), gradient cap 1.0, microbatch 16/accumulation 4, at most 1,600 updates, validation every 100 and patience 4. Class weights use training counts only. Selected seeds/checkpoints were fixed before any evaluation: loop32 seed 17 update 100, loop64 seed 43 update 100, flat64 seed 43 update 100.

The early selections and worsening later development loss are retained as evidence of overfitting/seed sensitivity. Cached action usability is not theorem certification. Per-class precision/recall, confusion matrices, delta MAE, branch/single losses and all family counts are in receipts/metrics.json.

| Family | Positive training probes | Positive development probes | Selected loop32 dev positive recall | Selected loop64 recall | Flat64 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact_local | 0 | 0 | N/A | N/A | N/A |
| rw_local | 54 | 0 | N/A | N/A | N/A |
| rcases_local | 5 | 0 | N/A | N/A | N/A |
| induction_list | 53 | 27 | 100.0% | 100.0% | 100.0% |
| unfold_simp | 274 | 74 | 91.9% | 89.2% | 95.9% |
| all_simp | 162 | 51 | 76.5% | 72.5% | 80.4% |

Branch inputs and positive induction outcomes now occur in both splits. Coverage is still uneven: exact_local has no positive probes, and development contains no positive rw_local or rcases_local examples. Their unavailable recalls are reported as N/A, not zero or improvement. The fixed collection limits were not enlarged after these gaps were measured.

## Hardware and resource enforcement

Native Windows, RTX 3050 Laptop GPU with 4096 MiB, existing Python 3.13.7/PyTorch 2.6.0+cu124/psutil 7.1.3, two CPU threads. No dependency upgrades, FP16 or torch.compile.

| Calibration device | Timed updates | Seconds | Updates/s |
| --- | ---: | ---: | ---: |
| cpu | 30 | 0.7926 | 37.85 |
| cuda | 30 | 1.5482 | 19.38 |

The identical goal-aware loop64 calibration selected cpu at the frozen 1.15x CUDA speedup threshold. CUDA/CPU ratio was 0.512. This was one short calibration, not a robust repeated hardware speed comparison.

Sampled peak process-tree RSS was 1.164 GiB during training/calibration and 1.303 GiB during proof evaluation. Exact torch CUDA calibration peak was 68.73 MiB. No resource abort occurred.

Caps remained 6 GiB tree RSS and 2.5 GiB allocated VRAM, with 2 GiB available RAM/commit and 512 MiB dedicated GPU headroom. RSS/availability was sampled every 0.1s; nvidia-smi every 2s. Large artifacts and mailbox traffic stayed on D:. Resource peaks are sampled except the separately reported torch allocator peak.

## Certification and repair audit

The final audit checked 196 scored policy/theorem trajectories and 157 successful instances. These correspond to 8 distinct prospective theorems and 15 distinct FC-13 theorems. Duplicate policies/proof variants are not new theorems.

Every accepted result compiled the complete original theorem in its original preceding context, with no ProbeServer or sorry in the release script. Printed dependencies allowed only propext, Classical.choice and Quot.sound. The identical final source bytes were independently recompiled. Deduplicated sources and a model/theorem/script/hash index are under certificates/. Synthetic smoke proofs and original reference oracles are excluded from research coverage.

An initial evaluation attempt was invalidated after a source-scaffolding bug inserted a heartbeat option between some documentation comments and their theorem declarations. This caused valid tactic prefixes to fail final source parsing. The wrapper was repaired and fresh-mailbox and clean probe-exit checks were added. That audit also caught heartbeat usage accumulating across long interactive batches; the outer session now has no cumulative heartbeat cap, and each request independently receives the same 100,000-heartbeat limit and 3-second wall timeout. Every unchanged training observation was then independently re-audited using corrected complete sources and clean process exits. The dataset bytes, all checkpoint weights, seed selections, candidate grammar and registered budgets were kept frozen. evaluation-01 and its logs remain preserved. These implementation repairs occurred after initial prospective outcomes were visible and are explicitly disclosed; no claim of an untouched implementation is made.

The complete primary comparison comes from evaluation-02. Its secondary phase later experienced repeated initial-goal startup timeouts, independently reproduced by synthetic smoke-07. The root cause was not established. Fresh import/session checks and full smoke-08 subsequently passed with unchanged Lean code and timeouts. All failed attempts are preserved. Both policies and all 28 secondary cases were rerun as an entire comparison in evaluation-03-secondary; individual failures were not selectively retried. An explicit assembly receipt verifies matching fingerprints and complete case counts. The final artifact manifest includes both raw component runs and their initial startup failures.

The preserved incomplete secondary phase contains 17 recorded attempts, including 14 initial-state timeouts. These are operationally excluded from the final comparison as a whole, rather than treating successful retries as additional theorems.

Earlier interrupted collections were preserved after fixing Lean isolated streams and transient Windows file-sharing errors. A Python import-path collision was fixed before any evaluation. Neither transport failures nor timeouts were fabricated into negative training labels.

All 17 integrity tests pass. Fault injection additionally reproduced and fixed missing child/log cleanup during a startup resource exception after the scored evaluator had loaded its frozen implementation. No scored resource abort occurred; this exception-path cleanup does not alter successful observations or policies. A final real-Lean smoke checks the published runtime.

## Limits and reproducibility

There are only 27 training theorem roots, five development groups and twelve prospective declarations. Descendant states are correlated, not independent mathematical problems. The native comparator is stronger than RAM-02 because it receives the same new grammar and backend. The study bundles data, representation and search changes; it cannot isolate their causal effects without further ablations. Syntactic signatures are not semantic equivalence. The model sees pending conclusions rather than every sibling context, and token hashing/truncation lose information. Sparse family support, the fixed grammar, goal cap and bounded search remain limitations.

All 18 best/last checkpoints and complete JSONL data, mailbox messages, failures, compiler logs and replays are under D:\Research Engine\RAM-03\LoopedTransformers\experiments\lean_ram03\runs. Checkpoint locations and SHA256s are in receipts/checkpoints.json. The full artifact manifest accompanies the local outputs; compact receipts and accepted sources are tracked in Git. RAM-02 artifacts remain unchanged. README.md and run_windows.ps1 provide reproducible commands. No merge was performed.
