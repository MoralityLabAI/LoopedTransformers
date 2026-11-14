# RAM-02 measured results | 2026-11-14 (Asia/Taipei)

The local campaign completed all six required recurrent fits, then three flat controls. All nine used finite FP32 losses and passed exact CPU checkpoint reload. Total: 7,100 optimizer updates and 108.406 summed fit seconds, including in-fit validation/checkpoint work.

The post-hoc FC-13 comparison produced 65 independently replayed policy/theorem instances across eleven policies, covering 6 distinct original theorems. Native certified 6/16.

## Hardware and device choice

Native Windows, RTX 3050 Laptop GPU (4096 MiB, driver 610.62), approximately 24 GiB installed system RAM. Python 3.13.7 and existing PyTorch 2.6.0+cu124; two CPU threads. Existing environments were not upgraded. All repositories, dependency launcher, datasets and checkpoints were stored on D: after C: was found nearly full.

| Device | Measured updates | Seconds | Updates/s |
| --- | ---: | ---: | ---: |
| cpu | 40 | 0.6240 | 64.11 |
| cuda | 40 | 1.3775 | 29.04 |

CUDA/CPU speed ratio was 0.453; the frozen 1.15x threshold selected **cpu**. Both trials used the identical 198,148-parameter four-pass model, microbatch 16/accumulation 4, ten warmup and forty measured updates. This is one short calibration, not a repeated hardware benchmark.

Training/calibration peak sampled process-tree RSS: 1.085 GiB. Exact torch CUDA calibration peak: 68.45 MiB; peak sampled dedicated GPU use: 207.0 MiB. Minimum available RAM: 10.39 GiB. Lean build/regression peak tree RSS: 1.215 GiB. No resource abort occurred.

Caps were 6 GiB tree RSS and 2.5 GiB allocated VRAM, with 2 GiB available RAM/commit and 512 MiB dedicated GPU headroom floors. Monitoring includes all child Lean processes. RSS and hardware peaks are sampled; torch separately reports its allocator peak. FP16 and torch.compile were not used.

## Frozen input and RAM-01 reproduction

The original training ZIP and the original RAM-01 model ZIP matched their registered SHA256 values. All 16 RAM-01 checkpoint byte hashes were verified, and all eight original development top1 executable counts reproduced with the existing 2.6 runtime. The 51-state /961-probe corpus and 39-state/766-row training versus 12-state/195-row development split were unchanged. Native cached top1 executable was 3/12; original training-only family frequency was 6/12.

Only actual goal/context text, candidate script and family enter features. No original reference proof command, future action or held-out theorem label enters training. Compiler labels were reused without regeneration or relabeling. The independently recompiled FC-13 source-goal file matched the frozen SHA256 after LF normalization.

## Selected development checkpoints

| Model | Seed | Parameters | Updates run | Best update | Train CE | Dev CE | Cached top1 usable | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loop32 | 17 | 82,692 | 600 | 200 | 0.1100 | 0.2013 | 6/12 | 9.633 |
| loop32 | 29 | 82,692 | 500 | 100 | 0.3066 | 0.3816 | 3/12 | 8.319 |
| loop32 | 43 | 82,692 | 1100 | 700 | 0.0440 | 0.1673 | 7/12 | 17.735 |
| loop64 | 17 | 198,148 | 600 | 200 | 0.0685 | 0.2070 | 7/12 | 11.023 |
| loop64 | 29 | 198,148 | 700 | 300 | 0.0782 | 0.3135 | 8/12 | 13.124 |
| loop64 | 43 | 198,148 | 1100 | 700 | 0.0316 | 0.1275 | 8/12 | 20.288 |
| flat64 | 17 | 173,252 | 700 | 300 | 0.0676 | 0.2241 | 8/12 | 7.931 |
| flat64 | 29 | 173,252 | 700 | 300 | 0.0652 | 0.2875 | 6/12 | 7.926 |
| flat64 | 43 | 173,252 | 1100 | 700 | 0.0352 | 0.1839 | 7/12 | 12.427 |

Each fit used AdamW(.002,.01), clipping 1.0 and effective batch 64. The loss normalizes accumulated weighted CE over the entire effective batch. Maximum 2,000 updates, validation every 100; stop after four validations without a 0.0001 development CE improvement. The best checkpoint is chosen exclusively by development CE. Last and best optimizer/RNG checkpoints are preserved. No all 51-state refit was performed.

Cached usable means the selected fixed probe recorded progress, including branching or scratch closure. It is not a complete proof. Development contains zero positive multi-goal outcomes. Best-checkpoint counts vary substantially across seeds; no NaN/Inf or hardware memory failure occurred. Flat64 has 173,252 parameters and is a useful control, but is not exactly parameter-matched to loop64.

The frozen development grammar has at least one progressing/executable candidate for 8/12 states and a scratch-closing candidate for 5/12. Four states therefore have no observed progressing action in this fixed candidate set.

| Family | Positive train probes | Positive dev probes | Dev positive recall, loop32 seeds 17/29/43 | Dev positive recall, loop64 seeds 17/29/43 | Dev positive recall, flat64 seeds 17/29/43 |
| --- | ---: | ---: | --- | --- | --- |
| exact_local | 1 | 0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A/N/A |
| rw_local | 9 | 4 | 75%/50%/75% | 0%/75%/100% | 0%/75%/100% |
| rcases_local | 1 | 0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A/N/A |
| induction_list | 30 | 0 | N/A/N/A/N/A | N/A/N/A/N/A | N/A/N/A/N/A |

Positive recall here is binary predicted usability; class-wise precision/recall, supports, full confusion matrices, train/dev losses and every seed are in `receipts/metrics.json`. Zero-support recall is N/A in analysis; raw training receipts retain their zero-division=0 convention. Exact and rcases have only one successful training probe each. Development therefore cannot validate generalization for those families or for induction/branching.

## Complete original-theorem certification (post-hoc FC-13)

| Policy | Certified /16 | Search Lean calls | Replay calls | Search seconds | Search+replay seconds/certificate | Model-only vs native | Native-only misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 6 | 102 | 6 | 230.571 | 40.877 | 0 | 0 |
| frequency | 6 | 102 | 6 | 237.609 | 41.990 | 0 | 0 |
| loop32_s17 | 6 | 98 | 6 | 248.505 | 44.165 | 0 | 0 |
| loop32_s29 | 6 | 98 | 6 | 231.326 | 41.118 | 0 | 0 |
| loop32_s43 | 6 | 107 | 6 | 247.257 | 43.637 | 0 | 0 |
| loop64_s17 | 6 | 113 | 6 | 262.610 | 46.284 | 0 | 0 |
| loop64_s29 | 6 | 105 | 6 | 257.579 | 45.520 | 0 | 0 |
| loop64_s43 | 6 | 91 | 6 | 236.115 | 42.074 | 0 | 0 |
| flat64_s17 | 5 | 99 | 5 | 263.443 | 55.675 | 0 | 1 |
| flat64_s29 | 6 | 97 | 6 | 253.159 | 44.891 | 0 | 0 |
| flat64_s43 | 6 | 88 | 6 | 241.291 | 43.549 | 0 | 0 |

| Original theorem | Policies certifying it /11 |
| --- | ---: |
| MathCan/HookMultiTick.lean:before_swap_exact_input | 11 |
| MathCan/HookMultiTick.lean:specified_credit_restores_debit | 11 |
| MathCan/Settlement.lean:certificate_gap_identity | 10 |
| MathCan/Settlement.lean:exact_five_account_partition | 11 |
| MathCan/Settlement.lean:terminal_pool_floor_budget | 11 |
| MathCan/TOEJordan.lean:metric_derivative_vanishes_at_root | 11 |

All policies used the unchanged FC-12 candidate grammar, 20-second search deadline, 12 actual Lean calls including separate final compilation, and at most three successful transitions per path, including closure. All branches must close. No proof completed after the deadline received credit. Deadline termination has small process-kill overhead, recorded in raw trajectories. Fresh independent replay is subsequent verification and cannot improve the score; its time/calls are reported separately. In this run, every accepted instance also remained within 12 calls when its replay call was included.

Every accepted theorem was separately compiled in its original preceding declaration/import context. Printed dependencies allowed only propext, Classical.choice and Quot.sound. The exact final source bytes were freshly replayed and matched SHA256. Accepted sources are deduplicated by byte hash under `certificates/`; `certificates/index.json` maps every model/theorem/script to its certificate. Original source-oracle proofs and synthetic smoke tests are excluded from these counts.

All 16 reference proofs compiled in the separate source oracle. This is an already published FC-13 regression, not a pristine holdout. New theorem families were not evaluated. Comparison timing is one sequential local run per policy; scheduling/cache variation precludes a robust speed claim. Per-instance fit+search+replay costs are also in the JSON receipt. Cost per model-only theorem is undefined when no new theorem was solved.

### Family and branching observations

| Policy | Accepted multigoal transitions | Certified proofs using exact | Using rw | Using rcases | Using induction |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 0 | 0 | 0 | 0 | 0 |
| frequency | 1 | 0 | 0 | 0 | 0 |
| loop32_s17 | 3 | 0 | 1 | 0 | 0 |
| loop32_s29 | 1 | 0 | 0 | 0 | 0 |
| loop32_s43 | 2 | 0 | 0 | 0 | 0 |
| loop64_s17 | 4 | 0 | 0 | 0 | 0 |
| loop64_s29 | 4 | 0 | 0 | 0 | 0 |
| loop64_s43 | 2 | 0 | 0 | 0 | 0 |
| flat64_s17 | 4 | 0 | 0 | 0 | 0 |
| flat64_s29 | 4 | 0 | 0 | 0 | 0 |
| flat64_s43 | 1 | 0 | 0 | 0 | 0 |

The prototype stores complete ordered goal lists, remaining counts, goal hashes and canonical visible-state signatures, and preserves siblings after induction/rcases. The real Lean smoke suite exercises complete constructor and induction branching, plus rcases continuation; the custom-axiom negative compiles successfully and is rejected by the release gate. Unit tests cover partial-branch rejection, deduplication, deadline/final-call accounting and candidate injection.

Training remains the original four-outcome supervision on single-goal contexts. Frozen probe records contain outcome classes and remaining counts, but not complete post-transition branch text. No extra branch-state training labels were fabricated. The extension changes state tracking/search and training/selection duration, not the frozen supervision corpus.

Failures are preserved by theorem and policy with every source/log hash. Hard-deadline failures do not establish grammar impossibility: not every candidate was tested. Multi-goal trajectories that remain open demonstrate branch/budget pressure, while the three-transition cap prevents unrestricted continuation. Frozen cached probes identify states without a usable candidate in the registered grammar; no new outcome labels were generated to repair expressivity.

## Additional recurrent passes versus compute

Inference-only ablation on each selected four-pass checkpoint, using pre-encoded development batches, five warmups and three timing trials of twenty repetitions. CPU forward time excludes tokenization and metric bookkeeping. These are not separately trained one/two/eight-pass models, and no proof-coverage conclusion is drawn from cached scores.

| Model/seed | Passes | Cached top1 usable /12 | Dev CE | Median forward ms /195 rows |
| --- | ---: | ---: | ---: | ---: |
| loop32/17 | 1 | 5 | 0.2165 | 8.055 |
| loop32/17 | 2 | 5 | 0.2068 | 10.528 |
| loop32/17 | 4 | 6 | 0.2013 | 13.171 |
| loop32/17 | 8 | 6 | 0.2032 | 20.772 |
| loop32/29 | 1 | 3 | 0.4322 | 8.641 |
| loop32/29 | 2 | 3 | 0.4096 | 11.848 |
| loop32/29 | 4 | 3 | 0.3816 | 13.605 |
| loop32/29 | 8 | 3 | 0.3520 | 22.387 |
| loop32/43 | 1 | 7 | 0.1582 | 9.009 |
| loop32/43 | 2 | 7 | 0.1604 | 11.124 |
| loop32/43 | 4 | 7 | 0.1673 | 14.508 |
| loop32/43 | 8 | 7 | 0.1645 | 21.780 |
| loop64/17 | 1 | 7 | 0.2161 | 9.608 |
| loop64/17 | 2 | 7 | 0.2127 | 11.874 |
| loop64/17 | 4 | 7 | 0.2070 | 15.200 |
| loop64/17 | 8 | 7 | 0.2025 | 24.140 |
| loop64/29 | 1 | 8 | 0.3209 | 11.267 |
| loop64/29 | 2 | 8 | 0.3164 | 11.158 |
| loop64/29 | 4 | 8 | 0.3135 | 17.122 |
| loop64/29 | 8 | 8 | 0.3062 | 22.783 |
| loop64/43 | 1 | 7 | 0.1685 | 9.825 |
| loop64/43 | 2 | 8 | 0.1501 | 13.541 |
| loop64/43 | 4 | 8 | 0.1275 | 16.919 |
| loop64/43 | 8 | 8 | 0.1143 | 29.775 |

4 versus 2 inference passes: cached top1 improved for 1/6 checkpoints, declined for 0/6, and tied for 5/6. Median forward-cost ratio was 1.266x. These are fixed-checkpoint development diagnostics, not additional certified proof comparisons.

8 versus 4 inference passes: cached top1 improved for 0/6 checkpoints, declined for 0/6, and tied for 6/6. Median forward-cost ratio was 1.583x. These are fixed-checkpoint development diagnostics, not additional certified proof comparisons.

## Preservation and reproducibility

The campaign runs are under `D:\Research Engine\RAM-02\LoopedTransformers\experiments\lean_ram02\runs\`. `campaign-01` contains all 18 best/last checkpoints and complete training receipts; `fc13-regression-01` contains all 176 theorem searches, compiler sources/logs and independent replays. Full hash manifests and checkpoint locations accompany the output deliverables. Large artifacts remain gitignored. Previous FC-12/FC-13/RAM-01 results and checkpoints were not changed.

Training and protocol freeze: local commit 1d86ea5. Search freeze before regression: 0b44a15. Subsequent implementation additions provide memory backoff/recovery, portable checkpoint resolution, evidence audits and reporting; they do not change the recorded training tensors or frozen scoring policy. No memory backoff was needed in the measured campaign. The initial CLI encoding typo was corrected before calibration/fitting and before any checkpoint existed.

Commands and environment setup are in README.md. `reproduce_local.ps1` reuses this machine’s installed environment and creates a new timestamped output tree. `audit_ram01.py`, `smoke_branches.py`, `analyze.py` and `report_results.py` reproduce the audits and summaries. No merge was performed.
