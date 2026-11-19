# Deuterium final paper: scope and claim ledger

**19 November 2026.** The final manuscript is [deuterium_complete_20261119.md](deuterium_complete_20261119.md), with a [PDF edition](deuterium_complete_20261119.pdf). It integrates the historical audit and completed VPD comparisons with related BPD recurrent-model observations. The research record ends at the measured outcomes described below. Preparing this edition uses saved receipts and leaves all reserved scientific cohorts unopened.

## Completed claims

| Claim | Source and unit | Result and scope |
| --- | --- | --- |
| The six-seed replication contains 16 accepted grafts | 1,305,171 dependent candidate rows; six model seeds | All accepted rows tie their selected guarded random controls on task metrics. Smaller output movement determines acceptance. The preview retained 12 rows. [Historical audit](../../experiments/deuterium/HISTORICAL_AUDIT.md). |
| Historical analysis reproduces | Hosted audit of saved source records | 38 comparable outputs reproduced byte-for-byte. [Hosted audit](../../HOSTED_REPLAY_AUDIT_20261119.md). |
| Synthetic VPD steering under a matched search | Three model/decomposition seeds; 2,048 held-out examples per seed | Mean random-minus-VPD target NLL +0.0008899869669 nats; 95% t interval [-0.0001600500971, +0.0019400240309]. NO_REGISTERED_ADVANTAGE. [Hosted summary](evidence/hosted_gate/paper_summary.json). |
| Native recurrent-weight steering | Three previously examined teachers sharing 256 scored strings; 17 saved arms | Mean random-minus-VPD goal NLL -0.00002608100808 nats; descriptive 95% t interval [-0.0002873765254, +0.0002352145092]. DEVELOPMENT_NO_REGISTERED_ADVANTAGE. [Verdict](evidence/native01/registered_verdict.json). |
| Reused-layer mask semantics | Four calls to one target module; original fitted importance networks; eight old training strings | The stock cache retains the final input and reuses its mask at all calls. First-versus-final mask differences are 0.27464578, 0.26785272 and 0.29798546 across the fitted networks. [Receipt](evidence/cache_audit/old_train_source_only.json). |
| Adapter pointwise and optimizer checks | Excluded untrained seed 61077 | Pointwise checks pass. The complete 32-warmup/64-main-update trajectory exceeds its fixed 2e-5 tolerance. BLOCKED_UNMATCHED_OPTIMIZER. [Receipt](../../native_trm/adapter02/results/source_schedule_20261119/schedule_receipt.json). |
| DT04 state steering | 59 loaded root receipts; 56 audit-clean; 16 strictly eligible | Original INCOMPLETE_OR_INELIGIBLE. The post-lock all-root mean task-Gramian minus signed-sham effect is +0.0002648305 exact-board fraction; the selected 19-root competent mean is zero. Later REDESIGN_OR_PILOT_ONLY. [Source bundle](evidence/bpd_dt04/provenance.json). |
| DT05 online routing fidelity | Eight excluded engineering roots, 64 boards per root; five arms | At depth eight, online accuracy is 27.34375%, fixed 1.5625%, privileged schedule 38.4765625%, dense teacher 46.6796875% and random 0%. These are descriptive engineering results. [Receipt](evidence/bpd_dt05/08_autonomous_routing_study.json). |
| DT05 transported-state intervention | Eight excluded engineering roots | Mean transported-minus-static effect -0.0078125 exact-board fraction. REDESIGN_OR_PILOT_ONLY. [Receipt](evidence/bpd_dt05/07_dynamic_transport_and_evalmath.json). |
| DT05 causal-rank diagnostic | Three excluded roots; multiple depths within each root | Depthwise donor headroom ranges from -0.0208333 to +0.0208333 exact-board fraction. The future-response spectrum depends on its endpoint and horizon. [Receipt](evidence/bpd_dt05/10_causal_rank_study.json). |
| First-valid Sudoku9 retention | Two existing checkpoints on the same 64 exposed boards | First-valid solves 64/64 in both; terminal depth sixteen solves 60/64 and 64/64. Each BF16 compacted run records one policy disagreement. [Receipt](evidence/bpd_dt05/09_first_valid_stopping_7m_and_dt04.json). |
| FP32 training implementation | One excluded root, matched 480 updates and microbatch 128 | 146.30 to 15.12 seconds; maximum loss difference 6.44e-6. Both exact-board scores are zero. BF16 trajectories are separately labeled. [Receipt](evidence/bpd_dt05/06_ab_training_factors_and_probes.json). |

## Controls, resolution, and exposure

| Study | Physical intervention and controls | Resolution and comparison scope |
| --- | --- | --- |
| Synthetic VPD | Rank-one edit at 1% head Frobenius norm; symmetric signed candidate search, random bank, zero edit, task gradient | One registered primary contrast across three roots. The interval crosses zero. Per-example bootstrap describes variability within a fixed root. |
| Native VPD | Rank-one edit at 1% shared recurrent-weight Frobenius norm; learned and matched random search, sham, clean and task-gradient arms | One registered development contrast across three previously examined teachers. Shared inputs and prior exposure limit generalization. An equivalence margin was not registered. |
| DT04 | Protected rank-eight state intervention; eight signed-task-Jacobian shams per root, plus the archived control family | Registered mechanistic floor 0.10 exact-board fraction. The post-lock contrast falls below the floor; eligibility and selection prevent a prospective positive claim. |
| DT05 transport | Rank-eight task basis transported by JVP from the selected depth to the patch depth; static comparator, eight privileged shams and eight Haar controls per root | Eight actual engineering roots; 128 confirmation roots planned in the source study, zero opened. The pilot is below the 0.10 mechanistic floor and cannot calibrate the required 95/95 sham tail. |
| DT05 routing | Shared 24-component factorization; 12 active factors; online, fixed, privileged teacher schedule and random routes, plus the dense teacher | Five reported arms on eight excluded roots. Competence-selected three-root results are descriptive. A causal floor and confirmatory test family were not calibrated for this engineering routing comparison. |
| DT05 causal rank | Three roots; selected ranks, dose-matched shams, corrupted state and full-donor comparisons | Repeated depths, cases and ranks share roots. Four Gramian cases per root define a local sensitivity spectrum; 32 evaluation cases per root supply the intervention diagnostic. No population-level causal-power claim. |
| First-valid policy and training timings | Public-constraint stopping; full and compacted policies. Training recipes retain precision, batch and update-count labels | Exposed checkpoint and engineering comparisons. Aggregate score agreement leaves the recorded compacted-policy disagreements intact. Throughput ratios apply to their measured configurations. |

The BPD counts and Deuterium counts refer to separate model populations. They are never pooled into a common effect or confidence interval. The current comparison family and all relevant failures are reported beside each outcome. Source files preserve additional arms and diagnostics beyond the quantities summarized in the paper.

## Remaining evidence limits

The recovered 11,191,328-parameter classifier has 91 matching frozen tensors, but its original forward implementation, input encoding and independent reference logits remain unavailable. Its status is BLOCKED_ARCHITECTURE_PROVENANCE; 51 recovered source-task groups remain unscored. These gaps limit the historical claim.

The pass-indexed adapter's 12 reserved teachers and their evaluation sources remain unopened. Its maximum stock-versus-forced-last-repeat discrepancies are 0.0005836487 for loss, 0.0003585815 for parameters and 0.0001150885 for gradients. The failure is part of the completed report. Any future experiment would have its own scientific release gate.

The archived DT05 mask-overlap field measures ordered position-match fraction divided by 12. True set overlap cannot be recovered from that aggregate. The source correction and erratum are preserved, while corrected eight-root diagnostic measurements remain unavailable. Recorded online timing also includes extra route-comparison instrumentation. The paper therefore limits both claims to the quantities that were measured.

## Provenance and rebuild

The [render manifest](deuterium_complete_20261119_manifest.json) records every build input and output hash. The final renderer verifies copied receipts before checking DT05 per-root aggregates and DT04 summary consistency. It then regenerates the figures and PDF. The [cross-study summary](cross_study_summary.json) records the derived values and their evidence scope. These operations read archived outputs and perform no model inference.

The original evidence directories and optimizer trace archive retain their hashes. [The preceding completed edition](archive/closeout_before_integration_20261119/provenance.json) preserves the earlier PDF, manuscript, renderer and companion documents exactly. Its source commit is faddf809f9a85b64c9b36869963df1c480053675. Historical files outside this edition remain subject to their existing inventory checks.

From the repository root:

```bash
python -m pip install -r mechinterp/deuterium/experiments/deuterium/requirements-paper.txt
python -m mechinterp.deuterium.paper_closeout.render --verify-only
python -m mechinterp.deuterium.paper_closeout.render
```
