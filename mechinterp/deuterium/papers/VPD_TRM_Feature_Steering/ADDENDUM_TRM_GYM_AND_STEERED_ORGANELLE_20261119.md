# Cross-study addendum: recurrent steering, TRM gym evidence, and a proposed trainable organelle

**19 November 2026 | Source-linked interpretation and prospective research design**

This note accompanies the completed 22-page Deuterium report (`deuterium_complete_20261119.md` and its byte-frozen PDF). It adds independent findings from the BPD depth-steering gym, then proposes a new training study. The Deuterium renderer, scored outputs, original protocols, claim ledger and PDF retain their published hashes. No new scientific outcome is attributed to Deuterium.

## Evidence scope and cross-study ledger

| Source | Observation | Scope |
| --- | --- | --- |
| Deuterium synthetic gate steering | Random-minus-VPD target NLL +0.00088998697 nats, 95% t interval [−0.00016005010, +0.00194002403]; `NO_REGISTERED_ADVANTAGE` | Three synthetic seeds; completed historical/replayed comparison. |
| Deuterium native four-pass recurrent weight | Random-minus-VPD goal NLL −0.00002608101 nats, descriptive 95% t interval [−0.00028737653, +0.00023521451]; `DEVELOPMENT_NO_REGISTERED_ADVANTAGE` | Three previously used teachers and their shared 256-string scored cohort; development evidence. |
| BPD-DT04 selected rank-8 protected state steering | Mean task-Gramian minus eight signed-task-Jacobian shams +0.00026483 exact-board fraction at depth 8 over 59 roots; exactly 0 in the 19-root competent subset | Post-lock analysis; 59 completed, 56 original audit-clean, 16 eligible under every strict gate. Original `INCOMPLETE_OR_INELIGIBLE`, post-lock EvalMath `REDESIGN_OR_PILOT_ONLY`. |
| BPD-DT04 local recurrent geometry | One-step selected-subspace closure defect 0.9904; principal-angle drift 51.77° at depth 8 and 53.65° at depth 12 | Selected rank-8 basis, measured states and tangent map; geometric diagnosis, neither a universal impossibility theorem nor an isolated causal proof. |
| BPD-DT04 shared-operator fidelity | On a postselected set of 19 competent 4×4 Sudoku teachers, 12/24-component factor rollout: 94.90% exact at unseen depth 8 and 95.39% at depth 12 | V/U fit at depths 2,4,6. Masks were selected from the **clean teacher's later evaluation trajectory**, so this is teacher-conditioned replay fidelity, not a deployable autonomous operator or a measured causal gain. |
| BPD-DT04 official 7M TRM bridge | One 16-board checkpoint audit: 16/16 solved at step 4, 9/16 at step 24, 14/16 at step 32. Its matched random dark kick had the same observed early unstick outcome as the SVD dark kick in the separate second checkpoint. | Small, exposed engineering cohort, checkpoint-specific. Motivates first-valid stopping and matched random controls. |

The Deuterium loss and BPD exact-board effects use different architectures, tasks, root populations and units. Keep their intervals, negative results and eligibility conditions separate; this table is a conceptual comparison rather than a meta-analysis.

### Direct scientific interpretation

DT04 preserved immediate 4×4 Sudoku readout probabilities within 4.17×10⁻⁷ while perturbing the protected recurrent state. Its high one-step escape and depth-dependent basis rotation indicate that a fixed low-rank *state direction* can lose alignment during later recurrence. The signed-Jacobian shams reproduced its observed small benefit. A different basis, dose, adaptive control law, or training regime remains an open question.

The 12/24-component result suggests that **operator factors applied at each repeated block** can retain behavior across extra depth. Its future teacher-route information creates an important limit: the deployable hypothesis requires masks to be chosen from the compressed model's own current input. BPD-DT05 implements that online route, first-valid Sudoku stopping, EMA, a RAM teacher cache, and measurement guards, but reports **zero development roots, zero confirmation roots, and no measured CUDA speedup**. Its 70 passing CPU tests validate engineering behavior.

Numerical comparability also matters. Three DT04 roots exceeded the original 36-pass FP32 gauge tolerance without prediction flips. The Deuterium Native-02 full fitting trajectory independently exceeded the frozen 2×10⁻⁵ tolerance while its pointwise parity tests passed. Neither observation identifies the cause of the other's numerical discrepancies. Both counsel full-trajectory invariance checks before asserting mechanistic transfer.

The DT04 hardware probe measured only inference: the 6.83M-parameter official-style TRM at BF16, batch 16, used 61.23 MiB peak allocated and 82 MiB reserved VRAM, with a single measured depth-8 latency of 410.1 ms. The archived host-RSS zeros were invalid; training activations, optimizer state and derivative probes require separately measured budgets. DT05 includes corrected RAM sampling and throughput calibration, not scientific GPU measurements.

## Proposed next experiment: a trainable recurrent control organelle

**Hypothesis.** An online, state-conditioned low-rank operator controller with small persistent memory can improve hard-task solve probability under depth extrapolation and matched compute, after training from scratch. Recurrent *transport* of the controller's effective direction is measurable separately from task improvements.

A minimal implementation retains a shared TRM core Fθ and adds a controller memory mₜ, low-rank basis Bφ, and online mask/action aφ:

- zₜ₊₁ = Fθ(zₜ, x, mₜ) + λₜ Bφ(zₜ, mₜ) aφ(zₜ, mₜ)
- mₜ₊₁ = Gφ(mₜ, zₜ, zₜ₊₁)

The controller observes only current/past model states and public inputs. It may retain 8–16 latent memory coordinates and select k of r shared low-rank operator components per pass. Maintain tied recurrence and log every active component. Fix a physical edit metric and test immediate readout preservation at the actual intervention point when claiming a protected edit; a soft penalty alone is insufficient.

**Mechanistic diagnostic, separate from the deployment policy.** On controlled training/development states compute Jacobian-vector products JₜEₜ, one-step escape, singular-value decay and principal-angle rotation. The exact QR basis orth(JₜEₜ) is an *offline oracle*: assigning it as the next basis trivially gives perfect closure, so its closure metric cannot be reported as learned transport success. A learned online transporter Tφ(zₜ,mₜ,Eₜ) must be scored against the independently computed JVP target on unseen trajectories. Evaluate causal matched-dose interventions separately from this predictive geometry.

**Five matched arms:** (A) shared recurrent core with parameter-matched inert controller capacity, (B) shared fixed-factor sparse operator, (C) input-conditioned sparse operator with no persistent memory, (D) the proposed online sparse operator plus persistent memory, (E) an architecture/parameter/FLOP-matched controller with randomized routing. Add zero-edit, matched delivered-dose random, signed-Jacobian sham, and full-donor headroom conditions as appropriate. Record actual training FLOPs, gradient horizon, physical batch, optimizer state, recurrence depth, stop policy and task-data exposure for every arm. Train candidates under matched budgets and independent model roots. An HRM high/low-frequency architecture is a later matched architectural comparison, rather than an assumed improvement.

**Primary prospective endpoint:** heldout 9×9 Sudoku exact solve rate by a fixed *cumulative inference-FLOP budget*, comparing D with the strongest frozen matched controller. Register one primary horizon/budget, a meaningful positive effect floor, a root-level uncertainty rule and a sham-calibrated causal contrast before opening test inputs. Report raw fixed-depth accuracy *and* the retained first-valid solution policy. First-valid validity checks use public givens and constraints, never gold labels. Secondary outcomes: solve retention after additional loops, causal recovery under equal delivered dose, transport prediction, on-line mask stability, energy/FLOPs, and negative collateral. Use genuinely hard/stalled cases to avoid the near-ceiling competent 4×4 cohort. Later ARC-style tests must use task-program-disjoint heldout rules; public control-world completion is reported separately from hidden-cause inference.

### Silico resource policy

1. **CPU/GitHub Actions, zero Silico RUs:** freeze interfaces, source hashes, case generator, matched-dose samplers, numerical parity checks and independent receipt auditor. Resolve Native-02's numerical divergence on excluded engineering data only, as an independent maintenance task.
2. **Local RTX 5080, zero Silico RUs:** use the existing 242K and official-style 7M gym profiles for JVP probes, online route correctness, first-valid stopping, and measured forward/backward peak VRAM, RSS and cases/s. DT05's test-only outcomes remain separate from new model results.
3. **Capped Silico study only after local admission:** train a 7M-class pilot and a compute-matched control with full training/evaluation trace. Promote to an approximately 27M HRM or comparable looped organelle only after a credible pilot, including sham calibration. Freeze maximum RUs, duration, checkpoint frequency and stop conditions from a measured GPU throughput trial rather than inventing a cost estimate.
4. **200–400M scale:** consider only after the compact pilot supports a reproducible benefit under independent-root evaluation and useful transfer.

Keep prospective roots, harder puzzle cohorts, and other repositories' sealed data untouched until their own source-lock and release conditions are satisfied. A negative pilot is a publishable outcome. Success in static fidelity, online routing, task gain and mechanistic attribution requires distinct receipts.

## DT05 RTX 5080 engineering extension: measured results (19 November 2026)

The BPD [RTX 5080 sprint report](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/SPRINT_REPORT_DT05.md), [immutable JSON manifest](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/ARTIFACT_MANIFEST.json) and pilot receipts give local numerical evidence for experiment selection. All eight DT05 pilot roots were excluded engineering roots 2605900–2605907, and the released Sudoku9 replay cases were exposed engineering cases. The 59 reserved DT05 development roots, DT04 confirmation roots and unrelated seals remain untouched. The original Deuterium PDF retains its byte-frozen completed status.

### Autonomous routing: fidelity gain and an important diagnostic error

Same fitted V/U matrices (24 factors, 12 activated) were compared under online, fixed, seeded random, and privileged teacher-scheduled masks at depth eight:

| Metric | All 8 excluded roots | Postselected competent 3 roots |
| --- | ---: | ---: |
| Dense teacher exact board | 46.6797% | 89.0625% |
| Fixed-factor exact board | 1.5625% | 4.1667% |
| Online-factor exact board | 27.3438% | 58.8542% |
| Privileged teacher-scheduled exact board | 38.4766% | 81.7708% |
| Online minus fixed | **+25.7813 percentage points** | **+54.6875 percentage points** |
| Privileged schedule minus online | +11.1328 pp | +22.9167 pp |

The deployed online mechanism is substantially more faithful than a fixed 12-factor selection on these engineering roots, though the comparison has model-selection and population limitations. Mean per-arm forward times for all roots were 4.450 s online versus 1.711 s fixed, a 2.60× online runtime overhead in the recorded eager implementation. At depth eight, online KL relative to the dense teacher was 0.1201 nats, teacher-scheduled 0.0460 nats, and fixed 0.6104 nats.

**Post-hoc erratum: the reported mask "set overlap" around 0.019–0.024 was computed from identical component IDs at the same *rank positions*, not unordered set membership.** In the [DT05 source](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/causal_instruments.py) the vector \`topk().indices\` was multiplied elementwise after one-hot encoding without summing over selected positions. Reordering the same selected set can produce zero positional agreement. The implementation also divides each match indicator by k *before averaging over k positions*, so the archived figure is the **ordered position-match fraction divided by k**, capped at 1/k even for identical ordered masks. Multiplying the archived values by 12 recovers only the old ordered position-match rate. For independent uniform 12-of-24 selections, expected unordered intersection / 12 is 0.5. [BPD PR #136](https://github.com/MoralityLabAI/BPD/pull/136) contains a forward-code correction and regression test while preserving the original receipt as observed. A separate excluded-root replay must obtain new true set-overlap measurements. The current 11.13/22.92 pp deployment gaps must not be attributed mechanistically to lost **set membership** on the basis of the old overlap field.

### Dynamic transport and behavioral headroom

The eight excluded roots in [07_dynamic_transport_and_evalmath.json](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/07_dynamic_transport_and_evalmath.json) gave a mean **−0.0078125 exact-board fraction** for the JVP-transported task basis versus the static matched comparator, with eight privileged shams and eight Haar controls per root. EvalMath returned \`REDESIGN_OR_PILOT_ONLY\`: small negative pilot effect, insufficient independent sham roots for a 95/95 floor, and low causal headroom. JVPs themselves agreed with small finite perturbations. An accurate local derivative estimator therefore does not establish useful downstream steering.

[10_causal_rank_study.json](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/10_causal_rank_study.json) isolates different notions of effective dimensionality on three excluded roots: full-state statistical participation ratio roughly 21.44→19.73, a linearly decodable operator dimension proxy 14.26→18.33, and a native future-response causal Gramian participation ratio 2.95→4.71 across depths one through eight. Corruption-recovery oracle headroom is approximately zero to 0.02 exact-board fraction and 0.013–0.017 nats KL. The statistic is specific to a probe distribution and horizon. For a convincing steered-training test, deliberately admit unsolved/stalled states with measured donor recovery headroom before evaluating controller benefits.

### First-valid stopping, numerical fidelity, and compute admission

The exposed, solver-verified 64-board Sudoku9 checkpoint replay in [09_first_valid_stopping_7m_and_dt04.json](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/09_first_valid_stopping_7m_and_dt04.json) reports raw depth-16 exact accuracy 60/64 (93.75%) for \`yagizdevre\` and a gold-free first-valid 64/64, retaining four earlier valid solutions. It reports 883/1024 ACT case-steps saved and wall time 16.74→8.20 s. The second \`gaoxin492\` checkpoint already scored 64/64 at fixed depth; first-valid retained 64/64 and saved 893/1024 case-steps, wall time 16.15→5.53 s. Each BF16 compacted replay recorded **one changed trajectory due to batch composition**; preserve raw-policy/compacted-policy disagreement counters and test alternate microbatches before a deterministic equivalence claim. These cases are exposed and cannot establish a population-level gain.

Measured 242K-scale FP32 480-step training on identical microbatch-128 recipes improved 146.30→15.12 seconds (**9.67×**) with max training loss trajectory difference 6.44×10⁻⁶ and no observed depth-eight exact-board delta in that paired engineering comparison. The prominent 245.83→2.61 s (**94.23×**) 120-step benchmark also changed FP32→BF16 and microbatch 16→128: it is a **compound configuration improvement**, not an isolated acceleration from the new implementation. Same-microbatch BF16 480-step experiments showed a **0.661 maximum loss trajectory difference**, flagged as a changed optimization regime. The [A/B receipt](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/06_ab_training_factors_and_probes.json) must govern any parity-sensitive architecture comparison.

The 6.83M BF16 full-step calibrator admitted physical microbatch 32 through activation checkpointing at **3.61 GiB reserved** with **8.19 examples/s** and logical batch 32. As a rough throughput-only extrapolation, 480 updates × 32 cases ≈ 15,360 examples / 8.19 ≈ 1,876 seconds, or approximately 31 minutes *per model arm before evaluation and controller overhead*. This figure is a planning bound from a single measured configuration, **not** a promised runtime. The machine uses an 80W-capped RTX 5080 Laptop GPU with approximately 47.1 GiB system RAM. The sprint observed host available RAM fall below 0.5 GiB during concurrent workloads, which makes fail-closed admission and avoidance of simultaneous large agent jobs essential. Fast batched causal probes can now run locally: 16-case observer VJP 112.31→1.259 s and 7M rank-eight JVP transport 6.41→1.120 s, with the numerical comparisons archived.

### Revision to the organelle research priority

The first experiment should explicitly target the **11.13-pp all-root privileged-to-online routing gap**, or 22.92 pp on the selected competent subset, using self-conditioned masks and persistent controller memory. The teacher-scheduled arm supplies a *nondeployable upper-bound comparator*. Recover corrected set overlap and per-depth divergence before making a claim about why the gap emerges. Train memoryless online, memory-based online, fixed-factor, and parameter/FLOP-matched baseline models on unsolved or stalled examples where the dense teacher has measurable recovery headroom. Keep rank and dose shams in the separately registered causal test.

Retain root-level model inference and route logs in a fresh prospective study. Compare performance by fixed inference-FLOP budget, including policy overhead. Reserve Silico for a capped, measured novel-organelle training run only after locally validated transport, accuracy, causal-headroom and runtime gates.


## Traceable sources

- [Deuterium completed paper](deuterium_complete_20261119.md); [claim ledger](COMPLETION_AND_SILICO.md).
- [BPD-DT04 results, 19 November 2026](https://github.com/MoralityLabAI/BPD/blob/gym/TRM-Sodoku/mechinterp/bpd/experiments/depth_steering_gym/dt04/RESULTS_20261119.md); [original execution lock](https://github.com/MoralityLabAI/BPD/blob/gym/TRM-Sodoku/mechinterp/bpd/experiments/depth_steering_gym/dt04/EXECUTE_LOCK.json); [post-lock 59-root dissection](https://github.com/MoralityLabAI/BPD/blob/gym/TRM-Sodoku/mechinterp/bpd/experiments/depth_steering_gym/dt04/results/POSTLOCK_59ROOT_AND_COMPETENT_DISSECTION.json).
- [BPD-DT05 review and caveats](https://github.com/MoralityLabAI/BPD/blob/gym/TRM-Sodoku/mechinterp/bpd/experiments/depth_steering_gym/dt05/REVIEW_DT04.md); [engineering protocol](https://github.com/MoralityLabAI/BPD/blob/gym/TRM-Sodoku/mechinterp/bpd/experiments/depth_steering_gym/dt05/PROTOCOL_DT05.md); [source-only verification](https://github.com/MoralityLabAI/BPD/blob/gym/TRM-Sodoku/mechinterp/bpd/experiments/depth_steering_gym/dt05/VERIFICATION.json).
- [HRM](https://arxiv.org/abs/2506.21734); [TRM](https://arxiv.org/abs/2510.04871); [What Limits Recursive Reasoning Models (2026)](https://arxiv.org/abs/2609.39967); [Nested Shared-Operator Recursion (2026)](https://arxiv.org/abs/2610.26666); [Interpreting and Steering State-Space Models (ICML 2026)](https://proceedings.mlr.press/v306/mohan26a.html).

**Status:** retrospective cross-study synthesis plus a *proposal*. No Silico run requested or initiated by this file; no new development or confirmation cohort opened.
