# LSA-02F: CPU pretty-text dosage with an independent native uniform-menu baseline

**Date:** November 23, 2026. **Evidence class:** exploratory Lean-verified CPU development study. **Registered dosage development screen:** FAIL. **Native model competence gate:** BLOCKED_INCOMPETENT_CHECKPOINT. **Code:** [PR #327](https://github.com/MoralityLabAI/LoopedTransformers/pull/327), squash merged as `7a6a596c0c6dc45ffc122f0bfc5a5bc68d56b416`. **Source and experiment records:** [GitHub Actions run 45186851642](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642). No ordinary-test or OOD proof outcomes were opened.

## 1. Source and fixed CPU contract

The new Lean-CGRT theorem suite was frozen using generator seed `20261127`, with independent train, calibration, ordinary-test and OOD family shards. Real Lean 4.19 `lgym` extracted **48 independent train theorem roots and 282 shortest-continuation frames**; every generated training reference proof passed standalone, axiom-free compilation. **40/48** training reference proofs fit the eight-tactic budget. The trained models did not receive generator labels, oracle scripts or proof outcomes at inference.

The source teacher manifest SHA-256 is `912f4bdf0abd716b06640f1a731b20375e33638465a20fdc0bc13dd8ea4222bd`, linked to the frozen suite manifest SHA-256 `24350e15220db9e7f43c295fe7108d2a4e33f75b6e8538eca10feaac48711e67`.

All five input views used **Plan B, width 32, four attention heads, 57,475 parameters, two AB recurrent passes, 384-byte ByteEncoder** and 720 FP32 AdamW updates. The input was the only registered representation dimension. Training seeds `137` and `139` used exactly matched initial parameter tensors and the same root/frame sampler. Both seeds and all five views were fitted **twice**, sequentially on **one GitHub-hosted CPU worker**: **20 fits total**. Every within-worker duplicate passed exact tensor, checkpoint and optimizer-trace equality, using pinned CPU PyTorch 2.5.1, one thread, deterministic algorithms and disabled MKLDNN.

## 2. Bounded public pretty-text dosage

All four `pp` views preserve the same mandatory public goal, remaining budget and candidate hypothesis premise/head grammar. Their only registered difference is the maximum UTF-8 byte allowance for the *public Lean pretty-printed goal tail*, with optional background facts consuming any spare bytes. The historical raw JSON view is an additional anchor.

| Input | Mean training packet length | Model byte limit |
| --- | ---: | ---: |
| Raw canonical JSON | 3,314.48 bytes | Last 384 bytes |
| 0-byte pretty (`pp0`) | 210.97 bytes | Full 384-byte cap |
| 48-byte pretty (`pp48`) | 255.80 bytes | Full 384-byte cap |
| 96-byte pretty (`pp96`) | 298.26 bytes | Full 384-byte cap |
| 160-byte pretty (`pp160`) | 344.38 bytes | Full 384-byte cap |

The byte limits were enforced on every training packet, and the final aggregator verified all mandatory candidate hypothesis types survived each dose. Each `pp` checkpoint records its distinct schema, training teacher SHA and exact seed. The public-only encoder reads no shortest-proof labels or hidden generator metadata when constructing a model input.

## 3. Independent native uniform-menu proof baseline

This study introduces the previously missing **model-free native uniform-menu policy**, separate from the historical deterministic `auto8` policy. It samples public candidate tactics uniformly from the current Lean state after omitting already rejected choices, under an identical eight-tactic check budget. Each root's pseudorandom stream is keyed only by the fixed seed, root, visible packet, remaining budget and public menu; native proof answers and neural logits are excluded.

Three predeclared random seeds `17, 31, 47` generated **96 real Lean episodes** on the same **32 calibration theorem lineages** used for every model. All action transcripts were independently replayed, and every successful terminal script independently compiled in Lean 4.19.

| Uniform random-menu seed | Native standalone-certified proofs |
| --- | ---: |
| 17 | 5/32 |
| 31 | 6/32 |
| 47 | 4/32 |
| **Root-weighted mean over seeds** | **15/96 = 15.625%** |

The stochastic baseline made **zero neural-model calls**. It is a three-stream finite estimate of random-menu proof closure, rather than an analytically exact expected rate.

Compiler binary SHA-256: `92c3d35b5bfaa5e0fea413a775d504cf46cd95e1345df61c2274f76779e7e023`. Native uniform-menu evidence: [run artifact 14823593667](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642/artifacts/14823593667). Uniform episode file SHA-256: `ccf9501eb1a5b5720866667edad15b6b72c6bf1863d6d465d656298c86daf857`.

## 4. Full native Lean proof closure by dosage

Every checkpoint was evaluated on the **same 32 fresh calibration theorem lineages**. The score is successful **whole-proof completion** under the *trained AB* block order, accepted by native Lean kernel verification and independent standalone compilation. Model-free action replay passed for every cell. Independently compiled shortest reference proofs established **23/32** reachable within the eight-check budget; the public-order `auto8` control completed **3/32** on this cohort.

| View | Seed 137 | Seed 139 | Mean native proof closure |
| --- | ---: | ---: | ---: |
| Raw | 2/32 | 6/32 | 12.5000% |
| `pp0` | 6/32 | 5/32 | 17.1875% |
| `pp48` | 5/32 | 6/32 | 17.1875% |
| `pp96` | **7/32** | **7/32** | **21.8750%** |
| `pp160` | 5/32 | 7/32 | 18.7500% |
| Native uniform-menu reference | 5/32, 6/32, 4/32 over its 3 distinct seeds | | **15.6250%** |
| Model-free proof attainable ceiling | 23/32 | 23/32 | 71.8750% |

The observed dose response is **non-monotonic**. The 96-byte allocation performed best at both model seeds, with 160 bytes tying it at seed 139. These are two matched seed observations on the *same* 32 roots.

## 5. Registered developmental contrasts

The primary prespecified contrast was **`pp96` minus `pp0`**. Each theorem root's difference is averaged across the two matched model seeds, followed by a **2,000-draw descriptive root bootstrap** over the 32 unique theorem lineages.

| Paired dosage contrast | Certified proof closure difference | Descriptive 95% root-bootstrap interval |
| --- | ---: | ---: |
| **96 bytes minus 0 bytes (primary)** | **+4.6875 percentage points** | **−6.25 to +17.1875 points** |
| 48 bytes minus 0 bytes | 0.0000 points | −9.375 to +9.375 points |
| 160 bytes minus 0 bytes | +1.5625 points | −6.25 to +9.375 points |
| 96-byte typed versus raw JSON | +9.375 points | 0 to +20.3125 points |

The **predeclared development screen failed**: `pp96` had to gain at least **+5.00 percentage points** over `pp0` and avoid negative gains at either seed. It met the per-seed nonnegative condition, with a total gain of **+4.6875 points**, short by **0.3125 points**.

The descriptive intervals include zero or touch it. Their values are neither independent multi-seed confidence estimates nor confirmatory hypothesis tests; the same 32 theorem lineages are reused at every dosage. A confirmatory model search would need prospectively frozen independent theorem roots and correction for the number of configurations compared.

## 6. Actual native competence disposition

The strongest dosage was `pp96`, closing **7/32 = 21.875%**. The formal pre-gym model competence gate required:

- At least 32 independent eligible calibration roots: **PASS**.
- Trained-order proof closure at least **40%**: **FAIL**, 21.875% observed.
- Native proof closure at least **10 percentage points above independently sampled uniform-menu**: **FAIL**, `pp96` gained **6.25 points** over the 15.625% random baseline.

The registered result is `BLOCKED_INCOMPETENT_CHECKPOINT`; architecture promotion and larger-gym construction remain gated. The uniform control resolves a missing *experimental capability* from LSA-02C/E but its three random streams do not provide a low-variance estimate of exact policy expectation.

## 7. Research implications and next gate

The dose-response curve supports **96 bytes as a useful development setting on this cohort**, with no evidence that a larger pretty-text allocation improves full-proof success monotonically. The remaining **23/32 attainable versus 7/32 completed** gap is substantial. The next CPU research job should increase **decision competence on discriminating public candidate forks**, not increase layer count solely because this representation has a development maximum.

A useful prospective experiment is a *source-state action-selection preflight* with fresh native training and calibration theorem roots: check public candidate-premise retrieval and shortest-continuation rank accuracy at matched equal training steps, include separate native root-seeded uniform selection as the proof baseline, and consider adding independently verified **off-policy states** where the learner's own earlier action deviated from the reference shortest script. Any new training signal must preserve independent Lean certification and keep evaluation roots free of optimizer updates.

The present study opens no locked ordinary/OOD test results and makes no Mathlib transfer claim.

## CI and immutable artifacts

The following GitHub Actions workflows passed on the LSA-02F PR head:

- LSA-02F CPU pretty dosage and native uniform-menu baseline, including 48 compiled source proofs, 20 one-worker fit copies, 96 independently replayed random episodes, ten 32-root neural proof campaigns, model-free replay, and root-paired comparison.
- Existing PPT Gym CPU regression.
- Existing PPT G03 live Lean verification.
- Existing LSA-02C public-state CPU regression.
- Existing LSA-02E typed-plus-pretty CPU regression.

The [main run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642) stores the source, checkpoints, complete native episodes and final paired-root results. Artifacts have **14-day retention**. The principal bundles are:

- [Native source artifact 14823847703](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642/artifacts/14823847703)
- [One-worker exact-fit checkpoints 14823129563](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642/artifacts/14823129563)
- [Uniform-menu native baseline 14823593667](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642/artifacts/14823593667)
- [Root-paired dosage and competence report 14825120911](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186851642/artifacts/14825120911)

Every checkpoint SHA-256, per-theorem result, source receipt and native proof transcript is also bound inside those artifact manifests.

**Code:** [LSA-02F protocol](../docs/lsa02f_pretty_dose_cpu_protocol.md) · [dosage encoder](../research/lean_cgrt_hrm400/ppt_gym/typed_state.py) · [random-menu native checker](../research/lean_cgrt_hrm400/ppt_gym/uniform_baseline.py) · [paired-root competence audit](../research/lean_cgrt_hrm400/ppt_gym/dose_compare.py).
