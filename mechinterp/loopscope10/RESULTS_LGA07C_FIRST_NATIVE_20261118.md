# LOOPSCOPE-10C / LGA-07C: global readout-dark donor transfer succeeds at natural dose

**18 November 2026. First preregistered ENGINEERING development result: `DEVELOPMENT_ENGINEERING_POSITIVE`, all three original exposed teachers pass their source-level gates.** This is a frozen original first-run native result, following prospectively registered [LGA07C_GLOBAL_FIBER_PROTOCOL.md](LGA07C_GLOBAL_FIBER_PROTOCOL.md), first committed at `2ee61b288c173514d4714afe0b7a8e2a1f33b361` before any code/31061 source generation or task-outcome inspection.

The independently trained 44,001-parameter TRM teachers 103/211/307 are **the same previously examined fixed models** from LGA-02/LGA-05/LGA-06/LGA-07B. Only the 64 new *source strings* are fresh. Replication across these three frozen teachers does not establish population inference over new model initializations or pretrained LLMs. A source-specific finite gradient tangent is not a proof that the TRM itself computes a scalar objective.

## Background and causal question

[LGA-05](../loopscope08/RESULTS.md) found strong four-pass descent under a single external potential E(h) that was fitted to native activation updates, with no reliable native final-task CE benefit from a negative-energy-gradient edit against matched random directions. [LGA-06](../loopscope09/RESULTS.md) found the same potential's geometric/task-preserving transfer FAIL at twelve passes. [LGA-07B](RESULTS_LGA07B_FIRST_NATIVE_20261118.md) registered a negative donor-target result when editing only the goal token: +0.390625, 0 and +0.585938 percentage points over matched same-fiber random controls in old teacher seeds 103/211/307; even the larger natural goal-token donor patch recovered only 63.9–67.9% of jointly competent donor outcomes.

The new test asks if delayed donor-target information is carried by a **physically whole-state direction that is invisible to the current all-token linear head** and tangent to the frozen external learned scalar energy to first order. The independent alternative target comes from changing one predecessor input digit in a completely legal source string, prior to planning any internal edit.

## Frozen prospective design and source chronology

Original LGA-02 frozen model/SAE bytes SHA256:
- `bundle.pt` `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`
- `train.json` `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`

Original external LGA-05 shared scalar potential bytes SHA256:
- `potentials.pt` `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`
- `fit.json` `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`

The first immutable **source-only pretest lock** was uploaded from [Actions #44283810760](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760) before source generation, with `lock.json` SHA256 `79478f82dc7064650a89f031e15895c472d09d842b37c315641acbc60e42aac9` ([artifact #14418502023](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760/artifacts/14418502023)). It verified all original trained tensor fingerprints, frozen scalar normalization, complete registered protocol and code/workflow hashes; `new_test_generated=false`, `test_labels_read=false`.

Only after that lock, the Stage B first native job opened 64 new independent IID 81-position digit strings from fresh seed **31061**, excluding every previously generated source family and leaving separately reserved 31059 unopened. Base ordered int64 x SHA256 `87e3ef56464ce35b47c71cd25d0a726953ae48136b63786f1d9ccb9df0ae4809`.

Each base supplies exactly eight fixed goal positions j=(0,10,20,30,40,50,60,70), and a source-legal donor changes input `(j-1)%81` from digit d to `d%9+1`. Donor target is the right cyclic shift of its own input, changing exactly the one selected target digit. Thus 64 independent base IDs/teacher, eight correlated legal donor pairs/base, 512 source-goal cases/teacher, 1,536 across the three old frozen models.

## Whole-state energy-tangent physical geometry

At native k=2, independently reach H_recipient and H_donor with two actual original tied-block applications. The source donor difference `U=H_donor−H_recipient` lives in the **whole 81×48 state**, rather than one token. Let `W_head` be the original ten-class linear current-head matrix, `P_H` the physical projector onto its right kernel, and `g=∇E_shared(H_recipient)` the gradient of the full sum of per-token energy outputs.

Compute

```
U_head = U P_H
G_head = g P_H
U_K = U_head - (<U_head,G_head> / ||G_head||_F^2) G_head
```

with the frozen zero-row rule and rank cutoff. For rank-ten W_head and independent energy row, the global fiber has `81(48−10)−1=3077` physical dimensions. The initial current-head linear logits at all 81 tokens are protected to floating-point accuracy, and the energy derivative along U_K is zero to first order. Finite E changes must be measured separately because E is nonlinear.

Primary selected patch `rho U_K/||U_K||_F`, where `rho=||U||_F` is the **entire natural donor state displacement**, is compared with exactly eight independent model/source-seeded isotropic K-null Gaussian vectors at the identical physical L2 radius. All directions were selected from source/model states, trained potential and current head before scoring target labels. Every native edit continues two original tied blocks to the original four-pass final ten-class head.

Prespecified controls: head-only projected donor at same rho, exact natural full-state donor replacement (positive/oracle), goal-token-only natural donor, predecessor+goal pair natural donor, and energy-tangent patch at quarter rho. The random-sham controls are repeated within a source/model, not new source roots.

The first original native receipt preserves all **21,504 actual 81×48 physically applied FP32 edit vectors**, plus 1,536 source donor-delta/energy-gradient pairs, per-arm ten goal logits, all 81 predicted classes, natural whole-state donor full 81×10 logits for parity, physical doses, SHA256, first-order derivative and finite current head/E changes.

## Registered primary result: 3/3 old teacher gates POSITIVE

Every originally registered source-goal case was physically admitted. The engineering gate requires ≥80% clean recipient+donor competence, exact full-state donor-to-clean-donor numerical closure (max logit error ≤1e−4), ≥95% success for that whole-state positive control among jointly competent cases, and ≥15 percentage points correct-donor-target advantage versus the median of eight same-K matched-dose shams with a strictly positive 2,000-draw whole-64-base bootstrap lower bound in at least 2/3 teachers.

| Frozen teacher | Primary selected corrected donor targets | Median same-K sham correct equivalent | Mean ITT selected − sham | 95% whole-base-source bootstrap interval | Jointly clean competent | Exact full-state donor parity |
|---|---:|---:|---:|---|---:|---|
| 103 | **195/512** | 0/512 | **+38.0859 pp** | **[+33.9844,+42.5781] pp** | 495/512 | Max logit difference 0 |
| 211 | **224/512** | 1/512 | **+43.5547 pp** | **[+39.4531,+47.6563] pp** | 494/512 | Max logit difference 0 |
| 307 | **246/512** | 0/512 | **+48.0469 pp** | **[+43.7500,+52.1484] pp** | 502/512 | Max logit difference 0 |

Each full-state naturally reached donor replacement achieved 100% correct donor-goal output **conditional on both clean recipient and donor being competent**. The first native task result is therefore `DEVELOPMENT_ENGINEERING_POSITIVE` with **3/3 old teacher development gates**.

These are descriptive per-sequence bootstrap intervals within one trained model. Their three-seed replication is a development condition; a root-calibrated independent-model inference still requires newly trained roots and an EvalMath variation preflight. The same-K null family is intentionally stringent at a large physical dose, though it is not matched to the donor's future signed task gradient.

## Mechanism controls distinguish spatial support, dose and scalar energy

| Native donor-target rate | Teacher 103 | Teacher 211 | Teacher 307 |
|---|---:|---:|---:|
| Global energy-tangent U_K at natural full-state radius | **38.09%** | **43.75%** | **48.05%** |
| Global head-only U_head at same radius | 38.09% | 43.75% | 48.44% |
| Global energy-tangent at quarter radius | 0.98% | 0.39% | 0.98% |
| Goal-token-only reached-state donor | 66.02% | 63.67% | 63.09% |
| Predecessor+goal two-token reached-state donor | 96.48% | 97.27% | 95.90% |
| Whole naturally reached donor state | 98.24% | 98.24% | 98.44% |

The head-only and energy-tangent binary correct-target outcomes agree on **512/512, 510/512 and 510/512** cases for teachers 103,211,307 respectively. At natural dose, the extra energy-row constraint changes binary correct-digit recovery very little. The median norms ||U_head||/||U|| were 0.457/0.469/0.470, and ||U_K||/||U|| were 0.456/0.469/0.470. Thus the head kernel removes roughly half the donor-state difference; the energy-row projection changes the physical norm only slightly.

The **natural predecessor+goal pair** alone achieves 95.9–97.3% correct donor targets, nearly the whole-state 98.2–98.4% level, whereas the single goal token recovers only 63.1–66.0%. This narrows the missing spatial mechanism to at least the source predecessor/goal interaction in this cyclic-copy task, and supplies the most discriminating basis for another experiment.

The quarter-natural-dose global tangent condition recovers at most 1% of donor targets. As in LGA-07B, a small gradient-aligned state displacement can move logits and cross-entropy without producing exact label correction. The effective dose/future-Jacobian interaction is central.

The current **all-81-token linear head** change under the global tangent edit is about 9×10−6 L2 per case, consistent with head-null physical writes. The nonlinear learned scalar E behaves differently: its finite change at natural radius averaged **+18.36,+13.40,+21.01** units/sequence across the three teachers, despite vanishing first-order energy derivative. The head-only counterparts changed E by +12.11,+9.16,+13.56 on average. The natural full-donor control's mean E changes were approximately +0.135,+0.032,+0.060. Energy monotonicity is a trajectory-conditioned property; the larger tangential off-manifold edits can increase E substantially.

## What was established and what remains open

This native experiment demonstrates a strong **task-relevant delayed head-null physical state channel** in the already-trained tiny cyclic-copy TRM family: the whole-state donor difference has a component currently invisible to the complete linear head yet capable of changing a correct *future* target through genuine tied-block continuation. The matched-dose random K directions almost never recover the independently defined donor target. This is an intervention-based finding tied to actual physical model states; the selected hybrid remains off the natural trajectory after projection.

The evidence is weaker for the extra **energy-tangent** interpretation. Because the head-only edit has essentially the same behavioral efficacy, this experiment does not identify the scalar-energy constraint as a useful computational mechanism, nor validate an intrinsic internal SGD objective. The nonlinear finite E increase demonstrates that first-order energy tangent is not a Lyapunov-preserving finite step at natural donor radius. Unrestricted antisymmetric Q representations of tangent residuals remain algebraic, not causal circulation evidence.

The dominant semantic support is highly local: legal source predecessor+goal token pairs carry most of the whole-state donor future answer. A newly preregistered **two-token predecessor/goal physical nullspace study** with natural and matched fixed radii should isolate this geometry and compare global K, head-only, signed future-task-Jacobian matched nulls and the natural whole-state positive control. Across fresh independent teachers, introduce BPDModel/TiedComponentLinear gauge-transport and retained-physical-write controls rather than assuming this physical hidden-state projector corresponds to learned sparse BPD singleton components.

No result here proves language-model reasoning, larger TRM transfer, Sudoku causal mediation or an internally represented scalar objective. Original LGA-05 POSITIVE, LGA-06 FAIL, LGA-07B NULL and all protected cohorts remain unchanged.

## Immutable original source and receipts

- Prospective first registration: [LGA07C_GLOBAL_FIBER_PROTOCOL.md](LGA07C_GLOBAL_FIBER_PROTOCOL.md) first Git commit `2ee61b288c173514d4714afe0b7a8e2a1f33b361`.
- Executable source release: [LoopedTransformers PR #193](https://github.com/MoralityLabAI/LoopedTransformers/pull/193), one-time first-science merge `b490542bf81beb7e11c98c2e2fa32867f0ff77f7`.
- Source-only pretest lock `lock.json` uncompressed SHA256 `79478f82dc7064650a89f031e15895c472d09d842b37c315641acbc60e42aac9`; [artifact #14418502023](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760/artifacts/14418502023). New seed 31061 remained unopened here.
- FIRST original native science development [Actions #44283810760](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760): original data `full.json` SHA256 **`e48861f1834b7995be3aee1da91d84affd10218d92c54c4dbcead3a8349ce25e`**, `edits_f32.gz` SHA256 `c4969a9651274cb116e4ea5f813253a334c7de722d230becb6c70f086a3d1605`, `source_grad_f32.gz` SHA256 `4facab71876c8d670246aa7c78474b28b66dc1fe376f9498baf19905a68c6f75`. [Raw original archive #14418412541](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760/artifacts/14418412541), ZIP SHA256 `a076131031f58dabe5e33203fd6ff8b7696c7277356874bd9518a6b901aea1cf`.
- Independent **post-result NumPy QA**, [Actions #44287570753](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44287570753), [receipt artifact #14419951660](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44287570753/artifacts/14419951660), `audit.json` SHA256 `376c6036d70857353eebf6352cf94c4b9aa7ceb14cc434918a786541151475b4`: **PASS**, all 21,504 native edited vector hashes and projections, all 1,536 legal source gradients/donor goals, source-cluster 2,000-draw confidence intervals and original 3/3 development verdict. The QA implementation was created after inspecting the first result and is explicitly a supplementary instrument check, not an additional prospective science run.
- **Original first-science standard-library audit PASS** in the original [Actions #44283810760](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760), [artifact #14419069574](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44283810760/artifacts/14419069574), `audit.json` SHA256 `aaaed5f95c9c0241818ffeb5680c0fa7da0283266ddce4c337cc84cfd15ab7fc`. This was the **original independently specified Python-stdlib-only auditor**, completing all 21,504 physical vector checks, legal targets, source SHA, actual head/energy tangent, 64-base paired bootstraps, and original `DEVELOPMENT_ENGINEERING_POSITIVE` verdict, before the post-result QA was prepared. It is independent evidence on the same three exposed historical teacher roots.
