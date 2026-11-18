# LOOPSCOPE-09 / LGA-06: A learned recurrent energy loses its descent and its task meaning beyond four loops

**18 November 2026. Original preregistered scientific verdict: FAIL, zero of three frozen teacher seeds pass the jointly geometric and task-preserving OOD/extrapolation criterion.** First heldout [GitHub Actions #44239311003](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44239311003) passed the original SHA-locked source gate, the previously unseen 31058 evaluation, and an independent Python standard-library audit of every registered bootstrap/physical intervention/curl receipt. [PROTOCOL.md](PROTOCOL.md) was committed first at **31365ac501de34474f9e2a32bb9906dea30900ed**, before generating or scoring source 31058. The executable gym was merged as [PR #181](https://github.com/MoralityLabAI/LoopedTransformers/pull/181), squash **041d67506e01e176b4c7cf3d16635246e0e6d404**. This paper is post-result documentation. It does not retrain or reevaluate the original first heldout science run.

## Motivating observation and source isolation

Earlier [LOOPSCOPE-08/LGA-05](../loopscope08/RESULTS.md) found a source-trained **shared scalar E(h)** with gradient aligned to the native recurrent update F(h)-h for a 44,001-parameter four-pass TRM, and almost universal E(h_k)>E(h_(k+1)) across the four TRAINED application depths, on heldout IID cyclic-copy sequences. That earlier study reported POSITIVE for 3/3 independently initialized teacher seeds under its own predeclared four-depth gate. Its matching native edit did not reliably improve downstream task CE against random, and a source Jacobian skew witness constrained any interpretation as a globally conservative gradient field. LGA-06 asks how that energy behaves when the input distribution changes and the same F block is applied three times as long.

All current teacher weights, train-only SAEs and potential weights are EXACT first-science checkpoints. Teacher seeds 103,211,307 and their original 44,001-parameter 48D weight-tied token-mixing blocks were trained on 81-position one-step cyclic-copy strings with original four applications; no weights change in LGA-06. The scalar observer uses the same frozen 48D train-only normalization and original E_shared, with depth embedding index 0 for every input state including beyond the fourth pass.

Original source SHA256:
- Teacher/SAE bundle: **d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266**, training manifest **7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d**, from first LGA-02 run #44222863870.
- Shared/conditioned/untrained potential bundle **24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53**, fit manifest **69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d**, from first LGA-05 run #44235767614.
- Source-only first LGA-06 pretest/protocol lock **9100fbbac76a6dca5a6a8ea695f8c4e5a257f16ec830eaa0e03e3ab6f7883b06**, protocol SHA256 **7c94433e140be42f552eafd9d4de14531fe6af6c0ce6579baae28a2e8eaea01a**. All four file hashes and all model/dictionary tensor identities were verified before generating new test inputs.

The source-only step ran eight tests on separate 61058 smoke inputs and then uploaded the pretest lock. The subsequent GitHub job alone opened previously unobserved test generator seed **31058**.

## Four matched independent input distributions

Generate 256 independent base 81-position IID strings of digits 1..9 using generator seed 31058. Define four deterministic distributions from EACH ORIGINAL BASE string, preserving its index as a paired 256-source cluster:
- **IID**: original source unchanged, with target the previous cyclic input digit.
- **Period9**: repeat the first nine input digits nine times. The source is exactly periodic with period nine.
- **Block9**: sample nine digits at original positions 0,9,...72 and repeat each for nine consecutive tokens. The source has nine constant-length runs.
- **Zero20**: replace original positions j divisible by five with 0, i.e. **17 of 81 input tokens (20.99%)**. The model has an embedding/head for digit 0 but original recurrent supervised training sampled only digits 1..9. Its cyclic target legitimately includes class 0.

Every family has 256 unique full 81-digit strings and target y=roll(x,+1), with exact source hashes, no duplicates from earlier LGA-01..05 train/validation/test seeds or within the new independent base roster. The four shifted arms are **paired transformations of the same 256 base examples**, not 1,024 independent experimental sources. The independent auditor reconstructs all four transformations exactly from the original archived base-source byte tensor.

All three frozen teachers iterate the SAME existing F block for twelve actual steps h0..h12, with no new ACT learned halt gates, teacher BPTT updates or prospective stopping rules.

## Exact measured geometry and predeclared gates

At each of 12 native transitions, record the signed per-token energy drop under the SAME potential, D_k = [E_shared(h_k)-E_shared(h_(k+1))]/81, the indicator of strict positive drop and the mean tokenwise cosine between F(h_k)-h_k and −grad E_shared(h_k). Compare that cosine to the SAME INITIALIZATION untrained potential and a train-only normalized radial-bowl potential. A depth-conditioned E(h,k) is only reported over original k=0..3, evaluated with the SAME k on each side. Genuine 10-class head CE and accuracy are measured at every step.

The originally registered **H_G geometric gate** requires at least three of four original-depth steps to have positive 95% whole-sequence bootstrap lower bounds for mean energy drop, strictly positive-drop fraction lower >90%, and positive gradient-alignment advantage over both initial and radial potential controls. At least six of eight additional k4..11 steps must retain positive mean energy-drop and >90% positive-drop fraction lower bounds. A separate **H_T task gate** requires 4-pass accuracy >=60%, 12-pass accuracy within five percentage points of the 4-pass accuracy, and 12-pass CE no more than 0.05 nat/token above its 4-pass CE. Whole-study PASS requires BOTH H_G and H_T on ALL four distributions for >=2 of the three frozen teacher seeds.

Every teacher/distribution/step uses exactly **2,000 bootstrap resamples of 256 entire base sequences**, with immutable SHA256-derived LGA06|46058|seed|arm|k|metric resampling roots. These nominal two-sided 95% intervals are unadjusted across the large teacher×arm×depth family. No repeated token or matched distribution transform is counted as an extra independent source.

### Result: the four original steps replicate, the extra eight do not

**Every one of the 12 teacher/distribution cells passed all 4 original-depth geometric steps.** In the eight genuinely extrapolated steps, every cell passed at most three of eight, below the preregistered six-step threshold. Both the geometric-only program and stronger task-preserving program therefore FAIL (0/3 original seeds).

| Input family | Mean early geometric steps passing / 4 | Mean late steps passing / 8 | Mean all-12 energy-descent fraction | Accuracy at 4 passes | Accuracy at 12 passes | CE at 4 passes | CE at 12 passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | **4.00** | 0.67 | 0.0000 | **98.62%** | 10.21% | 0.060 | **4.716** |
| Period9 | **4.00** | 0.00 | 0.0013 | 97.36% | 10.69% | 0.104 | **4.704** |
| Block9 | **4.00** | 2.67 | 0.4766 | 56.60% | 9.63% | 1.544 | **4.568** |
| Zero20 | **4.00** | 0.33 | 0.0000 | 78.42% | 7.88% | 1.268 | **4.858** |

All percentages and geometric pass counts in this table are arithmetic means over the three ORIGINAL frozen teacher seeds, preserving their individual test outcomes.

Per-teacher original/extra geometric successes (early first, late second):

| Teacher seed | IID | Period9 | Block9 | Zero20 | H_G across all arms | Joint registered H_G and H_T |
|---|---|---|---|---|---|---|
| 103 | 4/4, 2/8 | 4/4, 0/8 | 4/4, 3/8 | 4/4, 1/8 | FAIL | FAIL |
| 211 | 4/4, 0/8 | 4/4, 0/8 | 4/4, 3/8 | 4/4, 0/8 | FAIL | FAIL |
| 307 | 4/4, 0/8 | 4/4, 0/8 | 4/4, 2/8 | 4/4, 0/8 | FAIL | FAIL |

The original 4-pass task was already weak on Block9: 53.2%–59.1% across three seeds, below the predeclared 60% task-capability gate. Even on IID and Period9, where 4-pass accuracy is about 97%–99%, continued iteration degrades the learned task readout to roughly chance accuracy by pass 12. Twelve-pass CE of 4.6–4.9 nats/token is substantially worse than the 10-class uniform-prediction CE of ln(10)≈2.303. The model is frequently **confidently wrong** after over-iteration.

### Finite energy can increase even where local gradient alignment is positive

Averaged across the three frozen teachers on new IID inputs, the exact same scalar potential evolves with the native recurrent transition as follows:

| Transition | Mean E_shared(h_k)-E_shared(h_(k+1)) per token | Fraction of sequences with positive energy drop | Accuracy after transition |
|---|---:|---:|---:|
| 0→1 | +14.193 | 100.0% | 46.7% |
| 1→2 | +3.323 | 100.0% | 85.0% |
| 2→3 | +1.116 | 100.0% | 94.9% |
| 3→4 | +0.163 | 100.0% | **98.6%** |
| 4→5 | **−0.108** | 0.5% | 94.5% |
| 5→6 | **−0.160** | 0.0% | 80.2% |
| 6→7 | **−0.148** | 0.4% | 49.7% |
| 7→8 | −0.090 | 7.6% | 25.6% |
| 8→9 | −0.018 | 42.2% | 15.0% |
| 9→10 | +0.027 | 77.6% | 11.3% |
| 10→11 | +0.044 | 84.9% | 10.3% |
| 11→12 | +0.045 | 84.9% | 10.2% |

**Positive energy drop means decreasing E.** The model initially approaches high task accuracy while decreasing E. It then begins **increasing the same E** for roughly five further passes while task performance rapidly deteriorates. Later the scalar resumes decreasing, but the model has reached a very poor task-readout regime. At k4→5 on IID, the mean cosine(v,−grad E) is still **+0.132** while the actual finite energy drop is **−0.108**. An instantaneous first-order alignment test therefore does not guarantee that a full nonlinear recurrent update descends the potential, particularly off the four-pass training trajectories.

On Block9, the energy keeps decreasing on a large fraction of extra-loop source trajectories even as task CE and accuracy deteriorate. Roughly 47.7% of Block9 base sequences have all twelve energy drops positive, but the 12-pass accuracy still collapses to 9.6%. This establishes an observed separation between geometric energy decrease and actual supervised task progress.

## Native physical causality at early and extrapolated depths

On the **FIRST 64 base sources of each distribution and each teacher** (the same matched 64 independent base IDs, four derived input arms), at k=2 and k=6 use eight fixed positions {0,10,20,30,40,50,60,70}. Every source-defined goal was retained. At k=2 the edited state continues two original F blocks to the standard four-pass head. At k=6 it continues six original F blocks to the extrapolated twelve-pass head.

Eight source-only physical edits, all with exact matched L2 radius rho=0.25||native update||, were frozen before task-label scoring: negative shared-potential gradient, negative untrained-potential gradient, negative train-only radial gradient, positive native F update, original k2 TopK4 SAE-active projection, its orthogonal remainder, head-null native-update projection and deterministic isotropic random. The SAME originally source-trained k2 SAE dictionary is deliberately used at k6 for an extrapolated representation stress; rank failure would be retained as ineligible, with no replacement goal.

**All 3×4×2×64×8 = 12,288 original source-goal cases** passed physical SAE/head-null/edit-norm admission, and **98,304 native edited physical arms** were measured. Each edited arm includes the actual full remaining recurrence, all 81 final output-token CEs, full ten-digit selected goal logits and clean competitor, original physical edit SHA, and energy-before/after for the EXTERNAL source edit.

Predeclared causal comparison: negative learned shared energy-gradient edit minus equal-L2 random edit. Positive means the learned edit causes MORE full-task CE loss than random; the 95% paired bootstrap clusters 64 entire base strings at each teacher/arm/depth.

| Depth and final readout | Mean learned-minus-random final CE | Nominal 95% cells with worse learned edit | Nominal cells supporting learned improvement |
|---|---:|---:|---:|
| k2 → final readout 4 | **+0.000408** | **9/12** | **0/12** |
| k6 → final readout 12 | **+0.000894** | **11/12** | **0/12** |

Nominal intervals are descriptive, with many interrelated comparisons and no multiplicity correction. Every one of 24 cell point estimates is either positive or statistically inconclusive with no beneficial confidence-supported sign. The potential-gradient edit has limited task value across the original matched source and extrapolated controls, even where its scalar decreases along the model's own native trajectory.

Relative to the untrained SAME-initialization potential edit, the trained shared-potential edit produced MORE final loss with nominal positive intervals in **11/12 early** and **10/12 extrapolated** comparisons. Relative to simply adding a native update, the shared-potential edit was worse in 3/12 early cells and 10/12 extrapolated cells at nominal 95%. These are comparisons of **external single-token edits** at fixed rho; they do not mechanically reverse the positive gradient/energy descent along the full ORIGINAL unedited recurrent trajectory.

The original SAE-active versus SAE-remainder and head-null versus random controls are fully archived by teacher/distribution/depth. At k2, active-minus-remainder has six nominally positive and three nominally negative source cells, with three inconclusive; at k6 there is one positive and three negative, with eight inconclusive. No uniform identifiable SAE/quotient causal transport across distributions and depths follows.

## Curl stress and integrability limitations

A true Euclidean negative gradient field has symmetric Jacobian wherever twice differentiable. Independently of the scalar critic, the source native vector field v(h)=F(h)-h is tested using two SHA256-seeded orthonormal 81×48 tangent directions u,w per source sequence. FP64 centered finite differences at ±0.001 estimate u·J_v w and w·J_v u. The source-defined normalized absolute skew ratio |u·Jw−w·Ju|/(|u·Jw|+|w·Ju|+1e−8) was recorded for the first 64 base IDs at k2 and k6 across all arms and teacher seeds.

| Frozen input family | Mean native Jacobian skew ratio at k2 | Mean at k6 |
|---|---:|---:|
| IID | 0.766 | 0.783 |
| Period9 | 0.729 | 0.786 |
| Block9 | 0.740 | 0.713 |
| Zero20 | 0.766 | 0.741 |

These are descriptive finite-dimensional random-plane diagnostics; a large normalized ratio can arise when the underlying bilinear terms are individually small, and does not give a population-level global curl norm. The nonzero measured antisymmetric contributions persist even after extrapolating the model to k6, consistent with a recurrent transition that contains nongradient components.

## Interpretation and preregistered verdict

The **stronger transfer hypothesis failed**. A single external scalar energy was a useful local description of native updates for the four depths on which it was trained, and that early alignment transferred to three significant input-structure changes and one unsupported input token. Extrapolating the same frozen recurrent block beyond its trained computation horizon produced energy overshoot, inconsistent monotonicity, and severe task-readout collapse.

This distinguishes four phenomena that can otherwise be conflated: local native vector-field alignment with an external negative gradient; actual finite energy decrease under one original recurrent step; task loss improvement at the current readout; and long-horizon stable task computation. These quantities agree under some source/horizon conditions and diverge sharply after untrained recurrence.

A subsequent prospective study should test a **source-trained stopping controller** against a fixed four-pass baseline on new registered distribution shifts. A strong optimizer claim would also require a genuine state potential whose predicted descent remains tied to task improvement, calibrated step control and an account of native nonconservative drift, preferably under independent larger TRM architectures. No currently measured result proves the frozen original model internally computes the external scalar potential.

## Immutable sources, original run, independent audit

- Original prospective protocol [PROTOCOL.md](PROTOCOL.md), first committed **31365ac501de34474f9e2a32bb9906dea30900ed**, before any source 31058 test data were generated.
- Executable LGA-06 experiment in merged [PR #181](https://github.com/MoralityLabAI/LoopedTransformers/pull/181), squash **041d67506e01e176b4c7cf3d16635246e0e6d404**. Includes original source_lock.py, eval_shift.py, audit_receipt.py, eight source-independent tests and a three-stage GitHub workflow.
- **First original complete [GitHub Actions #44239311003](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44239311003)**: all three jobs green, including eight unit tests, source-lock and novel OOD evaluation, and independent standard-library replay. No neural checkpoint retraining took place.
- [Original pretest source lock artifact #14398922309](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44239311003/artifacts/14398922309). Artifact ZIP SHA256 **e97363e9993b2998d3c07cfbd4a8acf85f7a736bde7df1093e9c0400ee229b43**; uncompressed source_lock.json SHA256 **9100fbbac76a6dca5a6a8ea695f8c4e5a257f16ec830eaa0e03e3ab6f7883b06**.
- [Original full 31058 science artifact #14398628707](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44239311003/artifacts/14398628707), ZIP SHA256 **f7f463315e12d3496d6b1b680e91d54641806177413d35144c881405fc91bf3b**; uncompressed **984,206,050-byte full.json SHA256 364dce0d03f3186afae0e4606a6d7acc057f90eadd94dd70d7d7006da90fdcf6**. Records all 36,864 model/distribution/step/sequence trajectory rows, original source SHA and task labels, all 98,304 physical native edit-arm receipts and all curl tests.
- [Independent source/energy/CE/curl audit artifact #14399246001](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44239311003/artifacts/14399246001), ZIP SHA256 **446237a7bd40747bd43eee64261bc902a3b32c5d1b90ef088a31a656d243c0aa**; uncompressed audit.json SHA256 **f217b52e4219bf75cc9cb2b597ad258cf6e71ed75b448789be6676cb26e27bef**. Independent Python stdlib checker reproduced original source transformations, all per-string mean-energy and CE identities, every original 2,000-draw bootstrap and registered FAIL 0/3, all matched native ten-logit CE/margins and all original Jacobian-skew arithmetic. It does **not** independently run the trained PyTorch F or source potential gradient.

A report-only update must use the above original immutable run and avoid recasting a second evaluation on public fixed seed 31058 as an additional prospective test. Prior LGA-01 through LGA-05 scientific verdicts remain unchanged.
