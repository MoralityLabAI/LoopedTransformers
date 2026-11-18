# LOOPSCOPE-07 / LGA-04: Calibration-only gradient matching improves terminal-loss sensitivity estimates

**Original science-producing date: November 18, 2026. Registered verdict: POSITIVE, 3/3 independent frozen teacher seeds passed.** This paper documents the original three-stage source-locked [GitHub Actions #44232573535](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232573535), which passed original calibration-only paired critic fitting, a new independently generated heldout 256-sequence terminal-gradient and native-causal evaluation, and an independent stdlib source/numerical auditor. The [prospective protocol](PROTOCOL.md) was committed FIRST at be9ac4e1541d6445458552e5665310fa1b2537ef, before calibration fitting or opening the new test source. The executable experiment was merged in [PR #177](https://github.com/MoralityLabAI/LoopedTransformers/pull/177), squash 71cb26727942ea7cf71b4ca424961c8133593df6. This manuscript is post-result documentation; its audit workflow reuses the original immutable artifacts rather than rerunning a model on observed test seeds.

## Motivation: current-head optimization and terminal-loss sensitivity differ

The preceding studies on independently initialized 44,001-parameter four-pass tied TinyRecursiveModel (TRM) cyclic-copy controls yielded three separate results:

- [LGA-01](../loopscope04/RESULTS.md) observed 3/3 model seeds whose recurrent state updates became aligned with negative **current-head** supervised CE gradients after ordinary shared-weight AdamW training. Native same-dose edit controls supported downstream sensitivity of those update directions.
- [LGA-02](../loopscope05/RESULTS.md) found only 1/3 model seeds whose recurrent updates met the registered criterion for alignment with **true final four-pass loss gradients**. The final-loss gradient nevertheless predicted small externally applied physical interventions accurately.
- [LGA-03](../loopscope06/RESULTS.md) fitted an external horizon-conditioned circular-convolution value critic on independent calibration labels, but its gradient matched the true terminal-loss gradient poorly (0/3 seeds), and negative critic-gradient native edits failed to beat equal-dose random controls.

LGA-04 intervenes on exactly one design dimension: the *critic's calibration training objective*. Does separately supervising the critic's derivative on original teacher-terminal-gradient labels allow it to generalize a better estimate of the final loss vector field? This is a result about an **external critic attached to a frozen TRM**, rather than direct evidence that the original weight-tied recursive block executes that critic or performs numerical gradient descent.

## Immutable teachers, new calibration and test chronology

Original teacher weights and six train-only TopK4 SAE dictionaries are the exact FIRST original LGA-02 model+SAE source archive [#14390144806](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870/artifacts/14390144806). Teacher seeds are 103,211,307; all share four actual repeated block applications, 81 token positions, 48D hidden channels and one 10-logit digit readout. These frozen teachers had been trained on original 512 cyclic-copy sequences with seed 31041, validation 31042, and original tests 31043/31044.

The two new critic arms initialize from the exact FIRST frozen LGA-03 value critic [#14393961360](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515/artifacts/14393961360), which was trained on older 31048/31049 calibration data and previously tested only on 31050. LGA-04 keeps the original state normalizer, shared depth embedding, circular Conv1D layers and positive 81-token output. The starting checkpoint and its original teacher-specific trained-critic SHA are identical across the two new training arms.

**Fresh calibration before test**: 256 independent 81-digit cyclic-copy strings seed **31052** for critic training and 64 strings seed **31053** for validation, each disjoint from the six earlier source/calibration/test generators and each other. Each frozen original teacher produces all four hidden states, the actual final per-token CE, and the exact **true terminal gradient of SUM of all 81 final output CEs with respect to each h_k**. These derivatives are permitted on calibration data and cached as teacher-forced targets. The full original source input/target and derivative-cache SHA are archived before testing.

Two exact paired training arms start at the same LGA-03 checkpoint, get identical 100 AdamW updates, source minibatches of 8 whole strings, depth samples, lr .001 and weight decay .0001:

- **value_only_continue**: train on per-token scalar final-CE MSE plus 0.10 mean-final-CE squared error.
- **value_plus_gradient**: same scalar objective plus lambda=1.0 normalized squared error between the gradient of the critic's 81-token predicted CE SUM and the original teacher's exact terminal gradient. Both are divided by one teacher-specific training-only RMS terminal-gradient element scale (minimum 1e-5); the critic receives second-order parameter gradients for the derivative-matching term.

Both undergo all 100 predeclared updates, with step 0/50/100 validation metrics **recorded but not used to pick a checkpoint**. No teacher, SAE or original critic is refitted after new-test source generation. The paired critic checkpoint and its training-only calibration receipt were uploaded as a distinct artifact before the next GitHub job could open the new 31054 heldout sequences.

Validation **normalized teacher-terminal-gradient MSE** at fixed update 100:

| Frozen teacher seed | Value-only continued | Gradient-matched | Relative change | Value-only scalar CE MSE | Gradient-matched scalar CE MSE |
|---|---:|---:|---:|---:|---:|
| 103 | 1.05129 | **1.00825** | −4.1% | 0.07075 | 0.07315 |
| 211 | 0.90084 | **0.81981** | −9.0% | 0.06566 | 0.06893 |
| 307 | 1.11570 | **1.08352** | −2.9% | 0.07565 | 0.07842 |

The gradient objective improved old-calibration derivative fidelity for all seeds, while making scalar CE regression MSE slightly worse in all three. This is a calibrated tradeoff of the registered training objective; generalization is evaluated separately.

The original frozen paired critic bundle SHA256 is **a9f553058f193421861b0d9d464001639bf0ab7be1fc45166820fb0e02486858**, with calibration.json SHA256 **73f803865d2c1a6c7584065a27c47eefc5e54307df6903dbe01cf6afab83920f**.

## Original fresh 256-sequence heldout gradient test

The new evaluation uses **256 independent IID 81-digit cyclic-copy input strings**, generator seed **31054**, with no exact string repetition against old original source, old LGA-03 calibration/test, new LGA-04 calibration, or within its own split. Exact little-endian int64 tensor SHA:

- test inputs: **94282760d573068210f8d4844b1b88c7aa7232b78910e11e0f959ceafda36c6c**
- cyclic target labels: **0fcd85b1b38021a85058720119813d14aafb49579e65725a19fbdf8ffa795460**

For every new test string at k=0..3, the scorer evaluates the true teacher-terminal gradient through the original remaining shared blocks, plus gradients of the frozen gradient-matched, equal-budget value-only and original unmodified LGA-03 critics. The teacher's true digit label enters only as a diagnostic gradient; each critic and its proposed physical edit is computable from source h without that label. The source unit for uncertainty is the **whole 81-token string**, with per-token cosines averaged inside strings before the predeclared 2,000 source-string bootstrap draws.

The registered *paired cosine* criterion requires the gradient-matched critic to improve on BOTH value-only continued and original unchanged LGA-03 critic with 95% source-cluster lower confidence bounds >0, plus a smaller normalized terminal-gradient MSE than the equal-budget value-only critic, at >=2/4 depths, in >=2/3 independently trained teacher seeds.

Actual test **mean cosine(critic gradient, true terminal gradient)**:

| Frozen teacher | k0: matched vs value-only vs old | k1 | k2 | k3 | Registered depths passing |
|---|---|---|---|---|---:|
| Seed 103 | −.01747 vs −.02055 vs −.01587 | −.01715 vs −.02017 vs −.01496 | −.00327 vs −.01024 vs −.00571 | +.00483 vs +.00066 vs +.00003 | **2/4** |
| Seed 211 | −.00645 vs −.01958 vs −.01827 | −.00101 vs −.02073 vs −.02017 | +.00766 vs −.01106 vs −.00976 | +.01082 vs +.00191 vs +.00191 | **4/4** |
| Seed 307 | −.00232 vs −.01370 vs −.01800 | +.00529 vs −.01375 vs −.01534 | +.01005 vs −.00554 vs −.00599 | +.00740 vs +.00297 vs −.00301 | **4/4** |

At seed 103 k0/k1, the matched critic improved versus the equal-budget value-only critic, but failed the preregistered paired lower-bound gain versus the original LGA-03 critic. All other ten seed/depth cells met the simultaneous paired-cosine plus gradient-MSE condition. Original study outcome: **POSITIVE, 3/3 frozen teachers**, with ten of twelve individual registered depths passing.

For example, at seed 211 k1, the 95% whole-sequence paired cosine gain intervals were **[+.01681,+.02246]** versus equal-budget value-only and **[+.01629,+.02195]** versus original LGA-03. At seed 307 k1 they were **[+.01597,+.02200]** and **[+.01785,+.02340]**, respectively. At seed 103 k2, the smaller gains were **[+.00476,+.00911]** and **[+.00026,+.00452]**. All other exact original paired intervals, numerical MSEs and all 256 per-sequence rows are retained in the source-locked full.json and independently replayed by the auditor.

**The absolute matched gradient cosine remains small**, ranging approximately −.017 to +.011. In every one of 12 cells, the true-label **current-head gradient** aligns more closely with the teacher's terminal gradient than this trained auxiliary critic does. Example at seed 211 k3: current-head/terminal cosine +.11657 versus gradient-matched/terminal cosine +.01082. The positive conclusion is a carefully defined *paired improvement* in an external critic, not a finding that an accurate gradient-flow optimizer has emerged in the original model.

## Native matched physical intervention controls

To ask whether the gain in heldout derivative fidelity predicts behavior, the original test scorer used the **first 64 of 256 new heldout strings** per teacher, at k=1 and k=2, with source-fixed token positions {0,10,20,30,40,50,60,70}. This yields **3×2×64×8 = 3,072 source-defined intervention cases**. All 3,072 passed the frozen physical admission. Every true native edit continues the exact original 4−k tied transformer-free recurrent blocks, preserving the full 81-digit final CE and the goal-token 10-logit readout.

Eight label-free physical edit directions were computed from the clean source h and the train-only model/SAE dictionaries: negative gradient-matched critic, negative equal-budget value-only critic, negative original LGA-03 critic, positive native update, equal-L2 source-seeded random, active TopK4 SAE decoder projection, SAE orthogonal remainder and instantaneous head-null projection. A ninth negative TRUE terminal gradient was computed **after the eight label-free physical plans were fixed**, solely for retrospective teacher-forced oracle comparison. All arms share radius rho=.25×||native proposed update||. Each admitted edit vector has its own SHA, recorded physical L2 dose, all 81 native final token CEs, original clean-rival true-label margin and full goal logits.

The primary native endpoint is final **whole-sequence mean CE change**, corresponding to a derivative of the SUM of all final token losses normalized by 81. Each source string contributes the mean of its eight fixed intervention goals before a separately registered 2,000-draw whole-string bootstrap.

Predeclared contrast: **gradient-matched critic edit minus equal-budget value-only critic edit** (negative favors the matched critic):

| Frozen teacher | Native edit at k1, mean global CE difference [95% CI] | At k2 [95% CI] |
|---|---|---|
| 103 | **−.0000811 [−.0001638,−.0000051]** | −.0000409 [−.0001062,+.0000252] |
| 211 | **−.0001139 [−.0001827,−.0000429]** | **−.0001102 [−.0001801,−.0000415]** |
| 307 | **−.0001203 [−.0002593,−.0000061]** | **−.0001244 [−.0002287,−.0000333]** |

**All six** matched-direction point estimates favor the gradient-trained critic, and **five of six** nominal predeclared 95% source-sequence bootstrap intervals exclude zero in that favorable direction. This is separate descriptive causal evidence that fitting the derivative can improve an external steering direction relative to fitting only values under the same optimization budget.

The **equal-L2 random edit** is a stronger reference for absolute causal benefit. Against random, gradient-matched mean native CE changes were:

| Teacher | Matched critic minus random at k1 [95% CI] | At k2 [95% CI] |
|---|---|---|
| 103 | +.0000586 [−.0000648,+.0001885] | +.0000418 [−.0000475,+.0001437] |
| 211 | +.0000007 [−.0000944,+.0000917] | +.0000050 [−.0000685,+.0000772] |
| 307 | −.0000244 [−.0001581,+.0001015] | −.0000167 [−.0001129,+.0000749] |

**All six intervals contain zero**. There is no registered evidence that the gradient-matched action has a reliable *absolute* native final-loss advantage over isotropic random intervention. These are small single-token effects diluted across 81 output positions. No multiple-comparisons-corrected native causal discovery was declared; the tests are secondary with nominal 95% intervals, and correlated token cases remain clustered within 64 strings.

Among the other predeclared physical controls, active SAE minus SAE orthogonal remainder was positive at nominal 95% for seed 103 k1, seed 211 k1, and seed 307 k2, indicating more loss caused by the active edit in those cells. Head-null minus random was inconclusive for all six teacher/depth contrasts. The original earlier LGA-02 and LGA-03 SAE/head-null results remain unchanged.

## Mechanistic interpretation and the next discriminating test

Gradient/Sobolev matching improves **external observer sensitivity** when a critic is trained on actual terminal derivative labels from a separate old calibration corpus, even while scalar loss-prediction MSE deteriorates slightly. The trained critic's gradient becomes more similar to the original teacher's terminal derivative, and its negative edit is less damaging than that of an otherwise identically trained scalar-only critic.

The original recursive transition F was trained **before** this study and remained frozen. The critic is an attached network that never controls F during the original teacher's inference. Consequently LGA-04 does not prove that the original TRM performs SGD on its own activations, uses a learned horizon-conditioned Lyapunov function, or organizes canonical SAE semantic units. The current-head gradient still gives a better absolute terminal sensitivity direction, and the native edit's advantage over random remains uncertain.

An LGA-05 next experiment should compare the gradient-matched external critic directly with a **source-only Jacobian-distilled current-head baseline**, fit on the same old calibration to infer terminal sensitivity from current logits. Freeze any new baseline before registering a fresh seed. A stronger internal-optimizer test would constrain a scalar potential whose *negative gradient predicts the actual native F(h)-h update*, and then use source-disjoint interventions to show task-relevant convergence under that potential. LOOPSCOPE-07 does not yet supply that evidence.

## Exact original science and audit provenance

- Preregistered [PROTOCOL.md](PROTOCOL.md) at commit **be9ac4e1541d6445458552e5665310fa1b2537ef**, before any new calibration or test.
- Executable code in [LOOPSCOPE-07 PR #177](https://github.com/MoralityLabAI/LoopedTransformers/pull/177), squash merge **71cb26727942ea7cf71b4ca424961c8133593df6**.
- First initial [Actions run #44232410472](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232410472) stopped at a syntax error in the auditor before the first paired fit or fresh test, with no scientific data inspected. The correction changes only the Python grouping of the predeclared gradient MSE condition.
- **Original full science-producing Actions #44232573535**, [link](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232573535): all seven CPU regression tests, source-only paired critic training, new heldout true-gradient/native experiment and standard-library independent audit **PASS**.
- Frozen paired critic/source artifact [#14396061237](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232573535/artifacts/14396061237), ZIP SHA256 **3afa407fde8585cef2b3345ec62cfc8abce0bd23b715f9ac864f2213bff608c1**. Uncompressed pair.pt SHA256 **a9f553058f193421861b0d9d464001639bf0ab7be1fc45166820fb0e02486858** and calibration.json SHA256 **73f803865d2c1a6c7584065a27c47eefc5e54307df6903dbe01cf6afab83920f**.
- New 31054 full test artifact [#14395786398](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232573535/artifacts/14395786398), ZIP SHA256 **78a97c470af3abf9ee4bf7603149d5346c08d9deec05171dd45a504dbc522f93**, uncompressed full.json SHA256 **bde2be845ca52c400ac0e753add7e0cb99ca10663c9dd5dd2139cca4e3280ad1**; the full JSON includes all 256-source per-sequence metric rows, all 3,072 original causal case plans and each recorded native edit.
- Independent stdlib audit artifact [#14394879454](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232573535/artifacts/14394879454), ZIP SHA256 **67232aab96b84d9ca69b234a1406915852c5493db71baef7fddc8d6993b5f8da**, audit.json SHA256 **f2f42f5a1425253c738ad6249e15d1a5fa38c434d2712b2399a520165aaa2ac8**. The auditor independently reproduces calibration/heldout source hashes and cyclic labels, gradient MSE/cosine means, paired 2,000-resample whole-sequence intervals, 3/3 scientific gate, original native full-digit logits and global/goal CE/margin differences for **27,648 native edited arms**, and all native paired causal intervals. It does not execute teacher or critic autograd.

Paper/README/audit-only GitHub changes should use a separate workflow pinned to the exact original source run, preserving this first heldout evaluation instead of retraining on an already examined public test seed.
