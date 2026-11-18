# LOOPSCOPE-06 / LGA-03: Value prediction does not recover the terminal gradient of a trained recursive model

**November 18, 2026. Registered scientific verdict: FAIL, 0/3 independently initialized teacher seeds passing the value-gradient fidelity gate.** The complete source-locked three-stage [original GitHub Actions #44228381515](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515) passed seven independent CPU tests, training-only critic calibration, new 256-sequence evaluation, native equal-dose causal interventions and an independent Python standard-library receipt audit. The original preregistration is [PROTOCOL.md](PROTOCOL.md), committed first at e01429a before any LGA-03 model fitting or new test data. Executable code was merged in [PR #175](https://github.com/MoralityLabAI/LoopedTransformers/pull/175), squash commit eef87c058eaefcd2b6fbec5e4723bae9d4b8dee8. This paper is a post-result documentation and source-audit record and does not reselect a model, checkpoint, training seed, or task.

## Question and prerequisite

Earlier [LOOPSCOPE-04/LGA-01](../loopscope04/RESULTS.md) trained a 44,001-parameter weight-tied TinyRecursiveModel(48,4) on a cyclic-copy task and found recurrent activation updates positively aligned with the **negative current-head supervised gradient**, with 3/3 final-only model seeds passing a prospective local alignment criterion. [LOOPSCOPE-05/LGA-02](../loopscope05/RESULTS.md) showed that the same updates were weakly or negatively aligned with the *true four-pass terminal-loss gradient*, with only 1/3 teachers passing its prospectively registered criterion.

LGA-03 tests a more flexible hypothesis: whether a **separately trained horizon-conditioned scalar value potential** can predict the original teacher's future loss surface and recover gradients that align with the exact four-pass terminal derivative. The value network is an external state observer, trained using supervised calibration trajectories from frozen teachers. Even a positive result would provide evidence about recoverable task sensitivity in an observer, with no claim that the original recurrent teacher implements that observer.

The original model/SAE source was locked to the **first science-producing LGA-02 training artifact**, rather than any post-result retraining:
- Original 44K teacher and six SAE checkpoint bundle SHA256 d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266.
- Original LGA-02 train manifest SHA256 7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d.
- Frozen teacher seeds 103,211,307; the original old-source models have exactly four shared recurrent steps and one 10-logit output head; their source checkpoint/SAE tensor hashes were revalidated with the existing LGA-02 loader.

The external value function has no access to any LGA-03 test labels during calibration.

## Chronological calibration and new test split

Train calibration: **768 independent cyclic-copy 81-token strings**, generator seed 31048. Validation: **192 independent strings**, seed 31049. Both exclude all previous LGA teacher training, validation and original heldout sources (generator seeds 31041/42/43/44). Old-calibration state records contain h0/h1/h2/h3 from a frozen native teacher, plus the exact correct-label fourth-pass per-token CE. The original model weights and pre-existing train-only SAEs remain unchanged.

For each frozen teacher seed, one shared horizon-conditioned critic takes the current 81×48 hidden state and an 8D learned loop-depth embedding, through two circular Conv1D/GELU layers (width 96 and 48) and a positive per-token Softplus output. The sum of all 81 predicted final token losses defines the candidate scalar potential. The critic is trained on those 81 output losses using exactly **300 AdamW updates**, batch 16, with fixed per-token Huber loss plus 0.10 times squared error in each sequence mean. All horizons share convolution weights and have separate embedded depth IDs. Select the minimum original-validation objective from fixed checkpoints 0/50/100/150/200/250/300; all three selected step 300. Freeze each model, its same-initialization **untrained** null, source-train normalization and an old-calibration constant-loss predictor, before generating new test data.

| Frozen receiving TRM seed | Critic selected step | Calibration validation token MSE | Frozen constant-mean validation MSE |
|---|---:|---:|---:|
| 103 | 300 | 0.06047 | 0.05950 |
| 211 | 300 | 0.07673 | 0.07757 |
| 307 | 300 | 0.06646 | 0.06590 |

The learned value predictor was only marginally competitive with a constant final-loss estimate. This is an important limitation for its expected gradient quality. The calibrated scalar-loss objective alone did not enforce agreement of the value network's derivative with the original trained teacher's derivative.

**Fresh new test**: 256 IID cyclic-copy sequences with generator seed **31050**. Their inputs, targets and source-disjointness were verified without refitting the value network. Ordered little-endian int64 SHA256:
- New test input: 4b48c5df61f87b30f9e841c74998b335b227c2ac4d8c8a082d0729cfbf709e3e.
- New test target: 9d7265640101f58c9eae2241caa689e1105aed1982f212989dc6435dd3f4eece.
- Value train x: 6338b29bc973783f79a3a37f54356f2b584162fd51401382327687857f2d53ea.
- Value validation x: b1ac9b905b81718a0acf9363c5c88140f3f094bfc33e18abe2608c12e08164b9.

All source/test groups use entire 81-token strings as independent bootstrap clusters; the 81 token positions are correlated measurements within a string.

## The measured gradients

At each current hidden state h_k and remaining-loop horizon 4−k, the **oracle terminal gradient** is the derivative of the SUM of all 81 correct-digit fourth-pass cross entropies after applying the genuine remaining tied F blocks. The local-head gradient is the current direct 10-logit head CE derivative. For the fitted and same-initialization untrained value critics, differentiate the SUM of their 81 predicted future token losses with respect to the same physical h_k. All derivatives are measured against the original frozen 48D model state. No heldout correct digit enters the fitted critic forward or intervention choice.

The registered primary statistic is the 256-sequence mean of the per-token cosine between the fitted value gradient and the true terminal gradient, with two paired tests: improvement versus an untrained same-architecture critic and improvement versus the original current-head gradient. The gate also required better heldout loss prediction than the constant calibration-loss baseline, at least two of four depths per seed, replicated by at least two of three independent trained model seeds. Each paired confidence interval was computed with exactly **2,000 source-sequence bootstrap draws**, SHA-locked to the original registered seed.

Actual source-frozen heldout **mean gradient cosine** with the oracle terminal derivative:

| Teacher seed | h0 | h1 | h2 | h3 |
|---|---:|---:|---:|---:|
| 103: fitted value | **−0.0155** | −0.0147 | −0.0051 | +0.0000 |
| 211: fitted value | −0.0210 | −0.0227 | −0.0101 | +0.0032 |
| 307: fitted value | −0.0182 | −0.0165 | −0.0069 | −0.0020 |
| Mean fitted value | −0.0182 | −0.0180 | −0.0074 | +0.0004 |
| Mean local current-head gradient | +0.0115 | +0.0183 | +0.0412 | +0.1124 |

Every one of the 12 fitted-value-vs-current-head gradient cosine gains was **negative**. At h0 and h1, the trained value gradient performed worse even than its same-initialization untrained critic in all three teacher seeds, with negative 95% paired bootstrap intervals for those trained-minus-untrained contrasts. The slight positive fitted-vs-untrained point estimate for seed 211 at h3 is a tiny effect and remains worse than the current-head gradient.

**No seed passed** the registered requirement of two depths with positive lower bounds versus both controls and improved scalar prediction MSE. Registered verdict **FAIL, 0/3**, on a fully populated test sample with the correct original model weights, calibration order and test split.

Fitted value predictions were also scarcely better than the constant baseline on heldout per-token CE MSE. At seed 103 the value model was better than constant on h1 only (0.06727 vs 0.06763), worse at h0/h2/h3; seed 211's value MSE beat constant at h1/h2/h3 by small amounts; seed 307 at h3 only. This helps explain the negative derivative result without requiring a claim that the recurrent teacher lacks a meaningful future loss function.

## Real native physical causal effects

A second frozen native experiment used the FIRST 64 new test strings per teacher at h1 and h2 and eight original fixed token positions each. For every original source goal, the direction planning used the clean model h, its already fitted value critic, its original train-only SAE and its output-head nullspace, without consulting any heldout answer:

- Negative fitted value gradient (an externally learned proposed descent action)
- Equal-L2 isotropic random edit, source-seeded
- Positive original native proposed update
- Current active SAE decoder span projection of the native update
- SAE-orthogonal remainder
- Instantaneous output-head null projection

An additional **negative oracle terminal-gradient** arm was evaluated as a teacher-forced, label-dependent diagnostic only, with no permission to select it as a deployable value policy. Every admitted arm had physical radius rho=0.25 times the norm of the source native update at that token. All **3×2×64×8 = 3,072** original goal-source cases were physically admitted under the registered checks. Every original 81-token final CE, goal full 10 logits, native CE change and 48D edit SHA was archived. The entire original remaining 4−k shared recurrence was run after each physical edit.

Predeclared **whole-sequence final CE** paired contrasts for negative fitted-value-gradient edit versus equal-norm random control:

| Teacher | Depth h1: gain vs random [95% CI] | Depth h2: gain vs random [95% CI] |
|---|---|---|
| 103 | −0.000009 [−0.000114,+0.000097] | +0.000052 [−0.000015,+0.000120] |
| 211 | **+0.000125 [+0.000027,+0.000236]** | **+0.000088 [+0.000013,+0.000166]** |
| 307 | **+0.000085 [+0.000008,+0.000168]** | **+0.000070 [+0.000016,+0.000130]** |

The sign is **value-gradient edit minus random** final 81-token mean CE change. Negative would favor the externally learned critic direction. Four of six nominal per-seed/per-depth 95% intervals are wholly **positive**, meaning the fitted-value edit caused MORE final loss than equal-dose random on those source/test cases. None excludes zero in the beneficial direction. These secondary comparisons are not a familywise discovery; 95% cluster bounds are descriptive over 64 independently sampled strings per trained teacher.

The matched original SAE-active minus SAE-remainder CE contrasts were unadjusted-positive in 2/6 seed/depth cells (seed103 h2 and seed211 h2), with 4/6 inconclusive. Head-null versus random showed no nominally supported effects in any of six comparisons. A value edit at h1 for seed103 was more favorable than simply adding the native update (−0.000192 [−0.000388,−0.000011]), but still had no supported benefit against random; the definition of a useful value-based direction depends on its physical control.

At these small per-token physical doses, the true terminal gradient's first-order prediction and the observed global CE difference are separately recorded for each matched arm. The critic's own gradient prediction was also archived before judging actual native change. These receipts can test local approximation error without claiming that low source value-prediction MSE implies derivative accuracy.

## Interpretation

The experiment provides a second independent restriction on interpreting the LGA-01 local-loss alignment as an internal optimizer. In LGA-03, an additional horizon-conditioned critic trained on calibration labels and frozen before test performed poorly as a terminal-gradient estimator. Moreover, physical interventions proposed by descending that critic did not reliably help the original teacher's final task loss.

The result is specific to the **value-only Huber+mean-loss objective**, lightweight local circular-convolution architecture, 768 calibration strings and small 44K tied recurrent model. Scalar value fitting is known to leave derivatives underconstrained on a high-dimensional manifold; low prediction loss is not a gradient matching certificate. A critic trained with **independently registered Jacobian/Sobolev gradient matching on old calibration only**, or with native intervention consistency under frozen physical controls, could recover better future directions on a truly new heldout split. Such a study must explicitly avoid using LGA-03's observed seed 31050 for critic or hyperparameter selection.

Even a successful value critic would be an **external approximation**, not direct evidence that original F applies the gradient of that potential. A stronger internal-optimization claim needs mechanistically identifiable energy/Lyapunov structure or intervention-mediated convergence in the original F dynamics.

## Immutable original science and audit

- Original code/registration [LOOPSCOPE-06 PR #175](https://github.com/MoralityLabAI/LoopedTransformers/pull/175), merged commit eef87c058eaefcd2b6fbec5e4723bae9d4b8dee8.
- Original full [GitHub Actions #44228381515](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515), source-only seven regression tests PASS, calibration-only freeze PASS, new 256-sequence native evaluation PASS, independent Python stdlib receipt audit PASS.
- Calibration-frozen learned critic artifact [#14393961360](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515/artifacts/14393961360), ZIP SHA256 217393f9367561793a4011a04d0964be0f2b7fd1da8113478661e9413c245cb5. Original value.pt SHA256 **4d1cb63d301f6cc1b1c90fea999a82de0fb4f077b1f1f2e3bb6f0047aead937a**, calibration.json SHA256 **267ffabb76e3a9fd7e42cf10b4eb31268d635f3bfe67a9c875ef461437ab189d**.
- Original fresh 31050 test/causal artifact [#14393479001](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515/artifacts/14393479001), ZIP SHA256 b077befa04bd3d029cdb60e9b8b6bc692ddca334089220e4ad5d7b649d2a2265, original full.json SHA256 **7418528839c681ffdf019b231650847f561377dd8e7afa3759641f09b9bee5fa**.
- Independent standard-library audit artifact [#14394326328](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515/artifacts/14394326328), ZIP SHA256 4afccef87d9b295672e41586c222c5bea0b8beffa4963a671ce3eb93dc4f714e. Original audit.json SHA256 **6b08fa03ef4982fd51b0c302d63ec83baf236bdd5ad0e2a533ed89ee039d291a**. Reproduced source/test cyclic-copy identities, 256-string gradient means and 2,000 paired resamples at all four depths, and actual full-sequence CE/goal-logit arithmetic in 3,072 admitted source intervention cases. Its reported scientific status was FAIL, 0/3 seeds.

The original first CI attempt [#44228179946](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228179946) stopped at a unit-test helper import before original teacher download, critic fitting or test generation. [ENGINEERING_AMENDMENT.md](ENGINEERING_AMENDMENT.md) preserves that import-only failure. Subsequent results-document changes do not trigger a new test on the already observed 31050 split.
