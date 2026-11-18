# LOOPSCOPE-05 / LGA-02: Recurrent updates follow local readout gradients, not the final-loss gradient

**Original prospective scientific verdict: FAIL, 1/3 source-valid final-only seeds pass the registered terminal-gradient criterion.** The first source-locked run is [Actions #44222863870](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870), whose distinct train-only and fresh-test jobs finished before any outcomes were inspected. Its original immutable 30.97 MB native receipt was independently audited in [Actions #44223836675](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44223836675), status PASS, reproducing the 1/3 result and source arithmetic. A separate full-QA rerun #44223836646 is engineering repetition on already examined test seeds, not another prospective experiment.

The [registered protocol](PROTOCOL.md), first commit 9890b8b, was committed before LGA-02 training, SAE fitting or generating any new heldout sequences. Code merged in [PR #173](https://github.com/MoralityLabAI/LoopedTransformers/pull/173), commit 8cb58bc3ac75a8feb6c60ec2e175fb4ff4b126d8. Original LOOPSCOPE-04 local-gradient POSITIVE and SMC-27 intervention-utility FAIL remain historically unchanged.

## Trained model and fresh test split

The three independent 44,001-parameter tied TinyRecursiveModel(48,4) weights were reproduced using exact LGA-01 source train seed 31041, validation 31042, initialization seeds 103/211/307, 200 AdamW updates and the same 81-position one-step cyclic-copy supervised task. Each original source checkpoint accuracy on old test seed 31043 was reproduced exactly to printed precision (98.7317%, 98.4713%, 98.6642%). This is a newly rerun deterministic weight reconstruction, not an inferred weight match.

Six separate 48→96 TopK4 SAEs were fitted solely to hidden states from the original 512 training and 128 validation strings, one dictionary for each seed and depth k=1,2. All six selected the 150th update by their train-only/validation-only protocol. Val normalized reconstruction MSE ranged 0.271–0.311. Their weights, train normalizers and source checkpoints were uploaded in a frozen artifact BEFORE the separate test job generated its 256 new 81-token strings (seed 31044). The previous LGA-01 test seed 31043 was excluded from new measurements except source checkpoint reproduction.

Fresh test input SHA256: e89160b7fe9e70df964dee6b9babb634e8cab4b3cb55b0230c47aaf5380d8743. Fresh cyclic target SHA256: 4f055bb9e2af78032708f9b7eb1fe7b0ff43307c2cf2ceebb81a29da5bf7bb55. Both hashes use ordered little-endian int64 tensors. The independent auditor reconstructs all target sequences from the original newly saved inputs, verifies the source digests and requires 256 unique new sequences.

## Gradients that answer distinct questions

For physical state h_k and native proposed update v_k=F(h_k)-h_k, the **current-head gradient** is the derivative of the local supervised readout loss CE(Hh_k,y). The **terminal gradient** differentiates the sum of all 81 final readout token losses CE(H[F^(4-k)(h_k)]_j,y_j) through all remaining tied blocks, including cross-token influences. Labels enter both gradient computations as evaluation-only diagnostics. The source update and SAE projection are label-blind. The two gradients answer different causal questions.

Original final-only mean tokenwise cosines, averaged within 256 heldout strings and then across the three independently initialized teacher seeds:

| Native update | Current-head negative-gradient cosine | Terminal negative-gradient cosine |
|---|---:|---:|
| h0→h1 | +0.2680 | +0.0123 |
| h1→h2 | +0.5689 | +0.0041 |
| h2→h3 | +0.5247 | −0.0142 |
| h3→h4 | +0.1725 | −0.0392 |

All three seeds reproduce strong current-head alignment. By contrast, the registered future-gradient-positive seed gate required at least two of four depth-specific 95% WHOLE-SEQUENCE bootstrap lower bounds greater than zero. Per-seed future cosine values with independently replayed 2000-draw 95% intervals:

| Teacher seed | k0 [95% CI] | k1 [95% CI] | k2 [95% CI] | k3 [95% CI] | Registered |
|---|---|---|---|---|---|
| 103 | +.0117 [.0094,.0139] | +.0030 [−.0003,.0060] | −.0161 [−.0195,−.0125] | −.0405 [−.0443,−.0369] | FAIL |
| 211 | +.0144 [.0123,.0164] | +.0066 [.0036,.0094] | −.0122 [−.0155,−.0088] | −.0385 [−.0419,−.0350] | PASS |
| 307 | +.0109 [.0089,.0130] | +.0028 [−.0001,.0056] | −.0144 [−.0177,−.0112] | −.0386 [−.0423,−.0349] | FAIL |

Study-level preregistered result: **FAIL_OR_NOT_ADMITTED, 1/3 positive seeds**. The only prespecified intermediate-supervision arm, deep_020 seed 103, had terminal-gradient cosines −.0029,−.0218,−.0524,−.0840 even as its local gradients remained strongly positive through the first three updates. These are complete measurements, with no model selection after new test results.

The fact that current-head CE decreases over successive loops does not mean a finite update descends the SAME terminal objective. Each intermediate readout is the result of a different number of remaining F applications, and a recurrent computation can improve a readout without being a step of terminal-objective gradient descent.

## Physical head-null and SAE-active geometry

At each new heldout token, the source SVD head rowspace and its complement were evaluated, alongside the train-only SAE decoder span of the original four active features. The SAE span is defined by that model/depth/token's four decoder columns; feature IDs are not compared across independent SAEs.

| Quantity (mean across three seeds, all 256 new strings) | k=1 | k=2 |
|---|---:|---:|
| Squared recurrent-update energy in direct head rowspace | 55.45% | 61.53% |
| Terminal-gradient norm in current head-nullspace | **80.03%** | **75.81%** |
| Squared recurrent-update energy in active SAE decoder span | 28.27% | 36.40% |
| Squared recurrent-update energy in SAE orthogonal remainder | 71.73% | 63.60% |
| Active Top4 feature-support overlap to next loop | 59.7% | 60.8% |

All tested SAE active-span projections passed the source rank/fiber nondegeneracy admission. These facts show that much future output sensitivity is carried in physical directions that are instantly head-invisible, while the small active SAE decoder span contains only about one-third of the recurrent update's squared norm. Because SAE reconstruction MSE is nonzero and dictionaries are fitted separately, the split is a measured projection rather than a canonical semantic decomposition.

## Native equal-L2-dose causal interventions

Using the first 64 new heldout source strings per teacher and eight fixed token positions in each, at k=1 and k=2, the scorer injected one of five source-defined 48D directions at exactly rho=.25*||native update||: original full update, SAE-active projection, SAE-orthogonal remainder, head-null projection or independent seeded isotropic random. It then continued the exact original 4-k native shared blocks. Each source case has the clean baseline, full ten digit logits, per-token correct-label CE/margin and **final 81-token mean CE**. The last endpoint matches the global terminal-gradient definition.

All **3,072** original seed×loop×sequence×goal comparisons passed geometry admission, with **15,360** equally normed edit-arm observations. Independent audit recomputed original readout competitors and margin differences from saved 10 logits, and the reported whole-sequence causal contrast means and 95% cluster bootstrap intervals.

| Seed | SAE-active minus orthogonal k1 final-global-CE change [95% CI] | Same contrast at k2 [95% CI] |
|---|---|---|
| 103 | +0.000067 [−.000112,+.000256] | +.000060 [−.000091,+.000195] |
| 211 | **+.000254 [+.000078,+.000440]** | −.000002 [−.000173,+.000168] |
| 307 | +.000061 [−.000092,+.000231] | +.000082 [−.000038,+.000213] |

Positive means the SAE-active direction caused more loss than the same-dose SAE orthogonal remainder on this task. One seed×depth comparison excludes zero under its unadjusted 95% interval. All six separately registered head-null-minus-random matched contrasts contain zero at 95%. A retrospective 99.58% Bonferroni CI across the twelve named contrasts includes zero even for seed211/k1. Hence the native SAE and head-null causal evidence does not establish a familywise robust positive effect.

The mean full-native-update physical arm raised final GLOBAL 81-token mean CE by about +.000187 at k1 and +.000106 at k2 at this quarter-update physical dose. LOOPSCOPE-04's larger native-full-update intervention reported goal-cell CE under a different dose and endpoint; its original result has not been revised.

## A useful mechanistic separation

Although the trained recurrent transition is nearly orthogonal to the negative terminal gradient, the computed terminal gradient accurately predicts **externally selected local interventions**. Post-result Pearson associations between first-order g_future dot physical edit and actual final GLOBAL CE change, pooled descriptively over 3 seeds × 64 strings × 8 goals per depth:

| Edit direction | k=1 Pearson r | k=2 Pearson r |
|---|---:|---:|
| Full native update | .978 | .986 |
| SAE active | .949 | .979 |
| SAE orthogonal remainder | .987 | .984 |
| Head-null | .984 | .994 |
| Equal-L2 random | .990 | .993 |

These 1,536 repeated goal observations per depth are correlated within 64 original source strings per seed; no independent-token significance claim follows. They show that terminal gradient-based local sensitivity approximations can be numerically predictive even when the model's own recurrent motion does not follow those directions. Learned loop dynamics can implement a useful token-routing computation rather than an explicit optimizer of terminal loss.

## Source receipts and immutable audit

Original source [LOOPSCOPE-05 code PR #173](https://github.com/MoralityLabAI/LoopedTransformers/pull/173) merged at commit 8cb58bc3ac75a8feb6c60ec2e175fb4ff4b126d8. Key source files: [fit_and_freeze.py](fit_and_freeze.py), [terminal_sae_test.py](terminal_sae_test.py), [audit_receipt.py](audit_receipt.py), [eight CPU tests](tests/test_terminal_sae.py).

- [First science-producing Actions #44222863870](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870), train-only and prospective test jobs success, original audit job stopped on too-strict FP32 arithmetic reassociation.
- [First frozen source bundle artifact #14390144806](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870/artifacts/14390144806), ZIP SHA256 218f323881c29dc34dac3a484faba5837c1aa3c002fcbf43e283c2c50b4dee23. Original bundle.pt SHA256 d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266. Source train.json SHA256 7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d.
- [Original new-test artifact #14391502274](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870/artifacts/14391502274), ZIP SHA256 6954a3e26818f33db424b6f5b77bedd9b15c0189b684703596ca144e01d29c03. Raw full.json SHA256 **9986c390798bfdaaf26ffc5089ba2ec8e2f57b941e89b983114a7c137a28d9f7**, 30,971,553 bytes.
- [Independent FIRST original receipt audit #44223836675](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44223836675), PASS. Audit artifact [#14392281078](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44223836675/artifacts/14392281078), original audit.json SHA256 78ace112c8ab7daf6bfa79909c60ab09251a5b48af2a0c143238995c7a2ecce8.
- [Later all-green engineering QA run #44223836646](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44223836646) repeated the original fixed seeds and rounded outcomes, and is not additional independent prospective evidence.

The independent audit permits source FP32-vs-stdlib-double reassociation error below 2e-6 on CE/margin scalar differences. Maximum observed source margin difference was 4.77e-7 and maximum native goal-CE delta difference 5.96e-8. The original native full logits, gradients, model parameter weights, projected activation directions, causal endpoints and science PASS/FAIL thresholds were unchanged.

## Next experiment

The immediate mechanistic question is whether the trained recurrent block performs a **moving-representation algorithmic state update**, whose useful potential differs from the fixed final-head loss. LOOPSCOPE-06 can test a learned horizon-conditioned value function on old train/validation rollouts, compare gradients against genuinely fresh terminal output, and intervene on SAE-support-switching coordinates at matched physical doses. A strong sign of internal optimization would require a potential function that transfers across depth/seed and predicts native remaining-loop causal gains, with independent heldout calibration and negative controls. LGA-02 restricts any claim that high current-head gradient alignment alone establishes internal gradient descent.
