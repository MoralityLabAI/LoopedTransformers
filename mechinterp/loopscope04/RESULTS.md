# LOOPSCOPE-04 / LGA-01: Supervised TRM activations acquire gradient-aligned update dynamics

**Measured 18 November 2026 Manila time. Registered result: POSITIVE on 3/3 final-only seeds, under the original local-descent classification.** Full source CPU run [GitHub Actions #44142413468](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44142413468) completed all nine regression tests, explicit ten-step pilot, and the 3-seed × 2-supervision-arm × 200-update prospective experiment. The source-locked [PROTOCOL.md](PROTOCOL.md) was committed at 81ab12d before training/test data. This manuscript is a post-result report, with no changes to source experiments or registered hypotheses.

## Scientific question

A weight-tied recurrent model is trained by AdamW on its parameter weights. At inference, the same reasoning block advances its hidden activation state four times. Does supervised learning make those activation updates locally point in the direction that reduces the supervised current-head loss? How does intermediate-loop supervision change the recurrence, and do the activation directions have a measurable native causal effect after later loops?

The model is an MIT source-attributed independent implementation of the original 44,001-parameter 48D four-step TinyRecursiveModel used in the related Compactification-Theory research. It has 81 tokens, one shared LayerNorm residual token/channel-mixing block, and one shared ten-class digit head. **This is an independently initialized TRM-inspired supervised routing control**, not an experiment on pretrained 7M attention TRM Sudoku checkpoints, Nanbeige, or an SAE feature decomposition.

The transparent task maps 81 IID input digits in {1,...,9} to their one-position cyclic shift. Train split: 512 independent strings; validation: 128; heldout test: 256. Target at j is input at j−1 modulo 81. All split RNG seeds, model initialization seeds 103/211/307, training budget, two objectives and inference gates were frozen before the run. Each entire 81-token sequence is one statistical cluster. Test inputs and targets have exact raw tensor byte fingerprints in the uploaded source JSON; the new source-independent auditor replays receipt arithmetic.

Two paired objectives have the same forward four recurrent passes, initial parameter seed, mini-batch schedule, optimizer and 200 full-BPTT AdamW updates:

- Final-only objective: cross entropy at the fourth output head.
- Intermediate objective: fourth-pass CE plus weight (0.20/3) on the CE at each of the first three readouts.

There is no validation-selected checkpoint. Model states are observed at optimizer steps 0,50,100,200.

## Three distinct ledgers

At a fixed trained checkpoint let h_k be the per-token state after k recurrent blocks, and let delta_k=F(h_k)−h_k. The local supervised head loss is L_k=CE(head(h_k),true target) and its current-state gradient is g_k=(softmax(head(h_k))−onehot(target))W_head. The per-token cosine of delta_k and −g_k, the first-order predicted loss improvement −g_k·delta_k, and the ACTUAL native head loss change L_k−L_(k+1) are separately recorded. These are not full BPTT-through-future gradients.

A separate optimizer ledger captures the TRUE AdamW parameter update Δtheta and pre-update training BPTT gradient G, reporting cos(Δtheta,−G) at optimizer steps 50,100,200. The recurrent activation step is not an explicit parameter optimizer.

The exact rowspace of the learned 10×48 output head, measured by FP64 SVD, separates each state update into current-head-visible and instantaneous head-null physical components. The latter may influence later native outputs after the next shared recurrence.

## Original heldout learning and loop-by-loop alignment

The three paired model seeds were evaluated on the same 256 unseen source strings. Average heldout token accuracy across the three seeds:

| Supervision | Checkpoint | Loop 0 | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
|---|---:|---:|---:|---:|---:|---:|
| Final-only | 0 | 10.19% | 10.07% | 9.94% | 9.94% | 9.99% |
| Final-only | 50 | 9.76% | 12.21% | 15.83% | 19.85% | 22.37% |
| Final-only | 100 | 10.08% | 39.73% | 74.81% | 82.82% | 85.05% |
| Final-only | 200 | 10.60% | 46.60% | 85.14% | 95.01% | **98.62%** |
| Intermediate 0.20 | 200 | 10.55% | **74.84%** | **89.73%** | **96.49%** | 98.16% |

The task admission was specified as final-readout heldout accuracy >20%, safely above the 1/9 random digit floor; all six trained models passed. The intermediate objective advances correct outputs to earlier loops, while final-only training slightly improves fourth-loop accuracy.

The registered primary activation-gradient statistic is the mean, across all 256 heldout strings, of each string's mean of 81 tokenwise cosines. At training step 200, average across the three seeds:

| Transition | Final-only mean cosine | Intermediate-supervised mean cosine | Actual final-only per-token CE gain | Intermediate per-token CE gain |
|---|---:|---:|---:|---:|
| h0→h1 | +0.2670 | +0.3623 | +1.2868 | +1.7655 |
| h1→h2 | **+0.5673** | **+0.6110** | +1.1490 | +0.8271 |
| h2→h3 | +0.5246 | +0.4019 | +0.4662 | +0.2735 |
| h3→h4 | +0.1736 | **−0.0154** | +0.1060 | +0.0391 |

All three final-only seeds have a strictly positive **256-string bootstrap 95% lower bound** on all four loop cosines, with all four cosines higher than their initial step-zero controls. The seed-level registered criterion required at least two positive lower bounds plus final accuracy >20%; **3/3 final-only seeds passed**.

Per-seed original step-200 results, with cosine values ordered by k=0,1,2,3:

| Seed | Training objective | Final accuracy | Four loopwise cosine values | Registered seed gate |
|---|---|---:|---|---|
| 103 | Final-only | 98.73% | +0.273, +0.568, +0.529, +0.176 | PASS |
| 211 | Final-only | 98.47% | +0.279, +0.570, +0.515, +0.166 | PASS |
| 307 | Final-only | 98.66% | +0.248, +0.564, +0.530, +0.179 | PASS |
| 103 | Intermediate | 98.22% | +0.375, +0.615, +0.398, −0.027 | Ancillary |
| 211 | Intermediate | 98.08% | +0.359, +0.607, +0.405, −0.004 | Ancillary |
| 307 | Intermediate | 98.19% | +0.353, +0.612, +0.403, −0.015 | Ancillary |

The intermediate-supervision arm makes the early state updates more closely aligned with local negative gradients while changing the last update's normalized geometry. In all three seeds its **average tokenwise cosine at loop 3→4 is negative**, although its **gradient-norm-weighted average first-order descent predictor remains positive** (+0.103 CE units) and its actual average CE gain is +0.039. This is expected mathematically because the unweighted average of per-token cosines and the average of gradient-norm/update-norm-weighted dot products are distinct statistics. It demonstrates the risk of equating average cosine with total loss descent.

At the trained final-only checkpoint, the fraction of update energy in the learned head's directly observable rowspace grows from **30.89% at k0** to 55.44%, 61.60%, and **66.49% at k3**. The remaining head-null energy is not by itself evidence of a causally indispensable dark fiber.

## Actual native recurrent counterfactuals

For exactly the first 64 heldout strings, at loops k=1 and k=2 and eight source-fixed token positions per string, insert an additional native next-step activation delta into the current recurrent state before all remaining original tied reasoning blocks. Paired controls subtract the same delta, use a source-seeded equal-L2 isotropic edit, or leave the state unchanged. The scorer fixes the strongest clean competitor separately for the true source digit, and reports actual **final fourth-pass** CE and margin differences from clean. Every source string contributes a mean of eight intervention outcomes. No edit direction, goal or dose is selected from heldout label information.

Average changes in final correct-digit CE across three seed checkpoints:

| Training arm | Injection loop | +Native update | −Native update | Equal-L2 random |
|---|---:|---:|---:|---:|
| Final-only | 1 | **−0.0263** | +0.1487 | +0.0110 |
| Final-only | 2 | −0.0022 | +0.1367 | +0.0106 |
| Intermediate | 1 | **−0.0188** | +0.1736 | +0.0165 |
| Intermediate | 2 | **+0.0387** | +0.1038 | +0.0193 |

Lower CE is favorable. The early +native-edit consistently reduces final CE relative to clean in all three final-only seeds, while its reversal makes final CE worse. The step-two positive edit is approximately neutral for final-only models and actually increases final CE for intermediate-supervised models, despite those models having strongly positive local cosines at this pass. This provides a useful **state- and horizon-dependent causal distinction** beyond a probe of the current head.

The source receipts include all 3 seeds × 2 training arms × 2 injected loop depths × 64 heldout strings × 8 fixed goals × 4 arms, with physical-dose parity and each full-logit endpoint. The complete 64-string per-seed paired-CI calculations are available to reproduce independently from the raw case records, rather than treating the eight positions in one string as independent observations.

## Optional BPD-OUT06 class-mass readout

After computing the proposed next activation delta, the experiment also measures the full nine candidate-digit **head-only** margin effects for +delta against −delta. A target-free softmax over the current readout's digit logits, temperature fixed at 1, yields classwise probabilities of the positive, negative and neutral effect classes at ±0.15. The original BPD-OUT06 reward yields expected utilities 2p_plus−1.1, 2p_minus−1.1 and p_neutral, with abstention on a tie.

This decision is computed **after F has already produced the proposed update**, so it is a post-compute update-selection diagnostic. It does not save a recurrence step or implement the SMC-27 full-native policy. Its utility is scored retrospectively with true test labels and remains secondary; a teacher can have a useful trained local state gradient even where an uncalibrated posterior intervention policy fails.

Descriptively, final-only models averaged head-only Bayes action utilities of +0.111, −0.025, +0.675 and +0.819 across k=0..3, while uniform-label Bayes utilities were +0.385, −0.451, −0.880 and −0.416. At k=1 the class-aware rule remained worse than always abstaining (−0.025 versus +0.003), despite the strong positive activation gradient cosine and actual native CE gain. This is another case where training optimization, activation descent, and cost-aware action utility must be kept distinct.

## Interpretation and research limits

The key mechanism is a **learned recurrent activation field whose updates often align with decreasing the supervised head loss**. The model is trained by ordinary AdamW BPTT on its shared weights; its activation trajectory was not explicitly optimized with SGD at inference, and an instantaneous negative-gradient cosine does not prove internal gradient descent or a globally decreasing Lyapunov function.

The native counterfactuals add causal evidence that at least part of the update direction is task-relevant after remaining loops. Additional intermediate supervision changes *when* this relevance appears: much better early readouts, a weaker late update and, at k=2, a positive edit that becomes harmful by final readout. Head-visible/hidden energy accounting gives a way to test later whether this comes from readout saturation, nonlinear recurrence, head-null transport, or geometry of a particular SAE feature dictionary.

This experiment uses a deterministic cyclic-copy token task rather than Sudoku constraint-solving, four exact recurrent passes rather than 16/336 ACT steps, and a small 44K source-attributed architecture rather than a 7M/3B model. The three parameter seeds share the same train/test generator. Source case-cluster confidence intervals quantify variation over input strings conditional on those three models; they do not prove transfer to new architectures or natural reasoning tasks.

A follow-up LGA-02 could train an SAE on h_k at a frozen checkpoint, project each actual update into SAE-active/dead directions and the head rowspace, then compare local gradient alignment to **native final-head causal edits** along the same aligned and orthogonal directions. Choose SAE fitting and all source goals on old train/validation, with source-disjoint tests and cluster-level intervals. A separate teacher-forced true-gradient versus target-blind candidate-mass decision audit can test the relationship to BPD-OUT06. Preserve the source SMC-27 registered FAIL and the current independent supervised routing result.

## Original source and receipts

- Prospective protocol [PROTOCOL.md](PROTOCOL.md), registered before LGA data at commit 81ab12d.
- Executable [gradient_flow.py](gradient_flow.py) and nine [source-independent tests](tests/test_gradient_flow.py); source-affecting changes and earlier pilot failures are documented in [ENGINEERING_AMENDMENT.md](ENGINEERING_AMENDMENT.md).
- Code merged in [LoopedTransformers PR #169](https://github.com/MoralityLabAI/LoopedTransformers/pull/169), squash commit **0a5640753539504cdd340e5a02e620328576e2f1**.
- Original successful source-locked full [Actions #44142413468](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44142413468), both pilot and full experiment jobs PASS.
- [Original full JSON artifact #14354897362](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44142413468/artifacts/14354897362), ZIP SHA256 72c9617cf3a06718281a3b1fc515ef1be6c364ae0a355caedb17dfc7cc46c1f2; uncompressed 25,406,824-byte full.json SHA256 **f4e5e92932f42ff362f1a69b0f72aa1f5408924453c36e1b98e4084f34b83c27**.
- Frozen exact split SHA256 of little-endian int64 ordered tensors: train x 12deb8f0b37bc4709f395786f455ee147b4fcf16606432fde8250d04818ad818, y bdc287fe1da93e4e5cacaf0b3537f3819100bcf225c380b86e5e0c5773bbc7dd; val x f1506d79fdc7daac45b2624297d9abc0634ad9f0b52c0f8c04d33e5853d71bc5, y 327708fcd22a23617b2fc23146a983a4f08bc4859f668eae1607b3009f99581d; test x e0980800dce2b03d5fe18afdaa0a71e873634826e7641c9ae24bb385ad592fac, y 2926721bd0bb2d10c9632c3960ef8a4a1c7a506af28eeef1360648f72f3e8f16.
