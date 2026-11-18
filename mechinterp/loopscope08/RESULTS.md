# LOOPSCOPE-08 / LGA-05: A shared scalar energy descends along TRM trajectories despite a nonconservative native field

**18 November 2026. Registered scientific result: POSITIVE, 3/3 independent frozen teacher seeds.** Original first complete [GitHub Actions #44235767614](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44235767614) passed eight source-independent unit tests, train-only fitting of a shared and conditioned scalar potential, fresh source-disjoint 256-sequence evaluation with native physical interventions, and separate standard-library receipt audit. The [prospective protocol](PROTOCOL.md) was committed FIRST as 338cb4847bd20928f60b91020d9b734d366f6244 before fitting or opening any fresh test data. The executable experiment merged as [PR #179](https://github.com/MoralityLabAI/LoopedTransformers/pull/179), squash a6a02a86406c7718eca95ac4752e7bbe240643ed. This is the original post-result manuscript, not an alternative scientific registration.

## Scientific question and prior constraints

We investigate whether a weight-tied recurrent 44,001-parameter TRM-inspired network has activation dynamics that can be described by a **single physically shared scalar energy**, rather than necessarily being gradient descent on the final supervised loss. The earlier independent experiments showed:

- LGA-01: all three seed-trained four-pass recurrent models learned a supervised cyclic-copy task and acquired positive alignment with the negative *current-head* task gradient, 3/3 registered seed successes.
- LGA-02: their recurrent updates aligned poorly with the true four-pass terminal-loss gradient, 1/3 registered successes.
- LGA-03: a separately trained scalar future-CE observer was a poor estimator of the true terminal-loss gradient, 0/3 registered successes.
- LGA-04: a separate future-CE critic trained on old true terminal gradients achieved paired improved gradient fidelity, 3/3 successes, while native edits against random remained inconclusive.

LGA-05 instead trains an **external scalar energy directly to the original model's native activation update field** on OLD source trajectories and tests whether that scalar remains coherent across different recurrent depths on unseen source sequences. A positive alignment is anticipated from the training objective; the heldout monotonicity and cross-depth coherence provide a separate generalization test. Since this scalar is external to the original trained model, it is evidence for an approximable *conservative component along its visited states*, not proof that the model computes an energy or implements SGD.

## Fixed teachers, fitting objective and chronology

Three original 44,001-parameter, four-pass weight-tied TinyRecursiveModel seeds **103, 211, 307** use exact original first LGA-02 teacher and six pretrained TopK4 SAE weights from [Actions #44222863870](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870). Original model/SAE bundle.pt SHA256 d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266, train.json SHA256 7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d. The original teacher and SAE parameters were frozen throughout; the independently trained original teacher seeds are the only model replication units.

The previous model was trained on an algorithmic 81-token one-position cyclic-copy task, with IID digits 1..9 and target at j equal to input digit at j−1 modulo 81. Earlier source/test seeds 31041/42/43/44/48/49/50/52/53/54 are excluded from new fitting and evaluation.

**Source-only fitting** used 256 IID 81-digit sequences from seed **31055** and 64 validation sequences from seed **31056**, all newly generated and source-disjoint. Each frozen teacher produced the original native states h0..h4 and updates v_k=F(h_k)−h_k, k=0,1,2,3, with the SAME trained block F applied at every depth. The original 48D physical states and native updates, not labels or final task gradients, provided the potential training targets.

Two scalar potential networks use the same architecture and identical random parameter initialization: two circular Conv1D/GELU layers (width96→48), Softplus scalar per token, 8D depth embedding. The summed 81-token output defines E(h,k). The **shared** potential always evaluates the depth embedding k=0, so it is ONE scalar E(h) across all four actual recurrent passes; the **conditioned** potential uses each native k separately. Both receive exactly 200 AdamW updates, identical 8-sequence batches and selected depth order, lr .001, weight decay .0001, using **second-order autograd** to match the physical negative scalar gradient to native v_k, both divided by the train-only RMS of v over all four depths. Every potential starts from the same initial weight state, frozen as an untrained random-potential control. A deterministic old-train-normalized quadratic radial energy E_rad(h)=0.5||((h−mu)/std)||² is another independent source-only control.

The original protocol **excluded any energy-decrease term from training**. A fitted field can align with update vectors without being globally integrable or yielding large finite-step descent. The actual heldout monotonicity is therefore a separate measured endpoint. The final source checkpoint is always step 200; validation gradient-MSE checks at 0,100,200 are diagnostic only, with no checkpoint selection.

Calibration normalized physical gradient MSE at step 200:

| Teacher seed | Shared scalar potential | Depth-conditioned potential |
|---|---:|---:|
| 103 | 0.60606 | 0.59807 |
| 211 | 0.58457 | 0.56156 |
| 307 | 0.60550 | 0.60013 |

The conditioned model has a slightly better validation MSE in every seed, as expected from its extra depth input. Both have the same parameter count and update budget. The entire fitting artifact was uploaded BEFORE generating the fresh test sequences.

## Fresh test on previously unseen 31057 sequences

The first new heldout cohort contains **256 independent strings** from seed **31057**, disjoint from every earlier source/testing split. Its ordered little-endian int64 input SHA256 is **6ba2988a35775d6861e10fe9ae57b38edf24ef39370eb96653c91f2cef2a962a**; cyclic target SHA256 **3a4da1ca797f3d32130ff107f80e81ae8082937b9dec97e5d0b4951b949e117b**. The auditor confirms all 256 target strings are right cyclic shifts of their respective 81-digit inputs and validates exact original SHA bytes.

At every k and each heldout 81×48 state, the evaluation computes v_k and the **whole-state gradient** of SUM of all 81 values of E(h). Per-token cosines between v_k and negative shared gradient are averaged within each entire string. For monotonicity, evaluate **the SAME E(h)** at both h_k and h_(k+1), retaining the same depth embedding k=0, and record per-source mean E(h_k)−E(h_(k+1)) divided by 81. No depth-specific change of scalar function enters the shared metric. The radial and conditioned controls are evaluated at their respective same fixed scalar function before/after each update.

The registered positive gate requires at least THREE of four depths per teacher to satisfy four conditions using 2,000 resamples of the 256 complete source strings: shared cosine gain versus same-initialization untrained and versus train-only radial potential has strictly positive 95% lower bounds, actual shared mean E decrease has a positive lower bound, and positive per-source energy-drop fraction has a lower bound >0.50. Whole-study POSITIVE requires >=2/3 frozen teachers passing. All three teachers passed at **all four depths**, 12/12 cells; the registered result is **POSITIVE, 3/3 teachers**.

### Heldout alignment and actual native energy decrease

| Teacher | k | Shared-gradient cosine | Conditioned cosine | Radial cosine | Shared E decrease per token | Fraction of 256 sequences descending |
|---|---:|---:|---:|---:|---:|---:|
| 103 | 0→1 | +0.7689 | +0.7773 | +0.6408 | +13.1119 | 256/256 |
| 103 | 1→2 | +0.5604 | +0.5628 | +0.2814 | +3.1988 | 256/256 |
| 103 | 2→3 | +0.4972 | +0.4901 | +0.2338 | +1.1200 | 256/256 |
| 103 | 3→4 | +0.2930 | +0.2715 | +0.1870 | +0.1506 | 256/256 |
| 211 | 0→1 | +0.7940 | +0.8034 | +0.6612 | +16.0020 | 256/256 |
| 211 | 1→2 | +0.5678 | +0.5929 | +0.2807 | +2.8545 | 256/256 |
| 211 | 2→3 | +0.4713 | +0.4770 | +0.2250 | +0.5709 | 256/256 |
| 211 | 3→4 | +0.2570 | +0.2606 | +0.1765 | +0.0627 | 255/256 |
| 307 | 0→1 | +0.7496 | +0.7592 | +0.6231 | +13.4746 | 256/256 |
| 307 | 1→2 | +0.5726 | +0.5771 | +0.2867 | +3.9153 | 256/256 |
| 307 | 2→3 | +0.5179 | +0.5071 | +0.2403 | +1.6603 | 256/256 |
| 307 | 3→4 | +0.3101 | +0.2743 | +0.1891 | +0.2831 | 256/256 |

Every original 95% lower cosine-gain bound versus BOTH untrained and radial is positive; the smallest lower bound versus radial occurs at seed 211/k3 (+.0789). Every per-depth lower mean shared-energy-drop bound is also positive; even the smallest seed 211/k3 energy decrease has interval [+0.0597,+0.0657] per token, and the fraction of positive drops has 95% lower 253/256 at that source/depth. Whole-trajectory four-step monotonicity holds for **256/256 (seed 103), 255/256 (seed 211) and 256/256 (seed 307)** independent new source strings.

Means averaged across all three seeds, in k=0,1,2,3 order: native update vs shared negative-gradient cosine **+0.7708,+0.5670,+0.4955,+0.2867**; native update vs conditioned gradient **+.7800,+.5776,+.4914,+.2688**; versus old-normalized radial negative gradient **+.6417,+.2830,+.2331,+.1842**. The untrained potential is approximately orthogonal, means −.0234,−.0072,−.0065,−.0083. The shared potential is competitive with the depth-conditioned control, especially at later loops, despite using only one scalar landscape.

Radial E decreases at k0 on average (+15.46 per token), then **increases** under the model's later updates (mean energy drops −.246,−.500,−.344 for k1/2/3). This is an important control: a generic centered quadratic produces some initial source contraction but does not reproduce the four-pass monotonic landscape.

Each seed and k has all original 256 per-sequence energies E(h_k) and E(h_(k+1)), signed source cosines, the exact raw gradients' normalization diagnostics, and original 2,000-draw source-cluster confidence bounds in the unaltered heldout receipt. The audit independently recomputes these scalar-energy differences and all registered intervals.

## Actual native matched-dose causal interventions

For exactly **64 first new independent source strings per teacher** at k=1 and k=2 and eight fixed positions per string, the original native scorer selects physical directions solely from the source hidden state, trained potentials and pre-existing SAE/head geometry. There are **3×2×64×8 = 3,072 original source-goal cases**, all meeting frozen rank-4/physical-dose admission. For each, eight source-only directions were computed: negative shared potential gradient, negative conditioned potential gradient, negative initial null potential gradient, negative radial potential gradient, actual native v, source TopK4 SAE active-span projection, SAE-orthogonal complement, head-null projection and seeded isotropic random. A ninth **negative true final-task gradient** was computed ONLY AFTER source plans were frozen, for teacher-forced diagnostic use. Counting all nine label-free directions and oracle, each source case therefore yields ten recorded edited arms: **30,720 genuine native physical edited readouts**. Every arm has physical L2 radius 0.25×||v||, edit vector SHA256, all 81 final token CE values, goal full ten logits, correct-digit fixed clean alternative margin and change in the fitted E under the physical edit.

The original full 4−k recurrent blocks are advanced after editing a single token h_k; global task CE is measured as the mean of all 81 final token cross-entropies. Each 64-source-string cluster contributes the mean of its eight selected token effects before the original 2,000 source-cluster bootstrap intervals. Negative paired CE differences favor the first arm.

| Teacher | Loop k | Shared E descent edit minus random: global CE [95% CI] | Shared edit minus native +v [95% CI] |
|---|---:|---|---|
| 103 | 1 | +.000032 [−.000113,+.000188] | **−.000081 [−.000145,−.000016]** |
| 103 | 2 | +.000020 [−.000050,+.000101] | +.000033 [−.000054,+.000136] |
| 211 | 1 | +.000014 [−.000095,+.000123] | **−.000100 [−.000173,−.000029]** |
| 211 | 2 | −.000017 [−.000074,+.000035] | +.000008 [−.000061,+.000081] |
| 307 | 1 | +.000070 [−.000034,+.000181] | −.000030 [−.000104,+.000045] |
| 307 | 2 | +.000048 [−.000040,+.000136] | +.000035 [−.000044,+.000120] |

**No one of six registered shared-potential-minus-random native causal comparisons excludes zero at nominal 95%**, despite the highly coherent scalar trajectory decrease. The shared potential edit is nominally less damaging to the final task than ADDING another full native update at loop 1 in two of three seeds. It has no robust task-CE benefit against the stronger random-dose control. All six shared-versus-conditioned native edit contrasts remain inconclusive.

Among secondary SAE/head physical controls, source-active SAE projection caused greater global final CE than SAE orthogonal complement at k2 for seeds 103 and 211 under nominal 95% intervals, with remaining four comparisons inconclusive. All six head-null versus random contrasts were inconclusive. None of these are familywise corrected causal discoveries. The small single-token global CE effects and correlated token goals limit their strength.

## Nonconservative original recurrence: native Jacobian skew

A genuine Euclidean gradient update field v(h)=−∇E(h) for twice differentiable E has a symmetric Jacobian J_v(h). The original LGA-05 source-only integrability diagnostic tests a **necessary** local condition on the TRUE ORIGINAL teacher recurrence F(h)−h, independent of the fitted potential. Using the original frozen FP64 copy of F, for 64 new source sequences, depths k1/k2, and two fixed-seed independently random unit full 81×48 tangent directions u,w per source, it computes symmetric finite-difference JVPs with step epsilon=0.001 and the bilinear asymmetry

  A = (u · J_v w) − (w · J_v u).

The diagnostic ratio is |A|/(|u·J_v w|+|w·J_v u|+1e-8). Exact symmetry would produce zero A apart from finite-difference roundoff. These original input-only measurements (mean across 64 separate new source strings per seed) are:

| Teacher | Native field curl ratio at k=1 | Native field curl ratio at k=2 |
|---|---:|---:|
| 103 | 0.7150 | 0.7508 |
| 211 | 0.7609 | 0.7673 |
| 307 | 0.7246 | 0.7535 |
| Across three seeds | **0.7335** | **0.7572** |

Source mean absolute asymmetric terms are approximately 0.008–0.011 per unit random-plane diagnostic, with mean absolute symmetric-bilinear magnitudes around 0.005–0.007. This is far above a tiny finite-difference rounding residual on a FP64 fixed-step check. The nonzero sampled antisymmetry is evidence that the full original F(h)−h field has **nonconservative components**, even though the separately fitted shared potential exhibits heldout monotonic decrease along its observed recurrent trajectories. Curl is a prespecified *descriptive* check with no acceptance cutoff, so it does not override the primary registered 3/3 scalar-energy field result.

## What the result supports

The trained TRM's recurrent motion contains a **substantial conservative-like component on its task-generated state manifold**. A single external shared potential learned on fresh calibration rollouts achieved high update-gradient cosine and descended on essentially all new rollouts through four tied iterations. The depth-conditioned potential did not need to dominate this shared model. This is stronger evidence for a low-dimensional descent-like structure of the recurrent trajectory than LGA-01's current-head gradient alone.

The original full recurrent field is substantially more complex: the native Jacobian is not symmetric in sampled planes, and an edit following the learned negative energy gradient does not reliably improve the *task's final CE* relative to equal-dose isotropic noise. The observed energy decrease can arise from a monotone component of a learned algorithmic update, a contraction on a limited manifold, or other geometry. These remain distinguishable hypotheses. Training directly to native update vectors also biases the learned observer toward alignment, so the original separate test and controls are essential.

An external fitted E is **not an intrinsic TRM state variable or proof of an internally computed objective**. No exact Lyapunov guarantee beyond source/task-conditioned heldout samples, cross-model semantic SAE identity, inference-time SGD algorithm or transfer to 7M+ Sudoku TRM/Nanbeige models follows.

An LGA-06 prospective follow-up should test whether the shared potential and its curl decomposition transfer across **OOD task distributions, sequence length and recurrent pass counts**, and distinguish pure contraction toward an attractor from a task-specific computation potential. A stronger internal objective would need an invariant potential across source variants and native interventions whose decrease predicts actual solver improvement. The original SMC-27 FAIL, LGA-02 terminal FAIL, LGA-03 scalar critic FAIL and LGA-04 external derivative-critic POSITIVE remain unchanged.

## Original source-locked receipts and independent audit

- Prospective first registration [PROTOCOL.md](PROTOCOL.md) commit **338cb4847bd20928f60b91020d9b734d366f6244**.
- Executable source and tests merged as [LOOPSCOPE-08 PR #179](https://github.com/MoralityLabAI/LoopedTransformers/pull/179), squash **a6a02a86406c7718eca95ac4752e7bbe240643ed**. Eight CPU source/autograd/SAE/geometry/curl tests passed.
- **First original science-producing [Actions #44235767614](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44235767614)** completed all three CI jobs PASS, including immutable calibration-only source fit, first source-disjoint 31057 test, and independent no-Torch audit. Registered status POSITIVE, 3/3.
- Source-frozen shared/conditioned potential artifact [#14396773191](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44235767614/artifacts/14396773191), ZIP SHA256 **86d8c518023c3fc43ea3ec375dd889add42c664848f12a935dcd7e48c9e9b47b**. Uncompressed potentials.pt SHA256 **24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53**, calibration fit.json SHA256 **69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d**.
- First genuine new-test/native/curl artifact [#14397576391](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44235767614/artifacts/14397576391), ZIP SHA256 **1b80faeee1c6667feeac61f9c36623c671a9d7443f5a8c6fc12f83341fc37f53**, original 299,160,482-byte uncompressed full.json SHA256 **8e401e9c23f38838af4fd3cf4512d9ff6dc4087ca9062dd36c2431b925085bcc**.
- Independent standard-library audit [#14397019375](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44235767614/artifacts/14397019375), ZIP SHA256 **72f120243ac60ed78cb4338dfd914e010ebc0dbf5e220ea4e06f5e7369344abe**, uncompressed audit.json SHA256 **287f448d10f6c78976f17b77f0eb70ccf8406c9e3927d7e6c6b8d7817a656f29**, audit PASS reproducing every 256-source energy and gradient statistic, 2,000 draws of all paired source-cluster bootstrap gates, true native goal ten-logit CE/competitor margins, all 30,720 physical edit-arm arithmetic receipts and original curl-plane bilinear results. The auditor reads original hashes without rerunning the teacher neural weights.
