# LS-SAE02: Native LoopedTRM SAE edits attenuate under recurrence

**Executed November 16, 2026.** Native LoopedTransformers study, [PR #113](https://github.com/MoralityLabAI/LoopedTransformers/pull/113). The [prospective registration](../native_trm_v1/PROTOCOL.md) was frozen at commit 3e5e29df0886c3754c2e5b9a07050953444f860b before heldout evaluation. The [hosted run #43810069425](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43810069425) passed its CPU regression checks, pretrained checkpoint pilot, complete evaluation and independent standard-library receipt audit.

**Registered scientific verdict: FAIL in both fresh Sudoku families.** The result is preserved with its original denominators, gates and trained checkpoint.

## Frozen primary result

| Depth-2 measure | Fresh 30-hole | Fresh 34-hole |
|---|---:|---:|
| Eligible goals / 32 preselected | **2 / 32** | **6 / 32** |
| Eligible independent puzzles / 16 | 2 | 6 |
| Mean selective kill minus best inactive-matched sham margin loss | −0.002158 | +0.003382 |
| 95% puzzle-cluster bootstrap | Abstained: 1,748/2,000 valid draws | [−0.004641, +0.010689] |
| Clean-correct-to-wrong flips, kill / best sham | 0 / 0 | 0 / 0 |
| Original registered gate | **FAIL** | **FAIL** |

The locked primary gate required at least 12 eligible goals from eight puzzles, all 2,000 puzzle-ID draws valid, a strictly positive lower confidence bound, and greater clean-correct-to-wrong flipping after kill. Both groups fail coverage, and the 34-hole confidence interval also crosses zero. A successful CI run is an execution statement; the scientific result remains negative.

## The native checkpoint, heldout families and fitted SAE

The actual 7M-parameter [gaoxin492 Sudoku TRM](https://huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att) step_21700 file was reloaded at SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**. Its weights remain frozen. This uses the LoopScope-02 native PyTorch model at FP32 CPU, a distinct numerical mode from previous BF16 work.

The calibration-only 512→48 PCA and 48→96 top-four SAE used 32 unique 30-hole fit puzzles and 200 AdamW updates (seed 71), followed by 16 separate calibration-diagnostic puzzles. Its final reconstruction-plus-activation loss is **0.38786286**; all 96 SAE atoms fired at least once on the independent diagnostic set. The frozen fit fingerprint SHA-256 is **f217453dc17e078b055e99f3cebc3cf736af7e7cfae2ab120353c91bf2554c9d**. No model/SAE weights were fitted or selected on either test family.

The new source-disjoint noncanonical Sudoku solution split manifests are:
- Fit 30-hole SHA-256: 915194003397d0fbc43bea30f13f6161af36b579fb9c1928d4d4fafc78c3128d
- Diagnostic 30-hole SHA-256: 1e2ccfb3a02ddc1aa7b63d3435c11dba0af254b4e2a1053f84ab70b417442eb8
- Heldout 30-hole SHA-256: 1124efeda7096c2269aa5b1b569a9514240cb7215155412523eeffa9a89922a3
- Heldout 34-hole SHA-256: 6ddc31e55206fbcb60b169f2715eb6c9a046815d46beb6ef2f78ceec3c7e2b83

Two blank goals per 16-puzzle family were chosen solely from the visible input mask. Only naturally highest-active SAE feature kills that passed the full original SAE reencoding checks were admitted.

## Mechanistic observations on eight eligible goals

The intervention removes one source-fitted SAE coefficient from a real board-token z_H state after the first ACT step. It preserves three other original active coefficients and compares against the best of 131 label-blind, equal-dose, inactive-preactivation-matched tangent shams. Native model continuation advances 0, 1 and 2 additional full ACT steps, or up to **42 L-level calls**. The task margin uses the correct digit and frozen strongest clean non-target rival.

| Measurement | Test30, 2 eligible | Test34, 6 eligible |
|---|---:|---:|
| Mean margin loss, k=0 immediate readout | +9.504718 | +7.976889 |
| Mean margin loss, k=1 | −0.038037 | +0.037205 |
| Mean margin loss, k=2 | −0.002158 | +0.003382 |
| Mean absolute per-goal effect at k=0 | 11.495615 | 11.186236 |
| Mean absolute per-goal effect at k=2 | 0.015933 | 0.007822 |
| Median rank-one SAE-versus-128-random p at k=2 | 0.6783 | 0.5388 |

For **all eight admitted goals**, the individual absolute kill-minus-best-sham effect at k=2 is less than **0.7%** of its own immediate k=0 effect. The largest individual ratio is 0.00691893. This is a descriptive posthoc observation among a small, strongly selected subset. It identifies a recurrence-dependent attenuation of this physical perturbation's task margin at these sites without identifying a unique reasoning variable.

Actual full-recurrence Torch autograd predicts the immediate k=0 perturbation almost exactly, with mean linear predictions +9.504722 / +7.976889. At k=2 the mean gradient predictions are +0.000346 / +0.004102. Mean relative nonlinear residuals at k=2 are approximately 0.00279 / 0.00164. The source-trained SAE direction does not systematically outperform 128 rank-one random directions on task-gradient alignment at depth two, as the median rank-based p-values show.

## Why the majority abstained

| Pre-outcome geometric decision | Test30 | Test34 | Pooled |
|---|---:|---:|---:|
| Feature kill failed full-SAE reencoding | 28 | 24 | **52** |
| Baseline equal-dose sham failed reencoding | 2 | 2 | **4** |
| Both edits admitted | 2 | 6 | **8** |

Removing the highest active SAE coefficient often causes a previously inactive atom to enter the top-four set or perturb the remaining active coefficients. That violates the original protocol's full-SAE guard and is scored as abstention before any output label or margin. Among all eight retained goals, the 92×48 inactive encoder matrix had full column rank **48/48**, giving the OLM-17 exact inactive-preactivation matching obstruction in this native network. The best candidate sham residual was approximately 0.730 / 0.689 times the original sham residual, averaged separately by test family; seven admitted cases selected the positive least-squares candidate and one the negative.

## Public receipts and proof boundary

[GitHub Actions #43810069425](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43810069425) ran all three stages successfully after an engineering-only source SAE autograd-scope repair. The registered protocol, sample seeds and gate remained unchanged. Native unit tests include differentiating through an actual small TRM and comparing with a finite difference.

- Full source-locked run: code head **d08e5aedd330b65caf9dcb3e12d3daade6e6117a**; complete raw/audit artifact **14200227290**.
- Artifact ZIP SHA-256: **c1d997b8de6dc062eff34c6ec91bdb3d4b0fe74235393894ce8f5956fbf1228a**.
- Full JSON (911,232 bytes) SHA-256: **f86ace0d910e091792f39c6cad3f6605c466ef5257c2a266734fd781fd42abbe**.
- Independent audit JSON SHA-256: **2c59a267f34f2a7cb2d92ee782c6643774920216d789178d4bcf91a29034f6c5**.

The independent standard-library auditor replays Sudoku uniqueness and roster hashes, the 96 encoder-shift coordinates, archived model logits for all four intervention arms, projected gradient arithmetic, random rank-one control p-values, puzzle-ID bootstrap and both registered failures. It **does not independently reexecute native model forward passes** or prove floating-point correctness with interval arithmetic. The [machine-readable summary](LS_SAE02_SUMMARY.json) preserves all source hashes and scientific decisions.

## Next native experiment

The useful next causal instrument is the **512D unrolled Jacobian-vector propagation** of the original feature-kill edit, rather than additional SAE dictionary refits on these already inspected puzzles. It can measure whether the sensitivity disappears because of contraction, feature redistribution or an insensitive final task readout. A separate prospective native campaign would need new Sudoku solutions, its own calibration set and fresh rank-matched controls. Any changed SAE rank, support count or selection threshold belongs in a new protocol. LS-SAE02 retains its two failed primary gates.
