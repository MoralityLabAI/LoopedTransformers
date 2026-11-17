# BPD-01: CUDA BF16 numerical closure and native online-SVD comparison | 2026-11-17

Branch `research/bpd01-cuda-bf16-closure-20261117`, cut from `feat/mechinterp-bpd` at
`92f93219` (PRs #156, #158, #159, #160 merged).

Hardware and software: NVIDIA GeForce RTX 5080 Laptop GPU (16 GB, 150 W limit), torch
2.11.0+cu128, TF32 matmul off, `allow_bf16_reduced_precision_reduction` True (default).
Jobs ran under vramorama leases of 8 GB.

Checkpoints verified by SHA-256 before every run:

| Artifact | SHA-256 |
|---|---|
| Stage-4 factors `bpd_b4_v2_stage4_weights.pt` | `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc` |
| gaoxin492 `step_21700` | `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3` |
| yagizdevre `model.pt` | `d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39` |
| `benchmark_bundle.json` | `5f9e262e63623e11ee0e1a431224e007497eca140c0b02bd0b85ff54b871f70f` |

The 16 boards are the first 16 WB-Stall evaluation boards and 63 invocations, read out at 20,
41 and 62. KL is the token-mean KL(reference || arm) over 81 cells and 9 digits. Full
per-board values for every arm are in the JSON receipts and in
[`results/v2/bpd01_tables.md`](../results/v2/bpd01_tables.md).

## Commands

```powershell
python mechinterp/bpd/scripts/audit_b4_v2_controls_and_ablation_split.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_closure_controls_corrected.json
python mechinterp/bpd/scripts/audit_b4_v2_closure_mechanism.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_closure_mechanism.json
python mechinterp/bpd/scripts/audit_b4_v2_online_svd.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_online_svd.json
python mechinterp/bpd/scripts/audit_b4_v2_online_svd.py --device cuda --precision fp32 --out mechinterp/bpd/results/v2/gate_b4_v2_online_svd_cuda_fp32.json
python mechinterp/bpd/scripts/reduce_bpd01.py
```

Each ran through `vramorama run --vram 8G` with `HF_HUB_OFFLINE=1`.

## Experiment A: the identity controls under native CUDA BF16

The registered controls script reproduced the historical receipt exactly where the arms
overlap. For example, the FP32-linear arm gives 0.309060 and 0.263795 at pass 62, and the
trained identity arm gives 0.365314 and 1.534241.

| Control vs native BF16 target | gaoxin492 p20 mean / max | gaoxin492 p62 mean / max (boards > 1e-3) | yagizdevre p20 mean / max | yagizdevre p62 mean / max (boards > 1e-3) |
|---|---|---|---|---|
| 1 same-dtype repeat | 0 / 0 | 0 / 0 (0) | 0 / 0 | 0 / 0 (0) |
| 2 target FP32 linear | 0.0031 / 0.0158 | 0.3091 / 4.0286 (4) | 0.0029 / 0.0095 | 0.2638 / 2.5376 (6) |
| 3 full FP32, original weights | 0.0038 / 0.0279 | 0.3159 / 4.3801 (4) | 0.0082 / 0.0409 | 0.7184 / 11.0780 (5) |
| 4 FP32, BF16-quantized weights | 0.0037 / 0.0278 | 0.0530 / 0.6914 (2) | 0.0048 / 0.0160 | 0.7306 / 11.1868 (5) |
| 5 SVD all 512, residual off | 0.0072 / 0.0514 | 0.0541 / 0.3787 (3) | 0.0080 / 0.0440 | 0.8186 / 6.7567 (5) |
| 6 SVD all 512, residual on | 0.0050 / 0.0386 | 1.0015 / 13.1288 (4) | 0.0036 / 0.0131 | 0.5024 / 6.7987 (5) |
| 7 trained all, residual on | 0.0059 / 0.0315 | 0.3653 / 5.3172 (4) | 0.0047 / 0.0337 | 1.5342 / 11.0486 (6) |

SVD weight reconstruction, measured independently: the relative Frobenius residual of the
full-rank SVD start is 3.19e-7 to 3.47e-7 across the four sites on both checkpoints.

### Identified failure mechanism: BF16 rounding accumulated through the recurrence on unconverged boards

`audit_b4_v2_closure_mechanism.py` separates the three candidate causes. Its decision rules
are in the script docstring. One rule changed after a 2-board smoke run, before the 16-board
execution. Raw per-element ulp error is unbounded at near-zero outputs: there, even the
correctly rounded FP64 product differs from cuBLAS BF16 by hundreds of ulps. The rule now
uses an effective ulp floored at the token RMS, plus relative error against FP64.

1. **No implementation mismatch.** Every identity path was teacher-forced on the clean
   target's own inputs, at every invocation and site. All of them, including the correctly
   rounded FP64 product, stay within 1.0 effective BF16 ulp of the native output. Their mean
   relative error against the FP64 product equals the native BF16 matmul's own,
   1.657e-3 to 1.663e-3, identical to four significant figures. The component paths
   change the BF16 output bits slightly more often than FP32 arithmetic does: 0.032–0.073%
   of elements against 0.021–0.054%. No path is less accurate than native.
2. **Arithmetic precision is sufficient to close the controls in FP32.** The FP32 rerun of the
   three identity arms (SVD residual off, SVD residual on, trained residual on) stays below
   the 1e-5 per-board tolerance at every readout, on all 16 boards, for both checkpoints and
   both weight sources:

   | Worst board identity KL | p20 | p41 | p62 |
   |---|---:|---:|---:|
   | gaoxin492, FP32, BF16-rounded weights | 3.4e-10 | 1.4e-9 | 3.2e-6 |
   | gaoxin492, FP32, original weights | 6.3e-10 | 1.5e-9 | 1.1e-6 |
   | yagizdevre, FP32, BF16-rounded weights | 7.2e-10 | 3.6e-8 | 2.3e-8 |
   | yagizdevre, FP32, original weights | 2.5e-10 | 5.1e-7 | 1.8e-7 |

   The FP32 target is within 3.7e-7 (gaoxin492) and 1.2e-8 (yagizdevre) worst-board KL
   at pass 62 of an FP64 rollout, with RMSNorm patched to run in FP64 too. The native BF16
   target is not. Its KL from the FP64 rollout of the same weights is 0.0044 / 0.0061 mean
   at pass 20 and 0.050 / 0.241 mean (max 0.46 / 3.05) at pass 62.
3. **No chaotic amplification of isolated perturbations.** A single seeded relative kick of
   size 1e-12, 1e-9, 2^-24 or 1e-5 to one `layers.0.o_proj` invocation (0 or 21) of the FP64
   model moves no board above KL 1e-3 at any readout. A single BF16-sized kick (2^-9) also
   moves none. The median state divergence decays, for example from 4.1e-4 to 8e-6 by
   invocation 62. Injecting 2^-9 noise at every invocation of that one site moves 2–4
   boards above 1e-3 at pass 62. Rounding only the four decomposed sites' outputs to BF16,
   in an otherwise FP64 rollout, does the same: up to 0.85 KL at pass 62 on gaoxin492.
4. **The affected boards are the unconverged ones.** In the exact FP64 trajectory, gaoxin492
   leaves four boards unsolved at pass 41: {0, 2, 7, 11}. Those are exactly its
   BF16-divergent boards, and exactly the boards that the persistent-noise FP64 run moves.
   Their minimum cell logit margins at pass 41 are 0.08–0.93, against 55–67 on the twelve
   solved boards. For yagizdevre the divergent boards {2, 3, 7, 10, 11, 13, 14} are all unsolved at
   pass 41. Solved boards stay at KL ~0 under every arm.
5. The cuBLAS reduced-precision-reduction flag has no effect: the BF16 target is
   bit-identical with it off.

Conclusion: the late-pass identity failure is arithmetic-precision noise. BF16 rounding
(~2^-9 relative) enters at every operation of every invocation. On boards whose search has
not converged, the recurrence accumulates it until near-tied digit commitments flip.
The component path and the native path are two equally accurate BF16 discretizations of
the same exact computation, and on those boards they take different branches. The
decomposition is not mis-implemented. The dynamics are not chaotic for isolated
perturbations.

### Gate consequences

- Under the 1e-5 per-board identity tolerance, CUDA BF16 does **not** close at any readout.
  Even pass 20 has identity-arm board KL up to 0.051, at mean 0.003–0.008. All BF16
  measurements at passes 41 and 62 are **uninterpretable for mechanism classification**. At
  pass 20, BF16 effects smaller than about 0.01 mean KL cannot be separated from rounding.
- **CUDA FP32 closes** the identity controls at passes 20, 41 and 62 on all 16 boards. It is
  the regime in which late-pass measurements become interpretable.
- The registered B4 final-KL criterion of 0.15 lies below the BF16 identity divergence, so
  it cannot be evaluated in BF16. Gate B4 stays INCOMPLETE: stages 5a–8c have not run.

### Corrected zero+sphere arm and the ablation factorial

Both `mean_comp_acts` and `mean_delta_out` are now zeroed for the zero+sphere arm. All four
cells use the same frozen clean-informed top-307 masks (CUDA BF16, mean KL; paired
contrasts over 16 boards with bootstrap 95% CI):

| Cell | gaoxin492 p20 | yagizdevre p20 |
|---|---:|---:|
| zero, no sphere | 0.0201 | 0.0062 |
| zero, sphere (corrected) | 0.2683 | 0.3965 |
| zero, sphere (historical, confounded) | 0.2748 | 0.4187 |
| pass-mean, no sphere | 0.4436 | 0.1623 |
| pass-mean, sphere | 0.6216 | 0.4372 |

| Pass-20 contrast | gaoxin492 | yagizdevre |
|---|---|---|
| sphere effect, zero fill | +0.248 [0.153, 0.353], 16/16 boards up | +0.390 [0.227, 0.594], 16/16 |
| mean fill, no sphere | +0.424 [0.209, 0.722], 16/16 | +0.156 [0.076, 0.267], 15/16 |
| sphere effect, mean fill | +0.178 [0.049, 0.316] | +0.275 [0.020, 0.572] |
| interaction | −0.070 [−0.178, 0.034] | −0.116 [−0.313, 0.082] |

The confound in the historical zero+sphere arm was small at pass 20, about 0.007 and 0.022.
Two corrections follow for the previous reading that "mean replacement drives the main
damage":
- Sphere rescaling on its own damages every board on both checkpoints.
- On yagizdevre its effect is larger than mean fill's.

The rescale pulls each token's write norm 40% of the way toward the population-mean write
RMS for that pass. That change is large even when the kept components carry most of the
write. These are BF16 values. The no-sphere zero arm is close to the BF16 identity floor on
yagizdevre (0.0062 vs 0.0080). The sphere and mean-fill contrasts are 20–60x that floor.

## Experiment B: online vs clean-informed SVD top-307

Static rank-307, clean-informed dynamic top-307 and online top-307 were compared with zero
ablation, residual off and no sphere. The online selector reads only the current perturbed
input of each wrapped module. Clean masks feed the Hamming telemetry only. A new arm records
the in-evaluator identity floor: all 512 components on, residual off.

| Pass 20 | floor mean / max | static S | clean-informed C | online O | O − C [95% CI] | online better / worse | M upper 97.5% |
|---|---|---:|---:|---:|---|---|---:|
| gaoxin492, CUDA BF16 | 0.0072 / 0.0514 | 0.0329 | 0.0201 | 0.0095 | −0.0106 [−0.0305, 0.0014] | 10 / 6 | −8.3e-4 |
| yagizdevre, CUDA BF16 | 0.0080 / 0.0440 | 0.0060 | 0.0062 | 0.0032 | −0.0030 [−0.0070, 1.5e-4] | 10 / 6 | −4.9e-4 |
| gaoxin492, CUDA FP32 | 7.5e-11 / 3.5e-10 | 0.0285 | 0.0092 | 0.0074 | −0.0017 [−0.0052, 6.0e-4] | 9 / 7 | −1.3e-3 |
| yagizdevre, CUDA FP32 | 5.7e-11 / 2.4e-10 | 7.2e-4 | 1.55e-4 | 1.68e-4 | +1.3e-5 [−8.9e-6, 3.8e-5] | 6 / 10 | −4.9e-6 |
| gaoxin492, CPU FP32, reserved 64 (PR #160) | | | 0.005518 | 0.005986 | | | |
| yagizdevre, CPU FP32, reserved 64 (PR #160) | | | 0.000162 | 0.000147 | | | |

| Pass 62 | floor mean / max | static S | clean-informed C | online O | O − C [95% CI] |
|---|---|---:|---:|---:|---|
| gaoxin492, CUDA BF16 | 0.0541 / 0.3787 | 0.0996 | 0.0629 | 0.0245 | −0.0384 [−0.1011, −1e-28] |
| yagizdevre, CUDA BF16 | 0.8186 / 6.7567 | 0.0990 | 0.4603 | 0.5074 | +0.0471 [−0.0375, 0.1820] |
| gaoxin492, CUDA FP32 | 5.0e-9 / 5.0e-8 | 0.9273 | 0.8861 | 0.9249 | +0.0387 [0.0073, 0.0767] |
| yagizdevre, CUDA FP32 | 3.6e-9 / 4.5e-8 | 0.4730 | 0.4265 | 0.2950 | −0.1315 [−0.4719, 0.0781] |

Mask disagreement between online and clean-informed selection is 0 at pass 0 by
construction. The mean Hamming fraction per module rises to 0.0015–0.016 at pass 20 and to
0.019–0.057 at pass 62. Component density is exactly 307/512 = 0.5996 for all three
selectors (measured for the clean-informed masks). Argmax cell flips against the clean
target at pass 20, summed over 16 × 81 cells:
- gaoxin492: BF16 floor 8, static 18, clean-informed 12, online 8; FP32 floor 0, static 19,
  clean-informed 11, online 10.
- yagizdevre: BF16 floor 10, static 5, clean-informed 11, online 6; FP32 floor 0, static 2,
  clean-informed 2, online 2.

Reading:
- **CUDA BF16.** Not interpretable at any readout, because the identity floor is not closed.
  At pass 20, yagizdevre's static, clean-informed and online KLs all lie below its own
  floor mean. Online is numerically lower than clean-informed on both checkpoints, and
  neither paired interval excludes zero.
- **CUDA FP32 (closed).** The pass-20 result reproduces the CPU FP32 reserved-64 finding.
  Online selection has no detectable cost against clean-informed selection on 16 boards.
  The reserved-64 margin statistic has a 97.5% upper bound below zero on both
  checkpoints. That is descriptive at n = 16, not a registered gate.
- **FP32 late passes.** These are now interpretable. They are dominated by one or two
  unconverged boards: gaoxin492 board 11 at KL 13.6–14.3 for every selector, and yagizdevre
  boards 10 and 14. Static rank-307 is no better than dynamic selection at pass 62. A 60%
  zero-fit selector fails on the boards where the search is still open, whichever
  selection rule it uses.

## Registered-gate decisions

| Item | Decision |
|---|---|
| BF16 same-dtype identity (repeat) | Exactly 0 on every board and readout |
| BF16 algebraic identity controls, 1e-5 per board | FAIL at passes 20, 41, 62 on both checkpoints |
| FP32 algebraic identity controls, 1e-5 per board | PASS at passes 20, 41, 62 on both checkpoints and both weight sources |
| BF16 passes 41 and 62 | Uninterpretable for mechanism classification |
| Online-SVD protocol (16-board CUDA BF16, descriptive) | Executed; no interpretable difference; identity floor dominates |
| Gate B4 | INCOMPLETE (stages 5a–8c not run; 0.15 final-KL criterion untestable in BF16) |
