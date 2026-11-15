# COE-07: Native LoopScope pretrained phase-gauge gate

**Protocol frozen before pretrained checkpoint inference.** November 15, 2026.

This is the native follow-up to `Compactification-Theory` COE-01..06 and MathCan's phase-blind observer capacity theorem. The existing COE-05/06 runs used small models trained expressly for that experiment. COE-07 uses the exact native checkpoint loader and recurrent state equations from **LoopedTransformers**, on a pretrained 7M Attention TRM.

## Immutable inputs and resource budget

- Underlying source branch pinned from `MoralityLabAI/LoopedTransformers@f504111e9985bd76bf321ed3968b46ac03a9811f`; native loader file `mechinterp/loopscope02/scripts/trm_sudoku.py` Git blob `a7396bc74a635b3ccf844e72cf48f73804e6ac5b`.
- Primary checkpoint `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att`, file `step_21700`, SHA-256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`. Use the existing `TS.load_trm_checkpoint`, which checks SHA before loading, with CPU float32 inference. Freeze weights.
- Benchmark source `mechinterp/loopscope02/results/benchmark_bundle.json`, internal bundle digest `59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811`. No external dataset fetch required. Select exactly the **first row of each digit 1–9** in each of four existing strata by committed file order, for 36 distinct boards. The selection depends on metadata alone.
- Fit on `S1_naked1` and `S3_prop3plus`, test on `S2_prop2` and `S4_stalled`. All four groups have one board per digit. Within each held-out group, score the complete 9-vertex evaluator graph, 36 edges each. Exactly 72 held-out pairs per head. Context/stratum shift is an explicit limitation.
- Capture 8 outer ACT visits. Each visit is `3` H cycles with `6` L cycles and two native physical transformer layers, reusing the checkpoint's time-homogeneous weights. Capture target-cell `z_H`, `z_L`, eleven full `lm_head` logits and the `z_H` complement orthogonal to the **entire** `lm_head` row space. Report numerical projector parity and row-space rank.
- CPU-only, at most four threads, one model checkpoint and no fine-tuning. CI download consumes network bandwidth; workflow caps inference job at 30 minutes and uploads receipts and compressed activation arrays.

## Frozen fit-only probe selection

For each of four representations `visible`, `dark`, `full`, `scratch`, fit one shared binary linear threshold \(b_t(x)=1\{w\cdot z_t(x)>\theta\}\), the same \(w,\theta\) at all eight steps. Candidate directions are all 36 pairwise differences of fit-only digit centroids, 96 seeded random Gaussian directions, and the first 12 fit-only singular vectors. Norm-normalize each, evaluate the unique fit-only projected quantiles at `[.15,.25,.40,.50,.60,.75,.85]`. Random seed `20261115`. No test activations or held-out labels are available to this fitting routine.

Select four separately reported heads from this same fixed candidate pool by lexicographic fit-only objectives:

1. `ordered`: ordered edge retention, then cyclic retention, then count retention.
2. `cyclic`: cyclic-invariant edge retention, then cyclic-above-count gain, then count retention.
3. `cyclic_gain`: cyclic-above-count gain, then cyclic retention, then ordered retention.
4. `count`: count edge retention, then cyclic retention, then ordered retention.

Tie order is the deterministic enumeration order; no test-based head changes. Each selected head has *one* fitted direction and threshold, with 8 bits of **readout over time**, not 8 learned linear directions.

## Metric and decision rules

For each held-out stratum and each of the 36 pairs of distinct target digits, compute the ordered 8-bit words \(w_i\), their canonical cyclic rotations \(N(w_i)\), and popcounts \(C(w_i)\). The exact order of information loss is

\[D_{\rm ordered}\ge D_{\rm cyclic}\ge D_{\rm count},\qquad
D_{\rm ordered}-D_{\rm cyclic}=\#\{\text{phase-origin-dependent distinctions}\}.\]

A cyclic-invariant advantage means `D_cyclic − D_count ≥ 8/72` on the **`cyclic_gain` head in the `dark` representation**, with at least one such preserved distinction in each held-out stratum. This is the one pre-fixed binary success gate. Other heads and representations are diagnostics with explicit numerators, denominators and no post-hoc significance claims. A base-model performance gate reports how many of nine target cells are solved by visit at each stratum. If native inference, source hash, or the fit/test firewall fails, label the cell **invalid**, with no substituted synthetic results.

Controls include the final-step one-bit projection on the same learned direction, the full `lm_head` visible channel versus its exact orthogonal complement, and a frozen selected-head replay from a saved activation file. All source model weights and source codes are preserved; no mutation of the pretrained repository's checkpoint or original LoopScope results.

## Reproduction

```sh
python -m unittest discover -s mechinterp/phase_gauge07 -p 'test_*.py' -v
python -m mechinterp.phase_gauge07.run_native --out /tmp/coe07_native --threads 4
python -m mechinterp.phase_gauge07.replay --input /tmp/coe07_native/native_activations.npz --receipt /tmp/coe07_native/receipt.json
```

The first command requires NumPy and no internet. The native command requires CPU PyTorch, Hugging Face Hub connectivity, and the pinned external checkpoint. Final outcomes are conditional on the actual hosted native run. The literature-wide novelty of any resulting mathematical formulation remains unassessed.
