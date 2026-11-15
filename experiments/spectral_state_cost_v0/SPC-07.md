# SPC-07: cross-checkpoint temporal observability and phase-transfer alignment

This experiment continues SPC-04/SPC-05/SPC-06 with their exact, unchanged trained
pointer-chase checkpoints. The implementation and numerical receipts live in
`spc07_transfer.py` and `receipts/spc07_summary.json`. For the gauge-
transport and rank-r Gramian mathematics, see
`patrickdugan/MathCan/research/spectral_aliasing/SPC-07.md`.

The full checkpoint-free source runs deterministic CPU geometry tests. The
trained-data experiment requires locally supplied SPC-04 checkpoint files,
verified by SHA256; CI makes that trust boundary explicit.

## Experiment: frozen transfer across two separately trained LoopedChase models

Saved checkpoints from SPC-04 are loaded using `torch.load(...,weights_only=True)`, with model-weight SHA256 verified internally. Both have eight pointer nodes, one tied layer, 48 answer-token channels, and 1,200 updates of three-hop supervision. No weights are modified in SPC-07.

- **Temporal observable directions:** at depth 32, collect central-difference gradients of all seven output-logit margins through eight future loops, with 48 answer-state perturbation directions. Form a Frobenius-normalized Gramian per prompt and average over 16 frozen source-only prompts (`920731`). Top-eight eigenvectors define the observable subspace. The same estimator on the target checkpoint is a within-model comparison.
- **Phase subspace:** use SPC-05's uncentered answer-token phase-increment SVD from 256 prompts seeded `904001`, with training-only depths 32, 48, and 64. Eight directions were selected in the prior SPC-05 work.
- **Cross-model alignment:** build 768 paired phase-increment vectors from the same 256 training prompts, then solve orthogonal Procrustes `min_{Q^TQ=I} ||X_s Q-X_t||_F`. Fit a negative control after shuffling the target rows before Procrustes. Apply both Qs unchanged to source subspaces. The target's phase-increment SVD remains a separate fitted reference.
- **Held-out:** use eight seeds `910701` through `910708`, with 64 fresh input graphs each per target checkpoint (1,024 inputs total). Phase is evaluated at depth 64, following an answer-token-only intervention for 12 native recurrence steps. The 790 stable nonfixed task cycles satisfy the preregistered ground-truth pointer period, `relative_cycle_error<1e-4`, `relative_one_step>0.05`, and nonzero phase separation. The other 234 inputs are retained in total-count accounting and excluded from phase-rate denominators.
- **Controls:** source raw basis, paired-aligned source, shuffled-pair-aligned source, target-trained basis, four-to-eight random orthonormal bases, full phase update, and zero update. Both raw projection magnitude and full-update-norm-matched variants are recorded. Cross-checkpoint states themselves are never transplanted.

The native analytic instantaneous readout derivatives and two central-difference step sizes agree at relative errors of order `1e-10`. Train/heldout exact prompt-hash overlap is zero.

### Results: full-displacement phase-change subspaces (r=8, unscaled projection)

| Intervention | Source 41522 → target 41523 (406 cycles) | Source 41523 → target 41522 (384 cycles) | Combined |
| --- | ---: | ---: | ---: |
| Raw source phase basis | 16 | 27 | 43/790 |
| Shuffled-pair Procrustes phase basis | 318 | 294 | 612/790 |
| Properly paired Procrustes phase basis | **404** | **382** | **786/790** |
| Target-fitted phase basis | 405 | 383 | 788/790 |

The paired-alignment advantage over shuffled pairing is 174/790 = 22.0 percentage points. The paired, 16-batch cluster bootstrap (4,000 resamples, seed 771407) yields a 95% interval of 20.5–23.5 points. The paired contingency table has 175 gains and one loss. These are empirical observations on frozen synthetic models.

### Results: temporal-observability subspaces (r=8)

| Intervention | Source→41523 | Source→41522 | Combined |
| --- | ---: | ---: | ---: |
| Raw source temporal basis | 14 | 6 | 20/790 |
| Shuffled-pair aligned temporal basis | 126 | 229 | 355/790 |
| Paired-aligned temporal basis | **192** | **326** | **518/790** |
| Target-fitted temporal basis | 369 | 212 | 581/790 |

With the projected updates rescaled to match the norm of the natural full phase step, the aligned temporal basis shifts 624/790, the shuffled-pair basis 511/790, and the target-fitted basis 638/790. The paired alignment beats shuffled alignment by 14.3 percentage points (95% cluster bootstrap interval 10.3–18.3 points). This norm matching can magnify small projected updates and is labeled as a distinct intervention.

The two raw top-eight temporal subspaces have projector overlap 0.187 in the common hidden-neuron indexing. Paired alignment raises it to 0.453; raw phase-subspace overlap 0.200 rises to 0.876 after alignment. Aligned source temporal directions capture 49.3% of the target Gramian in the 41522→41523 direction and 43.6% in the other, compared with 79.7% and 71.1% for each target's own temporal subspace. A positive causal phase-transfer result can coexist with imperfect temporal-observability transfer.

### Statistical scope and falsifiers

The two trained checkpoints use the same architecture and synthetic task. The paired alignment accesses target hidden activations on unlabeled calibration graphs. Consequently the 786/790 result is **calibrated transfer**, not zero-shot transfer of raw neurons, and it is not evidence about general language-model reasoning or safety. The shuffled-pair control itself performs well, showing that part of the geometry is population-level. To establish stronger task-specific transfer, add distinct task families, independently trained widths, graph perturbations, and held-out calibration distributions. The source paper's automaton lower bound is used only as motivation for deferred distinguishability, not as a theorem about these learned models. The phase-zero and phase-one reference states are measured using the whole hidden state, while the interventions themselves alter only the answer token. Consequently the full-answer-step control can occasionally miss a full-state phase classification even when a learned projected direction succeeds.

## Reproduction and trust boundary

Run from the repository root after providing the exact SPC-04 checkpoint files:

```bash
python -m experiments.spectral_state_cost_v0.spc07_transfer \
  --checkpoint-dir /path/to/spc04/checkpoints \
  --out /tmp/spc07_full.json \
  --summary-out /tmp/spc07_summary.json
```

The 1.2 MB per-condition audit receipt is preserved separately in the downloadable SPC-07 reproducibility archive. The source repository contains the compact summary and deterministic unit tests. The repository-wide MathCan tests need only the Python standard library; the new applied unit tests use CPU NumPy and PyTorch. Lean checks `MathCan/GramianTransfer.lean` with the pinned project toolchain. `#print axioms` is executed on its named theorems.
