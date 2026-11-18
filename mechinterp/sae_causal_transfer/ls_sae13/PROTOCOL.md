# LS-SAE13: ACT4 numerical sensitivity and batch-shape audit

**Engineering diagnostic registration: 2026-11-18.** This document was
committed before implementing or running the LS-SAE13 pilot. LS-SAE12's
prospective FAIL, cross-platform re-executions, and all prior source/puzzle
outcomes are already public. This is an **exploratory engineering diagnostic**;
it has **no confirmatory scientific success gate** and cannot rescue LS-SAE12.

## Motivation and scope

LS-SAE12's same-forward baseline is batch-invariant across the two CPU
platforms, while CUDA (a different PyTorch build) has only 9/48 exact matches
against batch-1 screening. On unsolved boards, cross-platform verified deduction
displacement (VDD) is unstable even when the edit vectors are float32-identical.
The next question is where this instability first enters the recurrent model.

The pilot deliberately uses **four previously studied positions: 23, 24, 28,
40**, in that order, from the already-admitted LS-SAE12 source receipt. Positions
23 and 40 were unsolved on the registered runner; positions 24 and 28 were
solved there. These were chosen using existing evidence, so this experiment
estimates no population effect, p-value, or new directional efficacy.

The checkpoint, implementation, source receipt and encoding are inherited
unchanged from LS-SAE12:
- checkpoint \`gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att\`,
  \`step_21700\`, SHA256
  \`932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3\`;
- float32, MATH scaled-dot-product attention, four CPU threads, no training;
- the 48-board source-only receipt at
  \`mechinterp/sae_causal_transfer/results/ls_sae12/source_48.json\`;
- an ACT is three H cycles with six L cycles each. Probe after ACT4.

Run on Linux CPU (Python 3.11, torch 2.7.1+cpu, numpy 2.1.3). The same
script also supports a local Windows CPU rerun. A later CUDA execution, if
undertaken, is a separate exploratory platform comparison.

## Matched perturbations and batch layouts

For each selected position, choose the **model-blind median blank cell** (by
sorted index), avoiding platform-dependent eligible-state and selected-site
changes. Produce one batch-1 ACT1..ACT16 clean trajectory and save its exact
ACT4 \`(z_H,z_L,x_emb)\` anchor. All continuations start from that anchor.

Intervene on precisely one \`z_H[16 + site]\` token after ACT4 with **fixed**
\`+/- 0.02\` L2 Rademacher perturbations. A domain-separated SHA256 bitstream
sets each of 512 signs, with a single pinned float32 component magnitude:
\`float32(0.0008838834764831844)\`. Store the resulting 2048-byte
little-endian float32 vector and its SHA256 digest. Every execution must check
that digest. There is no Jacobian or Gaussian sampling in this diagnostic.
Both signed edits use the same bytes on every platform.

Run continuation ACT5..ACT16 at batch sizes **1, 2, 21, 61**:
- B1: one clean row; one plus row; one minus row, in three forwards.
- B2: clean+plus in one forward; clean+minus in another forward.
- B21 and B61: clean, plus, minus, then identical unedited filler rows.

Batch-1 screening, batch-1 repeated clean and every batch's clean rows
are compared. In B2 the duplicate clean is checked, and in B21/B61 each
padding clean is checked against row 0. These controls directly expose
batch-shape dependence.

## Required measurements

At every ACT5..ACT16 step, for clean and the two edits:
- first-index argmax predictions on the 81 cells;
- SHA256 of the exact row's 81x9 little-endian float32 logits;
- minimum clean winner-vs-runner-up margin over the blank cells;
- per-signed-edit hidden state differences \`||delta z_H||_F\`,
  \`||delta z_L||_F\`, centered-logit RMS and max absolute logit change;
- per-cell sufficient argmax certificate \`margin_clean > 2*epsilon_cell\`
  with \`epsilon_cell = max_digit |logit_edit - logit_clean|\`, plus actual
  blank-cell argmax flips.

At ACT5, 8, 16, retain the exact 81x9 logits of clean, plus and minus as
float32-valued JSON arrays, so a future independent comparator can compute
cross-platform immediate effects without relying on aggregate statistics.
Include full ACT5..ACT16 digit trajectories for within-batch clean/edits, VDD
relative to the same-forward clean, exact ACT16 correctness, and the first
clean divergence from batch-1 screening.

The primary **diagnostic**, not a hypothesis gate, is agreement in ACT5
centered-logit perturbation and winning-margin shift across environments
for an identical site and edit. Secondary diagnostics include the timing of
the first clean and edited trajectory bifurcations, the margin/certificate
coverage, and ACT16 VDD.

## Audit and interpretation rules

Unit tests must exercise deterministic edit generation, exact edit hashes,
certificate sufficiency, all four batch layouts, within-batch controls and
per-board VDD. Receipts include OS, CPU, Python, torch, numpy, checkpoint SHA,
source SHA, process thread count, edit SHA, batch size, and the run parameters.

An engineering run fails on an incorrect checkpoint or source hash, nonfinite
logits, mismatched edit hash, an invalid batch layout, or inconsistent receipt
shapes. Any batch-1 rerun mismatch is recorded as an engineering failure.
A divergence at ACT5 or later is retained in the report, never filtered away.

Interpretation separates (1) floating point variation before a decision
boundary, (2) boundary crossing, and (3) recurrent amplification after a
crossing. The finite-difference \`||delta z_H||/0.02\` is descriptive,
not an upper Lyapunov exponent or a certified local Jacobian bound. A
successful unit-test or clean-batch audit establishes engineering integrity
only. A new confirmatory study would use a new preregistered source cohort
after this pilot.

LS-SAE12 files, evidence branches, thresholds, PR #164 and its registered
results remain untouched.
