# NBR-01/02: native neuron basis for recurrent Sudoku reasoning

Status: implementation and engineering pilot specification, 2026-11-25. No
registered outcome, no new source roster admitted, no checkpoint inference
performed in this branch yet. Freeze the source manifest and executable commit
in a follow-up registration before running new evaluation boards.

Motivation: Anthropic's [neuron basis theory](https://alignment.anthropic.com/2026/neuron-basis-theory/)
and [applications](https://alignment.anthropic.com/2026/neuron-basis-applications/).
Existing LS-SAE12 (PR #164, failed registered gate) shows substantial
cross-platform instability of hard Sudoku continuation; hidden amplification
and fixed output directions alone are insufficient evidence of computation.

## Model and unit of analysis

Reuse main's mechinterp/loopscope02/scripts/trm_sudoku.py and checkpoint
gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att / step_21700, SHA256
932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.
FP32, math SDPA, batch 1, four CPU threads, fixed seed. CUDA is a secondary
replay and requires separate reporting. No parameter training. An ACT step
contains three H cycles with six L updates followed by one H update each;
capture the final L update and final H update of the final H cycle at
ACT3, ACT6, ACT12. Store layer, role, step and physical neuron ID; never
confound a role change with a loop change. Site = median sorted blank cell,
chosen without model outputs or answers.

Let a_i be the actual SwiGLU scalar entering down_proj, w_i the corresponding
column of its matrix, and y the nine centered digit logits at the chosen
cell after the H commit. Add a zero, independently differentiable scalar
delta_i at that activation, retaining the rest of the forward graph.
J_i = d y / d delta_i is the local transported write direction.
C_i = a_i J_i is its signed first-order zero-ablation magnitude (removing
the neuron would predict -C_i). It includes downstream RMS normalization,
later block(s), and (for the L role) the subsequent H commit, rather than
just W_unembed w_i. This is a derivative, not a finite causal intervention
or a full ACT16 attribution. We do not differentiate through earlier ACTs.

## NBR-01: stability and the null

Primary comparison: last zL update, physical layer 0, ACT3 versus ACT6.
Within each board, retain neurons with finite direction vectors and norm
above 1e-8 at both steps. Compare mean same-ID cosine to the mean of
32 derangements within eight fixed write-norm rank strata (ties by ID).
Exclude singleton strata from both arms. Compare by board, not by neuron.
ACT6 vs ACT12 and the H role/later block are diagnostics. Also remove the
component along the centered static head times w_i and repeat the cosine
comparison; this describes dynamics beyond the direct readout. Report
coverage, static alignment and undefined cases. An identical direction
under reuse may be architectural; stability alone is not a causal result.

For a prospective run, require >=48 admitted calibration and >=48 in each
of two distinct evaluation families (eval, rep), with independently checked
unique solutions, no duplicate input/solution or symmetry orbit across
splits, no overlap with previous experiments. Freeze exact source file IDs,
hashes and split assignment before model access. Prior LS-SAE11/12 sources
and LOOPSCOPE-02 are engineering-only. Public dataset provenance does not
establish training-set independence. Proposed gate (register after source
admission, before inference): for each eval family, at least 40 valid boards,
mean paired cosine advantage >=0.10 and lower 95% board-bootstrap bound >0.
Report both even on failure; no replacements. Bootstrap 2000, seed 20261125.
Residualized test is a diagnostic; require a later causal experiment before
claiming a computational primitive.

## NBR-02: recovery detection and basis comparison

Engineering arm in native.py: on the same board site, predict recovery
conditional on ACT6 being wrong: the target digit must be correct at every
ACT from 7 through 16. Feature computed at ACT6 without solver answers:
C_i projected onto the clean winner-minus-runner-up logit contrast.
Control = signed activation a_i. Only calibration boards choose one neuron
and sign per arm by AUROC. Evaluate those frozen choices independently on
eval and rep; all cells of a board stay together. Report sample/class
counts, AUROCs and paired board bootstrap (resamples lacking both classes
discarded and counted). This is a conditional diagnostic, not a deployable
detector because eligibility uses verified truth. No claim of SAE superiority
is licensed until the following comparisons have actually run.

NBR-02 full comparison follow-up: fit existing SAE at the identical hook
site / role / layer using frozen weights or train only on calibration
activations; document SAE normalization, decoder dimensions and reconstruction
error. Use the same readout gradient on SAE decoder directions and activation
scores. PCA basis and isotropic basis calibrated on the same site, unit
decoder norms, same number of candidate directions or explicit budget curve.
Include direct static head x down_proj, matched shuffled labels and a
supervised linear probe as controls. Match number of selected directions and
calibration search budget; report training/inference time, memory, ROC and
PR curves, calibration, family-wise confidence intervals. Preregister
the full SAE test and its gate separately; do not silently change the
native-only test after evaluation. Causal test: scale selected activation at
one call/site only, compare equal pre-normalization residual-write L2
to randomized native directions and SAE directions, use same-forward clean
row, immediate logit shifts, verified future deduction schedule (ACT16),
and numerical sham replay. Check finite-difference agreement before any
long-run claim, and audit CPU/CUDA independently.

## Engineering execution

From repo root:

    python -m mechinterp.neuron_basis.native --self-test
    python -m mechinterp.neuron_basis.native --roster roster.json --out nbr_pilot

Roster JSON is an array of objects {id, split, question, answer}; question
is 81 chars in 0..9 or '.', answer is 81 digits 1..9; split is cal/eval/rep.
The runner validates completed grids and exact duplicate inputs/solutions.
Independent uniqueness, source disjointness and symmetry audit remain
required externally; this runner deliberately marks every result
engineering_only. Do not use existing pilot output to select the future
registered gate. Output directory must be new; one compressed raw receipt
per board and report.json with hashes and platform details. Unit self-test
uses a tiny randomly initialized model, checks identity against unmodified
calls and centered finite differences; it does not establish checkpoint
performance. Report failure instead of replacing an invalid board.
