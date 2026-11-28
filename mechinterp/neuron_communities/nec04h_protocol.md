# NEC-04H frozen Looper XOR phase routing

This retrospective developmental study uses original NEC-04F calibration roots 12–17 and their original 80 source and 24 owned validation contexts per root. Each context contains 32 focal masks. The root is the unit of analysis. The 24 reserved contexts per world, later calibration roots, and all development and confirmation worlds stay sealed. The evidence class is `FSI-01 REDESIGN_OR_PILOT_ONLY`.

The archive lock pins the seven ZIPs from the original NEC-04F workflow and the original training revision and registration hash. Each root binds the 900-step `aux_prefix` checkpoint to the archived native and auxiliary logits, source rows, immutable program and independent NEC-04F audit. The original negative final-task result remains historical evidence. NEC-04G #315 applied a Fourier-sign diagnostic to the original final logit and reported 61.88%. It is an independent archival reference for this experiment.

## Zero-forward feasibility

The independent scalar interpreter checks every source and validation label and every parity identity. The selector is the bit at `context_positions[0]`. The phase is the XOR of the bits at `context_positions[1]` and `context_positions[2]`. The raw phase-adjusted branch must agree with the scalar AST at every focal mask. Stage-5 XOR predictions are combined with the true phase on parity rows, and the true selector bypasses that branch on alternate rows. Results are recorded by root and selector before the first model forward.

## Frozen repair

The historical `AuxLooper` runs its original 12 visits on CPU. Its final memory is captured at the input of the frozen native head. Cached source and owned memory, stage-5 XOR logits and native logits are SHA-sealed. Recomputed native and all four auxiliary distributions must match the archives within 0.0001 absolute probability and preserve every archived class prediction. Absolute logit errors are also retained. The initial 1e-5 logit-only check exposed Windows-versus-original-run differences even with the original PyTorch release. The portability check was fixed after verifying frozen logits and before fitting any repair head. The original state digest and disabled gradients are checked before and after fitting. The feature caches hold no labels.

The selector reads 32D memory through a linear head. The two phase arms each use a 32→32→1 GELU head with identical initialization and source sampling. The control permutes phase labels across complete source contexts within each selector stratum by deterministic SHA rank. It preserves the number of phase positives within each stratum. Training sees only source memory, group identifiers, and phase or selector labels. One selector checkpoint is shared by both arms at each budget. The head fit uses binary cross entropy and AdamW with LR .003, batch 64, decay zero and clipping 1 for 900 steps. Step 240 is diagnostic.

Probability composition uses the frozen prefix-XOR distribution and learned phase through probabilistic XOR, then routes parity via the learned selector against the frozen final distribution via probabilistic MUX. A separate scalar enumeration reconstructs both output probabilities. True phase, true selector and their joint intervention are diagnostic upper-bound inputs. The output includes full and selector-conditioned correctness, alternate-branch retention, head accuracy, signed fifth-order Walsh signal on class-one minus class-zero log odds, and the ordered arrival stratum of each phase bit relative to the fifth focal bit.

## Design decision

The primary descriptive contrast at 900 steps is the equal-root mean parity accuracy of the learned phase arm against the shuffled phase arm sharing the same selector. Report the native contrast beside it, all six roots including failures, pooled counts for inspection and root-stratified outcomes. A useful result for preparing a separate prospective protocol requires at least +5 percentage points on the primary contrast, at least +5 points against native on parity, and alternate-branch retention within 5 points of native. These are developmental effect floors. The randomized-label arm is a matched head-training control. Six previously exposed roots supply neither a calibrated sham distribution nor prospective resolution. Walsh and chronological contrasts are secondary diagnostics.

To reproduce, place the seven GitHub Actions ZIPs named in `nec04h_artifacts.json` in one input directory.

```text
python -m unittest mechinterp.neuron_communities.nec04h_test
python -m mechinterp.neuron_communities.nec04h_archive INPUTS phase-a.json
python -m mechinterp.neuron_communities.nec04h_frozen INPUTS OUTPUT
```
