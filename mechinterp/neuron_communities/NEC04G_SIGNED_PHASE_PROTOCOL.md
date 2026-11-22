# NEC-04G: retrospective signed XOR phase in the six original NEC-04F calibration worlds

**Evidence class:** read-only analysis of already scored, independently replayed NEC-04F final native logits.
**Status:** posthoc developmental mechanism diagnosis. FSI-01 remains REDESIGN_OR_PILOT_ONLY.
**Population:** only original NEC04A calibration-world roots12–17, source contexts0–79 and their existing 24 within-world owned validation contexts. No extra model training, new inference, SAE, causal intervention, reserved contexts, development worlds, confirmation worlds or NEC03P recruits.

## Motivating fixed observation

NEC-04F's prospectively registered 900-step XOR-auxiliary training failed the original final-task parity gate. Six-world macro accuracy was 46.27% versus 49.98% under matched shuffled auxiliary supervision, a −3.7078pp difference against the frozen +10pp floor. However, auxiliary XOR-head stage-5 validation in original roots12,16,17 was 95.05%, 90.23%, 97.01%. Their original final native parity logit fifth-order **unsigned** Walsh-power fractions were 54.34%, 48.30%, 30.63%. This identifies a possible readout-sign or context-MUX problem without proving it.

The exact original NEC04A parity branch is rawXOR(focal mask) XOR context_1 XOR context_2. For each complete 32-mask original parity context the original native composed teacher's mask-zero label provides its context parity offset. If y0 is the class label at mask 0, then true labels must satisfy

    y(mask) = (mask.bit_count() mod 2) XOR y0.

Let m(mask)=final_logit_class1 - final_logit_class0 and chi5(mask)=(-1)^(mask.bit_count()). The fifth-order signed Fourier coefficient and its teacher-aligned orientation are

    a5 = sum(m(mask)*chi5(mask) for mask=0..31)/32
    teacher_alignment = a5*(2*y0-1).

Positive alignment means the fifth-order coefficient supports the context-correct parity sign. Negative alignment means an inverted phase. Its square contributes positive fifth-order power regardless of correctness. Normalized signed coefficient is teacher_alignment/sqrt(sum(a_S^2 for S=1..31)) when the nonconstant energy is positive.

## Registered read-only diagnostic family

Use every original root (12..17), each native arm (aux_prefix, aux_shuffled, vanilla), and both already archived 240-step and 900-step checkpoints. Root remains the independent reporting unit. For each original owned validation context whose original MUX=1, require the complete 32 masks in chronological focal-mask order, with the original SHA-verified teacher labels and all 32 archived final native logit pairs.

Report original native parity accuracy and mean unsigned fifth-order fraction, root-level mean signed normalized coefficient, mean signed raw fifth coefficient, number of phase-aligned / inverted / near-zero contexts. A fixed descriptive 10%-of-nonconstant-power threshold reports high-power inverted contexts, and 1e-9 absolute fifth coefficient classifies the near-zero case. These two thresholds are **posthoc descriptive** and have no promotion authority.

A deliberately teacher-informed **phase oracle** replaces only the sign of the archived fifth coefficient with its original true context sign while leaving all other Walsh coefficients unchanged. Mathematically the margin changes by (sign_target*abs(a5)-a5)*chi5(mask). Report its label accuracy and delta from original task accuracy as a **posthoc diagnostic upper-bound-style counterfactual**. It uses validation truth and cannot be implemented as a learned readout or causal model intervention. Improvements do not establish that the trained network knew the context phase.

All root, arm and checkpoint results remain in the denominator; compare six root macros descriptively, without fresh population p-values, bootstrap confirmation intervals, training claims or changes to original NEC04F gates. Preserve negative original training result and report unlearned roots13,14,15, including their near-chance auxiliary performance.

## Provenance and independent scalar audit

Download from the successful historical NEC04F hosted run [45015714628](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45015714628) only:

- Pretraining ledger and independently audited original source/owned teacher label rows, artifact 14751816377.
- Six root model result and independent audit artifacts for original roots12,13,14,15,16,17: artifact IDs 14751701971, 14752095879, 14751577430, 14751617531, 14752185980, 14751747110 respectively.

Check every original source/owned row-file SHA, root ID, world/program SHA, checkpoint model/optimizer identity, old independent full native + four auxiliary logit replay flags, original source cohort and reserved/confirmation access counts. Validate per-context actual native correct counts against the accepted original NEC04F validation parity score.

The first scalar implementation computes all 32 Walsh coefficients by direct character sums. The independent audit separately reconstructs them by a five-stage fast Walsh-Hadamard butterfly, checks every derived context phase, coefficient, power, full-context accuracy and teacher-informed phase-oracle prediction. Both analyses use **only immutable archived original logits**. All source and audit method tests are CPU-standard-library synthetic arrays; none need Torch or an Inkling model.

This is useful as an **air-support diagnostic**, designed to inform the NEC04F agent's next candidate training protocol. It should remain in a separate draft PR and avoid changes to the agent's already merged NEC04F source or any sealed world.
