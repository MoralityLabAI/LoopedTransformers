# NAN-GRAD-15: conditional real-checkpoint gradient/resource preflight

Freeze, commit and push before any GRAD15 pretrained forward/backward. Wait for
the actual COMP14 supervisor to terminate, including its final audits/report.
Both backends must pass COMP14's unchanged focused first-token loop2 screen.
Do not substitute legacy/bounded results or relax a threshold. Original FIT07
and algebraic failures stay unchanged; the original 168-case test stays sealed.

## Scope and fixed design

This is an engineering prerequisite to actual NF4 factor fitting. It is NOT
trained VPD/BPD: optimizer steps=0, no CI training, no donor transfer. Existing
random-model gradient fixtures do not certify native 3B feasibility, so measure
native full-vocabulary closure, gradients and memory at depths 1/2 first.

Use the first two COMP14 CAL roots in source order, BOTH worlds, focused arm:
four fixed prefixes, regardless of their individual correctness. NF4 aggregates
were already exposed during design; there is no best-prefix selection. Reuse
the exact prompts/tokenization and compare both baseline loops to archived
COMP14 greedy logits/normalizers. No host answer is parsed by the model worker.
The host eligibility receipt contains aggregate gate outcomes and audit hashes.

Each backend loads once, NF4 then matched BF16. Before components, obtain the
four native two-loop clean references and check full-vocabulary CPU/resident
head parity on the first prefix. Lock target parameters and backend-specific
cached SVD factors from GYM05 by byte/operator hashes. Keep actual NF4 kernels;
never substitute dense dequantized forward execution.

At layer 21 down_proj rank 8, initialize the actual pinned BPD framework's native
shared read/write factors. For each depth 1/2 and each fixed prefix:

1. Autograd-enabled all-one component/residual masks must reproduce every
   selected normalized boundary and full-vocabulary readout bit-exactly.
2. Delete that graph. Apply fixed component mask .35/residual mask .7, zero
   ablation, no sphere rescaling. Use the unchanged BPD readout-only trajectory
   KL, uniform pass weights, and native frozen-block recomputation.
3. Backpropagate once. V and U gradients must each be finite and strictly
   nonzero. Save full gradient matrices, clean/all-one/masked vocabulary
   readouts/boundaries and the scalar loss locally.
4. Zero gradients between prefixes; never update factors or target weights.
   Check factor values unchanged and the full target parameter hash unchanged.

T1 failure stops that condition and leaves T2 unrun under its prerequisite;
otherwise retain all four cases per depth. A T2 failure stops its remaining
cases. Backend exceptions, OOM, saved-tensor-cap or budget failures are
preserved; no neural retry, smaller substitute, changed mask or lowered cap.
BF16 is still attempted as its fixed control. No outcome-selected case removal.

## Audits and resources

Independent terminal CPU archive audit: upstream prefix/case hashes, baseline
greedy identities/full log-normalizers, exact clean/all-one arrays, all gradient
shapes/finite values/nonzero norms, saved-tensor limits and archive hashes.
Recompute archived-logit KL in float64 NumPy and compare the float32 primary
with absolute tolerance 5e-5/relative 2e-4. This is not a second full model
backward, finite-difference check or proof of the gradient's numerical fidelity.

One large model, one numerical CPU thread (OMP/MKL/OPENBLAS/NUMEXPR=1), Below
Normal. Preserve the separate math process. Existing pinned offline snapshot,
Torch 2.6.0+cu124/Transformers 4.46.3/bitsandbytes 0.50.2 and BPD 8f658b7 only;
no packages/weights changed or paid APIs. NF4 all 22 decoder layers/head on GPU;
BF16 four layers/stage/head. CPU embeddings, 2-GiB added saved-tensor CPU cap,
CUDA fraction .85, 8192 vocabulary chunks, 3600 s between-stage backend budget.
Masking is unmatched-dose engineering, not a physically matched causal test.
Allocated/reserved peaks and sampled dedicated/shared counters are distinct.

All archives/case receipts are write-once. Observe append-only journals or
actual process handles, never live atomic progress files. Publish only terminal
audited aggregates/hashes. Raw arrays/tokens/replies stay local. Future actual
factor training needs its own prospective registration; passing this probe
does not establish optimizer feasibility, component specificity or transfer.

Unit: two reused CAL roots/four world prefixes, one checkpoint; depths/masks
and gradients correlated. Primary contrast: native NF4/BF16 differentiation
and resource feasibility. Scientific causal floor/sham floor: uncalibrated.
Formal comparisons: 0; no power, significance, equivalence or mechanism claim.
This exposed engineering pilot supplies no sealed causal confirmation. The
broader continuous goal remains active after this finite prerequisite.
