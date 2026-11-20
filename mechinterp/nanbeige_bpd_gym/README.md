# Nanbeige BPD gym integration

This package connects the frozen NAN-DIAG-04 source-only component task to optional evaluation infrastructure and provides a physical-coordinate SAE/VPD/BPD cross-reference function. It does not change the registered diagnostic, enable BPD transfer, fit a dictionary, or claim wider-suite scores.

## Local diagnostic and evidence

The prospective registration is 10936e4caac06115e4a53b600cbab658295cc246. Read experiments/nan_diag04/PROTOCOL.md before inference. The checkpoint is 451ed48c3273ecef7ea8faaa43c31ce529763bb1. BF16 and NF4 remain distinct conditions. The original 168-case ARG-03 test is unsupported and stays sealed.

The progress command inspects root filenames and operational receipts only; it never grades incomplete predictions:

    python -m mechinterp.nanbeige_bpd_gym.progress --out reports/nanbeige_bpd_gym

The existing scientific protocol has 24 paths/root and eight DEV roots, not 192 independent examples. Assistance and copy arms intentionally carry gold information only in their own prompts. Task loaders require the exact registered source-seal SHA before opening the public or host-gold source files, and support only CAL/DEV.

## Optional Prime Intellect adapter

Reviewed SDK: https://github.com/PrimeIntellect-ai/verifiers/tree/ed70053b6a7a4079d6976e03bdcd7d420356d594
API documentation: https://github.com/PrimeIntellect-ai/verifiers/blob/ed70053b6a7a4079d6976e03bdcd7d420356d594/docs/tasksets.md

Install that exact SDK in a separate environment; do not upgrade the pinned 3050 inference environment. Import NanbeigeDiagnosticTaskset and NanbeigeDiagnosticConfig from prime_taskset.py, then instantiate the taskset with a chosen component and split. TaskData private answer fields are grader data; only its prompt is supplied as the user prompt. Verify offline construction and correct/invalid reward examples with:

    python -m mechinterp.nanbeige_bpd_gym.validate_prime --out prime_sdk_validation.json

The offline check does not launch an agent, model, sandbox, cloud API, training job or billable evaluation. On Windows, importing the SDK's real Trace/runtime path currently requires the Unix fcntl module; deferred type annotations allow source loading and direct reward checks without pretending that the full runtime was validated. Use a compatible Linux deployment for actual runtime validation.

The adapter scores the COMPLETE final reply using the anchored bounded grammar, whereas the registered local diagnostic stops at the first complete generated prefix. These are different endpoints: any future Prime rollout must be registered separately, and cannot be pooled into NAN-DIAG-04's 15-comparison family.

## SAE towers and VPD/BPD coordinates

crossref.probe_and_crossref accepts physical residual states and frozen SAE normalization/encoder/decoder arrays. Required metadata binds the exact checkpoint, BF16/NF4 condition, zero-based layer (0..21), native loop (1 or 2), residual_after_layer site, width 3072, file SHA and calibration-fit manifest SHA. A dictionary fitted on DEV is rejected.

The function reports top-four nonnegative SAE codes, physical/normalized reconstruction error, encoder/decoder direction cosines and encoder preactivation sensitivity to supplied VPD output directions. Optional BPD directions add geometric cross-references only. No model state is edited, and geometry does not demonstrate causal fidelity or transfer.

Existing exploratory VPD down_proj right/input factors are width 8192, while physical output directions are width 3072. Equal width alone is insufficient: backend, layer, loop and site must also match. Post-final-norm directions require an explicit Jacobian/coordinate bridge and are rejected here. Zero-norm rows are counted and their undefined cosines are represented by zero for serialization; they must not be interpreted as orthogonality evidence.

Only synthetic numerical fixtures have exercised the cross-reference function so far. No compatible trained Nanbeige SAE tower was located in the scoped local asset search. Native training/probing and causal BPD transfer require their own prospectively frozen CAL/DEV protocol and actual assets; no existing Sudoku/TRM dictionary is relabeled as Nanbeige.

## Wider-suite status

| Suite | Integration status | Neural scores claimed here |
|---|---|---|
| NAN-DIAG-04 | Frozen native local BF16/NF4 diagnostic | See its audited report |
| NAN-GYM-05 | Frozen native wider-suite and read-only tower pilot in progress | See streaming aggregate report |
| Prime verifiers v1 | Discoverable wider-suite Taskset; pinned real-SDK iteration/reward checks | No hosted/model rollout yet |
| research_gym Sudoku/ARC/routing/storyworld | Native full-grid, full-plan and routing adapters in NAN-GYM-05 | Running, clearly partial |
| SAE/VPD/BPD cross-reference | Same-site physical geometry function, numerical fixtures | None |

CPU contracts:

    python -m unittest discover -s mechinterp/nanbeige_bpd_gym/tests -v

These tests validate interchange and coordinate safeguards, not language-model competence.

## Registered wider-suite adapter

NAN-GYM-05 registration: 18fb185dcda884473d51fe9c420654418c6a9f49. Its 322 contexts have 32 CAL pointer roots and 57 DEV roots. [Streaming report](../../reports/nanbeige_bpd_gym/NAN_GYM05_PROGRESS.md) and [frozen protocol](../../experiments/nan_gym05/PROTOCOL.md). The local routing corpus is excluded from public fixtures and raw uploads.

The official verifiers plugin ID is the top-level nanbeige_bpd_suite entry point, exporting exactly NanbeigeSuiteTaskset via __all__. Task definitions live in mechinterp.nanbeige_bpd_gym.prime_suite. Default scope public contains 89 DEV main task rows from 41 root groups; local scope adds 16 routing trajectories when the exact registered local files are available. Test/unknown source scopes are refused. The complete-reply grader uses native full grids and full storyworld rollout verification; it never scans explanations for labels. Private TaskData labels stay on the evaluator.

    python -m mechinterp.nanbeige_bpd_gym.validate_prime_suite --scope local --out prime_suite_validation.json

All 105 local DEV main rows passed real pinned-SDK construction, iteration indices, correct-reference reward and invalid-reply checks on Windows. This is software validation, not a model/hosted benchmark. An isolated Ubuntu Python 3.12.12 runtime now passes the official SDK discovery/import path and all 89 public DEV main-row gold/noop CLI checks, plus a separate two-row setup check. The existing pinned Windows inference environment remains unchanged.

Pinned model geometry matters: the actual checkpoint has residual width 3072, down_proj input width 10752 and o_proj input width 6144. An older 8192-wide factor from another setup cannot substitute for this effective operator. The native source confirms down_proj output is added to the layer residual; o_proj output still passes through the MLP before the layer-output SAE site.

Run the official CLI from this checkout with its root on PYTHONPATH (a console script does not automatically add the working directory):

    PYTHONPATH=. vf-validate nanbeige_bpd_suite --runtime.type subprocess --max-concurrent 1

This command performs local grader setup/gold/noop checks without an agent or model. The explicit subprocess setting avoids the cloud runtime. Model/hosted rollouts are a separate pending requirement, not inferred from these checks. [Full goal requirements](../../reports/nanbeige_bpd_gym/ROADMAP.md).

## Observational tower reports after a completed native backend

The post-registration audit/probing helpers are separate from the 15 frozen inference/fitting files. They were added after partial BF16 behavior was visible and before any SAE fitting; their outputs are exposed development diagnostics, never confirmation or causal mediation. Every registered source, layer, loop and rank remains included.

    python -m experiments.nan_gym05.grade --runs runs/nan_gym05 --out runs/nan_gym05/graded --allow-partial
    python -m mechinterp.nanbeige_bpd_gym.audit_native --backend bf16 --out runs/nan_gym05/audit/bf16_independent.json
    python -m mechinterp.nanbeige_bpd_gym.probe_towers --backend bf16

Run probe_towers only after the frozen SAE fitting command completes. It requires a passed backend audit. Aggregate normalized/physical reconstruction, initial-dictionary controls, all-feature activation frequencies, loop gaps and SVD/Haar output-direction geometry are publishable. Per-record codes and indices in probes/local stay local, including all routing derivatives. No trained VPD/BPD or causal-transfer result is inferred from SVD geometry.

Once both complete backends pass independent audits and all eight SAE fits exist,
run `python -m mechinterp.nanbeige_bpd_gym.compare_native`. It pairs all 322
contexts in physical residual coordinates, reports root-macro state/loop drift,
cross-dictionary physical reconstruction and SVD output-subspace principal
cosines. Separately fitted sparse feature indices are never subtracted. This
post-registration diagnostic was specified after BF16 results and before
completed NF4 outcomes; it has no inferential or causal claims.

NAN-PRIME-06 is the separate [frozen full-reply condition](../../experiments/nan_prime06/PROTOCOL.md).
Its finite queue waits for native inference/fits to exit, then executes official
local Prime NF4 and BF16 rollouts without another scope question. Use
`audit_prime --backend nf4 --snapshot <pinned-local-snapshot> --out <aggregate.json>`
in the pinned model Python environment after completion. The audit only loads
the tokenizer, matches every real SDK episode to fresh worker evidence, checks
whole replies with an independent parser/grader, and verifies all physical
state dimensions. Mock transport receipts are software evidence only.

## Native BPD engineering bridge

`native_components.py` reuses the actual BPD framework at
`8f658b789138d73f497b1a2a05f5783745147d40`. Its lock verifies all ten root Python
files against commit bytes, allowing only Git CRLF-to-LF conversion. It retains
the real BF16/NF4 module kernel, applies shared read/write factors around its
output, preserves all-one clean closure, and exposes pass-indexed reached-input
controller callbacks. Physical-module hooks preserve the BF16 staging layout.
The BPD model's placement method moves only added factors and buffers, leaving
the full output head and embeddings on CPU.

The differentiable loop reuses verified immutable CPU weight views for staged
BF16 operators, caches other frozen weights once and snapshots activations under
a shared 2-GiB live-added-CPU-memory cap. Its output
head streams all vocabulary rows in forward and backward; input derivatives
accumulate in FP32. Seven CPU engineering tests cover genuine-framework output
and factor-gradient parity, pass-mean/sphere ablations, selected-stage hooks,
reached-state callbacks, diagonal/permutation gauge relabeling, and full-head
derivatives. General rotations/hard-mask gauge identities are not established.

`validate_native_grad` checks a small random two-layer Nanbeige model against an
all-resident graph, including two recurrent passes, native all-one closure,
staged/NF4 backward parity and frozen target parameters. It has a 10% allocator
cap and loads no pretrained tensors. Passing these engineering tests does not
constitute trained Nanbeige VPD/BPD, competent causal transfer, or confirmation.
Native fits still require their own prospective recipe and source manifests;
the original algebraic competence gates and 168-case test seal remain intact.

The [v1](receipts/native_gradient_random_v1.json) and
[CPU-weight-reuse v2](receipts/native_gradient_random_v2.json) random GPU fixtures
both passed. BF16 and NF4 all-one state/head closure, masked forward parity and
factor-gradient errors were exactly zero against reference graphs. Peak CUDA
allocation was 75,769,344 bytes. V2 verified nine BF16 staged-weight bindings and
used 26 CPU aliases. These are two-layer random-model engineering checks;
pretrained Nanbeige factor fits and causal benchmarks remain separate stages.
