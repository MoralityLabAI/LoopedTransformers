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
| Prime verifiers v1 | Optional source/task/reward adapter; offline SDK check | None |
| research_gym saved Sudoku/ARC/routing/storyworld comparisons | Existing suite source reviewed; native Nanbeige task adapters pending | None |
| SAE/VPD/BPD cross-reference | Same-site physical geometry function, numerical fixtures | None |

CPU contracts:

    python -m unittest discover -s mechinterp/nanbeige_bpd_gym/tests -v

These tests validate interchange and coordinate safeguards, not language-model competence.
