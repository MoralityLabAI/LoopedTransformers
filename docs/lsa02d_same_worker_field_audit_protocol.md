# LSA-02D: deterministic same-worker fits and exact public-field survival

**Status:** prospective engineering preflight, November 23, 2026. This follows the [LSA-02C raw/typed development pilot](../reports/lsa02c_typed_public_state_cpu_20261123.md), which observed native full-proof closure of 7/32 and 4/32 at seed 107, and variable outcomes at seed 109 across GitHub CPU runs. The model remained below the registered 40% competence threshold. This preflight precedes any new architecture, model scale-up, or gym.

## Fixed source and experimental scope

Reopen only the frozen, publicly generated **48-root, 259-frame native Lean training teacher** from [the corrected LSA-02C CPU run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45172467970). The training teacher is bound to its original standalone Lean 4.19 proof certificates, SHA-256 files, root census and explicit training-only source split. Calibration and all locked test roots remain unopened by the new scripts.

Every run uses **one GitHub-hosted CPU worker**, running the existing Plan B 57,475-parameter architecture with the same two-pass AB dispatch. Seeds are 107 and 109. For each seed, train two raw and two typed replicas, each from freshly seeded initial tensors and the identical theorem-root/frame sampler, with 720 FP32 AdamW updates and the same 384-byte encoder budget. This is 8 fits total, sequentially on one CPU host.

Determinism is tightened with `torch.use_deterministic_algorithms(True)`, one intra-op thread, disabled MKLDNN, fixed `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. The reporter saves the CPU platform, Python and Torch version and build checksum, initial and final state hashes, exact tensor equality, largest tensor difference, checkpoint bytes, and optimizer trace SHA-256. A same-view repeat passes only when tensor values **and the optimizer loss/draw trace** match exactly. The raw and typed first replicas must start with equal initial parameters at the same seed.

**A success proves same-worker repeatability under this exact environment.** It does not establish cross-runner or cross-Torch-build reproducibility. Any mismatch is archived as a failure; a subsequent rerun must be separately identified rather than replacing the original receipt.

## Exact field-retention audit

The [source-level audit](../research/lean_cgrt_hrm400/ppt_gym/field_audit.py) re-serializes each *public* Lean state using the existing canonical JSON routine and checks its exact UTF-8 bytes against the source packet. It records byte spans for nested fields so that a raw tail string matching some different hypothesis is never credited as preserving the desired relation.

The actual `ByteEncoder` takes the **last 384 bytes** of raw JSON. A field counts as raw-visible only if its **entire canonical JSON value** begins at or beyond that byte boundary. The audit separately tracks the current goal, budget, failed choices, pretty-printed text, each candidate's complete public hypothesis type, noncandidate background types, and additional pending goal types.

For the typed representation, the audit counts the current goal and remaining budget in its explicit header and counts only complete semicolon-terminated public type entries produced by the already-registered `lsa02c.typed_public_state.v1` serializer. This makes the compared signals operationally explicit; the raw and typed measures are whole-field visibility tests under their respective grammars, not an assertion that their encodings are semantically equivalent. The raw `pp` field is deliberately omitted in the current typed view. The audit reports root-weighted and frame-total counts.

Only `goal`, `candidates`, lineage and frame index are read for representation reconstruction. The collector authenticates the teacher manifest and checks all source labels exist, while the audit itself never uses shortest-proof labels to choose fields. It runs with **zero native Lean calls, zero optimizer updates and no sealed evaluation exposure**.

## Gates and decisions

- **Source custody:** 48 native training lineages and 259 frames; all teacher hashes must match the immutable artifact.
- **Deterministic fit:** eight full 720-step CPU runs; exact duplicate saved tensor values and optimizer traces for each seed/view, plus same-seed raw/typed initialization.
- **Representation audit:** every typed input preserves the current goal and remaining check budget, and every per-frame public-field visibility measurement follows the exact byte-span rule. Record candidate and background type retention as the main diagnostic.
- **Model scientific competence:** remains **`BLOCKED_INCOMPETENT_CHECKPOINT`**. This study contains no new native proof-completion readout and no mechanism claim.

A complete raw-vs-typed field-retention report can motivate a next prospective variant that preserves type structure *and* selected public pretty-print context. The evidence may also show the raw tail retains more useful semantic information than its byte fraction suggests. The preflight does not select or train that future representation, and it leaves the LSA-02C calibration and locked test sets unchanged.

## Reproduce

The [GitHub CPU workflow](../.github/workflows/lsa02d-same-worker-cpu.yml) installs pinned CPU PyTorch 2.5.1, tests the field-span and checkpoint contracts, downloads only the original source artifact, then executes:

```bash
cd research/lean_cgrt_hrm400
python -m ppt_gym.field_audit \
  --teacher audit/source/lsa02c-teacher --out audit/field-results --max-bytes 384
python -m ppt_gym.determinism_audit \
  --teacher audit/source/lsa02c-teacher --out audit/deterministic-fits \
  --steps 720 --seeds 107 109
```

The complete model states and JSON diagnostics are retained as a short-lived Actions artifact. Following successful verification, commit a report with the exact run URL and field-retention counts. Nothing in this experiment modifies the prior frozen evidence.
