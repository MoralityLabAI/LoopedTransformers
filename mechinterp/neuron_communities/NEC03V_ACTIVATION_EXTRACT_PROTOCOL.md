# NEC-03V registered source-only activation extraction

**Pre-execution protocol.** Freeze this protocol, both extraction methods and original source-model dependencies by Git blob SHA1 in NEC03V_ACTIVATION_EXTRACT_LOCK.json BEFORE any new model inference. FSI-01 remains REDESIGN_OR_PILOT_ONLY.

## Data scope and purpose

Recover the original source-trained SAE's missing code-activation evidence: 1,280 source-fit TopK16 positive-code incidence sets and 256 original source-training case positive coefficients per model arm and registered recurrent visit. Use only historical dev5-000..007, the original NEC03T/NEC03U byte-pinned native and SAE checkpoint families, the prior 128-by-32 independently audited decoder columns and the original source-fit groups 0..39. The intervention cases are original training groups 24..31, 32 masks each, all contained within the source-fit subset.

This is **new native forward inference on previously opened training-source histories**, with explicit receipts. The source model and SAE weights are frozen. No root 8..95 recruits, development-evaluation groups 80..103, reserved groups 104..127, embargoed worlds or old confirmation worlds may be submitted for inference.

## Model and decoder contract

- CPU-only original Looper with 12 recurrent visits and memory-token 64D GELU pre-mlp_down activation at the originally registered visit.
- Frozen SAE normalization mean/scale and original encoder: ReLU followed by TopK16. Positive activations are feature values above 1e-7. Some of the 16 selected TopK entries may be zero: retain a variable length 0..16 positive-code list. A case with fewer than 16 positive entries is a registered invalid pair and gets zero source and sham perturbations at all visits without losing denominator membership.
- Original projected decoder32 columns, already independently audited, are read by byte identity. No new source site, native hub or decoder can be selected.
- Fixed source fit 40 complete groups times 32 masks = 1,280 rows per arm. Registered intervention rows are fit indices 768..1023 (groups 24..31).
- Fixed CPU batch size 32. Up to 40 new full 12-pass native forward calls per original arm with registered selected sites, and zero calls for an arm lacking all source sites. Count both forward calls and per-visit SAE code applications. Record model and SAE weights unchanged, zero optimizer updates, zero task labels accessed and zero model logits stored. The native forward computes logits transiently as a byproduct of the original GELU hook; its output is discarded and must be reported as such.
- Enforce process-tree RAM/time budgets via the existing CPU harness. Never allocate GPU, use paid cloud, train weights, compute scoring losses or print prediction labels.

## Phase gates

1. Commit the exact extractor source, separate independent auditor and this pre-execution protocol. Pin their Git blob SHA1s and all original decoding/generator/source-loader dependency blobs in a distinct commit containing NEC03V_ACTIVATION_EXTRACT_LOCK.json. Native source-only forward execution must reject missing or mismatched lock bytes.
2. Execute only after the registration commit exists. Require explicit CLI --allow-registered-source-forwards. Default dry-run must validate the original source and decoder chain with zero forwards.
3. Independently recompute original native GELU with a separately implemented forward hook, then reimplement ReLU+TopK from original SAE encoder tensors and compare 1,280 source-fit positive ID lists and all 256 original-case coefficients. All 12 visits of the native model are preserved, with exact source-only forward accounting and SHA-pinned original source identities.
4. Run the existing pure-standard-library four-code coherent group/disjoint sham physical geometry preflight on the newly authenticated data, preserving invalid cases as paired zero perturbations. Recompute physical 32D doses with a second independent geometry auditor. No behavioral scoring is authorized by this preflight.

## Versioned outputs

One original-world output directory with composed.json, additive.json, extraction_receipt.json and manifest.json. Evidence class is registered_source_only_activation_extraction_unscored. Each shard contains only original source-fit positive-code IDs, original source case positive ID/value pairs and already authenticated 128x32 decoder columns. The receipt includes model/SAE source checkpoint attestations, decoder audit digests, original source-fit input SHA, source group membership, native model forward counts, registered lock SHA and explicit no-label/no-optimizer/no-protected-world fields. An independent audit separately binds the genuine original source inputs and recomputed coefficients. Preserve audit JSON SHA and frozen method Git blob SHA1.

These input geometry records are still only source-training mechanism data. Any new behavioral source-group necessity experiment requires a distinct frozen scorer, independently audited recurrent transport, physical sham-resolution requirement and original eight-root root-level correctness test with meaningful +2pp floor. NEC03P recruits (69 source-eligible among 88) have zero new causal sham outcomes, and the 59-root 95/95 tolerance campaign remains suspended. Original NEC03T and NEC03U negative results stay immutable.
