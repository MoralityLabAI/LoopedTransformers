# NEC-03W source-only calibration coefficient extraction, preregistration

**Status:** PRE-EXECUTION CPU SOURCE-ONLY EXTRACTION. FSI-01 REDESIGN_OR_PILOT_ONLY.

The factorial-complementarity gym uses an outcome-blind source feature quartet menu ranked only on original source-fit groups 0..15. The original source-trained SAE fit already saw source contexts 0..39. A later source-only functional selector needs positive TopK16 **coefficient values** for calibration groups 16..23, whereas the original NEC03V archive stored per-case coefficients only for previously behavior-scored groups 24..31. Positive feature IDs from all 0..39 were already independently verified in that archive.

This protocol recovers exactly **256 original source-fit calibration histories** per model arm, groups 16..23, all 32 five-input focal masks. It is an additional, precisely counted CPU-only inference of the frozen native original model and SAE encoder. It produces only source-positive SAE code coefficients and a source-custody receipt. Task labels, errors, logits, group/sham behavioral outcomes, task gradients and optimizer updates are never evaluated or saved.

## Locked use of existing sources

- Original dev5-000..007 only. The external source pack must have the exact original manifest SHA listed in NEC03W_CANDIDATE_LOCK.json and the original NEC03V source/model/SAE checkpoint receipts. The independently audited original source sites and original source-five-input registry/split must remain unchanged.
- Calibration input is source catalog train rows **512:768 only**, the 8 source groups16..23. The source-fit discovery 0..15, previously NEC03V scored groups24..31, source query groups32..39, original source rank groups40..79, new development evaluation, reserved, recruited, embargoed and confirmation contexts are excluded from this extraction.
- Each original model arm with registered source sites receives eight native model forward batches of 32 histories. An original source-ineligible arm with no source SAE receives zero forwards and a complete 256-case empty-site record. Each original native forward executes its 12 tied recurrent visits and transiently computes original binary output logits, which are discarded; the saved output contains only registered-source SAE code IDs and positive float coefficients.
- The original source-model and SAE weights are frozen. GPU allocation, further training, model-gradient tracing, loss computation, prediction scoring and source-root extensions are prohibited.
- Use the native original mlp_down memory-token GELU pre-forward hook and the frozen SAE mean/scale plus trained TopK16 encoder. Positive codes have activation >1e-7. Original TopK16 can contain exact zero features, so an original source history can have fewer than 16 positive codes; preserve it exactly rather than padding.
- Every new positive ID list must match the corresponding independently archived source-fit positive ID list at rows512..767 before a coefficient can be saved.

## Independent mechanical audit

A separate independent auditor computes each native 12-pass memory-token GELU activation with its own hook and manually reconstructs frozen SAE encoder affine map, ReLU and TopK16 from the original checkpoint tensors, comparing every saved source-fit row ID and coefficient with an absolute 2e-6 and relative 1e-6 tolerance. Model weights must remain unchanged and all model forward batches counted. A duplicate, absent or changed original source group/visit fails the batch.

The extraction and audit use two distinct CPU model forward passes per original eligible teacher arm; their outputs must match independently. The exact method source blobs, this protocol, source source-fit catalog, original checkpoint loader and parent original-source code registration are sealed in NEC03W_CALIBRATION_CODE_LOCK.json before new original-source forwards.

## Output and future gates

The output calibration_codes.json contains:
- Original world ID, 256 original case identity by absolute index within groups16..23 and selected source visits;
- Exactly 256 rows of original positive IDs/coefficients per selected visit and model arm;
- SHA256 of the original source-fit 256 input-bit rows, byte-anchored native/SAE/source-roster provenance, old source-pack manifest hash;
- Exact new CPU model forward counts per arm and independent audit receipt;
- Explicit zero labels or behavior scores, optimizer steps, query accesses, recruits or confirmations.

The query group32..39 positive ID list is already present in the original source-fit archive but its new positive **coefficient values and behavioral outcomes remain out of this calibration extraction**. A later functional candidate selection must be preregistered using calibration groups16..23 and the fixed candidates, with minimum 64 admitted cases and strong sham controls, then SHA-seal the selected quartet/partition before doing a separate source-query extraction and native behavioral test.

**This stage cannot establish joint necessity, successful rescue, or population generalization.** Original NEC03V negative evidence stays frozen, and all 88 recruited roots plus held-out confirmations remain sealed.
