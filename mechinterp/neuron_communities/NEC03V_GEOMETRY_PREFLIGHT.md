# NEC-03V: source-only geometry feasibility preflight

**Status:** DESIGN / NO BEHAVIORAL SCORING. FSI-01 REDESIGN_OR_PILOT_ONLY.

Companions: NEC03V_SOURCE_GROUP_REDESIGN_DRAFT.md, NEC03T_SOURCE_LOCK.json, and NEC03U_ARCHIVE_PROTOCOL.md.

This is a **pure Python standard-library** preflight. It reads only user-supplied source-geometry JSON. It never loads PyTorch/native models or sparse autoencoders, makes no model forwards, reads no task labels, and opens no recruited/embargoed/confirmation cohort.

## Missing original source artifact

The immutable NEC03T results contain active code IDs and aggregate physical doses, but do not include all individual positive code coefficients and projected 32D SAE decoder columns needed for the proposed four-code intervention. A real feature-geometry pack must first be independently authenticated from the exact source-fit/source-rank data and source-trained checkpoint. **The pack hash alone is not an authenticity proof.** If regenerating those vectors requires new inference, register and separately approve that source-only extraction and its frozen code before executing it. This implementation authorizes no such extraction.

## Input contract

A local directory contains manifest.json, composed.json and additive.json. The manifest schema is NEC-03V/label-blind-source-geometry/v1; it names one of the original historical roots 0..7, declares source groups 24..31, binds the immutable NEC03T source lock SHA-256, declares both model/SAE/source-site upstream receipts and provides SHA-256 for each arm shard.

Each arm shard records precisely 256 source-history cases, enumerating eight groups times 32 masks. For each previously selected recurrent visit it supplies 128 projected decoder columns of 32 coordinates and 1,280 label-free source-fit TopK16 active ID sets. Each per-case visit supplies 16 sorted active positive-code IDs and activation coefficients. Extra fields, task labels, logits, protected-group indices and malformed source identities are rejected.

Original source-ineligible root 3/5 arms may supply an empty source-site set. Those 256 records remain invalid zero interventions.

## Frozen feature-group and physical-null computation

Within an original model arm/root/visit, form a pairwise graph of 128 SAE features from source-fit coactivation Jaccard and positive cosine between projected 32D decoder columns, with equal 0.5 weights.

Enumerate every four-code group among the 16 active positive features. Choose the highest-coherence connected group, with feature-ID tie breaks and a minimum average six-edge coherence of 0.10. Choose a disjoint four-code sham from the same admissible active-code set, prioritizing closest source-fit coherence, then closest projected norm, then IDs. Require structural coherence gap <= 0.35.

For projected vectors g and h, compute d = min(norm(g), norm(h)). Require d >= 1e-4 and use attenuation gains d/norm(g) and d/norm(h), each no greater than one. Independently verify physical output L2 matching within 1%. The locked intervention amplitudes are 0, 0.25, 0.5 and 1, with full dose the proposed primary. This preflight computes **geometry only**, never task outputs.

If any selected visit is invalid, both effective removal vectors become exactly zero on every selected visit of that case, retaining the case in the original 256-case denominator. The preflight records source/sham IDs, candidate counts, graph SHA-256, structural scores, physical dose, gains and effective intervention vectors.

## Commands

From the repository root, using Python 3.11+:

    python -m mechinterp.neuron_communities.nec03v_geometry preflight --pack PATH_TO_AUTHENTICATED_SOURCE_PACK --out preflight.json
    python -m mechinterp.neuron_communities.nec03v_independent_geometry_audit --pack PATH_TO_AUTHENTICATED_SOURCE_PACK --preflight preflight.json --out independent-audit.json

A second, separately implemented auditor reenumerates the selected and disjoint sham groups and recomputes the physical 32D dose, attenuation, selected-site membership, zero-case behavior and complete denominator. It explicitly reports external_source_provenance_authenticated=false until the original geometry source has an independent upstream attestation.

For numerical tests only:

    python -m mechinterp.neuron_communities.nec03v_fixture --out /tmp/nec03v-synthetic --world-index 0
    python -m mechinterp.neuron_communities.nec03v_geometry preflight --pack /tmp/nec03v-synthetic --out /tmp/nec03v-preflight.json
    python -m mechinterp.neuron_communities.nec03v_independent_geometry_audit --pack /tmp/nec03v-synthetic --preflight /tmp/nec03v-preflight.json --out /tmp/nec03v-audit.json

Synthetic fixtures are unmistakably labeled and cannot establish an actual native neuron effect. Hosted CI runs these synthetic tests without Torch or model downloads.

## Required gates

**Source identity.** Before scientific use, independently authenticate each original 0..7 root, both teacher arms, original selected visits, native/SAE/decoder source checkpoints, and 24..31 source-only histories. If any individual decoder contributions are missing, register a dedicated, label-blind extraction step with model-forward accounting in a separate PR. No current archive output may be silently invented.

**Physical null.** After a real original-source geometry pack exists, independently audit all eight roots and both arms, original root3/5 fixed ITT zeros, valid/no-op rates, nonamplification, matching tolerance and selected group ownership. Freeze the minimum physical coverage and sham variation-resolution thresholds before viewing any outcomes. Physical dose variation cannot by itself establish behavioral sham variation.

**Behavioral scoring.** A later, separately preregistered native scorer and independent auditor must precede a source-only pilot. It retains the original root-level composed-minus-additive physical sham-versus-source correctness contrast and the +2 percentage-point meaningful floor. All five-input recruited worlds, original reserved data, 32 embargoed five-input roots and 64 four-input confirmation worlds remain sealed. The 69 recruited source-eligible roots still have zero causal-sham outcomes; the planned >=59-root tolerance study remains suspended.

**No model inference, new source-panel access, external data opening, behavioral effect estimate or statistical equivalence is claimed here.**
