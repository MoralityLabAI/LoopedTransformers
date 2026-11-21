# NEC-03K source-only angular control feasibility

**Stage:** source-training-only engineering and mechanistic control availability, before any new causal response on previously evaluated worlds. This protocol uses the eight frozen NEC-03H-v2 development models, source rank SAEs and native lesion sites. Their original correctness results remain unchanged.

## Research question

NEC-03J showed that equal immediate MLP-down output L2 norm does not imply comparable effect on the final margin. The hubs-only margin gradient predicted the sign of observed nonlinear donor differences on 89.44% of 24,576 previously exposed, correlated focal cases. Source-ranked active TopK donors performed better than four different active hash-sham controls in dev-009 and worse in dev-013.

The proposed null matches the *projection direction toward the hubs-only, clean-chosen model margin* as well as component count and physical dose. Matching a mediator can remove real causal influence, so this experiment keeps the prior unconstrained hash-control comparison as a separate descriptive reference. It cannot establish general source feature superiority.

## Exact source allocation

Input worlds are **dev-008 through dev-015**, already used in the NEC-03H-v2 and NEC-03J experiments. Model checkpoints are reused from GitHub Actions run 44878411694; each is a previously frozen 900-step PEF-07-style tied recurrent model. Its source-only TopK16 128-feature SAE, 64D native GELU basis, 32D downstream linear projection, per-visit source feature ranks, native graph/fallback sites and hub masks are reused byte-for-byte.

This stage uses only the first **16 of the original 160 parameter-training context quartets**, each with sixteen focal-bit counterfactual histories, per world. These were already used for source-only model parameter learning and SAE training. The original 48 development-scoring context quartets, original 48 *reserved* context quartets, untouched dev-016..063 teacher worlds, and all 64 SHA256-committed original confirmation worlds remain **unopened by NEC-03K**. The existing 128-world registry digest and confirmation full-spec commitment are checked before any source computation.

## Label-blind per-input angular matching

For each training input, fix the clean model's argmax predicted class. On the source-selected recurrent visits compute the exact gradient of that same class's final binary logit margin on the **hubs-only-lesioned** reference trajectory, with respect to the post-MLP-down 32D memory-token output. This step consults the current input and the frozen model, not its true label. The 64D/32D Jacobian basis and finite-difference contracts remain inherited from NEC-03J.

The treatment is the sum of the **n highest source-ranked currently active SAE codes**, where n is the original native community count at that visit and lies between one and eight. Each source-trained TopK SAE produces 16 currently active decoder code indices. Exclude the treatment's n codes from the control pool.

For each of the complete **C(16−n,n)** possible disjoint, equal-cardinality candidate control sets, form the *physical 32D output direction* by summing the selected active SAE decoder contributions and projecting with the frozen 32×64 MLP-down matrix. Compute the cosine between the 32D output direction and the hubs-only clean-chosen class gradient. Select the control with the **smallest absolute difference between its cosine and the source treatment's cosine**, breaking ties by sorted native feature indices. This exhaustive algorithm is fixed before evaluating any new context and never scores downstream correctness during selection.

Per example and recurrent visit, require **absolute cosine difference ≤0.25**, nonzero donor output norms and a valid gradient norm. Independently compute both donor output L2 norms, attenuate both to the smaller available direct MLP-down output norm with gains in [0,1], and require the previously established physical feasibility criteria: ≥99% site/example dose matching within 1%; ≥80% positive shared dose; median fraction of original source donor dose retained ≥25%. The combined angle+positive-dose match fraction is required to be **≥80%** at the world level, without discarding any failed source sample.

Angle matching is input-dependent and uses clean/hubs gradients computed on the input being considered. The **algorithm and thresholds** are source-frozen; the identities of the selected active control codes legitimately vary with each input. This is a label-blind conditional control, rather than a fixed source-only list of code IDs.

## Audit and hard stop

For every case, the runner stores the entire gradient vector (32D), source output vector (32D), all 16 currently active physical decoder output contributions (32D each), selected source and control feature IDs, candidate subset count, cosine matching gap and decision, original and retained physical L2 norms, actual gains, and direct physical dose-matching telemetry. All sixteen focal masks in each of sixteen source contexts are retained, nested within one frozen trained teacher/world root.

An **independent pure-stdlib auditor** reconstructs every possible active disjoint control subset from the 16 per-feature 32D contribution vectors and proves the chosen candidate minimizes angular mismatch. It also recomputes the source and control direction cosines, direct L2 norms and gains, every physical admission gate, and the source-only context assignments. Any tampering with the frozen model/SAE/source lock or logged case source causes failure.

This initial stage is **calibration only**. The 8-root source panel establishes whether 80% angle matched controls and physical-dose adequacy can be reached *before* deciding whether scoring any previously exposed development-evaluation contexts is worthwhile. If matching is mostly impossible, any alternative with widened angular threshold or active support becomes a distinct versioned development protocol. Do not use heldout correctness to tune thresholds.

The scientific **FSI-01 verdict remains REDESIGN_OR_PILOT_ONLY**, because eight old worlds, even with exhaustive source-sham construction, cannot calibrate a 95/95 independent-world sham distribution. Any next experiment on new 5-input Boolean task families requires an independent source registry and new training/checkpoint verification, with the original 64 committed confirmation worlds still sealed.
