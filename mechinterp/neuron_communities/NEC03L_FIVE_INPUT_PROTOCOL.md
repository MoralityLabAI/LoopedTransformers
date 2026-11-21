# NEC-03L: Independent five-focal-input Boolean-world registry

**Registration stage only, November 21, 2026.** This is a new task family distinct from the earlier four-focal-input NEC-03G/PEF-07 worlds, intended for a future development-world matched-sham calibration. No native model, SAE or reserved/confirmation world is trained or scored in this stage.

## Motivation and independence claim

The completed NEC-03J and NEC-03K source-only studies isolated a physically feasible method to match both the immediate output dose of a sparse TopK donor and its local direction relative to the damaged recurrent model's margin gradient. Its original eight Boolean worlds have all been examined, and their source/SAE and matched-control results are development-only evidence. The original NEC-03H-v2 root-level causal ITT is +0.423 percentage points, below the registered +2pp minimum; only two original root-level sham families were fully feasible in the early controls.

A nonparametric one-sided 95/95 upper sham-tail tolerance bound requires at least **59 independently calibrated sham-world roots** even when using the largest sample. Only 48 previously unexamined original dev-016..063 worlds remain, giving at most 56 v2 worlds in that old finite grammar. A distinct five-input Boolean-program family offers 96 prospective development roots in a different task set before any old confirmation source is touched.

Each new root has its own five-essential-variable truth function, chronological focal-bit arrangement, source-context salt, and independently seeded PEF-style model checkpoint. This varies task complexity and input placement; all worlds still share a small synthetic Boolean alphabet and a compact recurrent architecture. Independence therefore applies to the newly generated *task/checkpoint roots*, not to arbitrary language-model generalization.

## New 128-world procedural registry

Enumerate three different five-input Boolean-expression tree shapes composed from AND, OR, XOR, NAND, NOR and XNOR gates, with four gate choices and independent input-literal inversion masks. Deduplicate by the **complete 32-bit truth function**. Accept only functions for which all five input variables are causally essential, 10–22 of the 32 focal states are positive, and the teacher is distinct from the proposed class-frequency-matched additive control. Canonicalize identical truth functions to one explicit teacher program. Deterministically sort with SHA256 under the new NEC-03L salt. Focal-position quintets are also distinct, with both earlier and later sequence positions represented.

Assign **96 full-development root specifications** and **32 additional embargoed roots** whose IDs and SHA256 spec commitments alone appear in the public registry output. Root details are procedurally regenerable from this open code, so the embargo is an **execution and outcome boundary** rather than a cryptographic secret. A separately authorized workflow is required before any embargoed root is trained or scored.

The original NEC-03G registry remains SHA256 **8816304a7db020bd0ad016acca7d9195dfd4633e3316a17bd8e4b8e912fd25b4**, with its existing 64 confirmation-world full-spec commitment **670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39**. The NEC-03L generator verifies these original commitments and leaves them unchanged.

## Complete 12-bit input closure and split geometry

Each five-focal-bit world has **32 complete focal counterfactuals** within each fixed context, and the seven nonfocal bits produce **128 separate source-context groups**. Deterministically SHA256-assign:

- **80 source-training groups** (2,560 input histories).
- **24 development-evaluation groups** (768 histories).
- **24 reserved groups** (768 histories).

No focal input counterfactuals are shared between source groups. These groups partition all **4,096** twelve-bit histories exactly once per world. Within the 80 source-training groups, the first **40** are available for training the source-only TopK16 decoder, while the remaining **40** are assigned to the source-attribution rank learner. The reduced source size per step, compared with the old 80+80 SAE source split, is explicitly a design change and requires independent source-model competence/SAE fit preflight before causal comparisons.

Each world has a class-frequency-matched additive teacher using unique binary weights [1,2,4,8,16] on the five focal bits. It produces exactly the same number of positive focal states as the composed teacher, with a separate independently seeded model if a later experiment trains it.

## Source selection, registration and scientific stop

This PR generates the deterministic 128-world source registry, writes 96 full development specs with 32 execution-embargoed hash commitments, and tests five-input teacher truth functions, independent root seeds, original confirmation commitment preservation, 4,096-history closure, full 32-state group membership and embargoed-world evaluation refusal.

**No five-input model training or causal readout occurs in this registration stage.** A future pilot must first preregister a single fixed training schedule, source-native graph/fallback site selector, TopK16 dictionary fit and 40-group rank policy, local output-dose and gradient-angle-matched null, exact root-level causal endpoint and conservative ITT gates. The first small new-root CPU pilot should establish clean model competency and source-angle feasibility before consuming the available 96 development worlds for an attempted 59+ eligible sham-tail calibration.

The original NEC-03H-v2 0.423pp effect, its two-percentage-point science floor, and the FSI-01 **REDESIGN_OR_PILOT_ONLY** verdict remain unchanged. The new registry makes a larger independent-world test possible; it does not authorize confirmation, infer a new effect size, or open the original 64 confirmation worlds.
