# NEC-03X: frozen continuous disjoint physical-subspace sham

**Prospective source-only registration.** Prior [NEC03V source coherent quartet study](../../reports/NEC03V_COHERENT_SOURCE_GROUP_RESULTS.md) had +1.220703 percentage points on eight original source-trained worlds, versus a +2pp meaningful threshold. Its strong active-code sham admitted only 517/1536 composed and 931/1536 additive source cases across six previously paired-source-eligible worlds. The independent [NEC03W read-only geometry audit](../../reports/NEC03W_ORIGINAL_SHAM_COVERAGE_RESULTS.md) traced most attrition to physical angle and decoder-group cohesion constraints. NEC03X registers a **new** and more flexible physical-null family before evaluating any newly selected source group.

## Frozen source worlds and historical source-only data

All original eight five-input Boolean development world roots dev5-000..007 with paired composed/additive source-native teachers and model-specific 128-feature TopK16 autoencoders, historical frozen reset1800 native checkpoint and original first40-fit/last40-rank source feature partition. Source graph visits, source rank and graph hubs are untouched. Register *only original source-training context groups 40–47 inclusive*, eight complete historical seven-bit contexts ×32 five-bit focal masks = **256 already-trained source inputs per teacher**, two original teachers per world, 16 arms attempted. All eight original roots remain in denominator. Root indices 3 and 5 are historic paired source-ineligible **fixed-zero primary ITT roots**, with their native and SAE arms still scored as diagnostics. No native or SAE optimizer updates or changes to original source data.

Original twenty-four development-evaluation and twenty-four reserved context groups per world, 88 independent NEC03P newly recruited source-world evaluation/reserved data, 32 five-input embargoed worlds, and the old 64 four-input confirmation worlds stay unopened. The 69 independently source-eligible newly recruited roots still have zero causal-sham outcome measurements.

## Physical native source restoration and fixed coherent source quartet

At every original source-selected recurrent visit, preserve live original graph-hub channels. Construct native 64D GELU nonhub donor with *all sixteen positive SAE code decoder contributions, original source-mean/standard-deviation normalization and learned decoder bias*, fully restoring the physical affine SAE state. The label-blind native clean-chosen class defines margin orientation. Compute the 32D physical post-MLP-down margin gradient of the full-affine restored **12-visit nonlinear recurrent trajectory**.

Use the exact NEC03V coherent quartet rule: seed at the highest historical source-ranked positive active SAE feature, then choose the three positive active feature decoder columns with highest signed cosine to that seed after original native nonhub masking and projection through frozen 32D MLP-down weight; tie-break by historical source rank then feature ID. The source removal uses the original activation-weighted sum of these four physical decoder columns, masked to nonhubs. This source feature family is fixed independent of the new control.

## New continuous physical disjoint control, completely deterministic

The source support consists of four active feature IDs; sham supports are **all 495 four-feature subsets of the remaining 12 positive active features**, in fixed sorted lexicographic order. Each support spans four original nonhub-masked physical 64D SAE decoder columns with actual original source positive code activations z_f. Instead of using just their natural original coefficients, fit four *continuous nonnegative* coefficients w_f inside the bounded box **0 ≤ w_f ≤ 3 z_f**, with exactly **one initialization w=z, one restart**, and **32** deterministic projected gradient steps minimizing half the squared 32D output-vector error between the candidate physical sham and the source physical removal:

\[
L(w)=\tfrac12\Big\|\sum_{f\in S}w_f\,P(W_f)
-\sum_{j\in G}z_j\,P(W_j)\Big\|_2^2.
\]

Here P is original native nonhub mask followed by frozen MLP-down 64D→32D projection, G is the original fixed coherent source quartet, and S is one disjoint candidate support. With physical candidate matrix B in shape 4×32, the gradient is \((BB^\top)w-B\,d_{\rm source}\); each step uses learning rate **1/max(∑_f||B_f||²,1e−10)**, followed by projection/clipping into [0,3z]. No random seeds, accelerated adaptive momentum, outcome labels, SGD checkpoint updates or score-dependent restarts are involved. Every candidate support is optimized with the **same** 32 steps and original coefficient box.

For each optimized candidate, register these independent quality measures in physical output32:
- absolute *full-affine clean-chosen margin-gradient cosine gap* between sham and source removal, **≤0.25**;
- absolute difference in *unweighted mean six pairwise decoder-column cosine* (source vs candidate group cohesion), **≤0.75**;
- mean absolute **4×4 cross-group decoder-column Gram cosine** between source and sham support, **≥0.12**;
- cosine of the total optimized source/sham physical output vectors, **≥0.25**;
- positive native projected source and sham output doses, each **>1e−6**, and gradient norm >1e−7.

Null quality is part of source-blind selection. Among all candidate supports meeting every limit, choose the **first lowest** scalar objective in lexicographic order:

\[
J=|\cos(d_{\rm sham},g)-\cos(d_{\rm source},g)|
  +0.25|\operatorname{cohesion}_{\rm sham}-\operatorname{cohesion}_{\rm source}|
  +0.25(1-\cos(d_{\rm sham},d_{\rm source})).
\]

If no candidate qualifies, save the first global objective minimizer and mark that input/site **unavailable**. Record all 495 candidate counts, number qualified, selected index, source IDs, distinct disjoint sham IDs, all four coefficients and box bounds, solver objective, quality metrics, and source/sham projected norm and subsequent dose gains. A control with high coverage obtained by violating the registered physical quality constraints **remains ineligible**.

## Physical equal dose, invalid inputs and secondary original discrete null

Remove source and sham from the *same* full-affine physical background. First compute their native direct 32D output L2 norms and take the **minimum** as a shared physical dose, applying attenuation gains in [0,1], no amplification and **≤1%** relative realized dose mismatch, with minimum common norm >1e−6. Both modified trajectories use the same native recurrence sites and all twelve recurrent visits. A failure of positive TopK16 support, the quartet, any selected visit's continuous physical-null quality/dose or a missing original source site makes both source and sham trajectories exactly the **unablated full-affine state at every selected site for that complete input**, giving zero correctness difference without dropping any case.

The old NEC03V discrete sham is a *secondary, newly scored original source-panel comparison*: its source quartet is identical; sham uses the original natural positive SAE code activations in one of 495 disjoint supports chosen by minimizing absolute gradient-cosine gap +0.5 absolute cohesion gap, with original ≤0.25 angle and ≤0.50 cohesion gates, and its own output-L2 matched attenuation. It is scored on the new already-trained source groups as a historical-null diagnostic and **cannot replace the continuous primary**.

The complete **nine** modes are clean, hubs-only, native nonhub restoration, full-affine SAE restoration, all16 contribution-only, continuous source/sham quartet removal, and original discrete source/sham quartet removal. Save every full 12-visit model output, raw case prediction and recurrent memory shift, both sham availability masks, and original frozen native+SAE hashes in SHA256-sealed JSONL. Truth task labels enter only *after* all sham/source routes are set.

## One primary, root-level power/uncertainty and stopping rule

The **single preregistered primary** is source-coherent quartet necessity relative to the new optimized physical disjoint sham. For each qualified source world, let H be 256-case correct-count under continuous sham removal and S under matched source removal. Each world's paired effect is

\[
D_r=\frac{(H_{\rm composed}-S_{\rm composed})
-(H_{\rm additive}-S_{\rm additive})}{256},
\qquad
\overline D =\frac18\sum_{r=0}^7D_r.
\]

Retain original source-ineligible root3/root5 zero. Historical minimum meaningful source-trained correctness effect **+2pp** with exactly **one primary family**. The original eight roots, two models per root and 256 masks per model are clustered; root count is eight, and only six roots are original paired source-eligible. Report root exact exploratory two-sided sign-flip p, conditional on root-level sign symmetry; repeated source input masks **must never** be treated as independent replicates.

For a developmental mechanism-resolution gate require all conditions: **8/8 independent root auditors**, macro effect strictly **>+2pp**, **at least five of six eligible root effects positive**, **≥205/256 positive-dose, physically qualified complete source inputs in each of the 12 eligible model arms**, and exploratory original root sign-flip **p≤0.05**. A source-only developmental pass would still require an additional independent-root powered study under FSI-01. With eight roots the distribution-free confidence bound is extremely wide; an eight-root source-only positive does not establish population equivalence or calibrated tail.

Raw teacher-signed margins are secondary only, normalized within each frozen model by max(1,median absolute original full-affine teacher-signed margin over its 256 source-training cases). Explicitly retain original marginal scale limitations. The old discrete-sham comparison, full SAE reconstruction accuracy, source-vs-control dose gains, individual solver saturation and physical null metrics remain secondary exploratory diagnostics.

## Independent replay, native model ancestry and future cohort protection

Freeze all source scorer/solver and **separate independent** source audit, stdlib eight-root aggregate, tests, source protocol and hosted CPU workflow Git blob SHA1 in a source-lock JSON *before any source scoring*. Retain five-input registry SHA256 **6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217** and old four-input confirmation spec SHA256 **670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39**.

Hosted preflight checks NEC03O original A/B independently audited reset1800 native and SAE ancestry, original source cohort membership and complete nine-mode/disjoint/solver/physical-dosing tests. Then eight CPU jobs each restore both native model+SAE frozen checkpoint arms, score only complete source groups 40–47, and produce raw SHA256. A **separate** independent auditor reenumerates all 495 disjoint candidate supports, independently reproduces the 32 projected iterations and nonnegative coefficient boxes, recomputes all quality metrics/physical shared doses/source labels and full nine 12-visit nonlinear logits, and verifies every raw case prediction and final root effect. A pure-stdlib final aggregate retains 8 attempts and original ITT zeros, requires 8/8 passed independent audits for publication, and stops on any failed preregistered scientific criterion.

The 69/88 independently source-eligible recruited worlds remain protected, with **zero causal-sham outcome roots**. No source-only pilot authorizes scoring them or any other held-out population. The nonparametric 95/95 upper-tail calibration still requires ≥59 independent **eligible causal-sham root** outcomes under a separately registered, scientifically meaningful intervention.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
