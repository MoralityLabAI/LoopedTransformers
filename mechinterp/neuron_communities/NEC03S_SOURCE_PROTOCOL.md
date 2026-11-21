# NEC-03S: frozen source-only affine-offset × active-code-budget factorial

**Experimental registration: 21 November 2026, before new source-only outcome scoring.** Follows the independently audited [NEC-03R result](NEC03R_SOURCE_RESULTS.md), where the full affine SAE recovered 322/323 composed-source correctness gaps yet the ranked n-code sparse donor recovered only 65/323 at natural dose and 26/323 when equalized to its disjoint active sham. The study tests whether the missing physical affine mean/decoder bias accounts for feature-specific sparse-donor recovery when the output-L2 dose and control policy are held fixed. A second active-budget test probes the cost of restricting the 16 active codes to the original n.

## Exactly frozen cohort and source-training selection

- **Statistical unit**: one independently seeded five-input Boolean world root, dev5-000 to dev5-007 inclusive, with the previously trained composed and additive teachers paired within each root. Both checkpoint/SAE arms are retrieved by immutable NEC03O A/B run IDs. Native and SAE training updates: zero.
- **Inputs**: only original native and SAE training-source groups **16–23 inclusive**, eight complete original seven-bit context groups ×32 focal masks = **256 previously trained histories per arm**. The first 40 original source groups fitted the 128-feature TopK16 SAE, while the last 40 supplied its frozen feature rankings. These source outcomes are in-sample.
- **ITT**: all eight original world roots, all 16 teacher arms and all 256 histories per arm stay in the denominator. Roots **3 and 5** are predeclared zero primary effects from the original paired-source-instrument admission failure; their already-trained model arms are still scored for descriptive source correctness, with the missing source-site case retained.
- **Protected**: score zero new development-evaluation or reserved context groups, score zero newly source-recruited worlds dev5-008..095, leave 32 five-input embargoed roots and all old 64 four-input confirmation worlds unopened. The NEC03Q negative developmental causal report is unchanged.

## Frozen physical donor coordinates and feature budget

The original source-selected sparse native nonhub recurrent sites and their graph hubs remain fixed. Every group and input uses the same frozen **clean activation donor** at each selected native 64D GELU pre-mlp-down memory-token input, with the 32D mlp-down output weight from that frozen native model. All hub channels remain live from the perturbed recurrent trajectory. For this factorial every injected source, sham, offset and native donor is **masked to the same original nonhub coordinate subset**, a new source-only condition registered before outcomes. Recurrent forward propagation includes all twelve tied visits.

Use original source-mean μ, scale σ, frozen TopK16 sparse codes z, physical columns of the learned decoder W, learned decoder bias b and physical affine offset o=(μ+bσ) masked to original nonhubs.

- **Budget 0**: offset-only donor o.
- **Budget n**: source top n positive active codes by the original per-visit source attribution ranking, with n equal to the number of original native source-selected neurons at that visit. Both feature-only donor and feature+offset donor are evaluated.
- **Budget wide**: k=min(2n,8) highest source-ranked active codes, with offset absent and present, scored under its own **four-way common dose** separately from budget n. If k=n, retain the duplicate by the original registered naming with no extra candidate search.
- **Full 16 positive codes**: sum of all sixteen contribution columns, with and without affine offset, at natural physical dose as **descriptive reconstruction ceilings**. Disjoint equal-16 active null is combinatorially impossible; no null or primary inference is claimed for full16.
- **Additional natural-dose reference**: clean unmodified, hubs-only lesion, native full nonhub clean donor; source-ranked n and wide k at natural output dose with offset absent/present, separately named descriptive recoverability endpoints.

A missing native source site yields no manipulation, all variants return the clean baseline and active-sham availability is explicitly false. Native heads under original 90% source competence remain in the descriptive analysis. A source input lacking exactly 16 strictly positive TopK codes is retained with an unavailable sparse control.

## Exhaustive disjoint active-code null, random control, and one common dose

For each complete source group, focal mask and source-selected recurrent visit:

1. Compute the label-blind **hubs-only clean-chosen margin gradient** in physical post-mlp-down 32D coordinates. Source ranks and all original decoder weights are frozen. The source subset is top k positive active codes. Every k-code subset of the disjoint remaining 16−k active codes is enumerated, using a fixed canonical lexicographic order.
2. For every candidate sham, calculate the absolute difference between its output-32D cosine to the gradient and the source cosine both **without** the affine offset and **with** the same offset added to source and sham. Choose the sham minimizing the **maximum** of these two absolute gaps. This is a joint-minimax angular comparator chosen before outcomes. Its ID is identical in both offset conditions. Predeclared angular admissibility requires both gaps ≤0.25 and a gradient norm above 1e−7.
3. Construct **four unattenuated nonhub donors** (source n-code, disjoint angle sham, source n-code plus the affine offset, the *same* sham plus the *same* offset). Project each to physical 32D output, compute their four L2 norms and take their **single minimum** as the dose anchor. Every vector is attenuated, never amplified, to this identical output norm, with allowed physical mismatch at most 1%. The anchor must exceed 1e−6 to count as a matched positive-dose case. The same minimax sham identity and source/control feature budget are used across offset levels. The offset interacts with the raw donor before physical dose equalization.
4. For a separate **secondary disjoint active control**, hash world SHA, teacher arm, source group, mask, visit and budget name with salt NEC03S-fixed-hash-disjoint-source-budget-v1, and choose one element of the same sorted exhaustive disjoint k-code set. Source and random donor are attenuated to their pairwise minimum physical output norm within each offset condition. This random pair has its own dose anchor and cannot substitute for the primary four-way matched angular null.
5. A failed code support, sham angular, gradient, or four-way dose match at **any selected visit** causes **all four** primary source and sham donors to be zeroed across every selected visit for that entire input. Both offset and no-offset comparisons consequently share the same hubs-only trajectory and contribute zero signed correctness difference. Random controls have separate all-visit availability. Never drop inputs or select another budget/candidate after seeing teacher correctness.

Store all active/source/sham/hash-random IDs, all candidate search counts, offset-level angular gaps, four raw physical output norms, the **single four-way anchor**, dose attenuation gains, original native/SAE checkpoint SHA256, per-input masks, all 26 full recurrent logits and memory-state drifts. Teacher Boolean truth labels enter **only after** the entire donor and sham routing is frozen.

## Primary eight-world comparison and secondary diagnostics

Let C(r,t,b,condition) be correct task labels among the 256 original source histories for world r, native arm t, budget b and donor condition. For the original source-site feature budget n, the fixed per-arm **offset-by-source-specificity interaction** is:

\[
I_{r,t}=[C_{r,t,n,\mathrm{source+offset}}-C_{r,t,n,\mathrm{sham+offset}}] -
[C_{r,t,n,\mathrm{source}}-C_{r,t,n,\mathrm{sham}}].
\]

The one and only primary root effect is \((I_{r,\mathrm{composed}}-I_{r,\mathrm{additive}})/256\), retaining root3/root5 as original **zero ITT** effects. Its primary eight-root macro mean is the equal-root mean, with signed positive direction and original **+0.02 absolute correctness fraction** meaningful floor. The root-level two-sided exact sign-flip statistic is descriptive and assumes sign symmetry; an eight-root source-trained sample is not an independent confirmation sample.

Wide-k interactions, full16 restoration with/without offset, offset-only response, source-ranked donors at natural dose, native clean restore and hash-active matched random controls are **secondary physical diagnostics**. These comparisons are deliberately non-equivalent in dose outside the primary four-vector family, and none can replace a failed primary.

## Independent audit, materiality gate and preservation

Before source scoring, freeze the Git blob SHA1 of this protocol, the scorer, the entirely separate replay auditor, the full eight-root aggregator, and preflight tests in an explicit NEC03S_SOURCE_LOCK.json. Pin original NEC03L registry SHA256 6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217 and old NEC03G four-input confirmation spec SHA256 670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39. Prerequisite GitHub Actions preflight must pass source cohort/treatment/physical dose tests and validate the original NEC03O A/B independent audits before allowing any scored root job.

Each of the eight fixed root jobs restores both source native/SAE arms and scores only original complete source groups 16–23. The independent auditor **rebuilds** all 64D donor/offset vectors, reenumerates every disjoint active-code candidate, checks both offset-level cosine minima and four-vector shared dose, recomputes full 12-visit logits and every source truth label, and matches every case-level raw JSONL SHA256. A stdlib eight-world aggregator retains all root attempts, including original source-ineligible roots and any failed worker (as unavailable with zero primary effect), while full publication requires **8/8 independent root audits**.

If the source-only primary effect remains below the predeclared +2pp or lacks physical-sham resolution at root level, **stop before using the 69 NEC03P recruited independent source-eligible worlds**. Those still have **zero causal-sham outcome measurements**. The nonparametric 95/95 59-world sham tolerance bound must be estimated in a later separately registered conditional-root causal study if a resolvable intervention emerges.

**Scientific evidence gate: FSI-01 REDESIGN_OR_PILOT_ONLY.**
