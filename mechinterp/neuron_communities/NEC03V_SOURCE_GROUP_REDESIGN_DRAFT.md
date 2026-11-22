# NEC-03V: source-only causal feature-group redesign, pre-execution draft

**Status:** DESIGN_ONLY / REDESIGN_OR_PILOT_ONLY. 2026-11-22.
**Related decision:** [NEC03V issue #279](https://github.com/MoralityLabAI/LoopedTransformers/issues/279).
**Prior evidence:** [NEC03T source-only correctness](../../reports/NEC03T_FULL_AFFINE_NECESSITY_RESULTS.md) and [NEC03U immutable margin audit](../../reports/NEC03U_ARCHIVAL_MARGIN_RESULTS.md).

This document is a candidate physical intervention and experimental admission contract. It is **not an executable registration, source-panel lock, scored cohort, or permission to open development, recruited, embargoed or confirmation worlds**. A separate protocol/scorer/auditor/CI pre-execution commit and SHA-locked panel manifest must precede even a new source-only forward pass.

## Retired hypothesis

The original NEC03T necessity arm removed an n-code source-ranked subset of the 16 active TopK SAE features, matching its projected 32D physical removal norm to disjoint active-code controls. Across eight historical five-input roots, with root 3 and root 5 fixed intention-to-treat zeros, the primary composed-versus-additive correctness effect was **+1/2048 = +0.048828125 percentage points**. The preregistered useful-effect floor was **+2 percentage points**.

NEC03U recovered all 4,096 original training-source case-arm records across 11 saved modes without model forwards. The posthoc equal-root signed-margin contrast was -0.023045856 uncalibrated logit units; its median root mean was +0.000620761 and its root signs were heterogeneous. Root 7 was heavily affected by a few negative-tail masks. Source correctness was saturated in many cases. This neither proves equivalence nor supports an original-n feature-necessity confirmation campaign.

Retire **the original n-code causal-necessity inference and its right to recruit new worlds**. Preserve the historical reports and raw artifacts unchanged.

## New, distinct scientific question

Can a **coherent functional group** of mutually coactive sparse features, or the physical decoder subspace spanned by such a group, be selectively necessary for a source-trained recurrent teacher's task behavior, at the same *physical* nonhub output-32D removal dose as a matched, disjoint active group?

Single-code necessity can be weak under redundancy. A frozen feature group provides a falsifiable joint intervention. The new test must distinguish a group-specific effect from the generic cost of removing any equally energetic, similarly organized active features.

## Selection and scope: source-only, label-blind

Maintain the original 128-feature TopK16 SAE, frozen source-trained native composed/additive teachers, original nonhub native coordinate mask, full-affine baseline donor and exactly the original 12 recurrent visits.

For each teacher/world, use **only previously opened source-fit and source-rank data** to construct a deterministic feature-coactivation graph. A candidate weight for features i,j is a predeclared nonnegative combination of (i) source-fit positive-code coactivation frequency and (ii) cosine coherence of their **masked 32D** decoder contributions. Normalize the two components from source-only statistics, freeze tie-breaking by feature index, and store the exact graph/mask hash.

Candidate group G consists of **four** active codes forming the strongest source-coherent connected group at a selected visit. Its membership is computed from the frozen graph and the current case's positive TopK16 support, using a predeclared deterministic algorithm. Labels, teacher correctness, test-world outcomes, and post-ablation logits must not influence group selection. If no valid group exists, that case is a zero-intervention case.

For the main physical sham H, search disjoint four-code active groups *within the same case and visit* using an identical algorithmic-complexity budget. Rank admissible shams by their source-only structural coherence and projected decoder-vector norm proximity to G, with deterministic hash-independent tie breaks. Do not select a sham by its effect on task labels, error rate, or output margin. Record the full candidate count and the exact chosen code IDs.

Keep an alternative, predeclared geometry-matched active sham and an optional orthogonal subspace perturbation as **separate sensitivity studies**; neither may replace the main physical sham after outcome inspection. In particular, matching the label-signed margin gradient may itself remove the hypothesized directional distinction, so any such gradient-matched control is diagnostic only and must be frozen separately.

The original eight historical roots are the only worlds eligible for source-only design. Historical source groups **24–31** have already been opened. A prospective **32–39 source-history panel**, if it is within the authenticated original source-training provenance, may be used only after registering every exact root, group, mask, arm, checksum and permitted checkpoint. No recruited-source roots, non-source evaluation cases, reserved groups, 32 embargoed five-input worlds or old 64 four-input confirmation worlds may be queried.

## The physical intervention

At each registered recurrent visit, let the full-affine SAE nonhub donor be B, and let g and h be the decoder-projected 32D contributions of group G and its active-code sham H, respectively.

Set the common nonamplifying removal dose to

  d = min(norm(g), norm(h)).

For fixed lambda in {0, 0.25, 0.5, 1}, the source-removed and sham-removed donor states use

  B_G(lambda) = B - lambda * (d / norm(g)) * g
  B_H(lambda) = B - lambda * (d / norm(h)) * h.

The controls share the same B, active native hubs, decoder mean/bias, nonhub mask, full 12-visit recurrent transport and true-data denominator. The ratios are at most one, so neither arm amplifies its native physical source vector. Each dose must match in actual 32D L2 within **1%** at every selected visit, with independently computed receipt hashes.

If a registered visit lacks an allowed disjoint group, nonzero dose or the physical matching conditions, apply the **exact same zero perturbation** for G and H at all registered visits of the entire case. Preserve that case in the 256-case denominator. Publish eligibility as its own feasibility metric. Never silently drop invalid masks or adjust the matched dose after inspecting logits.

The full-dose lambda=1 is the only proposed primary readout. The other doses provide a frozen, descriptive dose-response study. The lambda=0 source/sham trajectories must equal the full-affine baseline bitwise within the registered numeric tolerance.

## Primary outcome and statistical unit

Retain a single source-only development primary: the difference-in-differences of **kernel-recomputed correctness**, first between physical sham and group removal within each teacher, then composed minus additive within each original world:

  D_r = ((correct_comp_sham - correct_comp_group)
        - (correct_add_sham - correct_add_group)) / 256.

Every source group is enumerated with its 32 five-bit focal masks. Roots 3 and 5 retain their registered original intention-to-treat zero rule; their raw physical and behavioral diagnostic data remain visible, never substituted into the primary. The eight-root descriptive macro is mean(D_r over original eight roots), with the **same +2 percentage-point minimum meaningful effect** inherited as the clinical/engineering benefit floor. Root is the independent experimental unit. Treat 32 masks within each source group and recurrent visits as correlated measures.

The source-only experiment is developmental, including any 32–39 panel. It cannot supply a population p-value or confirmation by reinterpreting the original root signs. Declare positive/negative source-only feasibility separately from the behavioral primary.

**Secondary only:** within each root and teacher, compute a teacher-signed margin M from the true label after all intervention parameters are locked. Scale each within-model signed margin by a **frozen source-fit-panel logit-gap IQR**, with a numerical floor chosen before scored-panel access. Compare same-teacher G/H normalized margins first, then show root-level cross-teacher contrasts. Report the per-root median, mean, heavy-tail exposure, and baseline-margin strata as descriptive diagnostics. Never apply the uncalibrated NEC03U raw-scale eight-root margin result as a new confirmatory endpoint.

## Pre-forward feasibility gates and independent audit

Before any new source-panel inference, freeze an explicit V protocol, scorer, independently implemented auditor, no-model preflight tests, source membership manifest and SHA/Git blob identities in a separate commit. The preflight reads only previously authenticated source input IDs, source activations and decoder geometry; it may compute **physical matching eligibility and dose only**, without labels, logits or model output trajectories.

Require the preflight to report:
- Per-root and per-teacher fraction of complete 32-mask groups with a valid group, disjoint active control and L2 match across all intervened visits.
- Source and sham original 32D norm distribution, nonamplification factors, group coactivation/coherence, structural-sham similarity and invalid/no-op denominators.
- The number and range of distinct physical sham candidates and deterministic pairing stability.
- Per-root expected raw causal-sham variation at fixed physical dose under the admissible source-only preflight, with the exact measurement specified before looking at behavioral labels.

Decide and lock the minimum acceptable physical coverage and **physical null-resolution threshold** from the label-blind preflight. A regime with poor coverage, near-zero removal dose or unstable matching is **physically uninformative**, regardless of posthoc margins. This design may be retired before any model-forward scoring.

After a full-source-only V pilot, the independent auditor must verify every one of the eight source-world roots, the fixed 256-case denominator, complete intervention vector construction, nonamplification, original recurrence visits, valid/no-op masks and separately compiled original teacher labels. All model forward counts and every changed input/checkpoint hash must be reported, and any incomplete root fails the eight-root audit.

A separate causal-sham *behavioral* spread can only be estimated on the source-only pilot after the method and primary are frozen; it cannot be inferred from unlabeled physical dose statistics. Freeze the behavioral null-resolution rule before opening the source-only evaluation panel, or explicitly mark that analysis exploratory.

## Promotion gate and protected cohorts

A source-only result becomes a *design continuation candidate* only if its root-level group-specific correctness effect clears **+2 percentage points**, exceeds the independently registered sham-resolution floor, is not driven by a single world root, and passes all eight independent physical/neural audits. These are engineering promotion criteria, not a population confidence statement.

Even if these gates pass, design a new independent-root power calculation with EvalMath FSI-01, including selection conditioning, root sign heterogeneity, intended familywise comparisons and explicit handling of recurrent model covariance. A new prospective registered causal-sham intervention and a separate executable review are required before any NEC03P recruitment root is scored.

The **69/88 source-eligible recruits still have zero causal-sham outcomes**. The original 95/95 nonparametric tolerance construction still needs at least **59 independent eligible root outcomes** under a scientifically resolvable physical null. Keep all 88 recruited roots, reserved data, 32 five-input embargoed roots and 64 old four-input confirmations sealed. FSI-01 stays **REDESIGN_OR_PILOT_ONLY**.

This draft authorizes **zero new native or SAE training, zero model inference, zero new score reads, and zero confirmation-set opening**.
