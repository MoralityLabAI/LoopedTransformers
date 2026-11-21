# NEC-03Q: first frozen eight-root development-only causal contrast

**Pre-outcome execution registration.** This experimental implementation is separately SHA-locked after the [NEC-03P eight-root causal design](../mechinterp/neuron_communities/NEC03P_CAUSAL_PILOT_DESIGN.md), which was registered and merged before any development contexts were scored. All native weights and SAEs come from prior immutable source-only runs.

The study covers the original five-input development worlds dev5-000 through dev5-007. Six roots, 0, 1, 2, 4, 6 and 7, have paired native/SAE source-instrument admission. Roots 3 and 5 failed the preregistered source policy and contribute exactly **zero** to the eight-root intent-to-treat macro effect; their development contexts remain unscored.

## Exactly one primary contrast

For each admitted root, both its composed and additive arms receive the same complete 24 original development-evaluation context groups, with 32 focal counterfactual masks each, yielding **768 paired development cases per arm**. On every case compare the binary task correctness of the SAE-feature donor and its exhaustive disjoint active-code sham, at the same physical 32D output L2 dose.

The signed root difference-in-differences is:

\[
D_r = \frac{(C_{r,\mathrm{comp},\mathrm{SAE}}-C_{r,\mathrm{comp},\mathrm{sham}}) -
(C_{r,\mathrm{add},\mathrm{SAE}}-C_{r,\mathrm{add},\mathrm{sham}})}{768},
\quad \bar D = \tfrac18\sum_{r=0}^7 D_r.
\]

The two source-ineligible roots have D=0. The familywise comparison count is **one** and the historical minimum scientifically meaningful effect is **+0.02** correctness fraction. Two arms, visits, masks, contexts and feature codes are repeated observations within a world, rather than independent statistical roots. All descriptive clean/hubs correctness, per-arm outcomes and recurrent-memory distances are secondary.

## Frozen control and physically matched dose

Each arm is reconstructed from the original source-world model digest, exact frozen900 checkpoint ancestry, explicit AdamW reset900 second stage, 128-feature TopK16 SAE checkpoint and 40/40 source-only rank. Original source candidates use the same graph/nonhub fallback, and their recorded source-only sham geometry passed independent prior audits.

At each development input, the clean recurrent prediction defines the margin orientation. In a hubs-only trace, measure the physical 32D post-MLP-down gradient. For each preselected recurrent visit, choose the n highest source-ranked *active* TopK16 codes and exhaustively enumerate disjoint n-code subsets of the remaining active codes to minimize the absolute cosine gap to that gradient. The direction threshold is **0.25**. Compute donor vectors in original 64D GELU coordinates and scale each pair down to the smaller original 32D physical output norm. Gains are in [0,1], dose equality within 1%. All such directions are determined without teacher truth labels.

Inputs with insufficient positive TopK support, weak/no gradient, negligible physical donor dose, an angle gap beyond .25, or invalid direct dose equality at **any** selected recurrent visit receive two identical hubs-only trajectories, making their source-minus-sham paired effect zero. All 768 cases remain in the denominator, with a full boolean validity mask. The model then performs the complete 12-visit nonlinear recurrent forward for source and sham arms. Task labels are consulted only after these intervention routes have been selected.

## Independent audit and stop gates

The hosted experimental preflight checks the original frozen eight-root development-only design manifest, complete source-only audits of the four phase-A and four phase-B worlds, model/SAE/source hash lineage, all six paired-admission flags, the registered 2-point effect floor and one-primary-comparison rule. It tests label-blind fallback on 32 **source-training** cases, verifying it creates exactly zero causal contrast for unsupported active codes. This preflight must pass before any development scoring job begins.

Each of six independently seeded hosted CPU workers retrieves exact source checkpoint artifacts by immutable run ID. The runner records source/sham/clean/hubs logits, true task labels, site/feature selections, disjoint sham identities, matched-dose gains and invalid input coverage for all 24 complete development groups. A separate auditor reloads checkpoint and SAE bytes, **independently enumerates every eligible disjoint active-code subset**, checks physical 32D output doses, replays every model trajectory, rederives true labels and per-case correctness, and verifies the signed root effect.

The pure-standard-library eight-world ITT aggregate includes the two registered zero roots, every failed or missing worker, the single macro effect and an **exploratory exact root sign-flip p** under explicitly declared sign-symmetry assumptions. Complete independent audits of all six admitted roots are required to pass CI. A missing audit counts as an ITT zero and blocks full campaign completion.

## Scientific boundary

This is a *development-only causal pilot*, using models whose original source-training outcomes have already been inspected. None of the 24 reserved groups, none of the 32 five-input embargoed roots, and none of the original 64 four-input confirmation roots may be evaluated. The separate NEC-03P 88-root recruitment has shown 69 paired source-eligible worlds and provides only **source feasibility**. The requisite **59 independently observed causal sham root outcomes** for the 95/95 tolerance bound have yet to be measured.

At n=8 independent development roots with a single comparison, the FSI-01 conservative positive-effect margin is approximately **86.54 percentage points**. Any finding remains exploratory, with **FSI-01 REDESIGN_OR_PILOT_ONLY**. This run informs a later independent-world causal mechanism/power design and cannot claim population-level mediation or equivalence.
