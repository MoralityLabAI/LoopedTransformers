# NEC-03W/F01: root0-only source functional calibration scorer preregistration

**Status:** PRE-EXECUTION; source-fit developmental experiment, REDESIGN_OR_PILOT_ONLY.
**Original root authorization:** ONLY dev5-000 at this stage.
**New data scope:** original SOURCE-FIT group16..23, 8 groups times 32 masks = 256 source-training cases per teacher arm. All other eight-root behavioral scoring requires its own registration.

## Prior source and physical-geometry custody

The original source-trained Looper native checkpoint and TopK16 SAE on dev5-000 are already byte-anchored to NEC03T and NEC03V, with independently verified 128x32 decoder columns and source-fit original training inputs.

The original unlabeled source quartet selection was frozen before F01, using feature IDs from original source groups0..15. The calibration physical shortlist used positive code support and 32-case physical feasibility only on source groups16..23. It selected one original recurrent visit and one pair partition per arm without task labels.

The **full 256-case per-arm physical roster** was built separately before any F01 functional outcome access and independently audited in [zero-neural-inference workflow 45002476714](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45002476714), artifact [14746158588](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45002476714/artifacts/14746158588).

Frozen source data:
- Original source pack SHA256: ef79f9816a0143bc65aaa11d03626f5380dacd76e8b5c3f11437739b928e8c61
- Original source calibration positive-code file SHA256: 688f7c92beb7015b575d1addff606a357dbf2af360c81b048cc5bd271af9191d
- Original label-blind selected roster SHA256: 9cce6566bd0e019bca14374a4c8213ff80fde0e4267045d67aca6c94a3140030
- Original *full* F01 physical-roster SHA256: b3f13ab106b0d60f5c7d97e8bd521fa5890195fe8f025cf843126169e0f58513
- Original physical-roster file SHA256: 2e7f12838b265c05d825233dfd0387c69cb1e1f3e4cb7968b69a434f27128455
- Admissible: composed **256/256**, additive **152/256**, with 104 additive invalid paired-zero cases. The physical audit independently recalculated **78,336** 32D coordinates.

These exact source files and the independent physical audit must be verified by file digest and original source identity before any native task evaluation. They cannot be regenerated from task outcomes after opening this calibration.

## Fixed independent variables and mode budget

Use the original 12 tied-weight recurrent passes on CPU. Original source memory-token 64D GELU at all previously selected SAE visits is replaced with frozen full-affine source donor; one preselected original recurrent visit is manipulated at a time. Preserve original source hub gates.

Ten registered modes: baseline; remove A; remove B; remove A+B; matched sham remove C; remove D; remove C+D; AB removal followed by delayed target 32D memory direction restoration after the NEXT tied recurrent block; AB removal followed by a matched sham direction at the identical visit; same-site exact addback A+B.

Every case uses the exact preregistered 2×2 physical matched pair geometry. Invalid source cases receive identical full-affine native donors and zero delayed injection in all modes. At a selected visit 11, delayed rescue is impossible; both rescue modes are identical to AB removal, with case retained. Same-site exact addback is solely an identity control and is excluded from claims of rescue.

Pre-budget: 8 groups × 10 modes × 2 teacher arms = **160 CPU model-forward batches** per complete original root0 scorer; each forward contains 32 source training histories and 12 native recurrent visits. Separate independent audit executes the same fixed original source budget. No optimizer updates, no GPU allocation, no train-time model changes.

## Endpoints and acceptance

Error L is binary incorrect teacher prediction relative to the original five-input Boolean task, recalculated on source-training groups16..23 after the physical roster is frozen. For each case:

    J_group = L(AB) - L(A) - L(B) + L(baseline)
    J_sham  = L(CD) - L(C) - L(D) + L(baseline)
    J      = J_group - J_sham

The **source calibration diagnostic** is mean J over each arm's complete original 256-case denominator, then composed minus additive. Root3/5 retain fixed zero in any later multi-root ITT. No root0 calibration effect can be cited as a population result or as an eight-root confirmation.

The delayed-rescue secondary is mean (L(late sham rescue) minus L(late target rescue)), with full 256-case denominators. Same-visit exact addback must reproduce the baseline class and match logits within 1e-4 absolute; every invalid-case mode must be identical to baseline within 1e-6 absolute. Independently replay all saved native logits within 1e-4 absolute plus 2e-5 relative. The original NEC03V root4 posthoc FP32/FP64 replay issue remains recorded and informs the separate numerical auditor.

**Planned promotion floor, only after a separately preregistered complete eight-root source study:** +2pp positive composed-minus-additive error interaction. Require a positive matched delayed-rescue effect and measurable functional sham behavioral variation before considering any prospective follow-up. That later eight-root source statistic must be preregistered before scoring roots1..7.

## Independent native audit

The first-party scorer uses the existing source-only native12 hook gym and 64D donors. The independent auditor uses a different memory-token GELU prehook and block posthook, separately rebuilds original SAE full-affine donors, per-pair lesion contributions and 32D late residual correction, then reconstructs every native output class/logit and original teacher label. A failed independent numerical replay, altered original source case, changed physical SHA, loss of 256-case denominators or model mutation means the root0 result is **not admitted**.

No source-query group32..39 coefficient extraction or task outcome is authorized by this protocol. Previously behavior-scored original groups24..31 are likewise excluded. All development-evaluation and reserved original contexts, 88 recruited original NEC03P roots, 32 five-input embargoed roots and 64 old four-input confirmations remain sealed. Scientific status REDESIGN_OR_PILOT_ONLY.
