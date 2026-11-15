# PEF-10 | Principal-axis ablation-angle recovery frontier

Registered on 2026-11-15 before running **any PEF-10 heldout angle sweep**. PEF-07, PEF-08, PEF-09 and PEF-09R results were already known. PEF-09's native-head collapse after ungated rank-one ablation, paired with PEF-09R's successful retrained ridge recovery for near-parallel pair-SVD directions, motivates this conditional test. Do not relabel PEF-09's failed criterion.

## Fixed provenance

This branch forks PEF-09 / PEF-09R draft PR #79, exact head `aaf57e7ccc555ab7bbe53a820c9fc5ce77ab64d7`. Only the original six 450-step PEF-07 CPU checkpoints (seeds 103/131/167, coalition and threshold arms) are used. Original source split has 160 trunk-train, 48 ridge-probe, and 48 heldout eight-bit contexts, each with all 16 focal-bit states. The original PEF-07 checkpoint and source checks from PEF-09 must pass before PEF-10. The **previously frozen probe feature ledger** `pef09_results/frozen_selection_ledger.json` must match SHA256 `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157`, and the original training ledger must match SHA256 `1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714`. PEF-09 evidence ZIP SHA256 `6ea4a2a9b9b51dbfd9ac40d2645dce3edc800ab75e45d791322c1f202575d511` identifies the local source package (path itself is not a scoring dependency).

## Hypothesis and exact angle construction

At **recurrent visit t=8**, extract the signed leading global principal component g from the frozen PEF-09 probe selection ledger; use the two previously frozen interaction SVD pair vectors v_01,v_23. Set each pair vector's sign to have nonnegative inner product with g. Define its orthogonal tangent u_p = (v_p - (v_p dot g)g)/norm(...). If this norm is below 1e-9, record geometric degeneracy, substitute a deterministic probe PCA-2 direction, and count the pair's primary criterion as failed. Also define a second probe-only tangent as the global PCA-2 eigenvector orthogonalized against g and a single seeded Gaussian tangent orthogonalized against g. Random generator: CPU torch.Generator manual_seed(10100+seed+10000*[task=='threshold']), draw one FP64 32-vector and Gram-Schmidt against g. Fix all signs deterministically using highest absolute coordinate positive for each independently selected PCA/random tangent.

For a chosen tangent u and sign s in {-1,+1}, define the *same ungated rank-one affine intervention* for every input:
`v(theta,s)=cos(theta)*g + s*sin(theta)*u`,
`h8' = h8 - ((h8-mu8) dot v)*v`, where mu8 is the **frozen PEF-09 probe mean**.
Apply to the entire 768-row 16-mask-closed probe cube, and independently to the whole 768-row heldout cube. Continue each with identical natural source bits 9–12, then original continuation bits (0,1) to h14. Use the original native h12 head, original PEF-09 h14 frozen ridge, and fit *new* h12 and h14 ridge decoders separately for each angle exclusively on post-ablation probe states.

Fixed angle grid in **degrees**: [0,0.25,0.5,1,2,4,8,16,32,64,90]. Both +/- orientation signs at each strictly interior fixed angle; at 0 and 90 count one projector. Add the unique pair-specific *exact original pair-SVD* angle phi_p=degrees(acos(|v_p dot g|)) with +/- signs, evaluated for pair tangents only. Deduplicate an exact numeric collision with the fixed grid. There are 82 angle conditions per model: four tangents x20 fixed-grid angles plus 2 pair-anchor conditions (the pair-specific +/- phi_p endpoints count as 2 additional conditions per pair, giving 84 total). `phi_p` is measured from probe-only vectors and is frozen with its source before heldout scoring. Both signs use the same orthogonal tangent, changing the projector angle relative to g. At 90 degrees +/- signs have identical projection and are deduplicated.

Important arithmetic note: fixed grid has endpoints 0/90 (2 cases) plus 9 interior angles x2 signs (18), hence 20 per tangent. With four tangents, there are 80 grid conditions. The two pair tangents each contribute two +/- phi_p anchors, totaling **84 conditions per task/seed**. This count overrides any descriptive shorthand earlier in this document.

## Frozen readouts and strict ordering

For each task/seed, (1) read and hash the original probe selection and checkpoint ledger; (2) compute the exact list of g, pair tangents, PCA-2 and random tangent, centers, angles and unit vectors **using only 768 probe examples**; (3) replay all angle-projected *probe* states to h12/h14, fit intercept-augmented standardized ridge with fixed penalty=1 using original PEF-07 `fit_ridge` and save each fit; (4) freeze and SHA256 the entire selection-plus-readout ledger on disk; (5) begin heldout replay and scoring only then. No model training, hyperparameter search, pair-by-pair manual adjustment, or heldout-guided angular selection.

Evaluate each heldout condition with original native h12 head, the previously frozen (natural-state) h14 ridge, newly fitted ablated-probe h12 ridge and newly fitted ablated-probe h14 ridge. Report correct /768, per-class recall and wrong-mask counts for each, original score checks, and the probe vs heldout ablation energy ratio. Preserve raw condition-level table and all 84 fitted coefficients for each model. The h14 newly refitted ridge asks what information remains accessible to a *fresh readout*, not what the original trained head learns on its own.

## Angle frontier outcomes and frozen decision

For each task/seed/tangent/sign identify the *earliest fixed angle* at which new h12 ridge accuracy reaches **>=720/768** (93.75%); report null if none. The rank-one PCA baseline at theta=0, and the exact pair-vector anchor phi_p, are reported independently. Record the minimum theta among +/- signs for each pair tangent and for PCA-2 and random tangents. The primary **geometric recovery** criterion: in at least 2 of the 3 coalition seeds, **both pair tangents**, using either sign separately per pair, cross >=720/768 by theta<=4 degrees at h12, while theta=0 fails the 720 threshold. This is a within-task diagnostic; the pair directions and task family were previously observed.

A second *pair specificity* diagnostic requires both pair-tangent minimum positive recovery angles to be **strictly smaller** than the random-tangent minimum angle within a seed. Report its 3-seed count without promoting it to the primary criterion. The PCA-2 tangent controls whether generic high-residual-variance directions recover at a similar angle. Fixed h14 and native-head results are reported without secondary success promotion.

## Exact geometry and interpretation

Let orthonormal g,u span a two-plane and write centered hidden x = a*g+b*u+r, with r perpendicular to g,u. For v(theta)=cos(theta)g+sin(theta)u, the ablated state is

`x - (x dot v)v = (-a sin(theta)+b cos(theta))(-sin(theta)g+cos(theta)u)+r.`

The portion of the leading PC coefficient entering the retained tangent has amplitude `-a sin(theta)`. Its energy grows as `sin(theta)^2 E[a^2]` when the cross-covariance vanishes, while residual signal is also present. This is a geometric identity and a testable residual-energy decomposition; it does **not** predict readout accuracy without conditional feature statistics or label correlations. Absolute angle, signed orientation and ridge decoder condition all matter.

The resulting experiment examines synthetic supervised Boolean-history functions in six frozen small tied-attention models. Matching a geometric recovery curve does not establish unique neurons, SAE features, new operator generalization, a Lean capability gain, or pretrained-model behavior.

## Commands

```bash
python -m pytest -q research/pef10/test_pef10.py
python -m research.pef10.pef10 \
  --evidence /path/to/registered \
  --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
  --out /tmp/pef10
```

Local Python 3.13 / PyTorch 2.10 CPU evidence and hosted Python 3.12 CI shall be distinguished. Unit-only CI cannot claim full six-checkpoint reproduction absent the original evidence archive.
