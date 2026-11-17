# LS-SAE09 preregistration: native hidden-state transport and decision-margin diagnostic

**Registered 2026-11-16 before any LS-SAE09 native neural output.** The LS-SAE08 full-test independent numerical replay at Actions run [43851165980](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43851165980) verified 16/16 exact ACT6 solutions, zero STATE_POWER/RANDOM missing-cell flips at physical ±8.00, and registered FAIL in both 8-puzzle families. LS-SAE08's originally registered FAIL and complete source denominator are immutable. This is a separately registered **mechanism diagnostic**, with no prospective accuracy-success claim.

## Fixed inputs and trusted boundaries

- **Checkpoint:** gaoxin492 original step_21700 pretrained TRM, source SHA256 \`932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3\`; CPU float32, \`torch==2.7.1\`, \`numpy==2.1.3\`, 4 threads, MATH attention.
- **Fit:** original LS-SAE04 frozen preheldout PCA/SAE NPZ SHA256 \`073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25\`; manifest SHA256 \`cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a\`; fit array SHA256 \`83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec\`; physical dictionary SHA256 \`14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb\`.
- **Exact source rosters:** LS-SAE08 first-admitted original 2+2 engineering pilots, then original 8+8 52/54-hole full cohort, with no cohort replacement. Keep deterministic source selection, median missing source token, ACT1 \`z_H\` injection site and untouched \`z_L\`/weights.
- **Frozen intervention vectors:** **read** 512D directions from original LS-SAE08 saved geometry receipts. Do not re-optimize from labels or refit. Every eligible STATE_POWER, LOGIT_POWER, SAE, RANDOM direction receives both signs of L2 doses 0.20, 2.00, 8.00. Ineligible directions remain explicitly in the denominator. Preserve original source-arm ordering. Original LS-SAE08 evidence ZIP: Actions run \`43851165980\`, artifact \`14220373987\`, full.json SHA256 \`69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9\`. Engineering pilot: run \`43851165980\`, artifact \`14219275250\`, pilot.json SHA256 \`3b26bec178fdea4a2120001b72f3c035a91e728642adf635b9b42af985a05824\`.
- Before reading new results, independently run unchanged LS-SAE08 standard-library auditor on the loaded full/pilot numerical receipts. Validate original SHA256 and all original sources, output logits, geometry transcripts and gate decisions.

## Measurement and hypotheses

Replay clean and signed finite interventions **batched in the same order** from saved ACT1 native anchors. At ACT3, ACT6, ACT12 save, for each puzzle/sign/dose/arm:

1. \`G_H=||z_H^edited-z_H^clean||_F / radius\`, \`G_L=||z_L^edited-z_L^clean||_F / radius\`, 97-token source-site and non-site H transport norms, plus blank/given board token norms. All H/L tensors are 97×512 and effects are finite.
2. Decompose 81 board-token \`delta z_H\` into centered 9-digit lm_head **rowspace** and its orthogonal complement. With \`W_c=(I-11^T/9) W_{digit[1..9]}\`, rank by fixed relative singular cutoff \`1e-7\`, record \`||P_W delta z_H,board||_F\`, \`||P_W^\perp delta z_H,board||_F\` and \`readout_fraction\`. The centered logit displacement must equal \`delta z_H,board W_c^T\` within 2e-4 absolute Frobenius error.
3. For each originally empty cell with clean winning digit p and every competitor j, record maximum **decision-boundary pressure** \`(delta logit_j - delta logit_p)/(clean logit_p-clean logit_j)\`; also record minimum clean winner-to-runner-up gap and centered output Frobenius norm. If a clean digit top-two tie is detected, report it separately and abstain from its ratio. Recompute missing-cell prediction flips from actual replayed 81×9 logits.
4. Compare **all** actual replayed clean and signed logits to LS-SAE08's original frozen logit receipts at ACT3/6/12; reject mismatches above a per-entry absolute threshold of 2e-4 and refuse to classify the mechanism if this replay fails.

**Registered classifications (per puzzle, arm, sign, radius at ACT6):** \`contracted\` if \`G_H < 0.1\`; \`persistent_output_silent\` if \`G_H >= 1\` and blank prediction flips = 0; \`readout_small\` if centered-rowspace fraction <0.1 (and total board displacement >1e-8). Record intermediate cases explicitly. These are descriptive mechanism bins; readout nullspace occupancy and decision margins are distinct possible explanations. Compare STATE_POWER with matched RANDOM, LOGIT_POWER and SAE under identical physical doses. Report per-family 8/8 observations, per-puzzle rows and source-only propagation subgroups, ACT3→ACT6→ACT12 transport, and pilot separately. No new primary p-value or posthoc selection of a successful sign.

## Evidence and audit

The replay saves original baseline file SHA, model/frozen-fit hashes, directional status, site, signed dose, per-horizon finite transport metrics, actual output comparison errors and source-only difficulty. Independent standard-library receipts auditor checks identifiers, original baseline hash, record coverage, geometry/radius identities, Pythagorean decomposition, original output flip counts, and classification arithmetic. The auditor cannot reconstruct high-dimensional model states from summary norms; report that limitation. CI runs syntax/unit tests, the pilot and then the unchanged full 8+8 denominator; pilot is excluded from the full analysis. Keep original LS-SAE08 and its FAIL unchanged.

## Pre-existing CI issue

The separate LS-SAE03 pilot on [run 43851165837](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43851165837) fails at \`native_jvp_v1.fit_lock.checked_fingerprint\`: recalculated calibration does not match archived LS-SAE02 fingerprint. This is outside the source of LS-SAE09, and must be resolved using historical calibration evidence, without relaxing the checksum.
