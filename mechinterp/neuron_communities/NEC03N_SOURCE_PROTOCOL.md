# NEC-03N: Exact 900-step replay followed by fixed 1800-step native source calibration

**Registered November 21, 2026, after NEC-03M's four-root source-only outcome and before training any NEC-03N native checkpoint.** This is a new **in-sample source-training calibration** on already examined five-input teacher worlds dev5-000..003. It never consults the new NEC-03L development-evaluation, reserved or embargoed contexts or the original NEC-03G four-input confirmation worlds.

## Scientific motivation

[NEC-03M](NEC03M_SOURCE_RESULTS.md) trained eight new native PEF-style models under a fixed 900-step source-only recipe, then fitted eight source-only TopK16 dictionaries and direction-matched controls. All eight SAEs and angular controls passed their source physical feasibility gates, but **two composed teacher models failed to reach 90% in-sample source accuracy**: dev5-000 scored 2242/2560 and dev5-003 scored 1567/2560. The others showed strong source competence. The present experiment tests whether an unchanged training curriculum run longer can overcome this *model-learning bottleneck* without selecting an outcome-dependent training length.

## Fixed source-only native training

For every originally opened world **dev5-000..003** and each arm, compose an entirely new native model instance with the **original NEC-03L seed** plus 100000 for the additive control. Use the same PEF-07 `Looper`, hidden size 32, 12 tied recurrent visits, native 64-D GELU memory-token MLP coordinates, and the same original complete **80×32=2560 training-source histories** and teacher truth table.

Reinitialize AdamW with learning rate **0.003**, zero weight decay, batch **128**, gradient clipping norm **1.0**, and the identical source sampling generator seed **`seed*7919+59`**. Training is a single continuous **1800-update run from initialization**, with complete optimizer state carried through step 900. Stop is fixed at 1800 for **all eight models**, including those that already scored 100% on source histories under the original 900-step training.

### Exact checkpoint ancestry at step 900

At the 900th update, calculate the full native weight SHA256 digest and require it to equal the original NEC-03M model digest. Independently reload the old immutable NEC-03M checkpoint from Actions run 44894980345 and verify that **every parameter tensor matches exactly**, with no tolerance for floating differences. Re-score all original 2560 training-source histories and compare correctness with the original NEC-03M receipt. An optimizer stream or model mismatch fails the entire root.

Resume the *same live model and AdamW state* for updates 901 through 1800. Do not rebuild the optimizer or move the source assignment. The milestone's `model.eval()` source-scoring operation changes no weights; before continuing training the model is returned to `model.train()`.

After update1800, freeze a new source-only checkpoint, weight digest, checkpoint bytes SHA256 and original training-seed/teacher identity. Re-score all 2560 original training-source histories and record both class recalls, confusion matrix, exact source correct count, initial and final training losses and source competency status. The same **90% in-sample accuracy** threshold applies to both milestones and cannot be adjusted according to observed results.

## Independent verification and inference unit

Public CPU GitHub Actions first verifies this protocol's registered source-code Git blobs and both old-world/five-input SHA commitments before entering the four-root training matrix. Every new source-model report is independently audited by reloading the **original 900-step checkpoint** and the **new 1800-step checkpoint**, recomputing every Boolean teacher source-training label, all 2560 source predictions and their exact per-class confusion counts, verifying both checkpoint byte hashes and the new/old model weight digests, and checking the milestone's exact parameter equality receipts. The final pure-stdlib aggregate includes all four roots even if longer training reduces their accuracy.

Each root is one originally sampled five-input Boolean function, focal-position quintet and two independently seeded native models. The 2560 training histories and both model arms share a root and are **in-sample** observations, not 2560 independent task worlds.

This test does **not** train fresh SAEs, generate an alternative source rank, or perform a new causal intervention. Any proposal to use the longer fixed 1800-step weights for future sparse interventions would have to fit a new SAE and source-ranking dictionary on these newly trained weights under a separately locked prospective protocol.

## Cohort seal and next-stage decision

The original NEC-03L full registry SHA256 **`6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217`** and the original four-input confirmation-world commitment **`670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39`** remain unchanged.

All **24 five-input development-evaluation context groups**, **24 reserved groups per new world**, **32 five-input embargoed task roots** and all **64 original four-input confirmation roots** remain untrained or unscored under this protocol. If the 1800-step source learning schedule is viable, a separate locked small pilot on *untouched* dev5-004..007 worlds can assess new root source competence without opening their development-evaluation data. If significant capacity failures remain, model architecture or fixed source curriculum experiments may be needed.

The original four-input NEC-03K posthoc +0.16276pp effect and NEC-03H-v2 +0.423pp ITT remain unchanged, both below the old +2pp scientific minimum. **FSI-01: REDESIGN_OR_PILOT_ONLY** until independently powered null roots and meaningful prospective behavior are established.
