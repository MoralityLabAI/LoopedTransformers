# Recurrent comparison within the affine recovery menu

The earlier affine reward correction gained five proofs over the calibrated planner. An exploratory chosen-route swap removed four of those proofs, and an equal-dose sham removed one. The independently scored recurrent model trailed the affine policy. This pilot starts every learned arm at the affine reward policy and gives a shared recurrent token for the saved witness access to all currently affordable candidate routes.

## Registered comparison

Use the same failed-prefix bank with 48 source and 40 previously examined development theorem pairs. During fitting, construct the affine policy out of fold. Hold each outer theorem pair out of both the affine guard-label fit and the affine reward fit. Train the outer reward fit on nested out-of-fold guard posteriors from its source complement. At evaluation, load the sealed original affine guard model and affine reward correction, and require exact reproduction of all original affine reward traces.

For each affordable menu, represent the retained Lean snapshot once and attach the existing candidate and cost information to three masked route tokens. The two-slot recurrent actor uses a tied block and gated messages between the shared saved-witness state and the pool of legal routes. A matched TRM receives the same menu and mask but uses its tied latent and answer updates. Start from zero score corrections and keep them bounded to ±.10 utility. Freeze the affine stop gate. At any state where the affine policy continues, rerank only its affordable Lean routes. At any state where it stops, no new model may act.

Compare the return-trained menu actor against the same menu actor with full-information source Q regression, TRM and affine menu controls trained on the same exact native return, and the unchanged affine reward policy. The source fit uses 360 shared batches of 16 worlds, AdamW at .002 and temperature .08 within the affordable menu. Development uses deterministic route selection. The primary outcome is root-macro native utility across three, five and seven recovery calls after the common rejected prefix. The descriptive floor is +.025 over every non-oracle control. An actual-truth upper bound retains the same stop gate and ±.10 action envelope.

At the first disagreement with the affine reward policy with three feasible routes, swap the two chosen learned score additions. For the matched sham, use the same L2 dose in the orthogonal direction that preserves their difference. Hold the actual paid prefix, costs, affine scores and native outcomes fixed. This is a development pilot with no independently calibrated sham envelope, under `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Three recovery calls / 80 | Five / 80 | Seven / 80 | Proofs / 240 | Checks | Root utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived supervised planner | 25 | 53 | 70 | 148 | 2,246 | 0.382708 |
| Archived independently scored TRM return | 23 | 54 | 69 | 146 | 2,248 | 0.374167 |
| Frozen affine reward policy | 24 | 55 | 74 | 153 | 2,264 | 0.401667 |
| Recurrent menu with policy return | 25 | 49 | 73 | 147 | 2,265 | 0.376563 |
| Recurrent menu with Q labels | 24 | 56 | 67 | 147 | 2,266 | 0.376458 |
| TRM menu with policy return | 25 | 50 | 72 | 147 | 2,263 | 0.376771 |
| Affine menu with policy return | 24 | 54 | 74 | 152 | 2,260 | 0.397917 |
| Recurrent menu, chosen-route swap | 25 | 52 | 74 | 151 | 2,263 | 0.393438 |
| Recurrent menu, orthogonal sham | 25 | 49 | 72 | 146 | 2,265 | 0.372396 |
| Actual-truth upper bound with fixed stop gate and rank envelope | 26 | 64 | 79 | 169 | 2,209 | 0.474063 |

The frozen affine reward policy reproduced all 240 archived traces. The recurrent return model changed a route on 36 attempts, closing six fewer proofs than the anchor. Its root utility improved on nine roots, tied on 20 and declined on 11 against that anchor. Recurrent return, full-information Q training and matched TRM all solved 147 cases. The upper bound still has 16 additional proofs within the fixed stop gate and score envelope. The largest recurrent loss came at the five-call budget.

The chosen-route swap changed all 28 eligible decisions, gaining seven proofs and losing three. The orthogonal sham changed four decisions and lost one proof. Both used the same per-decision L2 dose, averaging .2357 utility, and retained every paid prefix. The chosen swap improved on the learned policy, so this intervention supplies evidence of harmful placement on some development cases.

The sealed fit SHA-256 is `430595168e2e4fed55573d0a85a349c2dd897d9f6cba54d35c4b06745f8e1752`. The file records the hashes of all outer-fold affine label and reward fits. Lean independently compiled 112 successful scripts and replayed 12 selected attempts, including patched successes and failures. The terminal-only campaign used logical CPU 2 and the 2 GiB process-tree cap. Final evaluation peaked at 0.50 GiB committed. All 135 native-enabled gym tests passed. [Numerical receipt](results/ppt_hybrid_rl/ppt-failed-prefix-menu-dev-1130.json.gz)

```text
python -m configs.run_ppt_failed_prefix_menu fit BANK BPD MATHCAN WARM PRIOR ANCHOR CORRECTION FIT RESULTS
python -m configs.run_ppt_failed_prefix_menu eval BANK BPD MATHCAN WARM PRIOR ANCHOR CORRECTION FIT RESULTS --archive ARCHIVE
```
