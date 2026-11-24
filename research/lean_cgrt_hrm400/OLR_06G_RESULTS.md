# OLR-06G: source-calibrated native portfolio results

**Status:** completed native experiment; developmental learned-selector gate FAIL.
**Executed commit:** 462c3375cfe7bd724cf3f29bcc706715e0124cf0.
**Registration:** [06G](OLR_06G_REGISTRATION.md).
**Workflow:** [45323569279](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45323569279), success.
**Artifact:** [14889165921](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45323569279/artifacts/14889165921).

All protocol tests passed. The source fit used 64 training roots and 16 separate calibration roots, 960 fixed-schedule trials. There were 974 distinct native leaf replays and 380 standalone conjunction declarations, 1,354 axiom-free source declarations. Evaluation used 48 independent equation systems, 576 menu trials, 1,008 frozen selector trials and 48 unrestricted references, totaling 1,632 policy/root trials. 592 distinct evaluation leaf paths replayed natively and 243 conjunction declarations compiled, totaling 835 axiom-free evaluation declarations. Selector trials inherit the selected, native-certified menu row; repeated seeds are not new independent roots.

## Completion at 512 charged work units

| Policy | DEV /24 | Transfer /24 | Total /48 |
| --- | ---: | ---: | ---: |
| Cached independent | 19 | 15 | 34 |
| Demand ledger | 17 | 17 | 34 |
| Hamming weighted | 4 | 1 | 5 |
| Eager width2 | 17 | 14 | 31 |
| Eager width3 | 5 | 0 | 5 |
| Cyclic quarter | 12 | 7 | 19 |
| Demand quarter | 15 | 11 | 26 |
| Demand half | 18 | 14 | 32 |
| Cache128 -> demand | 21 | 15 | 36 |
| Demand128 -> cache | 17 | 15 | 32 |
| Demand128 -> eager2 | 16 | 11 | 27 |
| Eager192 -> demand | 18 | 11 | 29 |
| Source-fixed / ridge / kNN | 17 | 17 | 34 |

All six learned reader arms attain 51/72 completions in each panel, with each of the three seeds attaining 17/24. The source-fixed choice is demand in both panels. Every bidirectional, one-direction, one-visit, local and dense checkpoint selects mix zero. Full-packet seeds 17 and 29 select .25; seed 43 selects zero. Those two full-packet checkpoints make four alternative DEV choices in aggregate, without changing any completion count. Both ridge and kNN select zero.

Bi, one-way and bi1 each have 21,708 parameters; local 9,260; dense 19,852; full-packet 120,108. The registered three-visit versus one-direction initialization and sampler checks passed. At mix zero, prediction weights cannot affect the selected action. The zero paired contrasts establish policy identity under the registered calibration, not equivalence of latent computations.

The bidirectional policy has mean utility .92855 DEV and .93689 transfer, mean charged work 280.33 and 351.42. Its selector feature+inference overhead is 1.37 and 1.74 ms respectively, in addition to mean recorded selected-schedule search time 17.25 and 26.37 ms. Cached independent search averages 4.75 and 6.56 ms in the same run; these include bounded misses and are not matched latency superiority claims. Source-fixed avoids the learned feature/inference overhead. The best posthoc fixed single schedule closes 36/48 (75.0%), versus every learned selector at 34/48 (70.83%). Best fixed per-panel choices close 38/48. Hindsight selection among twelve schedules closes 45/48, a +7/48 (14.58 pp) gap above the posthoc per-panel baseline. Unrestricted independent search closes all 48 roots outside the cap. The portfolio gap is a privileged upper envelope, not a deployable policy result.

## Source-only common-prefix diagnostic

After the experiment, inspect only its certified TRAIN and calibration schedule rows for candidate suffix menus that share a first-stage search. The numbers below do not use 06G evaluation labels to choose a next policy.

| Common prefix and registered available suffix schedules | TRAIN DEV (32) best / union | TRAIN transfer (32) best / union | CAL DEV (8) best / union | CAL transfer (8) best / union |
| --- | --- | --- | --- | --- |
| Demand128: continue demand, restart cached, restart eager2 | 29 / 30 | 23 / 24 | 7 / 8 | 5 / 5 |
| Cached128: continue cached, restart demand | 27 / 29 | 18 / 19 | 7 / 7 | 5 / 5 |
| Eager192: continue eager2, restart demand | 25 / 27 | 17 / 18 | 7 / 8 | 5 / 6 |

For demand128, only three extra completions across the 80 source roots are available even to a per-root oracle over these three suffixes. A classifier over that fixed suffix menu has little source headroom. This motivates moving from whole-schedule classification to allocation over the actual live frontier. No new teacher label or evaluation result is claimed by this diagnostic.

## Custody

Archive SHA256 b09ab3b93b9f0d8ccaf2a4bf799385a1b33d275d634593f8e69bd2caf915eb2f. Independently verified all 24 manifest-listed files (4 registry, 9 fit, 11 evaluation).
- Registry manifest: 581f2a247e0e7cd2f8abd452dcd24cb46e2caf253d6e5dac9963c1f94a74b5ce
- Fit manifest: 80a433783d58c21fa23b8af744d4c8dc787dac37a474a7da38759c2304660f43
- Evaluation manifest: 2aee5290b1fb97273880e03cb5438d611a0274da5e71a2d7c144f1f7d970434c

## Decision

Close 06G as a negative one-shot selection result; do not force nonzero blending after scoring. The exact ledger and native proof transport remain useful. [06H](OLR_06H_REGISTRATION.md) tests state-dependent allocation, preserving actual discovered edges and pending obligations across a counterfactual allocation block. Establish the utility of that action space before fitting another neural model. No 5M promotion.
