# LOOPSCOPE-11 / LGA-08: pair dark-fiber transfer survives exact signed future-gradient matching

**18 November 2026. First source-locked, registered native DEVELOPMENT_STRONG_INCREMENTAL_PASS, 3/3 historically exposed 44K TRM teachers.** The original native GitHub test and separately registered independent Python-stdlib audit both passed. The scientific evidence is a controlled, source-conditioned causal mechanism in a tiny cyclic-copy recurrence; these three repeatedly examined teachers are a *development* sample, and no independent checkpoint-root confirmation is implied.

## Why this experiment matters

[LOOPSCOPE-08/LGA-05](../loopscope08/RESULTS.md) learned an external shared potential E(h) that descends along the original trained four-pass recurrent trajectories, with 3/3 teachers passing geometric gates but no successful matched causal task-loss steering. [LOOPSCOPE-09/LGA-06](../loopscope09/RESULTS.md) established its breakdown beyond four passes. [LOOPSCOPE-10B/LGA-07B](../loopscope10/RESULTS_LGA07B_FIRST_NATIVE_20261118.md) found negligible exact donor recovery from a low-dose, one-goal-token energy/head-tangent edit. [LOOPSCOPE-10C/LGA-07C](../loopscope10/RESULTS_LGA07C_FIRST_NATIVE_20261118.md) found a strong **whole-state** native donor transfer (3/3 exposed teachers), with an energy-tangent projection that behaved almost like an ordinary current-head-null projection. Its natural predecessor+goal two-token patch already recovered 95.9–97.3% of donor labels.

The remaining question is whether the source-legal donor-specific **two-token** information remains causally useful after controlling for the present readout, external-energy derivative, physical norm, *and the signed first-order future donor-target margin*. A correct-model gradient of the donor margin can predict some immediate logit improvement. The LGA-08 strict matched shams constrain that gradient contribution exactly, testing causal information beyond the local first-order account.

## Chronology, frozen weights and source

The first prospective [PROTOCOL.md](PROTOCOL.md) commit `538f01ad54d183c6adc42e07c794141990375248` predates source-only code and all new 31062 generation. The independent [POWER_PREFLIGHT.md](POWER_PREFLIGHT.md) froze the source-grouped statistical unit, strict +10 percentage-point gradient-matched development threshold, +15-point random-K supplementary gate, and exclusion of protected benchmark populations. No cutoff, arm, dose or source was changed after observing new results.

The exact original LGA-02 44,001-parameter frozen teacher/SAE artifacts were reused: model bundle SHA256 `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`, model metadata `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`. The original LGA-05 shared scalar potential weights were frozen at `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`, fit metadata `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`. No model, SAE or observer retraining or normalizer refitting occurred.

The separate Stage A verified all four original pretrained bytes, canonical source tensor hashes, registered protocols, test code and complete workflow bytes. Its **first pretest source-lock `lock.json` SHA256 `b195c62aff99aa9a98f537910b5c43e581d792b1eb9277d8afea3cdf24ce28ea`** recorded `new_test_generated=false` and `new_test_labels_read=false` and was uploaded BEFORE the reserved source was opened. Then Stage B generated exactly **64 new IID 81-position digit strings** with original `fixed_data(64,31062)`; full ordered little-endian int64 x SHA256 `55aff9ac35186800607d0c23bb4e35935583dc01286786a116b5a107d3f9d6ec`. Existing source seeds 31041–58,31060,31061 were excluded by original registered counts, and reserved 31059 stayed unopened.

For each source and eight fixed goals `(0,10,20,30,40,50,60,70)`, the legal donor changed only predecessor input `p=(j-1) mod 81`, from digit d to `d%9+1`. Thus the correct donor target is independently defined by the generator as the right-shifted legal input, changing just one goal. Every source/cell remains in the intention-to-treat denominator: 64 independently generated base strings ×8 correlated goals ×3 historical teacher seeds =1,536 source-goal cases.

## Mathematical physical intervention

At actual recurrent state `h_2`, the recipient and donor are independently reached by two executions of the same native tied-weight block. Restrict the physical edit to the ordered pair S=(p,j) with two 48-dimensional token-channel vectors, 96 physical coordinates. Current ten-class linear head weights `W_head` have rank 10, so the per-token physical head-right kernel has dimension 38. The pair current-head-right kernel has dimension 76.

Let `U=(h_donor−h_recipient)|S`, `g=(∇E_shared(h_recipient))|S`, `P_H` be the stable orthogonal head-right-null projector. Then

```
U_head = U P_H
G_head = g P_H
U_K = U_head - (<U_head,G_head>/||G_head||²) G_head
```

when the frozen E-row threshold admits an independent energy constraint. All **1,536 source-goal cases** admitted that independent row and gave a **75-dimensional pair K** satisfying current-head silence and first-order whole-external-energy tangency. Set the selected pair edit `z=rho U_K/||U_K||`, where `rho=||U||` is the *full natural two-token donor displacement*, not an arbitrarily chosen multiple of a small native update.

Eight source/model-seeded random-K shams match the same 75D K and same rho. The stricter, target-aware first-order control uses the legal donor and original source classes `d,o` fixed before any edited outcome:

```
m(h) = logit_4(d | h) - logit_4(o | h)
G_T = P_K grad_(h2[S]) m(h2)
a   = <G_T,z>
z_sham = a G_T/||G_T||²
         + sqrt(rho² - a²/||G_T||²) q
```

where each `q` is an independently drawn unit vector in K orthogonal to G_T. All **eight** exact future-gradient-matched shams share K, rho and the selected's signed first-order donor-vs-original margin `a`. These oracle-matched controls use the model and legal source classes to constrain a gradient; they are target-aware controls, not independent task predictors or a new causal training procedure.

Each of the 23 prespecified arms is inserted at the actual state h2 and continued by **two genuine frozen native recurrent blocks** to the h4 full ten-class head. Other frozen arms: pair-head-only projection at rho, unprojected natural pair, independent goal/predecessor natural donor tokens, quarter-dose pair-tangent, and exact **whole-state donor replacement** (a distinct natural large-dose positive control). The source and native execution record every applied physical 2×48 FP32 vector SHA, original donor pair/energy/future-gradient, full donor state delta, current-head change, finite E change, 81 predicted classes, full 10 goal logits, and original model clean donor all-81×10 logits for exact positive-control parity.

## Registered main result: strict incremental 3/3 development PASS

The registered development gate requires each qualifying teacher to have ≥80% clean recipient+donor competence, exact full-state donor replay parity, ≥95% whole-state donor correct among jointly competent, ≥90% natural two-token donor correct among jointly competent, then both primary task-correctness differences:
- **≥+10 pp** `pair_energy_tangent` minus median of eight exact signed-future-margin-gradient-matched same-K shams;
- **≥+15 pp** `pair_energy_tangent` minus median eight random-K shams.

Each contrast also needs a positive lower bound of the frozen **2,000-resample whole-64-base-source bootstrap 95% percentile interval**. The entire development study passes if at least 2 of the three historically exposed teachers qualify.

| Frozen teacher | Selected donor-correct | Median future-gradient-matched equivalent | Selected minus future-gradient match | Whole-base bootstrap 95% | Selected minus random-K | Random-K bootstrap 95% |
|---|---:|---:|---:|---|---:|---|
| 103 | 180/512 (35.16%) | 67.5/512 (13.18%) | **+21.97 pp** | **[+18.75,+25.20] pp** | +34.96 pp | [+31.45,+38.48] pp |
| 211 | 229/512 (44.73%) | 90.5/512 (17.68%) | **+27.05 pp** | **[+23.44,+30.57] pp** | +44.73 pp | [+41.02,+48.44] pp |
| 307 | 211/512 (41.21%) | 73/512 (14.26%) | **+26.95 pp** | **[+23.24,+30.47] pp** | +41.21 pp | [+37.30,+45.70] pp |

**First preregistered result: `DEVELOPMENT_STRONG_INCREMENTAL_PASS`, 3/3 original old frozen teacher seeds.** Source-admission counts are 512/512 per teacher; every entire-pair K has the expected 75D numerical rank. Each positive within-teacher paired contrast meets both fixed margins with descriptive lower bootstrap above zero.

The *actual FP32 applied* strict shams matched the selected pair's signed linearized future margin within a maximum absolute discrepancy of **4.06×10⁻⁷, 5.42×10⁻⁷ and 4.35×10⁻⁷** across 103/211/307. Current all-token linear head changes at the intervention had medians only about `7.56×10⁻⁷`, `7.43×10⁻⁷`, `7.40×10⁻⁷` L2. This separates the observed finite nonlinear donor-correctness gap from the *single frozen signed first-order margin derivative* and current linear readout visibility.

The first original independent auditor using **Python standard library only, no Torch**, passed after independently verifying all **35,328 original pair edits** and **1,536 donor/energy/future-gradient/full-state source vector records**, every legal source/target and SHA, whole-state donor output parity, matching derivatives and physical doses, all ten-class margins and CE, original 2,000 whole-source bootstraps, teacher gates and final verdict. This is the original preregistered audit, distinct from a later exploratory replay.

## Mechanism controls and limitations

| Donor-target correct rate | Teacher 103 | Teacher 211 | Teacher 307 |
|---|---:|---:|---:|
| Natural full-donor state replacement | 506/512 (98.83%) | 502/512 (98.05%) | 500/512 (97.66%) |
| Natural predecessor+goal pair | 501/512 (97.85%) | 494/512 (96.48%) | 489/512 (95.51%) |
| Natural goal-token only | 335/512 (65.43%) | 326/512 (63.67%) | 302/512 (58.98%) |
| Natural predecessor-token only | 67/512 (13.09%) | 69/512 (13.48%) | 52/512 (10.16%) |
| Pair head-null donor at natural pair rho | 185/512 (36.13%) | 242/512 (47.27%) | 225/512 (43.95%) |
| Pair head+energy-tangent donor at rho | 180/512 (35.16%) | 229/512 (44.73%) | 211/512 (41.21%) |
| Pair head+energy-tangent at quarter rho | 2/512 (0.39%) | 2/512 (0.39%) | 1/512 (0.20%) |

The two-token natural donor patch is a strong positive control, successfully transferring donor targets in **494/499, 490/498 and 483/494** jointly clean-competent cases. Full natural state replacement exactly reproduces the independent donor's clean 81×10 logits (max error zero) and scores 100% on the jointly competent cases. This makes the 75D projected pair intervention mechanistically interpretable, even though it falls below the natural pair's absolute donor-correctness rate.

A random 75D K edit of the same L2 norm almost never predicts the donor class; its matched controls provide a strong non-oracle baseline. The signed first-order oracle explains a nontrivial portion, yielding ~13–18% matched-sham correctness. The selected legal-donor direction retains an additional **22–27 percentage points** of correct target recovery after exact linearized-future-margin matching. The average donor-target CE of selected minus each case's median matched-sham CE was **−1.624, −1.645, −1.871 nats** across the three teachers, a secondary post-result descriptive measurement; no familywise inference is attached to it.

The pair projection retains median fractions **0.392/0.419/0.417** of the full natural pair direction's original physical norm (103/211/307). The candidate is rescaled back to the natural pair radius, a deliberately large off-manifold hybrid. The frozen learned E(h) undergoes mean *finite increases* of **+15.11/+12.42/+15.98 total energy units** under the nominally energy-tangent pair edit, with every individual source-goal case increasing E. First-order energy tangency therefore does not imply finite energy preservation at this dose.

The **pair head-null-only** arm is modestly better at correct-target transfer than the extra energy-projected arm (5,13,14 more donor-correct cases). The evidence identifies useful delayed information in the current-head-dark physical pair space. It does **not** establish added causal necessity for the external scalar energy constraint. It does not show the model internally implements gradient descent, an intrinsic objective or identifiable Hodge/antisymmetric component.

The strict matched shams fix one signed *first derivative*. Differences in Hessian curvature, higher-order terms, direction-dependent state routing and off-manifold effects remain candidate explanations for the finite causal advantage. The source classes are used to construct the oracle comparator, so this test does not establish label-blind discovery of semantic circuits. The cyclic-copy task has a known predecessor/goal local causal support and is not a general reasoning benchmark. BPD factor axes, gauge-covariant write-community discovery, SAE semantics, hard Sudoku/TRM tasks and transfer to large LLMs remain untested.

The independent variation units for these intervals are 64 new **base strings** per frozen teacher, with eight correlated donor-goal variants in each base. The teacher checkpoints 103/211/307 have already been selected/examined repeatedly. The confidence intervals are descriptive within-model bootstraps, not checkpoint-root significance or a distribution-free cross-model mechanism guarantee. New model roots, EvalMath variation floors and a separate sealed confirmation cohort are required before general claims.

## Future discriminating experiment

A second-order extension should test whether the mechanism survives **matched finite curvature** in addition to current head/energy tangent, physical pair dose and signed first-order donor-vs-original margin. At fixed h2, define the future signed margin Hessian H_T over the physical K. For selected z, the second-order quantity is `zᵀH_Tz/2`. Construct source-fixed shams in K that match `||z||`, `∇m·z` and `zᵀH_Tz`, then continue true remaining blocks to compare actual corrected donor targets. A preregistered local curvature control can distinguish second-order Taylor predictivity from higher-order or donor-path-specific finite computation. Benchmark engineering must evaluate whether the matched-null set is sufficiently large and whether Hessian-vector products are reproducible, before opening fresh sources.

Independently train new checkpoint roots under a **separate** frozen protocol and different source families for formal mechanism claims. A 59-root calibration plus separately sealed confirmatory population is required if using the registered distribution-free 95/95 maximum tolerance standard. BPD should test whether its learned rank-limited physical writes recover this same pair mediator under orthogonally transported factor gauges and closed-loop native execution. No new cohort or confirmation results are claimed here.

## Immutable science chronology, hashes and receipts

- First source-only preregistration: [PROTOCOL.md](PROTOCOL.md), commit `538f01ad54d183c6adc42e07c794141990375248`; prespecified [POWER_PREFLIGHT.md](POWER_PREFLIGHT.md).
- Native executable source release: [PR #197](https://github.com/MoralityLabAI/LoopedTransformers/pull/197), one-time first-science merge `d76a99c6659f8a009e362d33143e5939301f9ab6`.
- [Original complete GitHub Actions #44308907706](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44308907706): pretest model/observer/tensor/code lock PASS, first native 31062 test PASS (engineering execution), independent stdlib audit PASS (scientific registered development verdict positive).
- [First pretest source-lock artifact #14429167142](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44308907706/artifacts/14429167142): `lock.json` SHA256 `b195c62aff99aa9a98f537910b5c43e581d792b1eb9277d8afea3cdf24ce28ea` and original `new_test_generated=false`.
- [First native complete raw artifact #14429432108](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44308907706/artifacts/14429432108): `full.json` SHA256 **`4470544993ca4cad3aacbf67d0871a81c57bf3fe7366ba82ea585b675378d418`** (81,? bytes); `pair_edits_f32.gz` SHA256 `6c9b7b92a1ad206b4de235d2ac522049f672a6b5389bf2b833e1c10f76669cfa`; `pair_source_gradients_f32.gz` SHA256 `19b6d146bfee2868ef8e4ef8aa934475aeed84377ba40988fa2a8fee5e81bd89`. Source input SHA256 `55aff9ac35186800607d0c23bb4e35935583dc01286786a116b5a107d3f9d6ec`.
- [First independent Python stdlib audit artifact #14429204009](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44308907706/artifacts/14429204009): `audit.json` SHA256 `4e49d91b96482401fd09f2c5b173bca9d5ccd8951f357699e956fc3aef4de3ec`, `audit_status=PASS`, 35,328 verified pair edits, all 1,536 source records and 3/3 original strong incremental development gates.

This report documents the first original prospective experiment exactly. Future analysis or manuscript updates must preserve those hashes, failure cases, test-family limits and frozen registration chronology.
