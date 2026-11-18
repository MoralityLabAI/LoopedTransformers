# LOOPSCOPE-12 / LGA-09: delayed pair-fiber donor transfer survives matching every first-order future logit

**18 November 2026. First source-locked native study: `DEVELOPMENT_STRONG_FULL_J_PASS`, 3/3 original exposed 44K TRM teachers.** The first original independent Python standard-library audit **PASSED**, verifying all 47,616 edited physical 96D vectors, 1,536 full-Jacobian source records, original source roster, the ten-class derivative-match controls, native ten-class outputs, source-cluster bootstrap intervals and registered decisions. The original model roots were examined in prior experiments, so this is **development evidence within one tiny recurrent model family**, not population confirmation across new checkpoints, reasoning benchmarks or language models.

## Why full-J matching was necessary

[LOOPSCOPE-08/LGA-05](../loopscope08/RESULTS.md) fitted an external potential E(h) to native recurrent updates; energy descended along original four-pass trajectories for three teachers, while causal task-loss steering with negative-energy edits failed its matched-random gate and the native recurrence retained curl. [LOOPSCOPE-09/LGA-06](../loopscope09/RESULTS.md) found the frozen external potential could not maintain task-coupled descent beyond the trained horizon.

[LOOPSCOPE-10B/LGA-07B](../loopscope10/RESULTS_LGA07B_FIRST_NATIVE_20261118.md) found a near-null exact donor-target effect from low-dose single-token tangent interventions. [LOOPSCOPE-10C/LGA-07C](../loopscope10/RESULTS_LGA07C_FIRST_NATIVE_20261118.md) recovered +38–48 percentage points correct donor targets from **whole-state** natural-dose current-head/energy-tangent source-donor edits over matched random-K shams, and found a naturally reached predecessor+goal two-token donor patch nearly sufficient for the cyclic-copy target.

[LOOPSCOPE-11/LGA-08](../loopscope11/RESULTS_FIRST_NATIVE_20261118.md) restricted the edit to the **75-dimensional** predecessor/goal head-and-energy tangent pair fiber. It passed a strong predeclared development gate on all three old teachers, with +21.97/+27.05/+26.95 percentage points exact correct-donor recovery over same-K/same-dose random shams constrained to match the selected edit's *one donor-vs-original-class signed first-order future margin*. This did not control the first-order sensitivities of the other eight competing digit logits. The new study holds all **ten** future digit logits' first-order derivatives fixed, thereby testing whether the residual donor-specific effect survives the complete linearized task readout.

## Prospective source, weights and chronology

The **first registered protocol** [PROTOCOL.md](PROTOCOL.md) was committed at **`5719f8d29ed6c22d097222bbef549c3597a3a727`** before implementing LGA-09 or generating its fresh source. [POWER_PREFLIGHT.md](POWER_PREFLIGHT.md) fixed the three comparisons, practical effect thresholds, the independent source unit, root-level scientific hold and protected source exclusions. No dose, source family, comparison, exclusion threshold, observable, teacher weight or scalar E was modified after opening first data.

The three teacher checkpoints (103,211,307) are the original frozen **44,001-parameter** four-pass TinyRecursiveModel instances:
- original LGA-02 `bundle.pt` SHA256 `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`;
- original LGA-02 `train.json` SHA256 `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`;
- original LGA-05 `potentials.pt` SHA256 `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`;
- original LGA-05 `fit.json` SHA256 `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`.

The first original [GitHub Actions #44315883529](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529) first verified all frozen model, SAE and scalar-potential tensor identities; it produced an immutable **pretest** source lock `lock.json` SHA256 `4d0460c9bd6fd6e342c6f0282a7f9744a7fc5082d4bf7e1f2f55af0253097ed4`, with `new_test_generated=false` and `first_native_scored=false` before the new source was generated. Source/implementation/workflow/protocol SHA locks were checked again before model inference.

Only then did the first native execution open exactly **64 fresh independent IID 81-digit source strings** with the original Torch generator `fixed_data(64,31063)`, ordered little-endian input SHA256 `c3c8c2c2ff1ceb70f65834b9f4fdf66b3eb4026c908e4e3c45564b8414644d1d`. Previously generated LGA sources 31041–58,31060,31061,31062 were excluded from exact input duplication, and separately reserved 31059 remained unopened. Each source supplied exactly eight legal goal donors at indices `(0,10,20,30,40,50,60,70)`; the donor changes **only predecessor input digit** `p=(j−1) mod 81` by d→d%9+1, so independent correct donor label at j is fixed by the legal source's right cyclic shift.

The independent within-teacher sample unit is **64 base strings**. Each contains eight correlated goal/donor cases and 31 edited arms. There are **512 original source-goal cases per already-exposed trained teacher**, 1,536 cases across the three historical checkpoints.

## Exact native 75D pair and full future-logit control

At native state `h2`, obtain recipient and legal donor hidden states by running the same original trained tied block twice from their respective legal inputs. Let S=(p,j), with one 48-dimensional physical vector at each source token. The 10×48 current linear head has full row rank ten. Across S the head-right kernel is dimension 76, and intersecting one independent row `∇_S E_shared(h2)` produces a **75-dimensional** physical K.

For pair difference U=(h2_donor−h2_recipient)|S, current-head right projector P_H and restricted learned-energy derivative g, define

```
U_H = U P_H
G_H = g P_H
U_K = U_H − <U_H,G_H>/||G_H||² G_H
z   = ||U|| U_K / ||U_K||
```

with the registered numerical rank/degeneracy rule. The selected source-legal edit z belongs to K, has the **natural two-token donor physical L2 radius** rho=||U||, and changes neither current pair's class-head logits nor learned scalar E to first order.

At the recipient's original h2, differentiate the **actual two remaining frozen native shared blocks** and ten-class final head to obtain the complete 10×96 physical pair Jacobian `J=∂head(F(F(h2)))_j / ∂h2[S]`. Restrict its rows to K, `B=J|K`, and freeze the selected's full future linearized ten-logit response `t=Bz`. Let z0 be the orthogonal minimum-norm rowspace preimage of t, and draw eight independent q vectors in `K∩ker B`. Each strict sham is

```
z_fullJ = z0 + sqrt(rho² − ||z0||²) q,    ||q||=1.
```

The shams have the same K, Euclidean dose and **all ten first-order future digit-logit responses** as the donor-selected edit, while their first-order future-silent residual is independently randomized. All source planning precedes task-outcome scoring.

Every original source-goal was admitted: **1,536/1,536**, all three teacher restricted J matrices had full numerical rank **10** in the pair K, and therefore the future-first-order silent subspace had dimension **exactly 65** in each case. Applied FP32 physical shams' maximum absolute per-digit linearized future response discrepancy from selected z was **4.28×10⁻⁷ / 4.69×10⁻⁷ / 4.22×10⁻⁷**, respectively, for old teacher seeds 103/211/307.

Original arm family is **31**: selected K source donor; eight full-ten-J matched shams; eight source-defined single signed-margin matched controls; eight source-only random K shams; pair head-only, natural pair, natural goal, natural predecessor, quarter-dose tangent and exact naturally reached full-donor-state positive control. Every actual edit executes two **real original tied recurrent blocks** to h4. The raw receipt preserves every f32 physical pair edit (47,616 in all), all 1,536 original full J tensors/source gradients and native donor deltas, complete goal ten-digit logits and all 81 final predicted classes.

## First registered causal result and source-bootstrap intervals

The registered development gate demands ≥95% geometric/full-J admissions, ≥80% jointly clean recipient+donor competence, exact full-state donor/native-head positive parity and ≥95% correct donor goals among jointly competent, natural pair ≥90% among competent. The main comparison must give at least **+10 percentage points** correct donor-target recovery over the median of eight **full-ten-J-matched** shams with a positive 2,000-draw within-teacher **whole-64-base bootstrap** lower bound. Supplementary registered constraints require selected minus single-margin matched ≥+10 points and minus random-K ≥+15 points, both with positive source-cluster bootstrap lower. Study passes the development gate if at least 2/3 historically exposed teacher seeds satisfy every condition.

| Original frozen teacher | Selected donor-target correct | Selected − full-ten-J match | Full-J 95% whole-base bootstrap | Selected − single-margin match | Selected − random-K |
|---|---:|---:|---|---:|---:|
| 103 | **181/512 (35.35%)** | **+17.6758 pp** | **[+13.8672,+21.6797] pp** | +22.5586 pp | +35.1563 pp |
| 211 | **219/512 (42.77%)** | **+21.8750 pp** | **[+18.0664,+25.8789] pp** | +26.8555 pp | +42.5781 pp |
| 307 | **241/512 (47.07%)** | **+23.8281 pp** | **[+20.2148,+27.3438] pp** | +29.5898 pp | +47.0703 pp |

All three teachers meet **every preregistered strong full-J incremental engineering gate**, giving the first verdict **`DEVELOPMENT_STRONG_FULL_J_PASS`, 3/3 original exposed trained seeds**. For reference the source-level 95% bootstrap intervals for signed-margin matched advantages were:
- seed 103 [+18.4570,+26.8555] pp; seed 211 [+22.9492,+30.7617] pp; seed 307 [+25.5859,+33.3008] pp;
and for simple random-K advantages:
- seed 103 [+30.6641,+40.0391] pp; seed 211 [+38.4766,+47.2656] pp; seed 307 [+42.3828,+51.9531] pp.

The intervals are **descriptive source-string bootstraps within three previously chosen checkpoints**. They are not model-root-confidence intervals or proof of out-of-distribution transfer. The full-J oracle shams successfully transfer substantially more donor labels than random-K: each seed's first individual full-J sham recovers 89/512, 108/512 and 118/512 donor targets, compared with near-zero random-K recovery. Yet the selected source-legal donor directions outperform the within-case median full-J shams by 18–24 percentage points. This excludes all *first-order local future ten-digit logit differences* at the original recipient state as a sufficient explanation of the observed finite nonlinear donor-target effect.

## Physical controls and independent model parity

| Registered donor-target arm | Teacher 103 | Teacher 211 | Teacher 307 |
|---|---:|---:|---:|
| Source-selected head/energy-tangent pair K at natural rho | 181/512 (35.35%) | 219/512 (42.77%) | 241/512 (47.07%) |
| Head-null-only pair at same rho | 196/512 (38.28%) | 234/512 (45.70%) | 258/512 (50.39%) |
| Natural source-donor predecessor+goal pair | 489/512 (95.51%) | 502/512 (98.05%) | 491/512 (95.90%) |
| Natural full-state donor positive control | 500/512 (97.66%) | 504/512 (98.44%) | 506/512 (98.83%) |
| Source-selected K pair at quarter rho | 7/512 (1.37%) | 4/512 (0.78%) | 6/512 (1.17%) |

Among jointly correct clean recipient+donor cases, natural entire donor replacement yields **100%** correct donor goals and exactly reproduces the independently reached clean donor **all-81-token×10-class logits** (maximum discrepancy zero). Joint clean competence was 493/512, 492/512 and 501/512. The natural two-token donor also succeeded in **97.77%,99.59%,97.01%** of those jointly competent pairs, preserving a strong positive control for this local cyclic-copy mechanism.

Pair head-null-only transfer is modestly better than the head+E tangent source edit on **all three** teachers. Its corrected-label outputs agree with the selected K edit on 491/512,487/512,479/512 cases. The extra external scalar-energy constraint has therefore not demonstrated positive incremental utility. The E-tangent selected edits raise the frozen nonlinear scalar by mean **+14.89,+12.08,+15.73 total E units** across teacher roots and produce finite energy increases on **all 1,536 original source-goal cases**, even though source-level `dE(h2)·z≈0`. Median current all-token head-logit L2 displacement under selected edit was only ~`7.4–7.7×10⁻⁷`, consistent with true physical current-head-null directions.

Median selected L2 donor projected fraction `||U_K||/||U||` before rescaling was approximately **0.3993,0.4161,0.4150**; the natural pair physical radius medians were **8.745,8.791,8.705**. After matching complete ten-logit linearized response, the source-selected donor improved independently defined donor-target cross-entropy by mean **−1.217,−1.205,−1.367 nats** versus each case's median full-J sham CE. These continuous CE comparisons are descriptive post-result measurements and have no separate familywise-controlled scientific discovery claim.

## Interpretation and limitations

The result establishes that a **source-specific physically defined hidden-state donor direction** within a pairwise current-head-dark space can carry useful delayed information through a real native recurrent model, beyond all first-order future task-logit derivatives of that direction at h2. It is a source-conditioned causal intervention effect relative to a strong matched null in the frozen 44K cyclic-copy TRM family, with exact independent original source and physical-vector replay.

The experiment does **not** prove that the trained network internally minimizes learned E, that E is necessary for causal transfer, that a unique antisymmetric/Hodge representation exists, or that a trained BPD factor basis has discovered these physical directions. Head-only's slightly better performance and finite external E increases counsel against making an "internal energy optimizer" mechanism claim.

Matching first-order J fixes the full **local Taylor linearization**, while finite radius leaves Hessian, higher-order derivatives, nonlinear state-dependent routing and off-manifold effects available as explanations. Because the source-selected edit has natural donor-specific provenance and each sham receives an independently randomized future-silent residual, the result suggests the finite nonlinear computation distinguishes structured source-directions from alternative directions with identical linearized future outputs. This is not yet source-manifold semantic circuit discovery in large LLMs or transfer to hard Sudoku/Nanbeige models.

To identify the next mechanism, a preregistered follow-up should measure the **second-order finite Taylor contribution of all ten future goal logits** at the same physical support and dose, and compare source-selected K donor with shams matched in both first- and second-order output predictions. Alternatively test a multiple-dose response on entirely new sources with controls matched at each fixed radius, including independent source-conditioned reachability and actual finite-energy constraints. An independent new-model checkpoint-root program with EvalMath variation floors and sealed confirmation is still required for general claims.

## Original receipts and independent audit

- First prospective [PROTOCOL.md](PROTOCOL.md) committed at `5719f8d29ed6c22d097222bbef549c3597a3a727`; source-only preflight [Actions #44313607024](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44313607024) PASSED before first source release.
- Executable one-time first-science [PR #200](https://github.com/MoralityLabAI/LoopedTransformers/pull/200), merged at `f19db86154a2e990ebf34853603a9bac33db5e19`, under the first-release guarded commit title.
- First complete source-locked [Actions #44315883529](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529): original source artifact and frozen tensor/protocol/executable SHA lock PASS, original first native 31063 experiment PASS (engineering execution), and original independent Python stdlib-only audit PASS.
- Original pretest source-only [lock artifact #14432496514](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529/artifacts/14432496514): `lock.json` SHA256 `4d0460c9bd6fd6e342c6f0282a7f9744a7fc5082d4bf7e1f2f55af0253097ed4`.
- Original [full source/native artifact #14432048361](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529/artifacts/14432048361): `full.json` SHA256 **`480f5a5b7e133060602e1ba850e39bd25565313b68585103820f43b62305d65a`**, `pair_edits_f32.gz` SHA256 `6500f8da067eedb76eea37d5a0572ed45b5a0ca44905aa3deae2849e4fe48421`, `ten_jacobian_source_f32.gz` SHA256 `39b0cd14b4ad2a2df05b798da3d6419b263bc471494c2e3af4bfed0b34b8d75e`. All original 47,616 actual f32 physical pair edits and 1,536 complete ten-J/source gradient records preserved.
- Original independent [stdlib audit artifact #14431523538](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529/artifacts/14431523538): `audit.json` SHA256 `9a9c746bab12088ceb797d2c41db105a4b075d44f366b5e064412cf6a99ffa21`, `audit_status=PASS`, verifying every original physical edit, matching all ten future-J derivatives, legal target scoring, all base-cluster bootstraps and fixed 3/3 development verdict with **no Torch**.

This report preserves the original first registered positive result and all comparator limitations. Subsequent report revisions, audits or new hypotheses must retain the original source/artifact hashes and source/scoring chronology and may not reclassify the three already examined teacher checkpoints as independent scientific confirmation.
