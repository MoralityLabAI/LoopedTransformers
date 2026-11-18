# LOOPSCOPE-12 / LGA-09: native donor effects remain after full ten-logit first-order matching

**18 November 2026 · Original source-locked `DEVELOPMENT_STRONG_FULL_J_PASS` · 3/3 exposed old TRM teachers · original independent Python-stdlib audit PASS.**

The first prospective `31063` study found that a source-reachable predecessor/goal pair edit can transfer a correct alternative target through the native recurrent block even when eight controls match its current-head invisibility, scalar energy tangency, physical norm and **entire ten-class linearized future-logit response**. This is evidence of a nonlinear finite-state donor-specific effect in the trained 44K cyclic-copy TRM family. The evidence does not identify a unique second-order mechanism, a learned internal scalar objective or a cross-checkpoint generalization theorem.

## Background and experimental question

LGA-05 observed a shared external energy E decreasing across original recurrent trajectories; LGA-06 showed energy/task collapse after extrapolated passes. LGA-07B found a single-token low-dose donor-transfer NULL. LGA-07C recovered delayed donor information from global current-head-dark physical edits, and LGA-08 localized that transfer to the source predecessor/goal pair (two 48D token states). LGA-08 gave a preregistered 3/3 development advantage of +21.97/+27.05/+26.95 percentage points over controls matched on the **single signed future donor-versus-original margin gradient**.

Other rival digit logits could explain that signed-margin result through unmatched first derivatives. LGA-09 therefore freezes **all ten** future goal class derivatives and tests whether source donor geometry still adds correct target transfer beyond the full linearized response.

## Prospective source, model and validation chronology

First [LGA-09 PROTOCOL.md](PROTOCOL.md) registration commit `5719f8d29ed6c22d097222bbef549c3597a3a727` predates the implementation or any test on reserved seed 31063. [POWER_PREFLIGHT.md](POWER_PREFLIGHT.md) freezes the engineering threshold, old-root development-only status and independent later confirmation requirement.

First original merge [PR #200](https://github.com/MoralityLabAI/LoopedTransformers/pull/200), one-time native test in [GitHub Actions #44315883529](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529).

Original model and learned energy artifact identities:
- Frozen LGA-02 trained TRM/SAE `bundle.pt`: `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`; manifest `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`.
- Frozen LGA-05 external energy `potentials.pt`: `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`; fit manifest `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`.

The first source-only pretest artifact `lock.json` SHA256 `4d0460c9bd6fd6e342c6f0282a7f9744a7fc5082d4bf7e1f2f55af0253097ed4` verified original trained tensor hashes, protocols, source code and CI workflow bytes before recording `new_test_generated=false` and `new_test_labels_read=false`. Only then did Stage B first generate `fixed_data(64,31063)`, a new source disjoint from the exposed LGA IID rosters. Ordered source input SHA256 `c3c8c2c2ff1ceb70f65834b9f4fdf66b3eb4026c908e4e3c45564b8414644d1d`. Seed 31059 stays separately reserved.

Each of 64 independent base strings yields eight legal donor variants at goals (0,10,20,30,40,50,60,70), changing exactly predecessor digit `p=(j−1) mod 81` with `d→d%9+1`. The alternative correct donor label is the right-cyclic shift of the legal donor input and changes only at goal j. The eight goal variants are correlated within a base string. Three original teacher seeds 103/211/307 have been examined in the previous LGA series. The scientific population is therefore three **historically exposed models** with 64 new independent source bases per model, not 1,536 new independent model roots.

## Physical nullspace and full future-logit matching

At native recurrent depth k=2, the model has an 81×48 physical hidden state. The edit support S=(p,j) is a 96D pair. The complete current ten-class linear head has physical pair-right nullity 76. Intersecting the frozen external E-gradient row gives physical pair tangent K of **dimension 75**, with an independently computed orthogonal Euclidean projector.

For the original two more tied-block passes define the model-only future goal-logit vector `L(h2)∈ℝ¹⁰`. Its pair Jacobian restricted to K is `B=DL(h2)|_K`. The ten Jacobian rows have full numerical rank **10 on all 1,536 first source-goal cases**, leaving **65 dimensions** in the joint physical future-first-order-silent subspace `K∩ker(B)`.

A donor-specific source-reachable edit `z` is constructed by projecting the natural donor pair difference into K, then scaling it to the original natural two-token donor L2 radius `rho`. Given a basis R for the rowspace of B, define `z0=RRᵀz`. Every full-J-matched sham is
```
z_sham = z0 + sqrt(rho² - ||z0||²) * q
```
with an independent, seeded unit q inside `K∩ker(B)`. All eight full-J shams share the same K and L2 radius as z, with exactly matching **ten individual signed first-order goal-logit shifts**, not only a single future margin. Every direction and source acceptance decision was frozen before true task correctness scoring.

The actual applied FP32 edits matched the entire ten-class linearized responses with maximum absolute mismatch **4.28×10⁻⁷ / 4.69×10⁻⁷ / 4.22×10⁻⁷** for seeds 103/211/307 respectively. Current 81-token head-logit perturbation is near numerical zero. These checks apply to physical state edits and first derivatives; full nonlinear h2→h4 continuation defines the behavioral endpoint.

Prespecified controls are eight weaker single-margin-matched K shams, eight purely random K shams, head-only donor, natural pair donor, single goal and predecessor donors, quarter natural-dose tangent, and exact whole-state reached-donor replacement. In all, **31 edited arms** per goal run the same real two-block continuation.

## Registered original positive result

The primary endpoint is correct alternative donor goal after native continuation under source-selected edit minus the **median of eight full-Jacobian-matched shams**. Eight paired goals are averaged within each of the 64 base sources, with frozen 2,000-resample whole-base percentile bootstrap CIs. The preregistered STRONG_FULL_J_INCREMENTAL development gate requires ≥+10 percentage points and positive within-teacher base-bootstrap lower bound over full-J shams, plus ≥+10 over margin shams and ≥+15 over random-K shams, and competence/parity/admission requirements, in ≥2/3 old teachers.

| Previously exposed teacher | Selected correct donor target | Full-J sham mean-equivalent donor correctness | Paired selected–full-J | 95% whole-base bootstrap | Selected–single-margin | Selected–random-K |
|---|---:|---:|---:|---:|---:|---:|
| 103 | 181/512 (35.35%) | 90.5/512 (17.68%) | **+17.68 pp** | [+13.87,+21.68] pp | +22.56 pp | +35.16 pp |
| 211 | 219/512 (42.77%) | 107/512 (20.90%) | **+21.88 pp** | [+18.07,+25.88] pp | +26.86 pp | +42.58 pp |
| 307 | 241/512 (47.07%) | 119/512 (23.24%) | **+23.83 pp** | [+20.21,+27.34] pp | +29.59 pp | +47.07 pp |

All **1,536 source-goal cases were admitted**, with 75D pair K and full ten-logit rank 10 per case. Joint clean recipient/donor competence was **493/512, 492/512, 501/512**. Exact whole-state donor replacement reproduced clean donor output (max full-head logit difference zero) and correctly recovered 100% of the jointly competent donors. The natural two-token pair control also exceeded its ≥90% conditional competence gate in every teacher.

**Original registered verdict: `DEVELOPMENT_STRONG_FULL_J_PASS`, 3/3 old teachers.**

The **original preregistered Python-standard-library auditor** independently verified every legal donor source/target, 47,616 original applied 96D physical edits, all 1,536 archived full ten-logit Jacobian source records, head/energy/ten-J restrictions and doses, all final task logits/CE, 2,000 source-base bootstrap replicates, and the unchanged development verdict. Audit PASS was produced in the original Actions run, rather than relying on a later post-result validation.

## Mechanistic controls

| Exact donor-target correct rate | Teacher 103 | Teacher 211 | Teacher 307 |
|---|---:|---:|---:|
| Whole naturally reached donor state | 500/512 (97.66%) | 504/512 (98.44%) | 506/512 (98.83%) |
| Natural predecessor+goal pair | 489/512 (95.51%) | 502/512 (98.05%) | 491/512 (95.90%) |
| Goal token natural donor | 325/512 (63.48%) | 334/512 (65.23%) | 327/512 (63.87%) |
| Predecessor token natural donor | 70/512 (13.67%) | 61/512 (11.91%) | 57/512 (11.13%) |
| Pair head-only donor at natural pair rho | 196/512 (38.28%) | 234/512 (45.70%) | 258/512 (50.39%) |
| Pair E-tangent donor at natural pair rho | 181/512 (35.35%) | 219/512 (42.77%) | 241/512 (47.07%) |
| Pair E-tangent at rho/4 | 7/512 (1.37%) | 4/512 (0.78%) | 6/512 (1.17%) |

The head-only physical donor consistently recovers **more** correct targets than head+E tangent (15/15/17 extra correct cases). The scalar energy restriction has no demonstrated independent causal advantage. Under the selected natural-radius E-tangent pair edit, **every recorded case increased the finite external scalar E**, with mean changes **+14.89/+12.08/+15.73** across teachers, despite the first-order derivative being zero. Matched full-J shams also increased finite E; those mean differences by themselves do not establish a unique energy-mediated mechanism. A naturally reached pair donor changes E much less on average.

The full-J-matched sham baseline accounts for approximately 18–23% absolute correct donor rate under this natural-dose intervention, substantially higher than random-K noise near zero. It is a first-order-controlled causal comparator, although it still leaves the selected source patch's **18–24 point** behavioral advantage.

The interpretation is therefore restricted: **future target effects depend on the nonlinear finite recurrent response to donor-specific pair directions beyond the ten-logit Jacobian at the clean state**. Hessian curvature, higher-order Taylor terms, state routing and off-manifold interactions remain observationally entangled. Source-legal donor structure carries information, and the model's own goal-source cyclic-copy rule puts much of it into a small, known spatial support. This experiment does not demonstrate an intrinsic recurrent optimizer, a learned BPD factor dictionary, or general reasoning transfer.

## Next registered direction

A second-order causal test should distinguish whether the full-J gap comes largely from quadratic curvature of the eventual ten-logit map along the edited pair directions. For source-only feasibility, compute the ten output Hessian-vector products in K on excluded data, with numerical parity against symmetric finite differences and cheap low-rank HVP sketches. Within the 65D joint first-order future-silent space, seek shams that match the **same full ten-dimensional second-order logit response** under the natural pair radius, while preserving the exact ten first derivatives, K constraints and L2 dose. This requires a prospective feasibility/power gate; failure to admit matched shams would be an engineering result, not behavioral evidence.

A less expensive first step matches the single **legal donor-vs-original margin quadratic form** `zᵀH_m z` and keeps all ten first derivatives fixed, with the other class curvatures reported descriptively. Such a result would narrow the second-order explanation for one margin. A complete ten-output second-order match is the stronger scientific target because argmax depends on all rival logits. Train independent fresh teacher roots and benchmark source families for future scientific confirmation after the geometric instrument stabilizes, with EvalMath root-level calibration and separately sealed confirmation.

## Original immutable receipts

- Original first prospective protocol: [PROTOCOL.md](PROTOCOL.md), commit `5719f8d29ed6c22d097222bbef549c3597a3a727`.
- First-science native and audit CI: [Actions #44315883529](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529), 3/3 development PASS, independent stdlib PASS.
- Pretest [source lock artifact #14432496514](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529/artifacts/14432496514), `lock.json` SHA-256 `4d0460c9bd6fd6e342c6f0282a7f9744a7fc5082d4bf7e1f2f55af0253097ed4`.
- Original [full native data artifact #14432048361](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529/artifacts/14432048361): `full.json` SHA-256 `480f5a5b7e133060602e1ba850e39bd25565313b68585103820f43b62305d65a` (103,774,084 bytes); applied `pair_edits_f32.gz` SHA-256 `6500f8da067eedb76eea37d5a0572ed45b5a0ca44905aa3deae2849e4fe48421`; source `ten_jacobian_source_f32.gz` SHA-256 `39b0cd14b4ad2a2df05b798da3d6419b263bc471494c2e3af4bfed0b34b8d75e`.
- First independent [stdlib audit artifact #14431523538](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44315883529/artifacts/14431523538), `audit.json` SHA-256 `9a9c746bab12088ceb797d2c41db105a4b075d44f366b5e064412cf6a99ffa21`. `audit_status=PASS`, `n_original_edit_vectors=47616`, `n_original_source_jacobians=1536`, `n_old_teachers_strict_pass=3`.

All original outcomes and the preregistered development decision are frozen. The original independent audit gives strong reproducibility of this particular empirical result, with the three prior teacher models remaining the independent checkpoint-level sample.
