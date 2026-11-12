# GG-08 closeout: depth-eight competency and protected-tail harm-sign recovery

**November 12, 2026 | Registered result: BOTH PRIMARY GATES PASS.**

[GG-08 registration](../docs/rmp_goal_guarding_competency_recovery_gg08.md) was frozen before any new continuation training or new gold-scoring panel outcomes. GG-07's original physical-SGD study had produced capability-limited findings on eight-cue Bayesian boundary, rare, and late-evidence-reversal cases. GG-08 adds a sample-visit-matched four-cue continuation control and a curriculum continuation that exposes the native recurrent decoder to four-, six-, and eight-cue cases.

## Verification and source

- Source model: original GG-01 native \`EvaluationBeliefLooper\` with one reused \`ExplicitDecoderBlock\`, four heads, width 32. Source \`main\` commit \`97b3eba35a5e0c037fc4f5f65b962e68b07bbdbb\`, reproduced from scratch using seeds **11 and 17** and **400** original four-cue supervised updates.
- Paired training arms start from the *exact same original model weight SHA*: the frozen source; **300 AdamW updates** with 4-cue batch **192**; and **300 AdamW updates** cycling depths **4,6,8** with batch **128**, exactly **100 updates each**. Both continuations have **230,400** total unrolled sample-visits. This is matched sample-visits/optimizer steps, not guaranteed bitwise FLOPs or equal numbers of unique training cases. Source initial model weights remain unchanged by both continuation procedures.
- The continuation pools are new seeded exact Bayesian panels of **8,192 examples per depth** and model seed. Both continuations share the same four-cue pool. The original model and two continuation endpoints were frozen before any protected gold interventions.
- Protected gold: **new independent GG-08 scoring panels**, generated using the *unchanged GG-07 exact-Bayes challenge definitions*. At each depth 4,6,8, four replicate panels contain 64 natural, 32 near-decision-boundary, 32 exact-mixture-rare and 32 late-evidence-reversal examples. There are **640 gold examples per depth** (1,920 across three depths) per model's physical SGD state. The scoring panel seeds were registered before new training and are shared by every arm/seed.
- Physical perturbation: **one real PyTorch SGD step** on a separately cloned endpoint task model for each truthful or flipped four-cue reported label at LR **0.05**, global gradient norm clip **1**, evaluated on every registered gold scoring row. Source queries are the previously known 48 GG-04B heldout inputs. Total across both model seeds and all three arms: **576 actual SGD update steps** and **1,105,920 per-example post-SGD gold cross-entropies**. Raw truth/flip/gap CE for every scoring sample, exact query IDs and all model SHA-256 hashes are retained.
- [Original GG-08 GitHub Actions run 43164398756](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43164398756) **passed seven focused tests in 8.70 seconds**, both model seeds and all three arm experiments, and the source-bound full raw physical-gold ledger/schema checks. [Original run full source-bound artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43164398756/artifacts/13919600920) contains training/continuation checkpoints, physical SGDs and robust interval witnesses.

## Preregistered competency gate

The protected scorer was designed to require **at least 90% exact Bayesian task accuracy in each of 12 depth×cohort cells** before interpreting a harm-sign result as capability eligible. Each natural cell contains 256 distinct draws across its four replicates, and each challenge cell contains 128 draws with replacement. The input grammar is finite and shares the same underlying Bayes channel as the training examples.

| Model arm | Seed 11, minimum of 12 cells | Seed 17, minimum of 12 cells | 90%-per-cell gate |
|---|---:|---:|---|
| GG-01 original after 400 updates | **89.06%** | **89.84%** | Fail / Fail |
| Four-only continuation, 300 further updates | **92.97%** | **97.66%** | Pass / Pass |
| Mixed 4/6/8 continuation, 300 further updates | **100%** | **100%** | **Pass / Pass** |

The mixed-depth continuations achieve **100% on every one of the 12 scoring cohorts** for both model seeds. In particular, all depth-eight boundary, rare, and late-reversal challenge cells are perfect on this finite newly sampled gold roster.

Detailed eight-cue scoring:

| Seed and arm | Eight-cue natural | Eight-cue boundary | Eight-cue rare | Eight-cue late |
|---|---:|---:|---:|---:|
| 11, original | 98.44% | 89.06% | 90.63% | 92.19% |
| 11, four-only | 99.22% | 92.97% | 96.88% | 96.88% |
| 11, mixed-depth | **100%** | **100%** | **100%** | **100%** |
| 17, original | 99.61% | 91.41% | 94.53% | 89.84% |
| 17, four-only | 99.61% | 97.66% | 98.44% | 99.22% |
| 17, mixed-depth | **100%** | **100%** | **100%** | **100%** |

All 4-cue scoring cells were also 100% after either continuation, with no registered four-cue catastrophic-forgetting signal.

The **two-seed competency recovery gate PASSES**. The four-only control also exceeds the 90% competence threshold, so the additional eight-cue curriculum is best understood as improving the measured *margin of competency*, alongside a more substantial change in protected loss-gradient sign robustness.

## Original GG-07 protected-tail objective, unchanged

For each source four-cue query \(x\), and each depth/cohort/replicate, physically measure the wrong-minus-truthful report SGD CE gap \(D\). Aggregate using the *unchanged* GG-07 protected objective:
\[
B_{d,r}(\lambda)=(1-\lambda)\bar D_{N}
+\frac\lambda3(\bar D_B+\bar D_R+\bar D_L),
\]
with primary tail mass \(\lambda=1/2\). Each of the four independent cohort distributions may shift inside an empirical-support TV ball of radius \(\rho=1/64\). A subsequent product ambiguity chooses a depth-mixture weight in the simplex \(\alpha_d\ge1/6\), with \(d\in\{4,6,8\}\). This yields the same attained sharp lower and upper bounds as GG-07, using independent cohort mass transport and the depth-weight simplex vertices. The exact optimization is conditional on observed gold loss differences, without accounting for novel unsupported inputs.

A query is **robustly harmful** when all four replicate lower bounds exceed \(10^{-7}\), **robustly beneficial** when all four replicate upper bounds are below \(-10^{-7}\), and unresolved otherwise.

## Registered primary physical harm-sign results

| Arm and model seed | Robustly harmful | Robustly beneficial | Unresolved | Total signed |
|---|---:|---:|---:|---:|
| Seed 11, original | 19 | 9 | 20 | 28/48 |
| Seed 11, four-only control | 17 | 17 | 14 | **34/48** |
| Seed 11, mixed 4/6/8 | **48** | **0** | **0** | **48/48** |
| Seed 17, original | 5 | 7 | 36 | 12/48 |
| Seed 17, four-only control | 19 | 12 | 17 | **31/48** |
| Seed 17, mixed 4/6/8 | **48** | **0** | **0** | **48/48** |

**Preregistered strong harm-sign recovery criterion:** the mixed-depth arm must be capability eligible, have at least 24/48 robustly signed queries, and exceed the paired four-only control by at least 8/48 robustly signed queries **in each seed**.

- Seed 11: mixed **48/48**, control **34/48**, improvement **+14**.
- Seed 17: mixed **48/48**, control **31/48**, improvement **+17**.

The **two-seed strong robust-harm-recovery gate PASSES**, with all 96 seed/query combinations protected-harmful under the preregistered empirical scoring uncertainty. This is a measured effect of real native report SGD on a supervised exact Bayesian task with 48 already-known report inputs. The positivity reflects that, at these endpoint weights, wrong-label report training worsens the protected task more than truthful report training under each registered scoring ambiguity.

## Gold-weight and sign-stability controls

The all-four-replicate same-nonzero-nominal-sign count, and the number of query inputs that acquire an opposite sign when switching from natural-only to protected-tail gold:

| Seed and arm | All-replicate nominal-sign stability | Any natural-to-protected nominal sign reversal |
|---|---:|---:|
| 11, original | 43/48 | 38/48 |
| 11, four-only | 43/48 | 32/48 |
| 11, mixed-depth | **48/48** | **0/48** |
| 17, original | 32/48 | 38/48 |
| 17, four-only | 34/48 | 25/48 |
| 17, mixed-depth | **48/48** | **0/48** |

The GG-08 secondary nominal-sign-stability event also passes on both mixed-depth models. The complete artifact retains the *full original GG-07 five-point lambda grid and two-point rho grid*, including the shadow controls, while the primary conclusion uses only the frozen lambda 1/2, rho 1/64.

## Inference, confounders and next experiment

The experiment demonstrates a clear conditional neural result: **targeted recurrent-depth training changes both long-horizon task accuracy and the sign robustness of physically measured report-gradient effects** on a prescribed protected gold scoring objective.

The control uses equal optimizer-update count and total unrolled sample-visits. It processes a larger number of four-cue samples than the mixed arm and has a different batch size. Neither regimen has exactly identical FLOPs. Separate continuations use distinct registered RNG states and the same original source weights. These features limit how precisely the difference can be attributed to depth curriculum alone.

The new gold draws are independently sampled, yet both training and scoring share the small \(9\cdot2^d\) finite task grammar and the same exact Bayes channel. An eight-cue test passing on the fixed scoring draws does not prove generalization to unseen task families or real training-awareness signals. The robust TV certificate only reweights *observed* gold support. A larger empirical sample, more seeds, and an out-of-grammar stress set are required to make population-level claims.

GG-08 trains no reporting agent that can infer or change its own optimizer and contains no evaluation-disclosure cues during the continuation. It neither observes deceptive alignment nor tests a model strategically modifying its own parameters. The productive successor is to **freeze a wider out-of-grammar transfer test** before revisiting a self-update meta-learning protocol.
