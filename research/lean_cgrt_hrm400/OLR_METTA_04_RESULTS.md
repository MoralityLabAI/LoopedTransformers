# OLR-METTA-04: trained dependent-witness stage controller

**Status:** completed exploratory native pilot, audited 2026-11-24.  
**Registration:** [OLR_METTA_04_REGISTRATION.md](OLR_METTA_04_REGISTRATION.md).  
**Verified commit:** 42b0b4a45bd4e3df864942919831f3b191e87fe2.  
**Native stage run:** [45212958918](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45212958918), successful. [CPU regression 45212959178](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45212959178) and [PPT G03 45212959111](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45212959111) also passed.  
**Archived evidence:** [artifact 14837117274](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45212958918/artifacts/14837117274).

## Native grammar and model

The dependent proof binds predicates P,Q : Nat → Prop, witness h0 : P (a+b), and dependent implication h1 : ∀ n, P n → Q n. Both root rules require an equality guard and obligations P c, Q c indexed by the candidate c. The uniquely provable branch can be completed with the five-tactic manual proof or the four-tactic fast proof (using exact h1 _ h0 for the final dependent goal). The native 04A grammar gate [45208410419](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45208410419) passed all 24 executions and replays and 12 independently compiled declarations.

The 04 campaign evaluates root, open, guard-accepted, guard-failed, retry and terminal decisions with a common six-native-tactic cap. Hyperon 0.2.10 checks public source/shape/budget eligibility for the two-step repair; actual Lean tactics, kernel closure and an axiom audit certify the proof. One host restore is allowed. A wrong opening costs two native checks before an early failed guard; a four-call fast continuation after restoring can still finish within six. The longer manual proof no longer fits. The fast tactic is a hand-specified proposal, not a learned MeTTa inference discovery.

All policies receive six curated synthetic witness claims, public stage and budget, previous branch identity and repair eligibility. They do not receive theorem numerals. The larger-operand panel changes native theorem operands but does not deliberately shift the model-visible distribution; this is fresh-root transfer, not demonstrated neural-input OOD extrapolation.

Three seeds 17/29/43 train for 48 optimizer updates on the same 48 training worlds. The one- and three-pass tied attention models share 5,528 parameters. The direct full-packet MLP has 8,672 parameters. Source branch outcomes are native-checked and replayed; 48 positive manual source proofs compile independently. The 96 source sequences are **expert-constructed stage templates** (one correct-start and one counterfactual wrong-start per world). The root winner is obtained from native Lean. The stage targets are programmed from budget and eligibility; the wrong-start template contains the correct root label and then a counterfactual wrong branch. Consequently this is a teacher-imitation pilot, not exhaustive native action-value training or 96 actual end-to-end expert rollouts. No query labels enter training. Same-worker seed-17 three-pass source fitting reproduced every tensor.

## Original native results

Sixteen distinct proof worlds in each panel (eight pairs) are evaluated at three model seeds. The scripted references run once per world. All 480 policy episodes were independently replayed, and 41 distinct theorem/branch/tail declarations independently compiled without axioms. The numerical table below is read from the archived per-episode rows.

| Policy | DEV proofs | DEV utility | DEV calls | Larger operands proofs | Larger utility | Larger calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-pass tied | 43/48 | 0.57500 | 220 | 38/48 | 0.50292 | 198 |
| 1-pass tied | 38/48 | 0.48542 | 210 | 37/48 | 0.46896 | 207 |
| Direct MLP | 20/48 | 0.20667 | 144 | 23/48 | 0.26333 | 148 |
| Fixed r0 + adaptive scripted repair | 12/16 | 0.45250 | 68 | 12/16 | 0.45250 | 68 |
| Full-information exact | 16/16 | 0.68500 | 72 | 16/16 | 0.68500 | 72 |
| Historical majority (sign error) | 9/16 | 0.29563 | 61 | 11/16 | 0.37688 | 71 |

Utility = completion minus 0.07 per actual native tactic call. The three-pass policy completed 15/16,13/16,15/16 in development and 13/16,12/16,13/16 on larger operands. The one-pass control completed 14/16,12/16,12/16 and 13/16,12/16,12/16 respectively. The original one-pass and three-pass models have different active compute. The direct model has different parameters and no inter-decision carry.

Descriptive paired-root bootstrap (10,000 resamples, all seeds averaged within each of eight root pairs): 3-pass minus 1-pass completion +10.42 percentage points in development (95% interval −2.08 to +25.00 pp) and +2.08 points on larger operands (0.00 to +6.25 pp). The corresponding utility differences are +0.08958 (−0.01833 to +0.20917) and +0.03396 (+0.00875 to +0.06688). Neither supports a direction-specific carry or matched-compute claim.

### Failure decomposition

| Policy | DEV correct first choices | DEV closures after restore | Larger correct first choices | Larger closures after restore |
| --- | ---: | ---: | ---: | ---: |
| 3-pass | 35/48 | 8 | 32/48 | 6 |
| 1-pass | 26/48 | 12 | 25/48 | 12 |
| Direct | 39/48 | 0 | 36/48 | 1 |

The direct reader actually chooses more correct first branches. On development, 19 of its 28 failures end with a repair proposal at guard_ok, six with STOP after failure, and three with restore after failure. The direct reader likewise ends 14 of its 25 larger-operand failures with a repair proposal at guard_ok. The tied models learn stage eligibility more reliably under the short fitting schedule. This is an important controller-learning observation, not evidence that repeated reasoning improved root selection over the direct MLP. A budget/eligibility-aware scripted reference is required.

### Conditional continuation ceiling (posthoc diagnostic)

Hold each learned checkpoint's archived first action fixed. Under this exact grammar, probing immediately, restoring the opposite branch if an eligible fast repair fits the remaining budget, and choosing the proper fast/manual continuation gives a deterministic completion ceiling. Counting from the recorded first actions and verified unique branches produces the following **offline conditional schedule calculation**; these schedules were not independently rerun in the original experiment.

| Initial model | DEV predicted scripted completions / 48 | DEV predicted utility | Larger predicted completions / 48 | Larger predicted utility |
| --- | ---: | ---: | ---: | ---: |
| 3-pass | 44 | 0.59292 | 38 | 0.50292 |
| 1-pass | 38 | 0.48542 | 37 | 0.46896 |
| Direct | 45 | 0.61813 | 38 | 0.51458 |

The learned direct controller's root ranking has substantial headroom if coupled to a deterministic policy over the public stage and eligibility status. The one-pass controller already attains the conditional schedule calculation; the three-pass is one proof short in development and attains it on larger operands. No advantage over the best deterministic continuation is established. For a learned carry to earn a mechanism claim, a harder environment must present public histories with different remaining obligation choices that cannot be determined by this simple stage/budget table.

## Baseline correction

The historical _majority function maps nonnegative weighted evidence to r0. The claim features encode vote 0 as -1 and vote 1 as +1. Thus its sign is reversed for all nonzero scores. The original run is immutable and remains reported as such. From the archived registered claim packets, a correctly signed reference is calculated to choose 12/16 correct first branches on each panel (the original chose 4/16). The deterministic early-probe/recovery rule would reach 15/16 development worlds and 13/16 larger-operand worlds. These two latter counts are **offline schedule predictions**, not native rerun results. The corrected comparator will be executed natively on a fresh registry in [04D](OLR_METTA_04D_CARRY_REGISTRATION.md).

Because the original weaker controls do not settle the value of inter-decision memory, 04D pins the actual archived checkpoint rather than refitting. It compares natural carry, zero carry, and a norm-matched random perturbation while preserving model depth and root choice, and includes correctly signed majority and source-calibrated vote-likelihood references.

## Artifact identity

| Artifact | SHA-256 |
| --- | --- |
| Original registry manifest | 4e7f0b5c298f85dc82ec17be0b5b4e5e943059f9585bab01579607fdafdc3dc6 |
| Original fit manifest | 9e331941a41736ca719509a368f531b6b1c0f881efaf6c498414d0a910ebfed6 |
| Original model weights | 89050773cf3d63904f08149904201e177c3d2bdef4558513b4b414de4fbaae24 |
| Original eval manifest | 242f891571fa5420fbc71af8c5552b738edcca88268ff38294da7366eed30b75 |
| Native episodes | c45f9d9d1c42b047448bcb02b673d2bd328084d6d5efdd3593d804952ad97995 |
| Score rows | 1b32e8a7a95d7a4901bd2419b9148f62796388e2301d3320f9d7e432d34bb7a2 |

No architecture promotion. Retain all original negative controls and the baseline error; run 04D before any scaling decision.
