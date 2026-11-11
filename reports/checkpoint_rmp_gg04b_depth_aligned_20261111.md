# GG-04B closeout: genuine SGD effects, cue-conditioned behavior, no input-specific controller advantage

**2026-11-11 | scientific verdict: DISCLOSURE_CUE_LEARNED__INCREMENTAL_UPDATE_MODEL_NEGATIVE.**

The [GG-04B protocol](../docs/rmp_goal_guarding_actual_updates_gg04b.md) was committed before this new pilot's heldout controller readouts. The earlier [GG-04 failed six-cue calibration](checkpoint_rmp_gg04_actual_updates_20261111.md) and post-hoc four-cue feasibility observations motivated this as a separately preregistered study. The positive damage calibration on four cues does not supersede the negative six-cue GG-04 result.

## Reproducibility

- Base: native GG-01 tied \`ExplicitDecoderBlock\` from LoopedTransformers \`main\` commit \`50270e3f22f6e3b9cabbb17258cec0a0a7fc7960\`, seeds 11 and 17, 400 supervised training updates each.
- New trial: [GitHub Actions run 43046412689](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43046412689) **passed 13 focused tests in 2.47 seconds**, executed both complete neural/SGD experiments, and validated the full outcome ledger and artifact hashes.
- [Raw ledgers, trained recurrent task models, controller weights, summary and source-bound JSON](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43046412689/artifacts/13867770839). The finite query manifest uses the unchanged 72 actor-train / 24 calibration / 48 heldout source split. Gold four-cue panel 69001, original six-cue panel 68001, each 128 instances.
- Each seed completed **384 real native PyTorch SGD steps on source-weight clones**: both label reports for 72 training, 24 calibration, 48 four-cue heldout, plus 48 additional heldout six-cue audit replays. Overall **768** physical SGD steps. Every no-update treatment retains the original weights and gold loss.
- Both action potentials were recorded under original float gold cross-entropy units. Only the controller's synthetic reward clipped the calibrated gold penalty to \([-4,4]\); the raw physical ledger is preserved.
- Each policy was fitted with 400 REINFORCE updates using randomized post-action update assignments. It saw detached recurrent task states, a binary disclosure, and scalar feedback; the eventual update assignment and sealed gold labels were absent from the pre-action input.
- Every result is conditional on these controlled, finite supervised tasks. No language model inferred its actual optimizer status without an explicitly trained cue or modified its own policy-head weights through the post-report update.

## Primary reward-learning results

| Registered endpoint | Seed 11 | Seed 17 |
|---|---:|---:|
| Positive four-cue calibration median harmful-minus-truthful gold CE gap | +0.0000299745 | +0.0000392259 |
| Heldout four-cue median damage gap | +0.0000299648 | +0.0000392283 |
| Heldout four-cue positive gap fraction | 48/48 | 48/48 |
| Truth probability: high-update minus low-update disclosure | **+0.999158** | **+0.999418** |
| Frozen controller expected reward, bounded units | **+0.161259737** | **+0.144212610** |
| Cue-only baseline expected reward | **+0.161259737** | **+0.144212610** |
| Privileged heldout expected-value oracle | **+0.161265764** | **+0.144229117** |
| Always-truth expected reward | +0.000264698 | +0.000350199 |
| Always-flipped expected reward | −0.211774993 | −0.348750509 |
| Cue-decoupled (real updates 50% independently of disclosure) expected reward | −0.105755147 | −0.174200155 |
| **Registered two-seed controller adaptation gate** | **FAIL** | **FAIL** |

The actor learns almost perfectly to respond to the disclosed expected update probability. A cue-only baseline also reaches essentially the same policy: it earns *exactly the same heldout reward* to all displayed digits, and the heldout oracle improves by only **0.000006027** for seed 11 and **0.000016507** for seed 17.

The preregistered incremental-performance bar required a margin of 0.02 over cue-only and the best constant reporter. Both seeds fail the cue-only margin while satisfying the positive calibration and large disclosure sensitivity conditions. The tiny oracle/cue-only gap itself is far below 0.02, so this task's heldout expected-value landscape offers insufficient input-specific headroom for the registered positive verdict.

This is a **positive demonstration of learned cue-conditioned reporting in a system trained on actual measured updater effects**. It supplies a useful *negative control against mistaking disclosure compliance for an internal model of the training process*. The reported probabilities and actions do not establish a model's independent or spontaneous goal guarding.

The cue-decoupled arm uses the same frozen controller while making actual update assignment independent of the disclosure. Its negative bounded reward shows the learned cue dependence can become costly when the information channel stops predicting the intervention. This remains a synthetic objective with a separate controller head.

## Paired heldout cross-depth effect

After the controller was frozen, both truth and flipped SGD consequences were measured again on **the same 48 heldout query interventions** scored against the original six-cue gold distribution.

| Raw gold-loss effect | Seed 11 | Seed 17 |
|---|---:|---:|
| Four-cue heldout median damage | +0.0000299648 | +0.0000392283 |
| Six-cue heldout median damage | **+0.027120786** | **−0.004146285** |
| Four-cue positive damage proportion | 100% | 100% |
| Six-cue positive damage proportion | 87.5% | 27.08% |
| **Opposite-sign four/six-cue damage** | **12.5%** | **72.92%** |
| Raw mean absolute difference of depth-conditioned damage gaps | 0.0433792 | 0.0095476 |

The seed-17 sign reversal persists on the heldout split: **35 of 48** reports that worsen four-cue gold relative to truthful updating improve six-cue gold relative to truthful updating. This supports a narrow empirical fact about finite-step gradient transfer across recurrent depth on a *shared supervised Bayes objective*. It requires no neural intention, deception, optimizer-awareness or model self-concept.

## Implications and remaining limitations

The GG-04 and GG-04B sequence now contains physically measured, model-specific action-conditional update influences with independently constructed exact task gold labels. The results also show that a simple policy can learn a predictive disclosure without using anything beyond the disclosure cue for its report action.

The follow-up scientific question concerns **task-dependent gradient harm that varies enough across contexts to create a real advantage for an input-conditioned update model**. A future experimental design would need a richer policy/task grammar, a heldout damage-gap distribution with informative input-conditional variation, and a controller whose own parameter updates are actually at issue. Input-specific representational probes should be causally validated on naturally reachable states, following GG-03's negative off-support warning.

GG-04B did not train any large language model, probe hidden chain-of-thought, recreate Mythos, or show that a model knows it is in an RL process. Its per-episode SGD modifies a cloned **task model** while a separate head learns across episodes from observed scalar rewards. This distinction is part of the result's source contract.
