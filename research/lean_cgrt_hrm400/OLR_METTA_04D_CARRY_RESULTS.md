# OLR-METTA-04D: frozen inter-decision carry audit

**Date:** 2026-11-24  
**Status:** completed native developmental diagnostic; no mechanism promotion.  
**Registration:** [OLR_METTA_04D_CARRY_REGISTRATION.md](OLR_METTA_04D_CARRY_REGISTRATION.md), commit ec9a4a2.  
**Runner commit:** 8729d8f45314cae3bae926eed5664188224994c9.  
**Native workflow:** [45280288508](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45280288508), successful.  
**Archive:** [artifact 14870135161](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45280288508/artifacts/14870135161).

## Controlled question

Does the latent vector carried between actual Lean verifier decisions contribute to the native success of the three-visit OLR-METTA-04 checkpoint? The original 04 pilot mixes parameter count, architecture and active compute across trained comparators. The frozen intervention is narrower: the three-visit checkpoint, public claim features, budget and initial root choice are held fixed.

The archived 04 source weights were downloaded by the CI job from the exact original [artifact 14837117274](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45212958918/artifacts/14837117274). Original model, source-bank and fit manifest hashes were verified. There were zero optimizer updates. The source-verified 04 bank remains the only training source; a simple conditional likelihood reference is calibrated on its existing TRAIN receipts. The new registry contains 12 independent pairs in each operand band (24 worlds each), excluding all 40 original arithmetic lineages.

The natural controller receives its ordinary carried vector. Zero replaces that vector with zero before each later decision. Sham adds a deterministic random sign direction with perturbation L2 equal to the local carry norm. All three use identical three-visit inference per decision. The sham retains the original carry plus a perturbation and does not match the resulting carry vector norm to zero. Doses are matched locally, conditional on each policy's evolving trajectory; total episode inference passes and downstream states may differ. This is not a matched-logit-margin intervention or a sealed confirmation.

The corrected majority maps negative votes to r0 and positive votes to r1. The source-likelihood reference estimates per-claim correctness conditional on its public strength pattern, using Laplace smoothing on native TRAIN winners; it uses no query labels. Both references apply the same deterministic early-probe, restore-if-affordable, and eligible repair/manual continuation. The privileged full-information oracle is a ceiling, not a same-information comparator.

## Native outcomes

All **912 native policy episodes were independently replayed**, each under the six-tactic ceiling. Forty-eight distinct successful theorem/branch/tail scripts compiled independently in Lean 4.19 with no axioms. The artifact was independently downloaded and all manifest-listed file hashes verified. The same 24 worlds in each panel are evaluated by all three model seeds; reference policies run once per world.

| Policy | DEV proofs | DEV mean utility | DEV calls | Larger operands proofs | Larger mean utility | Larger calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-pass natural | 55/72 | 0.47806 | 294 | 63/72 | 0.56875 | 315 |
| 3-pass zero carry | 48/72 | 0.40125 | 273 | 53/72 | 0.45903 | 285 |
| 3-pass norm-matched sham | 54/72 | 0.46611 | 292 | 61/72 | 0.54389 | 312 |
| 1-pass original | 56/72 | 0.47444 | 312 | 57/72 | 0.48542 | 315 |
| Direct original | 34/72 | 0.25931 | 219 | 33/72 | 0.24833 | 216 |
| Fixed r0 + scripted tail | 18/24 | 0.45250 | 102 | 18/24 | 0.45250 | 102 |
| Corrected majority + scripted tail | 19/24 | 0.50292 | 99 | 21/24 | 0.57458 | 103 |
| Source likelihood + scripted tail | 22/24 | **0.61333** | 104 | 21/24 | **0.58042** | 101 |
| Exact full-information | 24/24 | 0.68500 | 108 | 24/24 | 0.68500 | 108 |

Utility is completion minus 0.07 per attempted native tactic. Natural, zero and sham model passes in development: 903, 882, 900; larger-operand: 918, 888, 921. The one-pass reader used 326 and 327 visits. These are unequal total inference compute budgets. The larger operands are hidden from the model, so this panel is not an intentional change to the model-visible evidence distribution.

### Seed-specific causal contrast

| Panel | Seed | Natural / 24 | Zero / 24 | Sham / 24 |
| --- | ---: | ---: | ---: | ---: |
| DEV | 17 | 19 | 19 | 18 |
| DEV | 29 | 17 | 17 | 17 |
| DEV | 43 | 19 | 12 | 19 |
| Larger operands | 17 | 22 | 22 | 22 |
| Larger operands | 29 | 19 | 19 | 18 |
| Larger operands | 43 | 22 | 12 | 21 |

All net natural-minus-zero completion differences come from seed 43. Natural versus sham gains only one development completion and two larger-operand completions across all seeds. The results show that erasing carry can disrupt an individual checkpoint, while a similarly sized random perturbation largely preserves its performance. They do not show that a specific carried direction is necessary across seeds.

### Descriptive paired-root contrasts

The bootstrap averages all three seeds and both worlds inside each of 12 independent pair clusters, then resamples clusters (10,000 replicates). These intervals are exploratory; no multiplicity-corrected or prospective FSI-01 decision is implied.

| Contrast | DEV completion difference, pp (95% interval) | Larger operand difference, pp (95% interval) |
| --- | ---: | ---: |
| Natural − zero | +9.72 (+2.78,+18.06) | +13.89 (+5.56,+22.22) |
| Natural − sham | +1.39 (0,+4.17) | +2.78 (0,+6.94) |
| Natural − one-pass | −1.39 (−12.50,+8.33) | +8.33 (+1.39,+15.28) |

Utility differences natural minus sham are +0.01194 (0,+0.03583) and +0.02486 (0,+0.06167). The natural-minus-zero utility intervals are +0.07681 (+0.02194,+0.13167) and +0.10972 (+0.04389,+0.17556). The pre-registered interpretation requires direction-specific gains and strong same-information controls. Those conditions are not met.

## Conditional scripted-tail diagnostic

A posthoc calculation holds each frozen first choice fixed and follows the deterministic early-probe/restore/repair policy. This uses the known native grammar and the registered unique correct branch. It does not count as additional independently replayed policy episodes.

| Frozen first-choice model | DEV correct opening | DEV predicted full completion | Larger correct opening | Larger predicted full completion |
| --- | ---: | ---: | ---: | ---: |
| 3-pass | 46/72 | 56/72 | 54/72 | 63/72 |
| 1-pass | 38/72 | 56/72 | 39/72 | 57/72 |
| Direct | 59/72 | 62/72 | 65/72 | 68/72 |

The direct model has better root ranking but makes avoidable downstream control errors. Its conditional scripted-tail utility calculation is 0.56944 and 0.63528. A native direct-ranker-plus-scripted-tail control should therefore be mandatory for any subsequent learning comparison.

## Research decision

Keep the learned architecture at research-only status. The full-information exact reference solves the grammar; the source-calibrated likelihood and deterministic controller also exceed the natural recurrent model's utility on both new panels. The natural-versus-sham completion intervals touch zero and the zero effect is entirely seed-specific.

The current public stage observation carries enough information to implement the optimal post-opening schedule as a small deterministic table. It does not force the network to remember a hidden earlier obligation. A meaningful next test needs **history aliasing**: two native-verified histories with the same current allowed observation and different correct pending-obligation actions, while a transparent explicit-history or sufficient-statistic control remains available. Add live native replay of the direct root reader with the scripted tail. Register fresh lineages, fixed inference budgets, matched intervention doses and the FSI-01 power/sham gate before training or scoring another model.

## Provenance

- Original bank manifest: 4e7f0b5c298f85dc82ec17be0b5b4e5e943059f9585bab01579607fdafdc3dc6
- Original weights: 89050773cf3d63904f08149904201e177c3d2bdef4558513b4b414de4fbaae24
- Fresh registry manifest: e1ea077a5b0c888fe09383218f38f28203c138636b7e4e2188c76afcd5db9ca9
- Fresh evaluation manifest: cc390b40b2dd7229b343b44a02e43498b0cf141669c59ee26464289c0ac1c054
- Frozen choices: d8526e6254cbf39e24d11ff23cd175094d76f037128d9d03ed46ad0ab2cf5825
- Native episodes: 681d93bbea9caae53011fa9ff017aa9247f27803565c2ccd55e0cebb9d754f57
- Score rows: 135918f17873610ca0bd0537af77e60a8cf1081485a878ca2afff3b402a79c3a
- Independent Lean source: 3e9b9b74363de74748f5911aaebb2d543af0161e7d4596a8ec6812389d2c5048
- Uploaded archive ZIP: fad7899fd3ef5fc35c53e3b12cc356b6468749e4c7349eadee6663006c275f30

Reproduce with the archived original 04 artifact unpacked under runs/olr04-archive and the pinned Lean/CPU/Hyperon runtime:

    python -m ppt_gym.olr_metta04_carry freeze --bank runs/olr04-archive/olr04-bank --fit runs/olr04-archive/olr04-fit --out runs/olr04d-bank
    python -m ppt_gym.olr_metta04_carry score --bank runs/olr04-archive/olr04-bank --fit runs/olr04-archive/olr04-fit --registry runs/olr04d-bank --out runs/olr04d-eval

The [workflow](../../.github/workflows/olr-metta-04d.yml) retrieves the exact source artifact by run ID and checks immutable manifest and model hashes. A source artifact that expires or changes blocks the run rather than substituting new weights.
