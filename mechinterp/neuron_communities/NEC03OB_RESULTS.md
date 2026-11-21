# NEC-03O B: four new five-input source-only worlds, fixed native reset and fresh SAE

**Executed 21 November 2026. Source-training-only prospective development pilot.** Registered, source-SHA-locked before any new-root training at commit `a0201bf53b9e9a29cbff17eabbcc68272a4feda1` on [phase B PR #262](https://github.com/MoralityLabAI/LoopedTransformers/pull/262). The four independent native+SAE root executions, independent source auditor checks, original-source preflight, and pure-stdlib aggregate all **PASSED** in [Actions run 44900628360](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44900628360). Summary artifact: `nec03ob-four-prospective-source-roots-audited`, ID **14705339651**. Full individually audited native and SAE checkpoint artifacts are `nec03ob-source-root-4` through `-7`.

These are the **first source-only training results** on the previously unopened five-input development worlds `dev5-004..007`. The first 900 native updates use fixed source AdamW lr0.003/batch128/clip1, with a byte-hashed checkpoint and a **fresh AdamW reset** for a second source-only 900 updates. This is the originally registered **900+reset900** condition, not the failed strict continuous-optimizer 1800 experiment. Both composed and additive arms are retained regardless of source competence.

## Independent prospective root results

Native source correctness per arm uses exactly **2,560 original source-training histories**, comprising 80 seven-bit contexts and all 32 focal counterfactual masks. The source-native criterion was frozen at **90%, i.e. at least 2,304/2,560**.

| Fresh five-input root | Composed native 900 | Composed native 900+reset900 | Additive native 900 | Additive 900+reset900 | Paired source-eligible |
|---|---:|---:|---:|---:|---|
| `dev5-004` | 2560 | 2560 | 2560 | 2560 | yes |
| `dev5-005` | 2240 | **2237** | 2560 | 2560 | **no** |
| `dev5-006` | 2560 | 2560 | 2556 | 2560 | yes |
| `dev5-007` | 2361 | **2560** | 2560 | 2560 | yes |

Native competence held at **7/8 arms** at both milestones, with **3/4 roots** possessing both competent teachers. The reset condition improved `dev5-007` composed from 2,361 to 2,560 and `dev5-006` additive from 2,556 to 2,560. It decreased `dev5-005` composed from 2,240 to **2,237** (87.38%), below the 90% native gate. This native miss is retained in the preregistered intent-to-treat denominator. Fixed training duration and other hyperparameters were never selected from any new-root outcome.

## Fresh checkpoint-specific source SAE and sham checks

Each of the eight new reset1800 native models yielded a qualifying original-policy native source graph candidate. Each received a **fresh 128-feature TopK16 SAE trained for 480 steps** on the original first 40 complete source groups and re-ranked on the disjoint last 40 groups. **All eight** independently verified source SAEs had exactly **16 positive active features on 100% of source-fit activation samples**, normalized fit MSE below 0.03, and passed the eight-source-context NEC03K exhaustive disjoint active-code gradient-angle control and physical dose criteria.

| Root | Arm | Fresh SAE normalized MSE | Source angular coverage | Median retained 32D output dose | Native/source eligible |
|---|---|---:|---:|---:|---|
| dev5-004 | composed | 0.015149 | 99.48% | 61.30% | yes |
| dev5-004 | additive | 0.002358 | 100.00% | 51.71% | yes |
| dev5-005 | composed | 0.006372 | 100.00% | 54.91% | **no (native)** |
| dev5-005 | additive | 0.005428 | 100.00% | 67.47% | yes |
| dev5-006 | composed | 0.006456 | 90.23% | 72.35% | yes |
| dev5-006 | additive | 0.014073 | 99.35% | 59.49% | yes |
| dev5-007 | composed | 0.011125 | 98.75% | 67.61% | yes |
| dev5-007 | additive | 0.006111 | 100.00% | 76.03% | yes |

Positive physically equalizable dose coverage was **100% for all eight arms**; minimum source angle coverage was **90.23%** and median retained source physical dose was **at least 51.71%**. All raw logged 32D decoder contributions, active TopK source/control selection, 40-group rank digests and source-only native predictions passed the independent checks in the corresponding root artifact.

**Source-control feasibility transferred in all eight new model arms**, compared with 7/8 on the original four already-opened roots in [NEC-03O A](NEC03O_RESULTS.md). The previous dev5-003 additive graph's visit-8 hub saturation and its source-instrument failure remain recorded. Because the development roots have distinct teacher functions and focal supports, this is descriptive source feasibility across **eight total** source roots; phase B contains four prospective roots and paired arms, not eight independent new roots.

## Scientific boundary

No source-training inputs are held-out development evaluations. **All 24 five-input development-evaluation context groups and 24 reserved context groups per root, all 32 five-input embargoed world roots, and the 64 original four-input confirmation roots remain unopened.** None of the new source-only outcomes is a causal treatment-effect estimate. The original strict continuous-AdamW failure and signed NEC-03H/03K causal effects remain unchanged.

**FSI-01 state: `REDESIGN_OR_PILOT_ONLY`.** A later intervention study needs a separate, prospectively frozen independent-root causal effect protocol; the registered 95/95 tolerance calibration also needs at least 59 independently eligible sham roots. The incomplete native competence of dev5-005 composed should remain in the same fixed denominator if that root is included in future analyses.
