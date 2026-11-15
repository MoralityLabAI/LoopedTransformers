# RAM-08 | Post-freeze independent development replay (2026-11-15)

**Status:** read-only, retrospective diagnostic on the *original older MathCan TRAIN/DEV artifact*. This is **not** the preregistered twelve-theorem PEF evaluation. Zero new-PEF proof cases have been scored.

## Byte-identical input and frozen predictions

- Original Actions run: `43689874169`; archived artifact ID `14152063191`.
- Original artifact `ram08-frozen-audited-train-43689874169.zip`, ZIP SHA-256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`, byte length `20496574`.
- Original 20,930 independently replayed compiler observations: `data/transitions.jsonl` SHA-256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.
- Original old-source training: 18,658 observations on 57 theorem roots and 640 states.
- Original old-source development: 2,272 observations on five disjoint theorem roots and 74 states. Exactly 14 of these states have at least one *observed* candidate whose original compiler label closes all goals. Source reference proofs are not used as model inputs.
- Model architecture reproduced from frozen `model.py`, with identical BLAKE2b token IDs, four-tied-pass recurrent cell and original 20 action-family order. Predictions used CPU-only PyTorch 2.10.0. Original training used PyTorch 2.6.0+cpu. The checkpoint bytes, model tensor shapes, `state_dict` and selected optimizer steps were checked.

| Fixed policy | Selected checkpoint | Top-1 useful of 74 | Top-1 all goals closed of 74 |
|---|---|---:|---:|
| Native original candidate family order | none | 3 | 1 |
| Original TRAIN-only global-frequency policy | none | 30 | 5 |
| Original TRAIN-only shape-frequency policy (prior = 8) | none | **38** | 6 |
| Looped32 / seed29 / step300 | `4a88bc80…c09778f` | 30 | **7** |
| Looped64 / seed43 / step100 | `5cbd3279…737dac` | 19 | 4 |
| Flat64 / seed29 / step200 | `2aa101ed…5303d8` | 31 | 6 |

The original three model top-1 metrics were **reproduced exactly** from the immutable `training_report.json`; each was checked against the original receipt's `dev.states`, `dev.top1_usable`, and `dev.top1_closed`.

For non-neural controllers, the ranking uses only independently compiler-labeled original TRAIN action observations and the same dataset of development candidate actions. The native metric uses the registered fixed action-family order and source-observation ordering. Reported values concern **one observed action chosen per development state**; they are conditional on the historical collected candidate pool and are *not* complete theorem search or a fresh PEF proof outcome.

## Scientific interpretation

The strongest source-only symbolic controller, shape-frequency, ranked a useful first action in 38/74 original development states. Looped32 selected a closing action in 7/74, versus 6/74 for both shape-frequency and Flat64. Looped64's development top-1 closure was 4/74.

These old-source diagnostics cannot establish the registered RAM-08 primary hypothesis. That requires a recurrent policy to independently certify an *original new PEF theorem* missed by **each** of native, global-frequency and shape-frequency under the 30-second, 48-action, eight-transition matched search and then pass a second in-budget compiler replay plus third external source/axiom audit. The preregistered 60-second secondary suite also remains pending.

## Hosted-evaluation status

GitHub Action run `43691976669` was explicitly retried once more after the original failed first evaluation attempts. The second attempt, job `132752423238`, also failed with `runner_id=0`, empty `steps=[]`, and no assigned runner (`ubuntu-latest` label). GitHub returned no actionable job log. It is recorded as another infrastructure failure, with **zero new PEF cases scored**. Precise cause remains unknown.

The original preregistration `b8001e50`, training dataset, frozen selection, source commits `c84e2025` (TRAIN) / `83f61997` (TEST), original evaluation code and 120-case budgets are unchanged. See [evaluation amendment A](EVALUATION_AMENDMENT_A.md) and [frozen Windows runner](evaluate_frozen_windows.ps1). The original PR remains draft and unmerged.
