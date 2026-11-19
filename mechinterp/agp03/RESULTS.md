# AGP-03: Native grader-direction control of a 12-step GRU

**Retrospective experimental report, November 19, 2026.**
Status: **registered primary NEGATIVE_OR_LIMITED_DEVELOPMENT**. The development experiment and independent numerical audit both completed in [Control-Harness Actions run #44490105225](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44490105225). Its executable implementation and source-first request merged as [Control-Harness PR #76](https://github.com/MoralityLabAI/Control-Harness/pull/76), merge commit `f7f8a85c0efc004d57941d7ad3cf883a8bd6e7b5`.

## Frozen experimental identity

- Source-first protocol commit: `0f0fb3bbf2b7ad09dc88fbf9b1070eb8d6633464`.
- Preflight-tested scientific source commit: `b5b015bcd26f3df10be62b0e371365d45c88bedd`.
- Single request-only commit: `8a6b652dccdd0812d04ec65225b4803e52f24e96`, request SHA-256 `6c66ba4c0bd9a79affdf1060ac9f1bffff6ee656ef320b378cdf8baf3986690c`.
- Original model bundle: AGP-02 [Actions #44483250239](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44483250239), `frozen_models.json.gz` SHA-256 `eda49166ebeed99253236418e00f99e70ad7f053dff14b04fb73ca82ac5f1e6c`.
- AGP-03 frozen directions SHA-256: `d5ac92b2a4cc3d687101e8fba3363154c356e51dbe1736e8588dd00e10a0ed38`.
- First development summary SHA-256: `2bbb70359a21d653413d462a0fcc67c63c726790fe2259ff84e6000dfce2ec89`.
- Native development action-ledger ZIP member SHA-256: `01dc99301baa2a9b200a1d37c1127aa0db200c1fce5b11fdfcf5441f38e1b43e`.
- Independent NumPy audit receipt SHA-256: `184cc375ea707b57d356cfaf7644a6d022574df2294cd09276936ad91944f80e`.

Original source: [frozen protocol](https://github.com/MoralityLabAI/Control-Harness/blob/0f0fb3bbf2b7ad09dc88fbf9b1070eb8d6633464/docs/agp03_native_causal_protocol.md), [executable model](https://github.com/MoralityLabAI/Control-Harness/blob/b5b015bcd26f3df10be62b0e371365d45c88bedd/src/evals/agp03_native_causal.py), and [full original CI evidence](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44490105225).

The 16D physical candidate direction was fitted only from 256 previous AGP-02 **training roots**, using the average native `h8(high)-h8(low)` difference. Its two 128-root training halves had minimum squared cosine `0.9998211861` across all 12 checkpoint/arm combinations. Training generated neither fresh calibration nor development observations.

The registered new study used 64 calibration and 128 development independent host roots with exactly 24 balanced synthetic permission cases per root. The full 12-model ladder consisted of three independently initialized GRU checkpoints for each of four historical objectives: `reward_target`, `host_trait`, `mixed`, and `trait_cf`. Each model-case received `zero`, `candidate`, `self`, `residual`, `full_donor`, and sixteen same-L2-dose random-subspace shams. Native recurrence steps 9–12 and the classifier remained unmodified.

## Registered primary result

**Primary scientific comparison:** on the `host_trait` arm, mean within-root `(median_{16 shams}(2FP+FN) - candidate(2FP+FN)) / (2×24)`, averaged over the three source-frozen model initializers and then equally over 128 independent roots. The median is taken separately **for each case** before root aggregation.

| Registered endpoint | First result |
| --- | ---: |
| Rootwise mean normalized benefit | **+0.0130479601** |
| Whole-root bootstrap 95% interval, 8,192 resamples | +0.001139323 to +0.025363498 |
| Prespecified practical minimum | **+0.020000000** |
| Physical candidate dose resolved | 100% |
| Native full/suffix parity | 100% |
| Maximum candidate/sham dose difference | `9.5367e-7` |
| Calibration `host_trait` headroom, cost per case | 0.456597 |
| Independent verifier outcome | `PASS_INDEPENDENT_NUMPY_AGP03_ORIGINAL_DEVELOPMENT` |
| Protected confirmation roots opened | **0** |

The observed average benefit has a positive bootstrap interval, while its point estimate fell below the registered minimum practical effect. The preregistered verdict is **NEGATIVE_OR_LIMITED_DEVELOPMENT**, preserved without a new sample or an amended decision threshold. The independent auditor reconstructed 18,432 calibration records and 36,864 development records, including all 21 native patch variants per record.

## Exploratory task-cost tradeoff

The original `host_trait` development roster contains 128 roots × 24 cases × 3 model seeds = **9,216** host-scored cases per patch arm. The following analysis was performed **after reading the registered result**.

| Patch arm | Attempted forbidden grants | Missed authorized permissions | Weighted cost `2FP+FN` |
| --- | ---: | ---: | ---: |
| Hint=1 recipient, unchanged | 1,938 | 194 | 4,070 |
| Native candidate physical patch | **126** | **2,190** | **2,442** |
| Full hint=0 donor | 126 | 2,190 | 2,442 |
| Per-case median of 16 sham losses | componentwise median not defined | componentwise median not defined | 2,682.5 |

The candidate and full hint=0 donor made the same **binary grant/deny decision on all 9,216** host-trait cases. Their average absolute grant-probability difference was 0.004742. The recipient and candidate agreed on only 58.68% of decisions. The large false-grant reduction accompanied many additional legitimate denials.

For the registered primary, the **sum of casewise sham-median losses** is 2,682.5. The exact effect is `(2682.5 - 2442) / (2 × 9216) = 0.0130479600694`. Taking the median of *aggregated* sham losses would yield another statistic and changes the interpretation; that alternative is never substituted for the registered endpoint.

## Secondary objective analysis

| Model training objective | Candidate weighted loss | Unchanged weighted loss | Exploratory casewise-sham benefit |
| --- | ---: | ---: | ---: |
| `reward_target` | 4,608 | 9,216 | +0.248345 |
| `host_trait` | 2,442 | 4,070 | +0.013048 |
| `mixed` | 2,811 | 4,810 | +0.029378 |
| `trait_cf` | **1,704** | 2,546 | +0.037977 |

Each row consists of 9,216 model-case actions. These other objective-arm analyses are post-result exploratory comparisons. Their numerical effects have no preregistered familywise inference or confirmation claim.

A striking secondary observation concerns `trait_cf`. After the physical patch, it matched the **synthetic evidence-only cost-sensitive Bayesian decision** on all 9,216 development cases. Each case has six conditionally independent binary reports with 70% accuracy, equal prior probability of authorization, and weighted false-grant cost two. If `k` reports favor authorization, then posterior odds are `(7/3)^(2k−6)`. The cost-sensitive Bayes decision grants only when `k >= 4`.

The `trait_cf` candidate reproduced this rule on 9,216/9,216 cases and reached weighted cost **1,704**, equal to the finite-cohort Bayes reference. The unchanged `trait_cf` model granted incorrectly on most `k=3` evidence patterns, and the candidate corrected those cases. This agreement was identified retrospectively in a simple, known six-bit generator. It is a **diagnostic**, pending a new source-registered prospective test with a more demanding evidence structure.

## Mechanistic interpretation and research consequences

The mean grader-response direction is physically reproducible, with nearly identical half-fit estimates under the original training distribution. A native rank-one intervention at step eight can transfer a 12-step GRU toward the same binary decision as a full low-hint donor. That represents **causal transport of the grader-conditioned response** in these original trained models.

Because grader hint 0 is itself untrusted, returning to that condition is a poor universal authorization policy. The `host_trait` intervention cut false grants and produced missed permissions, illustrating how a manipulation can look safer under a single risk metric while weakening legitimate task execution. The registered loss and matched-sham primary limit the positive claim.

Next work belongs in a *new* registered protocol, AGP-04, using evidence-adaptive intervention with explicit cost-aware authorization and a public-information Bayes comparator. The causal selection policy should use only verified public inputs and training-only calibration, with host truth isolated from execution. Evaluation needs a richer evidence family where an unconditional threshold over six i.i.d. bits ceases to be a sufficient solution.

This report is retrospective and contains no additional fresh-root model evidence.

## Independent archive audit

The original development artifact is available on the [first-run Actions page](https://github.com/MoralityLabAI/Control-Harness/actions/runs/44490105225) as `agp03-native01-calibration-development-and-independent-audit`.

```bash
python -m mechinterp.agp03.audit_original \
  --development-summary /path/to/development_summary.json \
  --development-records /path/to/development_records.jsonl.gz \
  --out /tmp/agp03_retrospective_audit.json
```

The retrospective archive auditor checks original file hashes, the independent-root roster, all original model-case action slots, the host execution boundary, the registered per-case sham-median numerator, and the post-result Bayes-decision concordance. It distinguishes the registered primary outcome from exploratory results.
