# AF-08: qualified-checkpoint noncollapse training, final scored report

**Registered scientific verdict: FAIL.** Both qualified original AF-06 warm-up seeds (17,29) and all six 160-update continuations satisfy the 90% training/development goal and exact joint-Q competence gate. Both eligible noncollapse models fail preregistered held-depth4 and held-depth5 locality improvement and absolute-control requirements.

**Engineering verdict: PASS on the repaired source.** The original numerical execution failed, was retained, and then received one source-limited Euclidean gradient repair before a new full run. Both independent evaluation jobs and the aggregate passed. The independent local aggregate is **byte-for-byte identical** to hosted `results.json`, SHA-256 `d1fc3cdd2b1f314850603f28c764a4a5266dd580cf269ea1d03a576c2831915a`.

## Source and preregistration

- [Original frozen registration](../../experiments/activation_funnel_af08/PROTOCOL.json), SHA-256 `6aad992f2172ff6529b341ac071bb3308e3c4e40c5149e12d20c49dca7778c62`.
- Initial experiment commit `bbfd96b70208f1d2ab032882ee3e8a3664ee9338`; **numerically repaired scientific source** [`44bc1dc76ba7bc087f5c3da8f8ce5b4e668f0593`](https://github.com/MoralityLabAI/LoopedTransformers/commit/44bc1dc76ba7bc087f5c3da8f8ce5b4e668f0593).
- Original qualified warmups: AF-06 commit `36b83eb83bc116dd4e938213edeba3057ffa9016`; training artifact IDs `14116102941` (seed17) and `14115887129` (seed29), verified exact ZIP hashes `c8cf6c6c8264cf313e755c4d3008c26ae72cdd6ef20583fadc112bf02169ffdd` and `660b71acd5dc95bbfd200e55f206f609dcf6cc141ccdbb131d74bb7e5f0d4ebf`.
- [Original failed run 43680485938](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680485938). [Repaired six-job CI run 43681114942](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43681114942): **all passed**.
- [Separate engineering incident record](AF08_ENGINEERING.md); [exact carried-code classification-margin transport theorem](AF08_Q_MARGIN_THEOREM.md).

The original hash-pinned TypedLoop and two-phase Horn proof graph are unchanged. Each seed's `status`, `locality`, and `noncollapse` arms inherit the same validated 100%-competent checkpoint, use the same train/development cases and matched recurrent forward calls for 160 continuation steps. Both causal arms receive identical addressed-donor exposure. The noncollapse arm adds only the registered class-prototype and changed-status code-margin constraints.

## Frozen scientific comparison

All six trained continuations retain **100% train/development goal and exact Q0/Q1 status accuracy**, with finite checkpoint tensors. The table reports actual last-step protected/untouched Q accuracy on original held-out depth4/5 graphs.

| Seed | Depth | Causal locality | Noncollapse | Noncollapse minus locality | Informative goal flip: locality to noncollapse |
|---:|---:|---:|---:|---:|---|
| 17 | 4 | 0.48046875 | 0.50000000 | +0.01953125 | 0.3671875 to 0.5000000 |
| 17 | 5 | 0.72656250 | 0.68750000 | -0.03906250 | 0.5937500 to 0.5390625 |
| 29 | 4 | 0.75000000 | 0.75000000 | 0.00000000 | 0.5937500 to 0.5937500 |
| 29 | 5 | 0.75000000 | 0.75000000 | 0.00000000 | 0.3125000 to 0.3125000 |

The required per-seed causal-locality improvement is >=0.05 at **both** depths, plus absolute protected-Q >=0.90 and irrelevant-patch goal-retention >=0.90, clean-goal retention and false-positive gates. Every eligible seed fails the required improvement and absolute-locality gates.

Additional negative controls: noncollapse irrelevant-patch goal retention is 0.5000 (s17 depth4), 0.6875 (s17 depth5), 0.7500 (s29 depth4) and 0.78125 (s29 depth5). On seed17, noncollapse clean-goal accuracy drops from locality 0.5625 to 0.5000 at depth4 and 0.7500 to 0.6875 at depth5, outside the allowed 0.03 loss. The result provides no positive causal-training advantage.

## Numerical incident and evidence integrity

The first execution used `sqrt(sum((a-b)^2))` for a Euclidean margin loss; zero-distance samples caused undefined gradients and NaN weights. Both training artifacts are retained, and the original evaluation's named-array/Torch consistency check rejected them. The repair substitutes `torch.linalg.vector_norm` with its defined zero subgradient and adds explicit nonfinite guards and regression tests. The scientific protocol, source checkpoint bytes, loss weights, margins and registered gates were preserved. Ten tests pass in both Python modes; the repaired full CI passed its source, Lean, training, independent evaluation and aggregate jobs.

The repaired matrix contains **6 models, 2,976 canonical clean held-out predictions, 5,952 donor counterfactuals and 9,024 downstream recurrence-stage records**. Every independent scorer reconstructed the Horn targets and checked the saved Torch/NumPy model identity and original carried-code patch replays. Lean 4.19.0's 4,176 generated closed Boolean-Horn certificates establish the synthetic oracle, not neural theorem proving.

**Frozen protocol case-count erratum:** The original PROTOCOL.json describes `held_layout=112, held_depth5=240`. The pre-existing AF-05 corpus receipt and unchanged AF-08 evaluator actually enumerate `held_layout=224, held_depth5=128`. Remaining held-out families are `held_composition=16, held_depth4=128`, for 496 distinct held-out inputs. The exact input roster and its SHA hashes were fixed before training. This is a documentation discrepancy; retain the original registration and use the actual committed manifest and scorer's enumerated case IDs for outcome claims.

## Mechanism interpretation

The AF-06 qualified Q codes already had true/false prototype distances around 1.9–2.8 and changed-status donor code distances above 1.88 on observed train/development states, exceeding AF-08 margins of 0.45 and 0.35. The new terms therefore had zero penalty on those sampled initial states. This helps explain why the margin constraints generated little pressure to reshape the deep causal transport. It does **not** prove they could never activate elsewhere.

The [exact code-margin certificate](AF08_Q_MARGIN_THEOREM.md) clarifies the failure: correct clean signed Q-logit margin must exceed actual patched-state code drift weighted by the decoder/readout sensitivity. Low code drift by itself does not establish semantic correctness. A **future separately registered** experiment could supervise the actual true-class signed margin under the patched downstream state and test stricter graph-depth transfer. This proposed change is outside AF-08's immutable scientific protocol.

## Original GitHub Actions artifact inventory

The repaired [CI run 43681114942](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43681114942) retains six SHA-256 pinned artifacts:

| Name | Artifact ID | ZIP SHA-256 |
|---|---:|---|
| af08-preflight-43681114942 | 14148930143 | `4e09daf750f0eddc1fd150e63851f0761107f2247af0cd47f82f399b87e96d3a` |
| af08-train-s17-43681114942 | 14148079340 | `4375ae03ad6c3428cf178aabdf746bb54856cf367e64d145204f9ce0d123ae0b` |
| af08-train-s29-43681114942 | 14148895161 | `cf21001f8091d59e1f56ee5667264ef40f3939f18436fd3dc9c3f8e27d974c1f` |
| af08-eval-s17-43681114942 | 14148408381 | `fc81d069a9106e3e9f9c21361c2d11413c6159b0d233e38a975765e5d92356b8` |
| af08-eval-s29-43681114942 | 14148089264 | `2966a53c2f99dd0d9389b04bb025a5927b0a52a0aa941268fc9612dd6b46ebe9` |
| af08-results-43681114942 | 14147874451 | `fccc98e53b94928fa3e188a1879b0170797ef402b06b4c46cc0e8ede22bfac26` |

The original failure artifacts remain accessible via [run 43680485938](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680485938). Keep [AF-08 draft PR #52](https://github.com/MoralityLabAI/LoopedTransformers/pull/52) **unmerged** and stacked on AF-07.
