# NRF-09: matched optimizer-step depth/leakage replication

Date 2026-11-15. The six-seed fixed protocol in `configs/nrf09_matched_optimizer_steps_v1.json` was committed **before any NRF-09 model training**. Results are exploratory because `lambda=1` was chosen after NRF-04.

## Train and evaluation

The six new paired seeds 131, 137, 139, 149, 151, 157 were independently initialized under the canonical 128-dimensional tied ToyLoop. R16 and R32 train on identical minibatch sequences for 16 and 32 optimizer updates. At 16 updates R16 sees 2048 state-visits and R32 sees 4096; at 32 updates R16 sees 4096 and R32 sees 8192. All states are measured at h_(R/2). Decoder training uses fold 1 (160 Gaussian inputs), held-out MSE uses new fold 7 (96), and local shadow Jacobian leakage uses new fold 8 (12). Every checkpoint has a task-supervised 8D decoder, eight random 8D controls, a shifted-label null and a coordinate observation.

The objective is `J=MSE+(mean shadow leakage)^2`, with random comparator the median of eight **complete** random-arm scores.

| Trained depth | Updates | Supervised wins | Mean supervised J | Mean random median J | Signed margin |
|---|---:|---:|---:|---:|---:|
| 16 | 16 | 6/6 | 0.461193 | 0.592610 | -0.131417 |
| 16 | 32 | 6/6 | 0.401956 | 0.561631 | -0.159675 |
| 32 | 16 | 0/6 | 1.282125 | 1.046324 | +0.235801 |
| 32 | 32 | 1/6 | 0.969615 | 0.913607 | +0.056009 |

Both precommitted >=5/6 directional gates pass at 32 updates. Seed 157, depth 32, reverses to a supervised win at 32 updates with a narrow margin -0.003091.

## Equal-exposure and equal-update decomposition

For m(R,u)=J_sup(R,u)-median(J_random(R,u)), exactly

```
m(32,16)-m(16,32) =
 [m(32,32)-m(16,32)] + [m(32,16)-m(32,32)].
```

The mean equal-4096-visit gap is 0.395476. The equal-32-update gap is 0.215683 and the observed depth-32 additional-training movement is 0.179793, or 45.46% along this accounting path. All six equal-update paired depth margins are positive. A one-sided 6/6 sign test is 1/64 under the independent exchangeable-sign null; because lambda and the synthetic task family were exposed previously this is exploratory. The 5/6 depth-32 random-win count has one-sided binomial p=7/64 against 50%.

At R32 and 32 updates the five still-random-winning models need 3.7%-7.0% lower supervised leakage to reverse the decision, if their task errors and random-arm objectives stay unchanged. Diagnostic lambda crossovers range from 0.556 to 1.024. One R16 model has **no finite nonnegative crossover** in the recorded arm-line envelope.

## Evidence and boundary

- `nrf09_matched_step_receipts.csv` contains all 24 checkpoint hashes and primary task/leakage/objective values, plus exact-decimal-derived first crossover penalties.
- `nrf09_matched_step_summary.json` contains groups and paired attribution terms.
- Full 11-arm JSON receipts, model/optimizer checkpoint binaries, and standalone CPU replay code are packaged in the conversation archive. Seed 157 R32 regenerated to an identical .pt SHA in the same local runtime.
- The private repository's GitHub Actions may fail before runner startup. Local independent sanity checks recomputed each of the 24 arm scores and verified all 24 checkpoint hashes.
- Applied pure-math companion: patrickdugan/MathCan PR #75, eight-arm median-margin robustness and the matched-update algebraic identity.

The inference is restricted to a synthetic signed-permutation family and previously selected lambda. Model performance and formal reasoning ability are not certified by these metrics.
