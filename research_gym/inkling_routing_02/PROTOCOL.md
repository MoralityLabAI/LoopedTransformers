# INK-ROUTE-02 confirmation design and promotion gate

**Status** Development protocol. Registration and sealed scoring are blocked by the development effect and matched-sham gate. The earlier 1,800-root budget was conditional. There is no supported confirmation sample size for the measured effect.

## Primary estimand and controls

Use one independently sampled public prose document per registrable domain, with one eight-token prefix per document and no reuse of earlier document, domain, exact text or prompt. The statistical estimand is the equal-document mean of the coordinate-2027 zero-minus-sham difference in pass-3 K-channel block-3 attention weight at HF layer 12. For the one-sided analysis, multiply that difference by −1 so larger values indicate less K routing after target removal.

At the last prompt token, zero the previous-pass block-3 bank coordinate 2027. Set the physical intervention dose to the L2 displacement after the actual bank dtype conversion. Pair the zero arm with the mean of 16 distinct deterministic uniform non-target coordinates edited in the same signed direction, all within 1% realized dose. Sample four additional disjoint coordinates for placebo-versus-reference sham contrasts. Per-root coordinate draws are fixed and replayed. The four placebo contrasts calibrate sham variability using their largest absolute value per independent root.

Calibrate Q, K, V and MoE at each reader pass, disclose all eight development comparisons and measure how many roots change sign across choice of reference coordinate. Diagnose sensitivity by leaving out one reference sham at a time. Include a dense random-vector intervention that fixes coordinate 2027, an equal-dose rotation that fixes coordinate 2027 and preserves the bank norm, norm-preserving target removal and clean reader-output rescue. Norm-preserving target removal can have a different physical dose. Record the distinction in the raw measurements.

## Meaningful effect, power and decision

The inherited scientific minimum is a 0.10 absolute coefficient change. This changes the mixture weight of the previous-pass block-3 source by ten percentage points. Use the maximum of that minimum, instrument resolution and a 95%/95% upper tolerance bound on the root-level placebo maximum. With 60 independent roots, the sample maximum has confidence (1-0.95^{60}) for bounding the 95th percentile under the sampling assumption. The measurement resolution is the greater of 0.001 and the worst observed paired contrast replay discrepancy. The pilot's eight original roots and the two smoke prompts contribute zero units to this calibration.

Promotion requires valid native instrumentation on every enrolled development root, a resolved sham floor, and a development mean above the effective floor. Run the pinned EvalMath strict preflight before freezing a confirmation. Require a planned alternative supported by the development roots. Use a conservative development lower estimate and show sensitivity to its uncertainty. If the eligible alternative `a` is at or below the effective floor `d`, the proposed design has no supported sample size. When `a > d`, the one-sided bounded-effect power calculation at 80% power is

```text
n >= ceil(2 * (sqrt(log(m / alpha)) + sqrt(log(1 / 0.20)))^2 / (a - d)^2)
```

Here the root outcomes lie in [−1, 1], familywise alpha is 0.05, and the prospective family is one contrast `m = 1`. For the historical conditional values `a = 0.20`, `d = 0.10`, the bound rounds up to 1,800 roots. The observed development data currently invalidate those values as a planning basis.

On a qualifying sealed run, report the mean, the one-sided 95% finite-sample lower bound `mean - sqrt(2 * log(m / 0.05) / n)` clipped at −1, and the two-sided 95% descriptive interval from 10,000 percentile bootstrap draws of complete roots using seed 20271122. The primary gate succeeds only when the finite-sample lower bound exceeds the effective floor and all planned roots and validation gates pass. The threshold-tail test uses the one-sided Hoeffding bound. No multiplicity adjustment is required for the singleton registered family. The other seven channel/pass contrasts, log-probability effects, vector controls, position or writer analyses and rescue outcomes are descriptive. Failure to clear the bound yields an unresolved evidence classification. Equivalence would require its own prospective margin and valid interval.

## Freeze and access order

After a qualifying development preflight, commit the code and record hashes for the exact sampler, checkpoint, tokenizer, analysis, comparison family, floor, supported alternative, planned root count, resource placement, and all failure rules. Have a fresh split sampled independently with development-domain and document exclusions and seal its content behind a manifest commitment. Keep prompts and outcomes inaccessible to design work until the code and plan are frozen. Score the fixed cohort once. Replays, arms, tokens, positions and passes are nested within a document.

Any dose, reconstruction, rescue, restoration or software identity failure blocks promotion. Reconstruction is required below 1e-3. Retain invalid arms and failures, including incomplete runs, and report exclusions and reasons. Retain all final-token log probabilities, flips, and null log-probability effects. A log-probability change with magnitude at most 1e-4 nats is a null at the instrument's declared tolerance.
