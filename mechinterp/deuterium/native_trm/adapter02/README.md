# DEUTERIUM-NATIVE-02: stock-last versus pass-indexed VPD

**Status: SOURCE-ONLY IMPLEMENTED; FRESH SCIENCE BLOCKED.** [Prospective design](PROTOCOL.md) was committed first as 0512c69d63a16b58fbcf838292d1394f4d8d18e3 on 19 November 2026. All newly registered teacher roots and data sources remain unopened. Pointwise equal-mask tests pass, including a regression for the original leaky importance range. The complete excluded-source optimizer schedule exceeds the fixed full-trajectory tolerance and retains `BLOCKED_UNMATCHED_OPTIMIZER`. The scientific source lock and independent scoring pipeline remain subsequent work.

## Motivation

The original four-pass recurrent VPD study used a stock optimizer whose importance network received only the last physical input of a weight-tied block invoked four times. [DEUTERIUM-ADAPTER-01](../adapter_audit/RESULTS_SOURCE_ONLY_20261119.md) independently verified one cached input, one repeated mask, and learned original CI masks differing from the final pass by 0.27–0.30 mean absolute magnitude at the earliest actual recurrent application.

The prior registered [native steering result](../stagec/recovery/README.md) remains **DEVELOPMENT_NO_REGISTERED_ADVANTAGE**, with the original across-old-teacher t interval crossing zero. This prospective study isolates whether changing the mask-source/application schedule improves held-out steering under matched training, rank-one physical dose and candidate search.

## Implemented source-only adapter

The current implementation remains **source-only**. It uses the pinned original VPD components and native four-pass recurrence on an excluded untrained 44K TRM at generator seed 61077. No teacher root or registered new data source has been opened.

- [pass_indexed.py](pass_indexed.py) captures all four clean pre-weight activations from one reused module and applies physical call-indexed component masks with a strict counter and hook cleanup. The same 16 factor matrices and one shared `block.chan_mix.2.weight` remain in place.
- [pgd_shared.py](pgd_shared.py) preserves the original stock PGD source shape `[batch,81,17]` for 16 factors plus one residual. It uses the pinned original PGD gradient-sign steps and KL denominator while interpolating one shared adversarial source with four clean-state CI floors. This keeps PGD source dimension matched across arms.
- [tests/test_pass_indexed.py](tests/test_pass_indexed.py) checks clean and masked native replay, four-call capture, the same CI network at each depth, full parameter and CI gradient equivalence under forced-equal masks, stock PGD gradient equivalence, normalized importance-loss scale, and a single AdamW optimizer-update equivalence test.
- [Source-only Actions](../../../../.github/workflows/deuterium-native02-source-only.yml) runs these tests on original pinned optimizer code and Python 3.12.8 CPU. All generated inputs use the excluded smoke seed 61077.

The equality mode in `source_only_objective(mask_policy="last_repeat")` is exclusively a numerical implementation check against stock. The prospective pass-aware policy uses `mask_policy="native_pass"`, which keeps separate clean-source CI masks on four repeated tied-weight applications. A successful source-only parity test establishes implementation validity and supports the next optimizer trainer stage; it cannot substitute for a new teacher-root scientific comparison.

## Full fitting schedule and present boundary

[source_schedule.py](source_schedule.py) executes the unchanged pinned upstream
optimizer on excluded untrained seed 61077. Stock, repeated-last-mask identity,
and native-pass arms each run 32 factor warmup updates and 64 main updates. The
receipt checks initial parameters, source batches, random states, shared PGD
draws, and the unchanged teacher, while retaining complete losses and gradients.
The original terminal evaluation gives 65 main loss evaluations and 67 PGD
initializations per arm.

The first [schedule receipt](results/source_schedule_20261119/schedule_receipt.json)
records a maximum stock-versus-identity loss difference of 0.000583649,
parameter difference of 0.000358582, and gradient difference of 0.000115089.
These exceed the unchanged 0.00002 full-trajectory tolerance. Warmup is exactly
equal, and the first main loss matches exactly. Later differences are compatible
with amplification of floating-point reductions through signed PGD updates;
that explanation remains a hypothesis. The pointwise tests and this stricter
full-trajectory check answer separate engineering questions.

The full-trajectory gate was added before any reserved source was opened. Its
failure is retained conservatively as `BLOCKED_UNMATCHED_OPTIMIZER`; it does not
establish that the mathematical objective or update budget changed. Further
source-only numerical diagnosis must precede a scientific release.

After source-only adapter tests and a separate frozen source-lock artifact, train the 12 registered new teacher roots and paired decomposition arms. Search 16 candidates under both signs for each arm and one shared random control bank, then evaluate first held-out original four-pass recurrent weights only after the predeclared release gate. Run an independent audit of every saved ten-class logit and root-level uncertainty.

The completed paper records this engineering limit alongside the previous
negative steering verdicts. A new experiment requires its own source freeze and
first-exposure receipts.
