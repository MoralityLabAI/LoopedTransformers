# LSA-02L: learned equality-search ordering, CPU developmental result

Date: 2026-11-24. Engineering/native proof gates PASS. Registered developmental learned-advantage gate FAIL.

Source: [PR #336](https://github.com/MoralityLabAI/LoopedTransformers/pull/336), evaluated commit dae88c2b1de843d777a67984fbfb0e26e2d598e2, [completed native workflow 45282924392](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45282924392), [artifact 14870402528](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45282924392/artifacts/14870402528). The three compatibility workflows also passed. Values below are from the actual native job's emitted manifest.

The original 64 training roots supplied 546 certified reference-action frames. The new 32 development systems are alpha-lineage-disjoint from all original shards. All model fits passed same-worker duplicate state-digest checks. Native Lean 4.19.0 certified and independently compiled 118 distinct successful paths. Original ordinary/OOD contents remained unopened by the evaluator.

| Planner | 16 expansions | 64 | 256 | 1024 |
| --- | ---: | ---: | ---: | ---: |
| Hamming greedy | 12/32 | 24/32 | 29/32 | 32/32 |
| Bidirectional BFS | 6/32 | 14/32 | 25/32 | 32/32 |
| Hamming A* | 8/32 | 12/32 | 15/32 | 26/32 |
| Edit A* | 7/32 | 12/32 | 14/32 | 25/32 |
| Positional A* | 9/32 | 12/32 | 16/32 | 27/32 |
| Pair-projection A* | 10/32 | 14/32 | 22/32 | 30/32 |
| Learned four-update | 5/64 | 16/64 | 26/64 | 36/64 |
| Learned single-update | 7/64 | 16/64 | 25/64 | 42/64 |
| Shuffled sham | 7/64 | 16/64 | 23/64 | 33/64 |

The two model seeds repeat the same 32 roots. At the preregistered 64 expansion cap the learned arm is 50 percentage points behind the strongest public control, with root-paired 95% bootstrap interval [-65.625, -34.375] pp. Learned minus sham and learned minus one-step are both zero, including zero learned-minus-sham differences in each seed. The registered gate fails. Exact unrestricted BFS closes 32/32, as does Hamming greedy by 1024 expansions.

At 64 expansions Hamming greedy uses mean 32.65625 expanded words, 183.875 enumerated edges and 0.840653 ms planner wall time. Learned four-step uses 55.03125 expanded words, 345.765625 edges, 185.6875 scored model states, 53.046875 model batches, and 8.064662 ms planner wall time (6.396192 ms inference/encoding). Pair-projection A* uses 42.4375 full-word expansions plus mean 104.71875 abstract setup expansions. The learned planner is about 9.6x slower than Hamming greedy at this cap. No computational or recurrence advantage is established.

The run manifest binds fresh-development SHA-256 afe98f4f6a8af265379d2a2f2c82477497727a07512a5aecf16e75e290da6444, train-source SHA-256 1ccaa3ce7c0bd348dc38059b06628216cf7704ed030af9511ec48e08ae8f8d38, and per-root results SHA-256 a17346bfdb64f6941c74852451d12d3400fbd60bf3ea1ade7fc0f87f0c681632. The original suite SHA-256 is d928d8b5ffbbd9760e02fc9af903704bdac0cea7adc0b6e5be9103e7f2153874. The exact reference path corpus supplies limited off-trajectory ranking evidence; the absolute learned score also changes the successful greedy prior and uses a different priority rule. These are hypotheses about the failure, not measured causal explanations. The next experiment should isolate residual ordering under matched greedy search and expand native-certified supervision to off-path sibling states.

Known reporting limitations: per-cap edge thresholds in the emitted summary are retrospective filters at each expansion cap; they are not independent edge-budget executions. Proof/replay time is recorded per unique certificate, with logical tactic counts per planner. No sealed or architecture-promotion eligibility.
