# AF-30: prospective legal-rule selection under matched global Horn reachability

**Official result:** \`RULE_SELECTION_TRANSFER_FAIL\` (0/8 source training seeds). Source and independent proof audit PASS across **27,648** complete full-policy episodes. The learned ranker and the hand-coded, equal-certificate PATH controller both solve **96/96** positive goals and correctly stop on **96/96** negatives at both held depths in every seed. PATH eliminates the required learned +10/96 margin.

See [the full report](../../docs/research/AF30_RESULTS.md), [the constructive PATH theorem](../../docs/research/AF30_BLOCKED_PATH_THEOREM.md), [immutable source evidence](../../docs/research/AF30_EVIDENCE_MANIFEST.json), and [scored CI #44028290737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44028290737).

## Prospective protocol

First frozen registration: \`78a5b1e\` before AF-30 source generator, ranker, and model scoring. Two dated pre-scoring amendments in [PROTOCOL.json](PROTOCOL.json) document the source-only structural template repair and the stronger \`PATH_CERT\` same-feature baseline. The original scientific thresholds remain 6/8 seeds, 60/96 positive complete proofs and 80/96 correct immediate negative stops at both depths, plus a 10/96 positive-proof gain over the best same-certificate nonlearned control.

There are 96 pairs in each of TRAIN, DEV, held depth 5, and held depth 6. TRAIN/DEV swap positions 1 and 2, while each held cohort balances positions 1, 2 and 3 at 32 pairs apiece. Each pair changes exactly two Horn premises while matching every root/role directed radius-two incidence, public root legal mask, physical node/rule permutation and directed degree profile. Source Python host search and independent public-physical bitset BFS must agree on each label and exact positive depth before any source model fit.

## What the policies do

The learned \`RANK_CERT\` controller is an eight-feature, 500-update, TRAIN-only listwise linear ranker selecting among currently legal *physical rule slots*. No original 6,036-weight AF-23 recurrent network is loaded or optimized in AF-30. It has a **once-per-episode exact public fixed-point certificate** for immediate negative STOP. This is a **symbolic hybrid**, and all certificate/reverse-feature work is included in the resource count. Identical exact negative certificates wrap the equal-feature untrained linear baseline and hand-coded path-cost/ancestry/fixed/random rule baselines.

The PATH control uses the same reverse blocked-premise feature already available to RANK. Its negative STOP is likewise certified by public symbolic computation. Public closure and BFS16/BFS64 are separate symbolic controls. The first proof-producing policy is selected independently at each reached host state, with only the encoded public physical graph/current known bits as model inputs; no source metadata, label or scorer-only proof budget enters policy code.

## Source reproduction

Requires Python 3.12 standard library only for this experiment; no GPU, NumPy or Torch. Source-only and training smoke:

\`\`\`bash
python -m unittest discover -s experiments/activation_funnel_af30/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af30.gym > af30_source_manifest.json
\`\`\`

One source-frozen seed's **complete** independently audited experimental run:

\`\`\`bash
python -m experiments.activation_funnel_af30.evaluate --seed 17 --out out/af30-seed-17
python -m experiments.activation_funnel_af30.audit --seed 17 --out out/af30-seed-17
\`\`\`

All eight \`SEEDS=(17,29,43,61,73,89,103,131)\` have independently trained, full-episode, audited output directories. Aggregate using the eight original \`af30-seed-*\` result folders from the scored CI, under one root:

\`\`\`bash
python -m experiments.activation_funnel_af30.aggregate --root receipts --out af30_results.json
\`\`\`

The [source-scored workflow](../../.github/workflows/activation-funnel-af30.yml) performs the source lock, eight independently initialized fits/heldout policy evaluations, full physical replay, exact fit receipt check and final seed-level gate aggregation. The [preflight](../../.github/workflows/activation-funnel-af30-preflight.yml) performs source-only paired-structure/BFS tests and ranker smoke. To reproduce the official science decision, compare SHA-256 receipts against the committed manifest, since later changes to Python source or environment could alter results.

## Scientific scope

The correct negative STOPs belong to the exact public closure certificate. The recurrent AF-23 original Direct scorer was not evaluated on this fresh AF-30 cohort. The eight independent linear optimizer seeds share the same encoded held pairs; these are seed-repeat counts, not eight separately generated holdout grammar rosters.

The strong PATH baseline reveals why both RANK and PATH reach 100% on this gym: the misleading goal shortcuts all need producerless atom 10. Reverse cost with a missing-premise penalty above 3 always prefers the productive chain. The next genuinely discriminating source distribution should contain producer-supported but unreachable cyclic dependencies, frozen with the same-feature PATH and closure controls before model scoring. Earlier AF-22 through AF-29 scientific verdicts are preserved; AF-30 remains draft and unmerged.
