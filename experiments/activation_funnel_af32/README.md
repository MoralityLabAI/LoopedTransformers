# AF-32: strict vs slack proof selection with two goal-visible root distractions

**Preregistered source-only experiment in LoopedTransformers.** The first protocol was committed at \`7493b184e350e572f8fee8b60a81f91114befd8d\` before AF-32 generator, ranker, or heldout scoring. AF-31's source-verified rule-selection and learned-only STOP failures remain intact.

## Source and motivation

AF-31 found that a hand-coded, backward public-graph ancestry controller completed all 96 reachable proofs per held depth under a shortest-plus-one action budget. This can hide a controller's single irrelevant firing. AF-32 introduces **two independently legal goal-visible distractions at the root**, while retaining supported inaccessible 2/3-cycles, and scores exact-shortest and shortest-plus-one proof budgets separately.

Each source split has **96 paired positive and negative graphs**, from two logical grammar motifs with physical relabelings. Within each pair, the physical atom/rule permutation, initial-known facts \`{0,1,2,3}\`, marked roles 6/13/14, whole directed degree profiles, root legal actions, and seven-anchor directed radius-two physical incidence are identical. Two unary Horn-rule premises change whether the GOAL is globally derivable. Independent physical-bitset and original host BFS confirm shortest positive proofs of exactly 5 or 6 firings, and unreachable paired negative goals.

Every reachable root has two distinct enabled nonproductive actions, one deriving logical atom 7 and one deriving logical atom 12. Both are connected to a visible path toward goal 14 through \`7∧10→13\`, \`12∧10→13\` and \`13→14\`. The source unsupported producer-supported cycle excludes atom 10 from real forward closure. The source proof oracle checks that selecting either distraction leaves shortest remaining cost unchanged; selecting **both** does too. Thus a controller taking one distraction needs at least \`d+1\` rule firings; taking both requires \`d+2\`. The [constructive strict-budget theorem](../../docs/research/AF32_STRICT_DISTRACTION_THEOREM.md) proves the corresponding general Horn bound.

TRAIN/DEV have 2-cycles and productive swap positions 1/2. Held depths 5 and 6 each contain 48 2-cycle/48 unseen 3-cycle pairs, with swap positions 1/2/3 at 32 pairs each. Full public graph hashes are split-disjoint and distinct from source AF-29/AF-30/AF-31 graphs.

## Frozen model and policy arms

The new eight-feature linear legal-action ranker retains AF-31's exact deterministic 500-update TRAIN-only softmax fit and fixed seeds \`17,29,43,61,73,89,103,131\`. It sees current physical known bits and publicly encoded Horn graph only. Its separate ten-feature logistic learned STOP head trains on TRAIN-only graph-state labels for 600 steps, and chooses a threshold from DEV-only roots. No learned-only policy calls exact symbolic closure or BFS during inference.

All \`*_CERT\` policies get a public full forward Horn closure certificate for negative STOP, including the learned \`RANK_CERT\`, same-feature untrained ranker, \`PATH_CERT\`, \`ANCESTRY_CERT\`, fixed-first-legal and seeded-random action controls. The closure work is charged once per episode. Public \`CLOSURE\`, \`BFS16\`, and \`BFS64\` are separate complete controllers. The trained \`LRANK_STOP\` policy uses its logistic termination decision with the same learned action ranker and **no symbolic certificate**.

All eight original immutable AF-23 **6,036-weight frozen checkpoint** instances are tested on these identical heldout physical graphs with native STOP, forced continuation, and exact negative certificate + original Direct legal rank. The original weights receive zero optimizer updates; real frozen model forwards are independently sampled against serialized prediction traces.

## Prospective scientific scores

Two scorer-only action budgets are registered:

- \`strict\`: maximum \`d\` positive legal rule firings for shortest proof length \`d\`, with two decisions for negative cases. A nonprogressing first rule makes proof completion impossible.
- \`slack\`: maximum \`d+1\` positive firings, matching AF-31. Both arms are evaluated on fresh source-identical policy episodes. No budget or true shortest distance is passed to a model.

The **primary** ranker gate is only the strict budget. To pass in a seed at both held depths, \`RANK_CERT\` must complete at least 60/96 reachable proofs, correctly initially STOP at least 80/96 unreachable goals and beat the strongest equal-certificate control including PATH and ANCESTRY by at least **10/96 positive proofs** with zero false accepted proofs. Six of eight distinct trained ranker seeds must pass both depths.

A **separate learned-only STOP gate** requires, under strict budget, at least 60/96 positive proofs, 60/96 correct negative initial STOP and no more than 10/96 false positive initial STOP in six of eight seeds. Full original Direct scores, all strict/slack inference work, per-case first-off-optimal action receipts and a 2/3-cycle by swap-position breakdown are preserved. The source-only first-optimal diagnostic uses independent BFS **after action selection**, never as an inference feature.

## Reproduction

Python 3.12 standard library supports the source/linear policy preflight:

\`\`\`bash
python -m unittest discover -s experiments/activation_funnel_af32/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af32.gym > af32_source_manifest.json
\`\`\`

One source-fitted seed, two-budget full episode roster and independent audit:

\`\`\`bash
python -m experiments.activation_funnel_af32.evaluate \
  --seed 17 --out out/af32-seed-17
python -m experiments.activation_funnel_af32.audit \
  --seed 17 --out out/af32-seed-17
\`\`\`

The frozen original AF-23 same-graph control additionally requires CPU Torch 2.5.1, NumPy and an exact SHA-verified original \`af23-branch-seed-17.zip\` in \`originals\`:

\`\`\`bash
python -m experiments.activation_funnel_af32.direct \
  --seed 17 --original-zip-dir originals \
  --out out/af32-direct-seed-17
python -m experiments.activation_funnel_af32.direct_audit \
  --seed 17 --original-zip-dir originals \
  --out out/af32-direct-seed-17
\`\`\`

The [source-locked CI](../../.github/workflows/activation-funnel-af32.yml) loads all eight original frozen AF-23 checkpoints by original artifact ID and ZIP hash, independently audits full source episodes, and aggregates both registered strict scientific gates plus their slack-budget controls. A complete source-qualified scientific FAIL remains a scientific FAIL even when CI engineering passes. Preserve the original AF-22 through AF-31 verdicts and all historical blocked CI runs.
