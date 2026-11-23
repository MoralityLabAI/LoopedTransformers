# LBF-09: Native multiple-shortest proof-branch supervision

**Registered 2026-11-23, before LBF-09 source collection, model training or native evaluation.**
**Status:** development-only, frozen-source intervention; locked ordinary/OOD tests remain unopened.

## Hypothesis

The LBF-08 model-visited source had **138 witnessed visit frames and zero multi-positive frames**. Its single-action and full-action-set learners saw the same labels, making the action-set hypothesis untested. LBF-09 changes the *theorem source*, rather than modifying verifier labels retrospectively, to create genuinely distinct but equally short tactic paths.

Two source-based causes can be tested:

- With actual multiple shortest continuations in a public Lean action menu, a ranking loss that credits the full action set should reduce arbitrary one-target pressure.
- That training change should improve full native kernel/standalone certified proof closure on a newly frozen development calibration suite.

## Registered theorem-source construction

Extend only the **optional** native-CGRT Horn generator argument \`equal_alt_p\`. Its default must be **0.0**, preserving legacy source theorems, tactic grammar and random stream exactly. For the LBF-09 suite, fix \`equal_alt_p=1.0\` with generator seed **20261130**, 64 roots per split.

After the existing source theorem is constructed, and after its public hypotheses have been alpha-renamed, enumerate original shortest *solution* rules. For each solution rule \`r: (premises)->H\` with finite minimum cost \`d(H)=1+sum d(premises)\`, choose its first premise \`P\` and build a new, distinct proposition \`P'\` with **the same finite native proof cost**:
- When P has a native fact, create a separately named native fact for P'.
- Otherwise copy a shortest-producing source rule for P to a fresh P', retaining its premises and step cost.
Then append \`r': (P', remaining premises)->H\` as another producer of H, with the same shortest cost. This creates an actual alternate proof branch with a distinct intermediate proposition, rather than simply duplicating a rule binder. Add one duplicate named fact hypothesis for each original source fact, creating a second one-step legal proof at leaves. Every appended rule and fact remains a genuine Lean hypothesis subject to the normal kernel. If an original rule fails the exact finite-shortest condition, exclude it from augmentation using a fixed source rule rather than inspecting model outcomes. Track every change in generator metadata and source hashes.

**Why equal-cost:** With source minimum \`d(P)\`, its new twin has \`d(P')=d(P)\`. Therefore \`1+d(P')+sum other premise costs = d(H)\`. Two public tactic names become distinct successful first moves and are credited only when actual Lean expansion confirms their successors and native shortest costs. Tactics on unsound or unreachable branches get zero, never model-positive labels.

The generator must continue producing shard-disjoint graph fingerprints and stable original \`equal_alt_p=0\` cases. The source adapter binds the alternate flag into the frozen suite identity, output schema, model query proofs and generator digest. Do not alter the closed ordinary-test or OOD shards after freezing.

## Evidence / exposure gate

Native Lean 4.19.0 must independently compile **48 train-root proofs**. Use the LBF-08 real-Lean model visitation collector on those train roots after a frozen seed-107, 360-step source-only behavior checkpoint. Require model-free replay of all behavior episodes, and independent standalone recompilation of teacher-forced complete recoveries. Keep autonomous closures distinct from teacher assistance.

Before training any LBF-09 comparison arm, audit source frames and **actual model-visited native frames**:

1. At least **48/48** train-root reference proofs kernel and standalone certified.
2. At least **12 independent train roots** with witnessed model-visited frames.
3. **At least 25% of witnessed model-visited training frames have two or more distinct shortest native actions** in the public candidate menu.
4. At least 12 independent visited root lineages contain at least one multi-optimal model-visited frame.
5. No labeling source or model-facing packet reads future proof verdicts. Optimal-action labels derive from native AND/OR table after model visitation and source compilation.
6. The final model-visited frame roster, exposure fractions, public-menu hashes and censored counts are emitted as immutable JSONL and manifest SHA-256.

If any threshold fails, report **INSUFFICIENT_MULTI_OPTIMAL_EXPOSURE**, retain all source and replay receipts, and stop before training/evaluation. Do not relax a denominator or replace a theorem root after seeing results.

## Fixed matched learner campaign

Reuse the CPU-only PPT-B / Lean4.19 pipeline registered in LBF-08, with **seed 107 and 109** and matched 720-step FP32 AdamW fits for:

- \`source_only\`: native-compiled teacher reference states, full shortest native action sets.
- \`visited_one\`: the **same model-visited** public packets, credit only the menu-first native shortest action.
- \`visited_set\`: the **identical** model-visited public packets, credit every native shortest continuation.

Same 57,475 unique parameters, width 32, 4 heads, 8 typed slots, 384-byte \`hybrid_rule_intact\` input, AB middle-block schedule, two recurrent passes, LR .001, one deterministic PyTorch CPU thread, identical root/frame draws and 720 updates per model. Repeat the seed-107 source-only fit and require exact named-tensor and optimizer-draw equivalence on the same physical worker.

An additional hard gate requires **at least 50 optimizer updates with >=2 positive model-visited actions per seed** in the \`visited_set\` arm. Both visited arms must have identical model-visited draw indices and counts, while the \`visited_one\` arm must have exactly one positive label. The source-only arm receives no model-visited updates.

## Native proof test / statistics

Use **32 new independent calibration roots** from the frozen seed-20261130 suite, 8 tactic checks/root and depth 2, with real Lean terminal-kernel validation, independent standalone proof compilation and model-free replay. The same roots, candidate grammar, source-trained model family and check budgets apply across arms. No evaluation-root optimizer updates.

Primary registered development effect: \`visited_set - visited_one\` **+0.05 absolute root-paired kernel/standalone completion**. Secondary \`visited_set - source_only\` threshold **+0.05**. Compute 2,000 descriptive root-cluster bootstrap draws with seed 20261123, averaging the two seed contrasts inside each theorem root. Report wins/losses/ties, actual tactic/forward calls, verified-native teacher reference ceiling and public-order auto8.

The experiment remains **development-only**, even on positive results. Population architecture claims would additionally need a competent model (>=40% proof closure), >=10 points versus a separately registered exact candidate-uniform native baseline, an independently powered fresh cohort and an EvalMath FSI-01 matched-sham preflight.

## Visual / measurement contract

LBF fresco panels may show state/plan/message light/glyph changes over recurrent passes, but all proof credit is in separate native Lean receipts. Record model-side zero/roll cross-pass perturbations on the frozen best arm, without calling Lean during rendering. Distinguish learned proof completion, native source oracle assistance and synthetic source-pattern guarantees explicitly.

## Deliverables

- Lean-CGRT optional equal-cost branch source generator with exact default-preservation and multi-shortest unit tests.
- Frozen multi-shortest source suite and source-only teacher compiler receipts.
- Hash-bound native model-visited / oracle-repair corpus, independently replayed model episodes, labels, censored cases and forced-recovery proof receipts.
- Pretraining **multi-optimal exposure audit** with hard stop rule.
- Six matched 720-step CPU fits, deterministic duplicate, native 32-root proof/standalone evaluation and root-paired comparator.
- Markdown research report, generated SVG fresco, full machine-readable CI artifacts.
