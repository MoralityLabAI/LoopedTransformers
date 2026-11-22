# NEC-04E: source-only 12-visit native Looper parity-memory readout

**Prospective source-only hidden-state probe protocol, 22 November 2026.** Parent [issue #302](https://github.com/MoralityLabAI/LoopedTransformers/issues/302). Original native training completed in [NEC-04C #296](https://github.com/MoralityLabAI/LoopedTransformers/issues/296). Read-only final-output spectral diagnostic completed in [NEC-04D #300](https://github.com/MoralityLabAI/LoopedTransformers/issues/300). Freeze this protocol, new source code, fixed controls and NEC04E_PROBE_LOCK.json **before any new hidden-state inference**.

## Original world identities and frozen checkpoints

Use original NEC04C Actions run [44993859839](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44993859839), old pretraining oracle artifact ID **14742886436**, and the six original root/model archives IDs **14742124547, 14743106877, 14743221555, 14742897690, 14743391255, 14742099591**. Exactly the six already-scored NEC04A CALIBRATION world roots **6–11**, two original trained Looper arms (vanilla, parity_curriculum), and two native optimizer budgets (240, 900), for 24 existing native model checkpoints. Source = original 80 complete context groups ×32 masks (2,560 histories per root). Owned validation = original 24 complete context groups ×32 masks (768 histories per root), within the same six calibration worlds. Reserved context groups stay unopened.

Original source SHA256 is 82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0. Original world registry SHA256 is 88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e. Original native pilot source commit was b6e7a9a2e2b530e738967aa15d37c265643e98b4. Every pre-existing native optimizer checkpoint file, previous independent native logit-replay receipt, original source and owned-validation context/label SHA and NEC04C training-code hash must verify before any new inference. This pre-inference phase reuses the pure-stdlib NEC04D historical independent archive auditor, records the entire current NEC04E code-file SHA set and the exact current Git commit, and seals that receipt for all root workers.

## Native recurrent states and exact old output checks

From each original frozen 32D, 12-loop PEF07 Looper, collect h1..h12 by applying the exact original Looper.step and bit_embedding chronologically once per input bit, starting at its frozen original memory_init. Use deterministic single-thread Torch 2.10.0 CPU, no_grad and original source plus owned-validation inputs only. Check that original Looper.memory yields h12 and original model.head(h12) recovers every saved two-class native logit with an absolute tolerance of at most **3e-4**. Match all 24 checkpoint file SHA256s, original Looper model weight SHA256s and native optimizer update counters. Compute same-seed random initial model hidden states and check their original initial-weight SHA. No new task-model optimizer steps are authorized.

Use the original outer MUX selector = 1 to restrict *both source fit and owned calibration-world validation* to complete original 32-mask five-focal parity context blocks. Original source root ownership stays fixed, without model-score-driven sample selection.

## Fixed linear readout conditions

For each original root, native arm, original checkpoint and visit 1..12, fit these four readouts, with one fixed **source-only** ridge recipe: trained_hidden (32D native memory), random_initial_hidden (same-seed untrained Looper), visible_raw_bits (original input positions strictly before the current visit), and source_shuffled_trained_hidden (trained memory with deterministic SHA256-seeded shuffle of the source training labels only). Source-fit features are centered and standardized using *source parity selector contexts alone*, with standard deviation floor 1e-4. Ridge lambda = **1.0**, intercept unpenalized, class threshold 0.5. Evaluate on the previously owned calibration-world parity contexts after fitting. Do not tune ridge penalty, seed, feature scaling, source/validation allocation or visit from measured validation accuracy. The shuffle seed domain is NEC04E/source-only-label-shuffle/v1, keyed only by root, visit and target.

For each original recurrent visit, derive the following labels exactly from the original five focal bit positions and original composed Boolean teacher:

- prefix1: first focal input bit already seen, available once one focal bit has arrived
- prefix2: XOR of first two observed focal input bits, available once two have arrived
- prefix_current: XOR of every focal bit observed so far, available once at least two have arrived
- future_unseen: XOR of focal bits yet to be observed, where any remain, a future-information negative control
- full_parity_teacher: original frozen composed Boolean target on selector=1 contexts, at every visit, including before all required bits are observed

Each target is exactly 50% balanced on every complete original 32-mask context block. Record all 12 recurrent visits, roots, arms, checkpoint budgets and targets, source fit correctness (and true source-label correctness for shuffled controls), owned validation correctness, balanced accuracy, majority baselines, prediction hashes and coefficients. The independent auditor refits each original source-only readout with separately constructed labels and checks predictions and original hidden-state/architecture hashes. A checkpoint readout audit is **not independent from-scratch native optimizer retraining**.

## Predeclared root-level descriptive diagnostic

Main readout: **900-update parity_curriculum model, prefix_current at the first chronological visit after the fifth focal bit has arrived**. That visit is determined only by the original focal layout: max(original five focal positions)+1. For one calibration world to qualify: trained 32D hidden readout owned validation accuracy **at least 70%**, and advantage **at least 10 percentage points** over the better of that same root's matched random-initial hidden and visible-bit linear readouts. A positive six-root pilot additionally requires **at least four of six** qualifying original calibration roots, six-root macro hidden accuracy at least 70%, and six-root macro advantage over the best raw/random control at least +10pp. Vanilla models, 240-update checkpoints, other visits and final original target at visit12 are all registered descriptive secondary observations. Report low-order positive controls and future-target negative controls.

All inference is **within calibration-world-owned contexts**, statistical unit one original world root, n=6. Correlated focal masks/contexts, probe choices and multiple visits do not create independent inference units. Detecting source-fitted linear parity information suggests a readout/optimization bottleneck; a negative probe alone cannot establish absence of nonlinear internal parity, limited capacity or population-level unlearnability.

## Protection and stop

The separate 24 new NEC04A developmental WORLD roots and 16 confirmation WORLD roots remain untouched, as do every calibration reserved context, old NEC03Z source groups56–63, NEC03P recruited-world causal outcomes, original embargoes and four-bit confirmation worlds. No task-model optimizer updates, source SAE fitting, sham interventions, causal-neuron necessity scores, or unseen world-root model outcomes. The native NEC04B and NEC04C headroom/necessity stop gates remain in force. **FSI-01: REDESIGN_OR_PILOT_ONLY**.

The dedicated workflow is .github/workflows/nec04e-hidden-probe.yml. It must independently audit the historical checkpoint and the source-only replay, run all six root matrix workers, and aggregate all 24 native checkpoints and every readout/negative-control result before claiming a prospective diagnostic outcome.
